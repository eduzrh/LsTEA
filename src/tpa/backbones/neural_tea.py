# -*- coding: utf-8 -*-
"""Neural TEA backbones (GCN-Align, TEA-GNN, RREA, TREA, Dual-AMN).

Report reference: Section III-D1 (Eq. 9), Appendix A-B9 (Eqs. 35-39),
Section IV-A3 and Appendix Appendix C-B for the backbone adaptation.

All neural backbones share the TPA training objective

    L_TEA = sum_{k in B_a} ( L_align(S_k^{a,(t)}; Psi)
                             + beta * L_con(H_k; Psi) ),                 (9)

with the margin-ranking alignment loss of Eq. (36) and the cross-KG
contrastive regularizer of Eq. (39); xi_1 = xi_2 = 0.5 gives beta = 1.0.
The backbones differ in their message passing:

  GCN-Align : two-layer relational GCN, cosine scoring.
  TEA-GNN   : temporal attention over timestamped facts (time embeddings).
  RREA      : relational reflection transforms in aggregation.
  TREA      : temporal-relational attention on (relation, time) pairs.
  Dual-AMN  : dual node/relation attention with hard negative mining.

Timestamp input follows Appendix C-B: TEA-GNN/TREA/Dual-AMN consume the
calibrated fact time through their temporal encoders; GCN-Align and RREA
receive the unified-profile features appended to the vertex input.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .subtask import SubTaskGraph, TIME_VOCAB


class RelationalGCNLayer(nn.Module):
    """One relational message-passing layer with backbone-specific messages."""

    def __init__(self, hidden, n_rel, variant="gcn_align", time_dim=16):
        super().__init__()
        self.variant = variant
        self.n_rel = max(1, n_rel)
        self.rel_emb = nn.Embedding(self.n_rel, hidden)
        self.time_emb = nn.Embedding(TIME_VOCAB, time_dim)
        self.lin = nn.Linear(hidden, hidden)
        if variant == "rrea":
            self.reflect = nn.Parameter(torch.randn(self.n_rel, hidden) * 0.1)
        if variant in ("tea_gnn", "trea", "dual_amn"):
            self.attn = nn.Linear(hidden + time_dim, 1)
        if variant == "dual_amn":
            self.rel_attn = nn.Linear(hidden + time_dim, hidden)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, edge_type, edge_time):
        if edge_index.shape[1] == 0:
            return x
        h, t = edge_index[0], edge_index[1]
        rel = self.rel_emb(edge_type)
        msg = x[h] + rel
        if self.variant == "rrea":
            refl = self.reflect[edge_type]
            refl = refl / (refl.norm(dim=-1, keepdim=True) + 1e-8)
            msg = msg - 2.0 * (msg * refl).sum(-1, keepdim=True) * refl
        if self.variant in ("tea_gnn", "trea", "dual_amn"):
            tfeat = self.time_emb(edge_time)
            gate = torch.sigmoid(self.attn(torch.cat([msg, tfeat], -1)))
            msg = msg * gate
        if self.variant == "trea":
            gate_r = torch.sigmoid((msg * rel).sum(-1, keepdim=True))
            msg = msg * gate_r
        if self.variant == "dual_amn":
            rel_gate = torch.sigmoid(self.rel_attn(torch.cat([msg, tfeat], -1)))
            msg = msg * rel_gate
        agg = torch.zeros_like(x)
        agg.index_add_(0, t, msg)
        deg = torch.zeros(x.shape[0], device=x.device)
        deg.index_add_(0, t, torch.ones(msg.shape[0], device=x.device))
        agg = agg / deg.clamp(min=1).unsqueeze(-1)
        return self.dropout(F.relu(self.lin(agg + x)))


class TEAEncoder(nn.Module):
    """Shared two-layer encoder; ``variant`` selects the backbone."""

    def __init__(self, feat_dim, hidden, out_dim, n_rel, variant,
                 time_dim=16):
        super().__init__()
        self.variant = variant
        self.lin_in = nn.Linear(feat_dim, hidden)
        self.layer1 = RelationalGCNLayer(hidden, n_rel, variant, time_dim)
        self.layer2 = RelationalGCNLayer(hidden, n_rel, variant, time_dim)
        self.lin_out = nn.Linear(hidden, out_dim)

    def forward(self, g):
        x = F.relu(self.lin_in(g.x))
        x = self.layer1(x, g.edge_index, g.edge_type, g.edge_time)
        x = self.layer2(x, g.edge_index, g.edge_type, g.edge_time)
        return self.lin_out(x)


class NeuralTEABackbone:
    """Trainer / inference wrapper for the five neural backbones."""

    def __init__(self, data, variant="gcn_align", dim=128, lr=0.005,
                 margin=1.0, temperature=0.1, n_neg=10, device=None,
                 seed=0, loss_margin=True, beta=1.0, grad_ctx=32):
        self.data = data
        self.variant = variant
        self.dim = dim
        self.lr = lr
        self.margin = margin
        self.temperature = temperature
        self.n_neg = n_neg
        self.seed = seed
        self.beta = beta
        self.loss_margin = loss_margin
        self.grad_ctx = grad_ctx
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = None
        self.graphs = []
        self.ctx_list = []
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    def _build_graphs(self, contexts, src_profiles, tgt_profiles):
        self.graphs, self.ctx_list = [], []
        for k, ctx in enumerate(contexts):
            g = SubTaskGraph(ctx, src_profiles=src_profiles,
                             tgt_profiles=tgt_profiles, feat_dim=64,
                             seed=self.seed + k, n_slots=self.data.N)
            g.x = g.x.to(self.device)
            g.edge_index = g.edge_index.to(self.device)
            g.edge_type = g.edge_type.to(self.device)
            g.edge_time = g.edge_time.to(self.device)
            self.graphs.append(g)
            self.ctx_list.append(ctx)

    def initialize(self):
        n_rel = max(2, int(getattr(self.data, "n_rel", 2)))
        self.model = TEAEncoder(64, self.dim, self.dim, n_rel,
                                variant=self.variant).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, contexts, epochs=50, src_profiles=None, tgt_profiles=None,
            log=print):
        if not contexts:
            return
        self._build_graphs(contexts, src_profiles, tgt_profiles)
        if self.model is None:
            self.initialize()
        train_ids = [k for k, c in enumerate(contexts) if c["seeds"]]
        if not train_ids:
            return
        self.model.train()
        for ep in range(epochs):
            self.opt.zero_grad()
            total, n_ctx = 0.0, 0
            for start in range(0, len(train_ids), self.grad_ctx):
                chunk = train_ids[start:start + self.grad_ctx]
                losses = []
                for k in chunk:
                    g = self.graphs[k]
                    if not g.seed_local:
                        continue
                    y = self.model(g)
                    losses.append(self._loss(g, y))
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                loss.backward()
                total += float(loss.detach()) * len(chunk)
                n_ctx += len(chunk)
            self.opt.step()
            if ep % 10 == 0 or ep == epochs - 1:
                log(f"      epoch {ep}: loss={total / max(1, n_ctx):.4f}")

    # ------------------------------------------------------------------
    def _loss(self, g, y):
        """L_align + beta * L_con (Eqs. 36 and 39)."""
        pairs = torch.tensor(g.seed_local, dtype=torch.long, device=y.device)
        ys = y[pairs[:, 0]]
        yt = y[pairs[:, 1] + g.n_s]
        pos = F.cosine_similarity(ys, yt)

        seed_src = set(p[0] for p in g.seed_local)
        seed_tgt = set(p[1] for p in g.seed_local)
        free_src = [i for i in range(g.n_s) if i not in seed_src]
        free_tgt = [i for i in range(g.n_t) if i not in seed_tgt]
        if not free_src:
            free_src = list(range(g.n_s))
        if not free_tgt:
            free_tgt = list(range(g.n_t))
        B = pairs.shape[0]
        n = self.n_neg
        src_pick = torch.tensor(
            self.rng.choice(free_src, size=(B, n), replace=True),
            device=y.device)
        tgt_pick = torch.tensor(
            self.rng.choice(free_tgt, size=(B, n), replace=True),
            device=y.device)
        neg_src_cos = F.cosine_similarity(ys.unsqueeze(1),
                                          y[src_pick], dim=-1)
        neg_tgt_cos = F.cosine_similarity(yt.unsqueeze(1),
                                          y[tgt_pick + g.n_s], dim=-1)
        neg_cos = torch.cat([neg_src_cos, neg_tgt_cos], dim=1)
        if self.variant == "dual_amn":
            # normalized hard-sample mining (Dual-AMN)
            neg_cos = torch.topk(neg_cos, k=n, dim=1).values
        d_pos = 1.0 - pos
        d_neg = 1.0 - neg_cos
        align = F.relu(self.margin + d_pos.unsqueeze(1) - d_neg).mean()

        other = F.cosine_similarity(ys.unsqueeze(1), yt.unsqueeze(0), dim=-1)
        a_pos = pos / self.temperature
        a_other = other / self.temperature
        eye = torch.eye(B, dtype=torch.bool, device=y.device)
        con = torch.log(1.0 + torch.exp(a_other - a_pos.unsqueeze(1))
                        .masked_fill(eye, 0.0).sum(1)).mean()
        if not self.loss_margin:
            return con
        xi1, xi2 = 0.5, 0.5
        return xi1 * align + xi2 * self.beta * con

    # ------------------------------------------------------------------
    def embed(self, ctx):
        """Return {('s'|'t', entity): vector} for one context sub-task."""
        if self.model is None:
            return {}
        self.model.eval()
        target = None
        for g, c in zip(self.graphs, self.ctx_list):
            if c is ctx:
                target = g
                break
        if target is None:
            return {}
        with torch.no_grad():
            y = self.model(target).cpu().numpy()
        out = {}
        for e, i in target.src_ix.items():
            out[("s", e)] = y[i]
        for e, i in target.tgt_ix.items():
            out[("t", e)] = y[target.n_s + i]
        return out

    def state_dict(self):
        return {} if self.model is None else self.model.state_dict()