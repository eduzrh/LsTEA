# -*- coding: utf-8 -*-
"""Torch graph of one bounded context sub-task (joint vertex space).

Source vertices occupy local ids 0..n_s-1, target vertices occupy
n_s..n_s+n_t-1.  Facts of each TKG are mapped to their own side of the
joint vertex space:

    (h, r, t) in G^s  ->  edge (src_ix[h] -> src_ix[t], r, time)
    (h, r, t) in G^t  ->  edge (n_s+tgt_ix[h] -> n_s+tgt_ix[t], r, time)

Every fact is inserted in both directions for message passing; relation ids
stay global so the shared backbone keeps one relation table across all
sub-tasks.  ``edge_time`` is the calibrated start slot of the fact, mapped
into the backbone's time vocabulary (the timestamp input of Section IV-A3:
TEA-GNN and TREA consume it through temporal attention, the static
backbones receive profile features in the vertex input).

Initial vertex features combine degree statistics with a deterministic
hashing of the entity's (updated) temporal profile, which is the temporal
feature input described in Section III-D: seed endpoints supply their
updated source query profiles p^(t) in later iterations.
"""
import numpy as np
import torch

TIME_VOCAB = 4096


class SubTaskGraph:
    def __init__(self, ctx, src_profiles=None, tgt_profiles=None,
                 feat_dim=64, seed=0, n_slots=1):
        self.ctx = ctx
        self.src_list = sorted(ctx["Hs"])
        self.tgt_list = sorted(ctx["Ht"])
        self.src_ix = {e: i for i, e in enumerate(self.src_list)}
        self.tgt_ix = {e: i for i, e in enumerate(self.tgt_list)}
        self.n_s, self.n_t = len(self.src_list), len(self.tgt_list)
        self.n = self.n_s + self.n_t

        edges = []  # (local_head, local_tail, relation_id, time_id)
        for (h, r, t), iv in zip(ctx["facts_s"], ctx["intervals_s"]):
            a, b = self.src_ix.get(h), self.src_ix.get(t)
            if a is None or b is None:
                continue
            edges.append((a, b, int(r), _time_of(iv, n_slots)))
        for (h, r, t), iv in zip(ctx["facts_t"], ctx["intervals_t"]):
            a, b = self.tgt_ix.get(h), self.tgt_ix.get(t)
            if a is None or b is None:
                continue
            edges.append((self.n_s + a, self.n_s + b, int(r),
                          _time_of(iv, n_slots)))
        self.edges = edges
        self.n_rel = max([rel for (_a, _b, rel, _q) in edges], default=-1) + 1

        self.seed_local = [(self.src_ix[s], self.tgt_ix[t])
                           for (s, t) in ctx["seeds"]
                           if s in self.src_ix and t in self.tgt_ix]

        rng = np.random.RandomState(seed)
        base = rng.randn(self.n, feat_dim).astype(np.float32) * 0.1
        deg = np.zeros(self.n, dtype=np.float64)
        for (a, b, _r, _q) in edges:
            deg[a] += 1.0
            deg[b] += 1.0
        base[:, 0] = np.log1p(deg).astype(np.float32)
        for profiles, ix, offset in ((src_profiles, self.src_ix, 0),
                                     (tgt_profiles, self.tgt_ix, self.n_s)):
            if profiles is None:
                continue
            for e, i in ix.items():
                chain = profiles.get(e) if isinstance(profiles, dict) else (
                    profiles[e] if 0 <= e < len(profiles) else None)
                if chain is None or len(chain) == 0:
                    continue
                hsh = (chain.astype(np.int64) * 2654435761) % feat_dim
                np.add.at(base[offset + i], hsh, 1.0)
        self.x = torch.from_numpy(base)

        edge_index, edge_type, edge_time = [], [], []
        for (a, b, rel, q) in edges:
            edge_index.append((a, b))
            edge_type.append(rel)
            edge_time.append(q)
            edge_index.append((b, a))
            edge_type.append(rel)
            edge_time.append(q)
        if edge_index:
            self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            self.edge_type = torch.tensor(edge_type, dtype=torch.long)
            self.edge_time = torch.tensor(edge_time, dtype=torch.long)
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_type = torch.zeros((0,), dtype=torch.long)
            self.edge_time = torch.zeros((0,), dtype=torch.long)
        self.n_edge = self.edge_index.shape[1]


def _time_of(iv, n_slots):
    """Map a calibrated fact interval to a timestamp id in [0, TIME_VOCAB)."""
    if iv is None:
        return 0
    lo = max(0, int(iv[0]))
    if n_slots <= 1:
        return 0
    q = int(lo * (TIME_VOCAB - 1) / max(1, n_slots - 1))
    return min(TIME_VOCAB - 1, max(0, q))