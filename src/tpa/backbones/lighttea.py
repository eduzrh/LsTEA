# -*- coding: utf-8 -*-
"""LightTEA backbone (non-neural, three-view label propagation).

Report reference: Section III-D1 ("For the non-neural LightTEA backbone,
its label propagation on the same sub-tasks replaces this objective") and
Appendix C-B (three-view temporal propagation).

Faithful re-implementation of Cai et al. (IJCAI 2023, reference release):

  * seed pairs receive the same random feature vector, the only cross-KG
    link; the two TKGs are separate components of one joint graph;
  * entity-entity adjacency with fact-count edge weights and degree
    self-loops;
  * directional relation ids (2r forward / 2r+1 backward) and per-fact
    time ids (t_begin and t_end both generate incidence);
  * relation view: T_prop rounds of rel_ent / ent_rel propagation,
    concatenated and L2-normalised, then relation-aware neighbourhood
    aggregation;
  * temporal view: the same propagation with time ids;
  * final features = (1 - alpha) * relation_view + alpha * temporal_view.

Within TPA propagation runs on each bounded context sub-task and the outer
loop supplies the iterative high-confidence seed expansion.
"""
import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy is a declared dependency
    sp = None


def _l2(x, eps=1e-12):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def _rand_proj(rng, x, out_dim):
    v = _l2(rng.standard_normal((x.shape[1], out_dim)).astype(np.float32))
    return x @ v


class LightTEABackbone:
    def __init__(self, data, n_prop=2, alpha=0.6, ent_dim=256, mini_dim=16,
                 seed=0):
        self.data = data
        self.n_prop = n_prop
        self.alpha = alpha
        self.ent_dim = ent_dim
        self.mini_dim = mini_dim
        self.seed = seed
        self.graphs = []
        self.ctx_list = []

    # ------------------------------------------------------------------
    def fit(self, contexts, epochs=None, src_profiles=None,
            tgt_profiles=None, log=print):
        self.graphs = []
        self.ctx_list = contexts
        rng = np.random.RandomState(self.seed)
        for ctx in contexts:
            self.graphs.append(self._prepare(ctx, rng))
        log(f"      LightTEA label propagation on {len(contexts)} contexts")

    def _prepare(self, ctx, rng):
        src_list = sorted(ctx["Hs"])
        tgt_list = sorted(ctx["Ht"])
        src_ix = {e: i for i, e in enumerate(src_list)}
        tgt_ix = {e: i for i, e in enumerate(tgt_list)}
        n_s, n_t = len(src_list), len(tgt_list)
        n = n_s + n_t

        # ---- joint edge list (both directions), directional rel ids ----
        edges = []          # (h_local, t_local, rel_dir)
        for (h, r, t) in ctx["facts_s"]:
            a, b = src_ix.get(h), src_ix.get(t)
            if a is None or b is None:
                continue
            edges.append((a, b, 2 * r))
            edges.append((b, a, 2 * r + 1))
        for (h, r, t) in ctx["facts_t"]:
            a, b = tgt_ix.get(h), tgt_ix.get(t)
            if a is None or b is None:
                continue
            edges.append((n_s + a, n_s + b, 2 * r))
            edges.append((n_s + b, n_s + a, 2 * r + 1))

        # time incidence: t_begin and t_end both generate time ids
        time_of = {}
        for (h, r, t), iv in zip(ctx["facts_s"], ctx["intervals_s"]):
            a, b = src_ix.get(h), src_ix.get(t)
            if a is None or b is None or iv is None:
                continue
            time_of.setdefault((a, b), []).append(int(iv[0]))
            time_of.setdefault((a, b), []).append(int(iv[1]) - 1)
        for (h, r, t), iv in zip(ctx["facts_t"], ctx["intervals_t"]):
            a, b = tgt_ix.get(h), tgt_ix.get(t)
            if a is None or b is None or iv is None:
                continue
            time_of.setdefault((n_s + a, n_s + b), []).append(int(iv[0]))
            time_of.setdefault((n_s + a, n_s + b), []).append(int(iv[1]) - 1)
        time_ids = sorted({q for v in time_of.values() for q in v})
        time_ix = {t: i for i, t in enumerate(time_ids)}
        T = max(1, len(time_ix))

        # ---- ent_ent adjacency: fact counts + degree self-loops ----
        deg = np.zeros(n, dtype=np.float64)
        pair_cnt = {}
        for (h, t, r) in edges:
            deg[h] += 1.0
            pair_cnt[(h, t)] = pair_cnt.get((h, t), 0.0) + 1.0
        rows, cols, vals = [], [], []
        for (h, t), c in pair_cnt.items():
            rows.append(h)
            cols.append(t)
            vals.append(c)
        for i in range(n):
            rows.append(i)
            cols.append(i)
            vals.append(deg[i])
        ent_ent = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) \
            if rows else sp.csr_matrix((n, n))

        # ---- directional relation incidence ----
        R_dir = max(2, 2 * self.data.n_rel)
        rrows, rcols = [], []
        for (h, t, r) in edges:
            rrows.append(r)
            rcols.append(h)
        rel_ent = sp.csr_matrix((np.ones(len(rrows)), (rrows, rcols)),
                                shape=(R_dir, n)) if rrows \
            else sp.csr_matrix((R_dir, n))
        erows, ecols = [], []
        for (h, t, r) in edges:
            erows.append(t)
            ecols.append(r)
        ent_rel = sp.csr_matrix((np.ones(len(erows)), (erows, ecols)),
                                shape=(n, R_dir)) if erows \
            else sp.csr_matrix((n, R_dir))

        trows, tcols = [], []
        for (h, t), qs in time_of.items():
            for q in qs:
                trows.append(time_ix[q])
                tcols.append(h)
        time_ent = sp.csr_matrix((np.ones(len(trows)), (trows, tcols)),
                                 shape=(T, n)) if trows \
            else sp.csr_matrix((T, n))
        vrows, vcols = [], []
        for (h, t), qs in time_of.items():
            for q in qs:
                vrows.append(t)
                vcols.append(time_ix[q])
        ent_time = sp.csr_matrix((np.ones(len(vrows)), (vrows, vcols)),
                                 shape=(n, T)) if vrows \
            else sp.csr_matrix((n, T))

        # ---- edge grouping for relation-aware aggregation ----
        pair_rels = {}
        for (h, t, r) in edges:
            pair_rels.setdefault((h, t), []).append(r)
        eh, et, er = [], [], []
        for (h, t), rs in pair_rels.items():
            eh.append(h)
            et.append(t)
            er.append(rs)
        pair_times = {k: [time_ix[q] for q in v] for k, v in time_of.items()}
        th, tt, tr = [], [], []
        for (h, t), qs in pair_times.items():
            th.append(h)
            tt.append(t)
            tr.append(qs)

        # ---- seeded shared random features ----
        D = self.ent_dim
        feat = np.zeros((n, D), dtype=np.float32)
        seed_pairs = [(src_ix[s], n_s + tgt_ix[t]) for (s, t) in ctx["seeds"]
                      if s in src_ix and t in tgt_ix]
        if seed_pairs:
            rand = _l2(rng.standard_normal((len(seed_pairs), D)).astype(np.float32))
            for i, (a, b) in enumerate(seed_pairs):
                feat[a] = rand[i]
                feat[b] = rand[i]
        return {"n_s": n_s, "n_t": n_t, "n": n, "ent_ent": ent_ent,
                "rel_ent": rel_ent, "ent_rel": ent_rel,
                "time_ent": time_ent, "ent_time": ent_time,
                "feat": feat, "src_ix": src_ix, "tgt_ix": tgt_ix,
                "eh": np.asarray(eh, dtype=np.int64),
                "et": np.asarray(et, dtype=np.int64), "er": er,
                "th": np.asarray(th, dtype=np.int64),
                "tt": np.asarray(tt, dtype=np.int64), "tr": tr,
                "R_dir": R_dir, "T": T}

    # ------------------------------------------------------------------
    def _propagate(self, g, rng):
        n, D = g["n"], self.ent_dim
        ent_ent, rel_ent, ent_rel = g["ent_ent"], g["rel_ent"], g["ent_rel"]
        time_ent, ent_time = g["time_ent"], g["ent_time"]
        f0 = g["feat"]

        # ============ relation view ============
        ent, rel = f0, np.zeros((g["R_dir"], D), dtype=np.float32)
        ents, rels = [ent], [rel]
        for _ in range(self.n_prop):
            new_rel = _l2(rel_ent @ ent)
            new_ent = _l2(ent_ent @ ent + ent_rel @ rel)
            ent, rel = new_ent, new_rel
            ents.append(ent)
            rels.append(rel)
        ent_f = _l2(np.concatenate(ents, axis=-1))
        rel_f = _l2(np.concatenate(rels, axis=-1))
        d_rel = max(8, D // 2)
        rel_proj = _rand_proj(rng, rel_f, d_rel)
        proj = _rand_proj(rng, ent_f, self.mini_dim)
        feat_r = np.zeros((n, d_rel * self.mini_dim), dtype=np.float32)
        eh, et, er = g["eh"], g["et"], g["er"]
        if len(eh):
            edge_vec = np.zeros((len(eh), d_rel), dtype=np.float32)
            for i, rs in enumerate(er):
                edge_vec[i] = rel_proj[list(rs)].sum(axis=0)
            mini = self.mini_dim
            for d in range(d_rel):
                contrib = edge_vec[:, d:d + 1] * proj[et]
                np.add.at(feat_r[:, d * mini:(d + 1) * mini], eh, contrib)
        feat_r = _l2(feat_r)

        # ============ temporal view ============
        ent, tim = f0, np.zeros((g["T"], D), dtype=np.float32)
        ents, tims = [ent], [tim]
        for _ in range(self.n_prop):
            new_tim = _l2(time_ent @ ent)
            new_ent = _l2(ent_ent @ ent + ent_time @ tim)
            ent, tim = new_ent, new_tim
            ents.append(ent)
            tims.append(tim)
        ent_f2 = _l2(np.concatenate(ents, axis=-1))
        tim_f = _l2(np.concatenate(tims, axis=-1))
        d_tim = max(8, D // 2)
        tim_proj = _rand_proj(rng, tim_f, d_tim)
        proj2 = _rand_proj(rng, ent_f2, self.mini_dim)
        feat_t = np.zeros((n, d_tim * self.mini_dim), dtype=np.float32)
        th, tt, tr = g["th"], g["tt"], g["tr"]
        if len(th):
            edge_vec = np.zeros((len(th), d_tim), dtype=np.float32)
            for i, qs in enumerate(tr):
                edge_vec[i] = tim_proj[list(qs)].sum(axis=0)
            mini = self.mini_dim
            for d in range(d_tim):
                contrib = edge_vec[:, d:d + 1] * proj2[tt]
                np.add.at(feat_t[:, d * mini:(d + 1) * mini], th, contrib)
        feat_t = _l2(feat_t)

        y = (1 - self.alpha) * feat_r + self.alpha * feat_t
        return _l2(y)

    def embed(self, ctx):
        target = None
        for g, c in zip(self.graphs, self.ctx_list):
            if c is ctx:
                target = g
                break
        if target is None:
            return {}
        rng = np.random.RandomState(self.seed + 17)
        y = self._propagate(target, rng)
        out = {}
        for e, i in target["src_ix"].items():
            out[("s", e)] = y[i]
        for e, i in target["tgt_ix"].items():
            out[("t", e)] = y[target["n_s"] + i]
        return out

    def state_dict(self):
        return {}