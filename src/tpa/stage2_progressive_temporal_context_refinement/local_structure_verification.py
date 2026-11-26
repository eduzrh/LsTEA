# -*- coding: utf-8 -*-
"""Stage II-4: Local Structure Verification (report Section III-C4, A-B7).

For each temporal/combined-route query the module extracts the h-hop ego
network from the retained events Theta_ij^(t) and builds the
D_sig = 13-dimensional signature s(e):

  [0:4]   temporal activity: length-weighted histogram over four equal-width
          bins of normalized slot positions (q-1)/max(1,N-1) in [0,1]
  [4:8]   incoming/outgoing retained-event degree means and population
          variances across the ego-network vertices
  [8:11]  relation usage: normalized Shannon entropy, head-diverse instance
          fraction, distinct-to-total relation ratio |R_h| / |Q_h|
  [11:13] minimum and mean shortest directed hop distances from e to the
          forward leaves reachable within 1..h retained-event hops

Using the diagonal IDF weight matrix W_ij (Eq. 34)

    w_d^ij = log(1 + |E_sig,ij| / max(1, df_d^ij)),

with df_d the number of candidate signatures with positive d-th feature,
the verification score is the IDF-weighted cosine

    SV_ij(e,e') = s(e)^T W s(e') / max(eps_norm, nu(e) nu(e')),         (7)

where nu(v)^2 = s(v)^T W s(v).  Candidates with SV >= tau_struct form
C5(e); structural-route candidates bypass this filter.
"""
import math
from collections import defaultdict, deque

import numpy as np

from ..stage1_time_aware_graph_partition.unified_temporal_profile import (
    histogram_over_support,
)

D_SIG = 13
EPS_NORM = 1e-8


class ThetaGraph:
    """Retained event graph of one partition pair, prepared once per pair."""

    def __init__(self, theta, rel_of_fact, head_div):
        self.theta = theta
        self.rel_of_fact = rel_of_fact
        self.head_div = head_div
        self.adj = defaultdict(set)
        self.out_edges = defaultdict(list)
        self.ego_cache = {}
        for (side, u, v, lo, hi, fi) in theta:
            a, b = (side, u), (side, v)
            self.adj[a].add(b)
            self.adj[b].add(a)
            self.out_edges[a].append((side, v, lo, hi, fi))
        self.edges_by_vertex = defaultdict(list)
        for ev in theta:
            side, u, v, _lo, _hi, _fi = ev
            self.edges_by_vertex[(side, u)].append(ev)
            self.edges_by_vertex[(side, v)].append(ev)

    def ego(self, root, h=2):
        """Retained facts incident to vertices at distance < h from ``root``.

        Appendix A-B7 defines Q_h(e, Theta) over vertices at undirected
        distance less than h (h = 2 throughout), not at distance at most h.
        """
        key = (root, h)
        if key in self.ego_cache:
            return self.ego_cache[key]
        if root not in self.adj:
            self.ego_cache[key] = []
            return []
        dist = {root: 0}
        queue = deque([root])
        while queue:
            x = queue.popleft()
            if dist[x] >= h - 1:
                continue
            for y in self.adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    queue.append(y)
        verts = {v for v, d in dist.items() if d < h}
        facts = [ev for v in verts for ev in self.edges_by_vertex.get(v, ())
                 if (ev[0], ev[1]) in verts and (ev[0], ev[2]) in verts]
        # deduplicate facts collected from both endpoints
        seen, unique = set(), []
        for ev in facts:
            if ev not in seen:
                seen.add(ev)
                unique.append(ev)
        self.ego_cache[key] = unique
        return unique

    # ------------------------------------------------------------------
    def _histogram(self, ego, n_bins, n_slots):
        support = [(lo, hi) for (_s, _u, _v, lo, hi, _f) in ego]
        weights = histogram_over_support(
            np.asarray(support, dtype=np.int32) if support
            else np.empty((0, 2), dtype=np.int32), n_bins, n_slots)
        return weights

    @staticmethod
    def _degree_stats(ego):
        out_deg = defaultdict(float)
        in_deg = defaultdict(float)
        verts = set()
        for (side, u, v, lo, hi, _fi) in ego:
            a, b = (side, u), (side, v)
            verts.add(a)
            verts.add(b)
            out_deg[a] += hi - lo
            in_deg[b] += hi - lo
        if not verts:
            return np.zeros(4, dtype=np.float64)
        ins = np.array([in_deg.get(v, 0.0) for v in verts])
        outs = np.array([out_deg.get(v, 0.0) for v in verts])
        return np.array([ins.mean(), outs.mean(), ins.var(), outs.var()])

    def _relation_stats(self, ego):
        freq = defaultdict(float)
        total = 0.0
        diverse = 0.0
        for (side, _u, _v, lo, hi, fi) in ego:
            r = self.rel_of_fact[side][fi]
            w = float(hi - lo)
            freq[r] += w
            total += w
            if self.head_div[side].get(r, 0.0) >= 0.5:
                diverse += w
        if total <= 0:
            return np.zeros(3, dtype=np.float64)
        rhs = len(freq)
        if rhs <= 1:
            entropy = 0.0
        else:
            probs = [c / total for c in freq.values()]
            entropy = -sum(p * math.log(p) for p in probs) / math.log(rhs)
        return np.array([entropy, diverse / total, rhs / total],
                        dtype=np.float64)

    def _forward_leaf_distances(self, root, ego, h):
        fwd = defaultdict(list)
        verts = set()
        for (side, u, v, _lo, _hi, _fi) in ego:
            a, b = (side, u), (side, v)
            fwd[a].append(b)
            verts.add(a)
            verts.add(b)
        if root not in verts:
            return 0.0, 0.0
        leaves = [v for v in verts if v != root and not fwd.get(v)]
        if not leaves:
            return 0.0, 0.0
        dist = {root: 0}
        queue = deque([root])
        while queue:
            x = queue.popleft()
            if dist[x] >= h:
                continue
            for y in fwd.get(x, ()):
                if y not in dist:
                    dist[y] = dist[x] + 1
                    queue.append(y)
        reachable = [dist[l] for l in leaves if l in dist]
        if not reachable:
            return 0.0, 0.0
        return float(min(reachable)), float(sum(reachable) / len(reachable))

    def signature(self, root, n_slots, h=2):
        """13-dimensional signature s(root) (Appendix A-B7)."""
        ego = self.ego(root, h)
        if not ego:
            return np.zeros(D_SIG, dtype=np.float64)
        hist = self._histogram(ego, 4, max(1, n_slots))
        hist = hist / max(1.0, hist.sum())
        deg = self._degree_stats(ego)
        rel = self._relation_stats(ego)
        dmin, dmean = self._forward_leaf_distances(root, ego, h)
        return np.concatenate([hist, deg, rel, np.array([dmin, dmean])])[:D_SIG]


# ----------------------------------------------------------------------
# stateless helpers kept for direct use on small inputs
def ego_facts(theta, root, h=2):
    graph = ThetaGraph(theta, {"s": {}, "t": {}}, {"s": {}, "t": {}})
    return graph.ego(root, h)


def full_signature(root, ego, rel_of_fact, head_div, n_slots, h=2):
    graph = ThetaGraph(ego, rel_of_fact, head_div)
    return graph.signature(root, n_slots, h)


def idf_weights(sig_map):
    """Diagonal W_ij from the deduplicated candidate signature set (Eq. 34)."""
    ents = list(sig_map.keys())
    n = len(ents)
    if n == 0:
        return np.ones(D_SIG, dtype=np.float64)
    S = np.stack([sig_map[e] for e in ents])
    df = (S > 0).sum(axis=0)
    return np.log(1.0 + n / np.maximum(1, df))


def verify_score(s1, s2, w, eps_norm=EPS_NORM):
    """IDF-weighted cosine reading of Eq. (7)."""
    num = float(s1 @ (w * s2))
    nu1 = float(s1 @ (w * s1))
    nu2 = float(s2 @ (w * s2))
    denom = max(eps_norm, math.sqrt(max(0.0, nu1) * max(0.0, nu2)))
    if denom <= eps_norm:
        return 0.0
    return num / denom

# ----------------------------------------------------------------------
# Paper routine names (Algorithm 3, Appendix A-B7)
# ----------------------------------------------------------------------
Signatures = full_signature
