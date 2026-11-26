# -*- coding: utf-8 -*-
"""Stage II: Progressive Temporal Context Refinement (report Algorithm 3).

For each retained partition pair (P_i, P_j) and iteration t:

  1. U = P_i \\ E^{s,a,(t)}                       (unaligned source queries)
  2. Route(e;i,j) selects tem / str / mix; Cascade supplies C3(e) and
     SeedVote supplies C_struct(e); C_route(e) is C3, C_struct, or their
     nonempty intersection with union fallback (Algorithm 2).
  3. Events builds the pair event graph, Fwd/Bwd retain Theta (skipped for
     pairs without seed pairs), and C4(e) = C_route(e) ∩ V(Theta).
  4. Signatures over Theta ego-networks and IDF-weighted verification
     against tau_struct yield C5(e).  Structural-route candidates bypass
     both filters.

The intra-partition event index of a pair is built once and reused by all
iterations; only the seed-dependent Fwd/Bwd closure is recomputed.
"""
import time
from collections import defaultdict

from . import cost_aware_routing as routing
from . import temporal_reachability_filter as trf
from . import local_structure_verification as lsv
from ..stage1_time_aware_graph_partition.relation_diversity import head_diversity


def _relation_sets(kg):
    rels = defaultdict(set)
    for h, r, t in kg.triples:
        rels[h].add(r)
        rels[t].add(r)
    return {e: tuple(v) for e, v in rels.items()}


def _relation_index(kg):
    idx = defaultdict(set)
    for h, r, t in kg.triples:
        idx[r].add(h)
        idx[r].add(t)
    return {r: tuple(v) for r, v in idx.items()}


def _undirected_adjacency(kg):
    adj = defaultdict(set)
    for h, _r, t in kg.triples:
        if h == t:
            continue
        adj[h].add(t)
        adj[t].add(h)
    return {k: tuple(v) for k, v in adj.items()}


class ProgressiveContextRefiner:
    """Per-pair refinement shared by all iterations (Algorithm 3)."""

    def __init__(self, data, stage_one, index, p_cal_s, p_cal_t, params):
        self.data = data
        self.so = stage_one
        self.index = index
        self.p_cal_s = p_cal_s
        self.p_cal_t = p_cal_t
        self.params = params
        self.src_adj = _undirected_adjacency(data.gs)
        self.tgt_adj = _undirected_adjacency(data.gt)
        self.rel_of_fact = {
            "s": [r for (_h, r, _t) in data.gs.triples],
            "t": [r for (_h, r, _t) in data.gt.triples],
        }
        self.head_div = {"s": head_diversity(data.gs),
                         "t": head_diversity(data.gt)}
        self.src_rels = _relation_sets(data.gs)
        self.tgt_rels = _relation_sets(data.gt)
        self.tgt_rel_index = _relation_index(data.gt)
        self._event_cache = {}
        self.timing = {"routing_cascade_s": 0.0, "batch_temporal_scan_s": 0.0,
                       "local_structure_verification_s": 0.0}

    # ------------------------------------------------------------------
    def _event_index(self, i, j):
        key = (i, j)
        cached = self._event_cache.get(key)
        if cached is not None:
            return cached
        facts_by_side = trf.build_pair_facts(
            self.data, self.so.src_members[i], self.so.tgt_members[j],
            self.data.N)
        index = {
            "facts": facts_by_side,
            "out": trf._out_adjacency(facts_by_side),
            "in": trf._in_adjacency(facts_by_side),
        }
        self._event_cache[key] = index
        return index

    # ------------------------------------------------------------------
    def refine_pair(self, i, j, seed_pairs, src_profiles):
        """Run Algorithm 3 for one partition pair.

        Returns ``(c5, theta, routes)``; c5 maps source entity -> verified
        target candidates, theta is the retained event list and routes maps
        source entity -> "tem"/"str"/"mix".
        """
        P_i = self.so.src_members[i]
        P_j = self.so.tgt_members[j]
        params = self.params
        seed_map = dict(seed_pairs)
        aligned_src = set(seed_map)
        aligned_tgt = set(seed_map.values())
        part_j_set = set(P_j)

        # ---- routing + retrieval (Algorithm 2) ----
        t0 = time.time()
        routes, croute = {}, {}
        static = self.data.N <= 1
        for e in P_i:
            if e in aligned_src:
                continue
            if static:
                routes[e] = "str"
                cstruct = routing.seed_vote(
                    e, self.src_adj, seed_map, self.tgt_adj, part_j_set,
                    params["b_struct"], aligned_tgt)
                structural = routing.structural_candidates(
                    e, self.src_rels, self.tgt_rels, self.tgt_rel_index,
                    part_j_set, params["budgets"][0], aligned_tgt)
                merged = sorted(set(structural) | set(cstruct))
                croute[e] = merged[:max(params["budgets"][0],
                                        params["b_struct"])]
                continue
            sup_e = src_profiles[e]
            b_temp, b_struct = routing.routing_scores(
                e, self.src_adj, aligned_src, sup_e, self.index,
                len(seed_pairs), len(P_i))
            forced = params.get("route_only")
            route = forced if forced in ("tem", "str") else routing.decide_route(
                b_temp, b_struct, params["lam"])
            routes[e] = route
            if route != "str":
                _c1, _c2, c3 = routing.cascade(
                    sup_e, self.index, j, params["budgets"],
                    params["quantiles"], aligned_tgt)
            else:
                c3 = []
            if route != "tem":
                cstruct = routing.seed_vote(
                    e, self.src_adj, seed_map, self.tgt_adj, part_j_set,
                    params["b_struct"], aligned_tgt)
            else:
                cstruct = []
            croute[e] = routing.route_candidates(e, route, c3, cstruct)
        self.timing["routing_cascade_s"] += time.time() - t0

        # ---- temporal reachability filter (Algorithm 3 lines 5-9) ----
        t0 = time.time()
        ev = self._event_index(i, j)
        _facts, _arr, _dep, theta = trf.scan_pair(
            ev["facts"], seed_pairs, self.data.N,
            out_adj=ev["out"], in_adj=ev["in"])
        if not params.get("enable_trf", True):
            # w/o TRF: routed candidates pass unchanged; Theta still feeds
            # context construction (all pair facts are available).
            c4 = {e: list(c) for e, c in croute.items()}
        elif static:
            c4 = {e: list(c) for e, c in croute.items()}
        elif seed_pairs:
            verts = trf.theta_vertices(theta)
            c4 = {}
            for e, cands in croute.items():
                if routes[e] == "str":
                    c4[e] = list(cands)
                else:
                    c4[e] = [c for c in cands if ("t", c) in verts]
        else:
            # no seed pairs: all events and routed candidates pass unchanged
            c4 = {e: list(c) for e, c in croute.items()}
        self.timing["batch_temporal_scan_s"] += time.time() - t0

        # ---- local structure verification (Algorithm 3 lines 10-12) ----
        t0 = time.time()
        if not params.get("enable_lsv", True):
            c5 = {e: list(c) for e, c in c4.items()}
        else:
            uti = [e for e in c4 if routes[e] != "str" and c4[e]]
            graph = lsv.ThetaGraph(theta, self.rel_of_fact, self.head_div)
            cand_union = set()
            for e in uti:
                cand_union.update(("t", c) for c in c4[e])
            sig = {}

            def sig_of(v):
                if v not in sig:
                    sig[v] = graph.signature(v, self.data.N, params["h"])
                return sig[v]

            for v in cand_union:
                sig_of(v)
            for e in uti:
                sig_of(("s", e))
            w = lsv.idf_weights({v: sig[v] for v in cand_union})
            c5 = {}
            for e in uti:
                s_e = sig[("s", e)]
                c5[e] = [c for c in c4[e]
                         if lsv.verify_score(s_e, sig[("t", c)], w)
                         >= params["tau_struct"]]
            for e, cands in c4.items():
                if routes[e] == "str":
                    c5[e] = list(cands)
        self.timing["local_structure_verification_s"] += time.time() - t0
        return c5, theta, routes