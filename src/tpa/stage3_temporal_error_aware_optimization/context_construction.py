# -*- coding: utf-8 -*-
"""Stage III-1: Context Construction and TEA Training Setup.

Report reference: Section III-D1, Eq. (8), Appendix A-B8.

For a sub-task k from retained pair (P_i, P_j), the core V_k^{0,(t)}
contains every source entity with nonempty C5^{(t),ij}(e), all entities in
C5^{(t),ij}(e), and the endpoints of the current seed pairs in
S^{a,(t)} ∩ (P_i x P_j).

If |V_k^{0,(t)}| > M, current seed pairs are retained first (at most
floor(M/2) pairs per bounded subgraph), then refined candidate pairs are
ranked -- by Score_3^{(t)}(e,e';P_j) on temporal and combined routes and
by vote^{(t)}(e,e';P_i,P_j) on the structural route -- with ties broken by
ascending source-target identifiers, and the longest prefix within M is
kept.

Core-internal facts and incident one-hop facts are admitted in priority
order (support in Theta first, then relation diversity, then a fixed
entity-identifier order) while the budgets

    |H_k^s| + |H_k^t| <= M,   |Q_{H_k^s}| + |Q_{H_k^t}| <= F             (8)

hold.  Each original quadruple counts once toward F irrespective of the
number of slots it activates.  The in-context candidate set is
C5,k(e) = C5^{(t),ij}(e) ∩ H_k^t for e in (H_k^s ∩ P_i) \\ E^{s,a,(t)}.
"""
from collections import defaultdict

from ..stage1_time_aware_graph_partition.relation_diversity import (
    relation_diversity_map,
)


def build_context(data, c5, routes, theta, seed_pairs, score_of, vote_of,
                  M, F):
    """Construct one bounded context sub-task for a partition pair.

    Parameters
    ----------
    c5         : dict src_entity -> verified target candidates (C5)
    routes     : dict src_entity -> "tem"/"str"/"mix"
    theta      : retained event ranges for the pair
    seed_pairs : current seed pairs inside (P_i x P_j)
    score_of   : (src, tgt) -> Score_3^{(t)}(e,e';P_j)
    vote_of    : (src, tgt) -> structural vote count
    M, F       : entity and original-fact budgets of Eq. (8)

    Returns a context dict or None when the core is empty.
    """
    core = []
    for e in sorted(c5):
        route = routes.get(e, "mix")
        for e2 in c5[e]:
            rank = vote_of(e, e2) if route == "str" else score_of(e, e2)
            core.append((e, e2, route, rank))
    if not core and not seed_pairs:
        return None
    core.sort(key=lambda x: (-x[3], x[0], x[1]))

    Hs, Ht = set(), set()
    kept_seeds = []
    for (s, t) in seed_pairs:
        if len(kept_seeds) >= max(1, M // 2):
            break
        kept_seeds.append((s, t))
        Hs.add(s)
        Ht.add(t)

    in_c5 = {e: [] for e in c5}
    for (e, e2, _route, _rank) in core:
        add = (0 if e in Hs else 1) + (0 if e2 in Ht else 1)
        if len(Hs) + len(Ht) + add > M:
            break
        Hs.add(e)
        Ht.add(e2)
        in_c5[e].append(e2)

    # Snapshot the core vertex set: incident facts are one-hop facts of the
    # core, not a recursive expansion through newly admitted endpoints.
    H0s, H0t = set(Hs), set(Ht)

    theta_facts = {"s": set(), "t": set()}
    for (side, _u, _v, _lo, _hi, fi) in theta:
        theta_facts[side].add(fi)

    rel_div = {"s": relation_diversity_map(data.gs),
               "t": relation_diversity_map(data.gt)}
    Qs, Qt = [], []
    seen_facts = {"s": set(), "t": set()}

    def fact_key(side, kg, fi):
        h, r, t = kg.triples[fi]
        return (0 if fi in theta_facts[side] else 1,
                -rel_div[side].get(r, 1.0), side, h, t, fi)

    def add_fact(side, fi):
        seen_facts[side].add(fi)
        if side == "s":
            Qs.append(fi)
        else:
            Qt.append(fi)

    # ---- core-internal facts first, with one shared F budget ----
    core_items = []
    for side, kg, H in (("s", data.gs, H0s), ("t", data.gt, H0t)):
        core_facts = set()
        for e in H:
            for fi in kg.facts_of(e):
                h, _r, t = kg.triples[fi]
                if h in H and t in H:
                    core_facts.add(fi)
        for fi in core_facts:
            if fi not in seen_facts[side]:
                core_items.append((side, kg, fi))
    core_items.sort(key=lambda x: fact_key(*x))
    for side, kg, fi in core_items:
        if len(Qs) + len(Qt) >= F:
            break
        add_fact(side, fi)

    # ---- incident one-hop facts, globally priority-ordered ----
    incident_items = []
    for side, kg, H in (("s", data.gs, H0s), ("t", data.gt, H0t)):
        incident = set()
        for e in H:
            incident.update(kg.facts_of(e))
        for fi in incident:
            if fi not in seen_facts[side]:
                incident_items.append((side, kg, fi))
    incident_items.sort(key=lambda x: fact_key(*x))
    for side, kg, fi in incident_items:
        if len(Qs) + len(Qt) >= F:
            break
        h, _r, t = kg.triples[fi]
        add = (0 if h in Hs else 1) + (0 if t in Ht else 1)
        if len(Hs) + len(Ht) + add > M:
            continue
        Hs.add(h)
        Ht.add(t)
        add_fact(side, fi)

    in_c5 = {e: [x for x in c5[e] if x in Ht]
             for e in sorted(c5) if e in Hs}
    return {
        "Hs": Hs,
        "Ht": Ht,
        "facts_s": [data.gs.triples[fi] for fi in Qs],
        "facts_t": [data.gt.triples[fi] for fi in Qt],
        "intervals_s": [data.gs.intervals[fi] for fi in Qs],
        "intervals_t": [data.gt.intervals[fi] for fi in Qt],
        "fact_idx_s": Qs,
        "fact_idx_t": Qt,
        "in_c5": in_c5,
        "seeds": kept_seeds,
    }


def context_seed_pairs(ctx):
    """S_k^{a,(t)}: context seed pairs whose endpoints survive in H_k."""
    return [(s, t) for (s, t) in ctx["seeds"]
            if s in ctx["Hs"] and t in ctx["Ht"]]


def subtask_coverage(contexts):
    """Sub-task coverage map src entity -> set of target entities (recall)."""
    cov = defaultdict(set)
    for ctx in contexts:
        for e in ctx["Hs"]:
            cov[e].update(ctx["Ht"])
    return cov

# ----------------------------------------------------------------------
# Paper routine name (Algorithm 4, Appendix A-B8)
# ----------------------------------------------------------------------
Context = build_context
