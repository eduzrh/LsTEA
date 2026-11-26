# -*- coding: utf-8 -*-
"""Stage II-3: Temporal Reachability Filter (report Section III-C3, A-B6).

Within a retained partition pair (P_i, P_j), the temporal event graph
D_ij^(t) = (V_ij^(t), A_ij^(t)) has V_ij^(t) = P_i ∪ P_j.  Each temporal
fact whose endpoints lie in the same partition contributes a directed
event (u -> v, q) at every active calibrated slot q in Slots(tau); current
seed pairs S_ij^{a,(t)} add zero-delay bridge arcs s -> t' and t' -> s
between the two TKGs.

A feasible temporal path visits events in non-decreasing calibrated-slot
order.  The forward scan records earliest arrivals

    arr(v) = earliest slot at which v can be reached from seed sources,
    relax (u -> v, q): arr(u) <= q  =>  arr(v) <= q,

and the backward scan records latest departures

    dep(u) = latest slot at which u can still reach a seed target,
    relax (u -> v, q): q <= dep(v)  =>  dep(u) >= q.

The retained event set is

    Theta_ij^(t) = {(u -> v, q) : arr(u) <= q <= dep(v)},

which restricts each temporal/combined query to C4(e) = C_route(e) ∩ V(Theta).

Implementation note.  A fact interval [q1, q2) produces one event per
active slot; enumerating them is quadratic in interval width.  Because
the scans only ask for the earliest arrival / latest departure, the
closure is computed in closed form: crossing fact (u -> v, [q1, q2)) at
arrival a yields arrival max(q1, a) when a < q2.  The Dijkstra-style
sweep over facts is exactly equivalent to the slot-expanded scan of
Appendix A-B6, while Theta is stored as retained slot ranges [lo, hi) per
fact instead of one tuple per slot.

The intra-partition event index (facts and adjacency maps) is static for a
partition pair and can be built once by the caller: pass ``out_adj`` /
``in_adj`` to skip rebuilding them for every iteration.
"""
import heapq
from collections import defaultdict


def build_pair_facts(data, src_members, tgt_members, n_slots):
    """Collect intra-partition temporal facts with calibrated slot ranges.

    Returns ``facts_by_side = {"s": [(u, v, q1, q2, fact_index)], "t": ...}``
    with half-open slot ranges [q1, q2).
    """
    out = {}
    for side, members, kg in (("s", set(src_members), data.gs),
                              ("t", set(tgt_members), data.gt)):
        facts = []
        for fi, (h, _r, t) in enumerate(kg.triples):
            if h not in members or t not in members:
                continue
            iv = kg.intervals[fi]
            if iv is None:
                continue
            lo, hi = iv
            lo = max(0, int(lo))
            hi = min(int(hi), n_slots)
            if lo < hi:
                facts.append((h, t, lo, hi, fi))
        out[side] = facts
    return out


def _out_adjacency(facts_by_side):
    adj = defaultdict(list)
    for side, facts in facts_by_side.items():
        for (u, v, q1, q2, fi) in facts:
            adj[(side, u)].append((side, v, q1, q2, fi))
    return adj


def _in_adjacency(facts_by_side):
    adj = defaultdict(list)
    for side, facts in facts_by_side.items():
        for (u, v, q1, q2, fi) in facts:
            adj[(side, v)].append((side, u, q1, q2, fi))
    return adj


def _bridges(seed_pairs):
    """Zero-delay bridges in both directions (s -> t' and t' -> s)."""
    fwd = defaultdict(list)
    for s, t in seed_pairs:
        fwd[("s", s)].append(("t", t))
        fwd[("t", t)].append(("s", s))
    return fwd


def forward_scan(facts_by_side, n_slots, seed_pairs, out_adj=None):
    """Earliest-arrival scan Fwd (Appendix A-B6)."""
    INF = n_slots + 2
    arr = {}
    heap = []
    for s, t in seed_pairs:
        for v in (("s", s), ("t", t)):
            if 0 < arr.get(v, INF):
                arr[v] = 0
                heapq.heappush(heap, (0, v))
    bridges = _bridges(seed_pairs)
    out_adj = out_adj if out_adj is not None else _out_adjacency(facts_by_side)
    done = set()
    while heap:
        a, x = heapq.heappop(heap)
        if x in done or a > arr.get(x, INF):
            continue
        done.add(x)
        for y in bridges.get(x, ()):
            if a < arr.get(y, INF):
                arr[y] = a
                heapq.heappush(heap, (a, y))
        for (_side, v, q1, q2, _fi) in out_adj.get(x, ()):
            if a < q2:
                cand = q1 if a < q1 else a
                key = (_side, v)
                if cand < arr.get(key, INF):
                    arr[key] = cand
                    heapq.heappush(heap, (cand, key))
    return arr


def backward_scan(facts_by_side, n_slots, seed_pairs, in_adj=None):
    """Latest-departure scan Bwd (Appendix A-B6)."""
    NEG = -1
    top = n_slots + 1
    dep = {}
    heap = []
    for s, t in seed_pairs:
        for v in (("t", t), ("s", s)):
            if top > dep.get(v, NEG):
                dep[v] = top
                heapq.heappush(heap, (-top, v))
    bridges = _bridges(seed_pairs)
    in_adj = in_adj if in_adj is not None else _in_adjacency(facts_by_side)
    done = set()
    while heap:
        nd, x = heapq.heappop(heap)
        d = -nd
        if x in done or d < dep.get(x, NEG):
            continue
        done.add(x)
        for y in bridges.get(x, ()):
            if d > dep.get(y, NEG):
                dep[y] = d
                heapq.heappush(heap, (-d, y))
        for (_side, u, q1, q2, _fi) in in_adj.get(x, ()):
            if q1 <= d:
                cand = (q2 - 1) if d > q2 - 1 else d
                key = (_side, u)
                if cand > dep.get(key, NEG):
                    dep[key] = cand
                    heapq.heappush(heap, (-cand, key))
    return dep


def scan_pair(facts_by_side, seed_pairs, n_slots, out_adj=None, in_adj=None):
    """Run Fwd/Bwd on both graph sides of one partition pair.

    ``facts_by_side`` comes from :func:`build_pair_facts` and is cached by the
    caller; ``out_adj`` / ``in_adj`` are optional pre-built event adjacency
    maps.  Returns ``(facts_by_side, arr, dep, theta)`` where ``theta`` is a
    list of retained slot ranges ``(side, u, v, lo, hi, fact_index)`` with
    half-open [lo, hi).  Pairs without seed pairs skip both scans and retain
    all intra-partition temporal facts as Theta.
    """
    if not seed_pairs:
        theta = []
        for side, facts in facts_by_side.items():
            theta.extend((side, u, v, q1, q2, fi)
                         for (u, v, q1, q2, fi) in facts)
        return facts_by_side, {}, {}, theta

    arr = forward_scan(facts_by_side, n_slots, seed_pairs, out_adj=out_adj)
    dep = backward_scan(facts_by_side, n_slots, seed_pairs, in_adj=in_adj)
    theta = []
    for side, facts in facts_by_side.items():
        for (u, v, q1, q2, fi) in facts:
            a = arr.get((side, u))
            d = dep.get((side, v))
            if a is None or d is None:
                continue
            lo = q1 if q1 > a else a
            hi = q2 if q2 < d + 1 else d + 1
            if lo < hi:
                theta.append((side, u, v, lo, hi, fi))
    return facts_by_side, arr, dep, theta


def theta_vertices(theta):
    """V(Theta): vertices incident to retained events."""
    verts = set()
    for (side, u, v, _lo, _hi, _fi) in theta:
        verts.add((side, u))
        verts.add((side, v))
    return verts


def theta_fact_indices(theta):
    """Fact indices retained per side (used by context expansion priority)."""
    out = {"s": set(), "t": set()}
    for (side, _u, _v, _lo, _hi, fi) in theta:
        out[side].add(fi)
    return out


def retained_slot_count(theta):
    """|Theta| in the slot-expanded sense (sum of retained range widths)."""
    return sum(hi - lo for (_side, _u, _v, lo, hi, _fi) in theta)

# ----------------------------------------------------------------------
# Paper routine names (Algorithm 3, Appendix A-B6)
# ----------------------------------------------------------------------
Events = build_pair_facts
Fwd = forward_scan
Bwd = backward_scan
