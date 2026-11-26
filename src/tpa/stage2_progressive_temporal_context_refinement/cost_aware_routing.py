# -*- coding: utf-8 -*-
"""Stage II-2: Cost-Aware Routing (report Section III-C2, Algorithm 2).

Routing scores (Appendix A-B3, Eqs. 24-27).  Let Buckets^(t)(e,1) be the
nonempty Level-1 query buckets of source entity e:

    sel^(t)(e) = 1[|Buckets(e,1)|>0] * (1 - (|Buckets(e,1)|-1)/K'_1)   (24)
    meanIDF^(t)(e) = mean_{b in Buckets(e,3)} IDF_global(b,3)           (25)
    B_temporal = sel(e) * meanIDF(e) / max(1, log K'_3)                  (26)
    B_struct = |N^s(e) ∩ E^{s,a,(t)}| / max(1,|N^s(e)|)
               * |S^{a,(t)} ∩ (P_i x P_j)| / max(1,|P_i|)                (27)

With the routing margin lambda > 1, ``tem`` is selected when
B_temporal > lambda * B_struct, ``str`` when B_struct > lambda * B_temporal,
and ``mix`` otherwise (this also covers entities whose two scores are zero).

The three-level cascade (Appendix A-B4, Eqs. 28-30) keeps

    C_1 = Top_{b1}{e' in P_j \\ E^{t,a} : Score_1(e') > 0},
    C_l = Top_{bl}{e' in C_{l-1} : Score_l(e') >= kappa_l},  l = 2,3,

with kappa_l the eta_l empirical quantile over C_{l-1} scores.  Levels 2 and
3 are evaluated only on the surviving candidates, so the finest level never
touches the full target partition.  With L = 1 or L = 2 (the index-level
ablations of Appendix E-C), the cascade stops at the finest available level and
C_L(e) is returned as the temporal candidate set.

Structural voting (Appendix A-B5, Eq. 31): every aligned source neighbour
u in N^s(e) ∩ E^{s,a} votes for the neighbours of pi(u) inside P_j; the
top-b_struct positively voted unaligned targets form C_struct(e).
"""
import numpy as np


def routing_scores(e, src_adj, aligned_src_set, sup_e, index,
                   part_seed_count, part_i_size, part_j=None):
    """Return (B_temporal, B_struct) for one source entity (Eqs. 24-27)."""
    n_l1, _sum1 = index.routing_stats(sup_e, 0)
    sel = (1.0 - (n_l1 - 1) / max(1, index.K_levels[0])) if n_l1 else 0.0

    finest = index.L - 1
    n_l3, idf_sum = index.routing_stats(sup_e, finest)
    mean_idf = idf_sum / max(1, n_l3)
    b_temporal = sel * mean_idf / max(
        1.0, float(np.log(max(2, index.K_levels[finest]))))

    neighbours = src_adj.get(e, ())
    n_aligned = sum(1 for u in neighbours if u in aligned_src_set)
    frac_aligned = n_aligned / max(1, len(neighbours))
    frac_seed = part_seed_count / max(1, part_i_size)
    b_struct = frac_aligned * frac_seed
    return b_temporal, b_struct


def decide_route(b_temporal, b_struct, lam):
    """Route(e;i,j): tem / str / mix (report Section III-C2)."""
    if b_temporal > lam * b_struct:
        return "tem"
    if b_struct > lam * b_temporal:
        return "str"
    return "mix"


def _quantile(sorted_vals, q):
    """Empirical quantile with linear interpolation."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _top_c1(sup_e, index, part, budget, exclude_tgt):
    """C_1 = Top_{b1} positive Score_1 candidates in P_j (Eq. 28)."""
    targets, mat = index.dense_incidence(0, part)
    if len(targets) == 0:
        return []
    w = index.query_weights(sup_e, 0, part).astype(np.float32)
    scores = w @ mat
    k = min(len(targets), budget + len(exclude_tgt))
    if k < len(targets):
        part_idx = np.argpartition(-scores, k - 1)[:k]
    else:
        part_idx = np.arange(len(targets))
    order = part_idx[np.argsort(-scores[part_idx], kind="stable")]
    kept = []
    for i in order:
        if scores[i] <= 0:
            break
        e2 = int(targets[i])
        if e2 in exclude_tgt:
            continue
        kept.append(e2)
        if len(kept) >= budget:
            break
    return kept


def cascade(sup_e, index, part, budgets, quantiles, exclude_tgt):
    """Multi-level temporal cascade C1 -> ... -> C_L (Eqs. 28-30).

    Score_l(e,e';P_j) = sum_{b in Buckets(e,l), e' in Index_l[b] ∩ P_j}
                        IDF_mix(b,l;P_j)                                 (6)

    Returns ``(C_1, C_2, C_L)``.  For L = 1, ``C_2`` is empty and ``C_L`` is
    ``C_1``; for L = 2, ``C_L`` is ``C_2``.
    """
    c1 = _top_c1(sup_e, index, part, budgets[0], exclude_tgt)
    levels = [c1]
    prev = c1
    for l in range(1, index.L):
        if not prev:
            levels.append([])
            prev = []
            continue
        budget = budgets[min(l, len(budgets) - 1)]
        vals = index.scores_against(sup_e, l, part, prev)
        eta = quantiles[min(l - 1, len(quantiles) - 1)]
        kappa = _quantile(sorted(vals), eta)
        pairs = sorted(zip(prev, vals), key=lambda kv: (-kv[1], kv[0]))
        kept = [e2 for e2, s in pairs if s >= kappa][:budget]
        levels.append(kept)
        prev = kept

    c2 = levels[1] if len(levels) > 1 else []
    c_final = levels[-1] if levels else []
    return levels[0], c2, c_final


def seed_vote(e, src_adj, seed_map, tgt_adj, part_j_set, b_struct,
              exclude_tgt):
    """Structural voting SeedVote (Eq. 31), top-b_struct by votes."""
    votes = {}
    for u in src_adj.get(e, ()):
        counterpart = seed_map.get(u)
        if counterpart is None:
            continue
        for e2 in tgt_adj.get(counterpart, ()):
            if e2 in part_j_set and e2 not in exclude_tgt:
                votes[e2] = votes.get(e2, 0) + 1
    if not votes:
        return []
    items = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
    return [e2 for e2, _v in items[:b_struct]]


def structural_candidates(e, src_rels, tgt_rels, tgt_rel_index, part_j_set,
                          budget, exclude_tgt):
    """Structural counterpart of the temporal cascade (RQ5 / no timestamps).

    Without timestamps the temporal profile space is undefined; the cascade
    degenerates to relation-profile retrieval: candidates in P_j are ranked
    by the Jaccard similarity of their incident relation sets, which is the
    structural analogue of the shared-slot score of Eq. (6).
    """
    rels = src_rels.get(e, ())
    if not rels:
        return []
    counts = {}
    for r in rels:
        for e2 in tgt_rel_index.get(r, ()):
            if e2 in part_j_set and e2 not in exclude_tgt:
                counts[e2] = counts.get(e2, 0) + 1
    if not counts:
        return []
    n_e = len(rels)
    scored = []
    for e2, c in counts.items():
        union = n_e + len(tgt_rels.get(e2, ())) - c
        scored.append((c / max(1, union), e2))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [e2 for _s, e2 in scored[:budget]]


def route_candidates(e, route, c3, cstruct):
    """C_route(e) for the three routes (Algorithm 2 lines 7-14)."""
    if route == "tem":
        return list(c3)
    if route == "str":
        return list(cstruct)
    inter = sorted(set(c3) & set(cstruct))
    if inter:
        return inter
    return sorted(set(c3) | set(cstruct))

# ----------------------------------------------------------------------
# Paper routine names (Algorithm 2, Appendix A-B3-B5)
# ----------------------------------------------------------------------
Route = decide_route
Cascade = cascade
SeedVote = seed_vote
RouteRetrieve = route_candidates
