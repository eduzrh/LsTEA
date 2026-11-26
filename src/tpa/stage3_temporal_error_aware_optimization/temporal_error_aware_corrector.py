# -*- coding: utf-8 -*-
"""Stage III-2: Temporal Error-Aware Corrector (TEC).

Report reference: Section III-D2, Eqs. (10)-(11), (40), Appendix A-B9/A-B10.

Pseudo-seed pair acceptance
---------------------------
For a candidate pair (e1, e2) in sub-task k, s+ = sim_k(e1, e2) is the
stabilized cosine of Eq. (35) and s- is the largest score of its retained
competitors (-1 when none).  Resolve keeps the highest-scoring proposal per
source entity and greedily removes proposals that assign different source
entities to the same target entity, producing a one-to-one set R.  The
prediction margin is

    conf(e1, e2) = max(0, s+ - s-),

and a proposal is accepted when conf > delta_c.

High-IDF gated propagation
--------------------------
For an accepted target e2, m(e2)[b] keeps its active Level-3 buckets whose
global IDF is at least IDF_med.  The slot-level mask m_hi(e2) retains slots
q in supp(p_cal(e2)) with m(e2)[bkt_3(q)] = 1.  Previously inactive slots in
supp(m_hi(e2)) \\ supp(p^(t)(e1)) are ranked by descending inherited
Level-3 IDF (ties by ascending slot), and at most

    Delta_max(e1) = max(Delta_0, floor(rho_b |supp(p^(t)(e1))|))          (10)

slots are kept, forming m~_bud^(t)(e1).  The source query profile is
updated by

    p^(t+1)(e1) = p^(t)(e1) OR m~_bud^(t)(e1).                           (11)
"""
import numpy as np

from ..stage1_time_aware_graph_partition.unified_temporal_profile import (
    merge_intervals,
    support_size,
)


def infer_proposals(ctx, emb):
    """Infer (Appendix A-B9): stabilized cosine top-1 / runner-up per query."""
    proposals = []
    for e1, cands in ctx.get("in_c5", {}).items():
        if not cands:
            continue
        y1 = emb.get(("s", e1))
        if y1 is None:
            continue
        n1 = float(np.linalg.norm(y1))
        scored = []
        for e2 in cands:
            y2 = emb.get(("t", e2))
            if y2 is None:
                continue
            n2 = float(np.linalg.norm(y2))
            sim = float(np.dot(y1, y2) / max(1e-8, n1 * n2))
            scored.append((sim, e2))
        if not scored:
            continue
        scored.sort(key=lambda x: (-x[0], x[1]))
        s_plus, e2 = scored[0]
        s_minus = scored[1][0] if len(scored) > 1 else -1.0
        conf = max(0.0, s_plus - s_minus)
        proposals.append((e1, e2, s_plus, s_minus, conf))
    return proposals


def resolve(proposals):
    """Cross-sub-task conflict resolution -> one-to-one proposal set R."""
    best = {}
    for (e1, e2, s_plus, s_minus, conf) in proposals:
        cur = best.get(e1)
        if cur is None or (s_plus, conf, -e2) > (cur[2], cur[4], -cur[1]):
            best[e1] = (e1, e2, s_plus, s_minus, conf)
    ordered = sorted(best.values(), key=lambda p: (-p[2], -p[4], p[0], p[1]))
    R, used_tgt = [], set()
    for p in ordered:
        if p[1] not in used_tgt:
            R.append(p)
            used_tgt.add(p[1])
    return R


def accept(R, delta_c):
    """S_acc^{(t)} = {(u,v) in R : conf(u,v) > delta_c}."""
    return [p for p in R if p[4] > delta_c]


def high_idf_mask(tgt_support, index, level=None):
    """m(e2)[b]: active Level-3 buckets with IDF_global >= IDF_med.

    When the nonempty finest-level bucket set I+3 is empty, IDF_med is zero
    by definition and the mask of Eq. (40) is empty (the indicator
    ``1[I+3 != empty]``); returning all buckets would incorrectly propagate
    temporal activity that the index cannot support.
    """
    level = index.L - 1 if level is None else level
    if not np.any(index.n_l[level] > 0):
        return set()
    buckets = index.buckets_of(tgt_support, level)
    return {b for b in buckets
            if index.idf_global[level][b] >= index.IDF_med}


def _subtract_intervals(base, cover):
    """Segments of merged ``base`` not covered by merged ``cover``."""
    out = []
    j = 0
    for lo, hi in base.tolist():
        cur = int(lo)
        while j < len(cover) and cover[j, 1] <= cur:
            j += 1
        k = j
        while k < len(cover) and cover[k, 0] < hi:
            if cover[k, 0] > cur:
                out.append((cur, min(int(cover[k, 0]), int(hi))))
            cur = max(cur, int(cover[k, 1]))
            if cur >= hi:
                break
            k += 1
        if cur < hi:
            out.append((cur, int(hi)))
    return out


def _interval_gaps(tgt_support, src_support):
    return _subtract_intervals(tgt_support, src_support)


def budget_mask(src_support, tgt_support, index, delta0=5, rho_b=0.1,
                idf_mask=True, budget_cap=True):
    """m~_bud^(t)(e1): previously-inactive high-IDF slots up to Delta_max.

    ``idf_mask=False``   -> w/o IDF mask ablation (no median-IDF gate).
    ``budget_cap=False`` -> w/o budget ablation (no Delta_max cap).
    """
    level = index.L - 1
    gaps = _interval_gaps(tgt_support, src_support)
    if not gaps:
        return np.empty(0, dtype=np.int32)
    delta_max = (max(delta0, int(np.floor(rho_b * support_size(src_support))))
                 if budget_cap else None)

    if not idf_mask:
        slots = []
        for lo, hi in gaps:
            slots.extend(range(int(lo), int(hi)))
            if delta_max is not None and len(slots) >= delta_max:
                break
        if delta_max is not None:
            slots = slots[:delta_max]
        return np.asarray(sorted(set(slots)), dtype=np.int32)

    masked = high_idf_mask(tgt_support, index, level)
    if not masked:
        return np.empty(0, dtype=np.int32)
    bnds = index._bucket_bounds[level] if hasattr(index, "_bucket_bounds") \
        else index.bounds[level]
    idf_global = index.idf_global[level]

    kept = []
    # descending inherited IDF, ties by ascending bucket id
    for b in sorted(masked, key=lambda x: (-idf_global[x], x)):
        lo, hi = int(bnds[b]), int(bnds[b + 1])
        segs = []
        for tlo, thi in tgt_support.tolist():
            a, z = max(int(tlo), lo), min(int(thi), hi)
            if z > a:
                segs.append((a, z))
        if not segs:
            continue
        segs = merge_intervals(segs)
        for glo, ghi in _subtract_intervals(segs, src_support):
            for q in range(int(glo), int(ghi)):
                kept.append(q)
                if delta_max is not None and len(kept) >= delta_max:
                    break
            if delta_max is not None and len(kept) >= delta_max:
                break
        if delta_max is not None and len(kept) >= delta_max:
            break
    if not kept:
        return np.empty(0, dtype=np.int32)
    return np.asarray(sorted(set(kept)), dtype=np.int32)


def update_profile(src_support, mask_slots):
    """p^(t+1)(e1) = p^(t)(e1) OR m~_bud^(t)(e1) (Eq. 11)."""
    if mask_slots is None or len(mask_slots) == 0:
        return src_support
    pairs = [tuple(x) for x in src_support.tolist()]
    pairs.extend((int(q), int(q) + 1) for q in mask_slots.tolist())
    return merge_intervals(pairs)


def correct(proposals, src_profiles, p_cal_t, index, delta_c=0.15,
            delta0=5, rho_b=0.1, enable_tec=True, accept_all=False,
            idf_mask=True, budget_cap=True):
    """One TEC application: Resolve -> accept -> gated profile updates.

    ``enable_tec=False`` reproduces standard pseudo-seed propagation
    (every resolved proposal is accepted, no profile update), used by the
    w/o TEC ablation.
    """
    R = resolve(proposals)
    if not enable_tec:
        return R, list(R), src_profiles
    s_acc = list(R) if accept_all else accept(R, delta_c)
    updated = dict(src_profiles)
    for (e1, e2, _sp, _sm, _conf) in s_acc:
        mask = budget_mask(updated[e1], p_cal_t[e2], index,
                           delta0=delta0, rho_b=rho_b,
                           idf_mask=idf_mask, budget_cap=budget_cap)
        updated[e1] = update_profile(updated[e1], mask)
    return R, s_acc, updated

# ----------------------------------------------------------------------
# Paper routine names (Algorithm 4)
# ----------------------------------------------------------------------
Infer = infer_proposals
Resolve = resolve
BudgetMask = budget_mask
