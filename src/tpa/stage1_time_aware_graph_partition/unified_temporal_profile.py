# -*- coding: utf-8 -*-
"""Stage I-1: Unified Temporal Profile Space Construction.

Report reference: Section III-B1, Eq. (1), Definition 4, Algorithm 1 line 1.

Every fact interval tau = [t_b, t_e] activates the calibrated slots it
overlaps:
    Slots(tau) = {i in 1..N : [t_b, t_e] ∩ [a_{i-1}, a_i) != ∅},
where a_0 < a_1 < ... < a_N delimit N calibrated slots at a common
granularity covering all source and target timestamps.  The calibrated
timeline is built once over both TKGs by :class:`tpa.dataload.TEAData`.

For an entity e the unified temporal profile p_cal(e) in {0,1}^N has
entry 1 at slot i exactly when an incident fact satisfies e in {u, v}
and i in Slots(tau).  Its temporal support supp(p_cal(e)) is the set of
active calibrated slots.

Implementation note.  Profiles are stored as merged sorted interval
lists (k x 2 int32, half-open [lo, hi)).  Every query in the paper only
uses set cardinalities, intersections, unions, slot histograms and
"slot lies in bucket" tests, all of which are computed exactly on the
interval representation.  This is semantically identical to the dense
0/1 vector and uses O(#intervals) instead of O(N) memory, which is what
makes the million-scale benchmarks (N up to 178,142) fit in host memory.
"""
import numpy as np


def merge_intervals(pairs):
    """Merge (lo, hi) pairs into sorted non-overlapping half-open intervals."""
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    pairs = sorted(pairs)
    merged = []
    cur_lo, cur_hi = pairs[0]
    for lo, hi in pairs[1:]:
        if lo <= cur_hi:
            if hi > cur_hi:
                cur_hi = hi
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    return np.asarray(merged, dtype=np.int32)


def slots_of_interval(tb, te, boundaries=None):
    """Slots(tau) for tau = [tb, te] (report Eq. 1).

    With the rank-compressed unit grid used by the loader, slot i covers
    the point interval [i, i+1); the overlap is the integer range [tb, te].
    ``boundaries`` optionally gives a_0 < ... < a_N for non-unit grids,
    in which case the returned range is the bucket/segment index range.
    """
    if te < tb:
        tb, te = te, tb
    if boundaries is None:
        return list(range(int(tb), int(te) + 1))
    lo = int(np.searchsorted(boundaries, tb, side="right") - 1)
    hi = int(np.searchsorted(boundaries, te, side="right") - 1)
    return list(range(max(0, lo), max(0, hi) + 1))


def build_profiles(kg, n_slots):
    """Build the unified temporal profile of every entity in ``kg``.

    Parameters
    ----------
    kg      : TemporalKG with ``triples`` and ``intervals`` (``None`` marks
              an untimestamped fact, handled by simply contributing nothing).
    n_slots : size N of the calibrated slot grid.

    Returns
    -------
    supports : list[np.ndarray], each (k, 2) int32 merged active intervals.
    """
    supports = [np.empty((0, 2), dtype=np.int32)] * kg.n_ent
    buf = [[] for _ in range(kg.n_ent)]
    for idx in range(len(kg.triples)):
        h, _r, t = kg.triples[idx]
        iv = kg.intervals[idx]
        if iv is None:
            continue
        lo, hi = iv
        if lo < 0:
            continue
        hi = min(hi, n_slots)
        if lo >= hi:
            continue
        buf[h].append((lo, hi))
        buf[t].append((lo, hi))
    for e in range(kg.n_ent):
        if buf[e]:
            supports[e] = merge_intervals(buf[e])
    return supports


def support_size(a):
    """|supp(p_cal(e))|: number of active calibrated slots."""
    if a is None or len(a) == 0:
        return 0
    return int((a[:, 1] - a[:, 0]).sum())


def support_intersection_size(a, b):
    """|U_a ∩ U_b| via a two-pointer merge of the interval lists."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        lo = max(int(a[i, 0]), int(b[j, 0]))
        hi = min(int(a[i, 1]), int(b[j, 1]))
        if hi > lo:
            total += hi - lo
        if a[i, 1] <= b[j, 1]:
            i += 1
        else:
            j += 1
    return total


def temporal_cohesion(a, b):
    """TC(e_i, e_j) = |U_i ∩ U_j| / max(1, |U_i ∪ U_j|)  (report Eq. 20)."""
    inter = support_intersection_size(a, b)
    if inter == 0:
        return 0.0
    union = support_size(a) + support_size(b) - inter
    return inter / max(1, union)


def partition_temporal_support(supports, members):
    """TP_k^x = union over e in P_k^x of supp(p_cal(e))  (Algorithm 1 line 6)."""
    pairs = []
    for e in members:
        s = supports[e]
        if s is not None and len(s):
            pairs.extend(map(tuple, s.tolist()))
    return merge_intervals(pairs)


def intervals_overlap(a, b):
    """True when two merged interval lists share at least one slot."""
    return support_intersection_size(a, b) > 0


def slots_in_interval_list(support):
    """Explicit slot list of an interval-represented support (small N only)."""
    out = []
    for lo, hi in support.tolist():
        out.extend(range(int(lo), int(hi)))
    return out


def histogram_over_support(support, n_bins, n_slots):
    """Length-weighted histogram of support slots over equal-width bins.

    Equivalent to counting every active slot in
    ``int(q / max(1, N - 1) * n_bins)`` but computed exactly on intervals.
    """
    weights = np.zeros(n_bins, dtype=np.float64)
    if support is None or len(support) == 0:
        return weights
    denom = max(1, n_slots - 1)
    for lo, hi in support.tolist():
        # split [lo, hi) at bin boundaries
        cur = int(lo)
        while cur < hi:
            pos = cur / denom
            b = min(n_bins - 1, max(0, int(pos * n_bins)))
            b_start = int(np.ceil(b * denom / n_bins))
            b_end = int(np.ceil((b + 1) * denom / n_bins))
            nxt = min(int(hi), max(cur + 1, b_end))
            weights[b] += max(0, nxt - cur)
            cur = nxt
    return weights

# ----------------------------------------------------------------------
# Paper routine name (Algorithm 1, line 1)
# ----------------------------------------------------------------------
Profiles = build_profiles
