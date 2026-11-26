# -*- coding: utf-8 -*-
"""Stage II-1: Multi-Resolution Temporal Index.

Report reference: Section III-C1, Eqs. (6), (21), Appendix A-B2.

The index groups calibrated slots by cumulative target activity into
equi-depth buckets at L = 3 levels (coarse to fine).  For slot i,

    act_i = |{e in E^t : p_cal(e)[i] = 1}|,                             (21)

and D(i) = sum_{q<=i} act_q.  Level l uses K'_1 < K'_2 < ... < K'_L
buckets with boundaries s_{l,0}=0, s_{l,K'_l}=N and
s_{l,k} = min{i : D(i) >= (k/K'_l) D(N)}.

For bucket b at level l and target partition P_j:

    IDF_mix(b,l;P_j) = alpha * log(1 + |P_j| / max(1, n_{l,j}(b)))
                     + (1-alpha) * log(1 + |E^t| / max(1, n_l(b))),     (6)

with n_l(b) = |Index_l[b]| and n_{l,j}(b) = |Index_l[b] ∩ P_j|.  Posting
lists are grouped by target partition so that retrieval scans
``Index_l[b] ∩ P_j`` directly.
"""
import numpy as np


def bucket_boundaries(activity, n_buckets):
    """Equi-depth slot boundaries s_{l,0} .. s_{l,K'_l} (Appendix A-B2)."""
    n_slots = len(activity)
    total = int(activity.sum())
    if n_slots == 0:
        return np.zeros(n_buckets + 1, dtype=np.int64)
    if total <= 0 or n_buckets <= 1:
        # degenerate target activity: one nonempty slot range at most
        return np.linspace(0, n_slots, n_buckets + 1).astype(np.int64)
    cum = np.cumsum(activity)
    boundaries = [0]
    for k in range(1, n_buckets):
        target = k * total / n_buckets
        idx = int(np.searchsorted(cum, target, side="left")) + 1
        boundaries.append(min(n_slots, max(0, idx)))
    boundaries.append(n_slots)
    # non-decreasing by construction; keep duplicates so that empty buckets
    # stay explicit (paper: "This range is empty whenever s_{b-1}=s_b")
    return np.asarray(boundaries, dtype=np.int64)


def _interval_bucket_range(boundaries, lo, hi):
    """Bucket ids intersecting the half-open slot range [lo, hi)."""
    if hi <= lo:
        return 0, 0
    b_lo = int(np.searchsorted(boundaries, lo, side="right") - 1)
    b_hi = int(np.searchsorted(boundaries, hi - 1, side="right") - 1)
    b_lo = max(0, b_lo)
    b_hi = min(len(boundaries) - 1, b_hi + 1)
    return b_lo, max(b_lo, b_hi)


class MultiResolutionIndex:
    """Fixed target posting lists at L resolutions (report Section III-C1)."""

    def __init__(self, tgt_supports, tgt_membership, K_ticks, alpha=0.9):
        self.alpha = alpha
        self.K_levels = [int(k) for k in K_ticks]
        self.L = len(self.K_levels)
        self.n_tgt = len(tgt_supports)
        self.tgt_membership = np.asarray(tgt_membership, dtype=np.int32)
        self.tgt_supports = tgt_supports
        self._idf_arr_cache = {}
        self._part_cache = {}
        self._dense_cache = {}
        self._prefix_cache = {}
        self._range_cache = {}
        n_parts = int(self.tgt_membership.max()) + 1 if self.n_tgt else 1
        self.part_sizes = np.bincount(
            self.tgt_membership, minlength=n_parts).astype(np.int64)

        # act_i via a difference array over interval endpoints (exact).
        n_slots = 1
        for s in tgt_supports:
            if len(s):
                n_slots = max(n_slots, int(s[:, 1].max()))
        diff = np.zeros(n_slots + 1, dtype=np.int64)
        for s in tgt_supports:
            if len(s):
                np.add.at(diff, s[:, 0].astype(np.int64), 1)
                np.add.at(diff, s[:, 1].astype(np.int64), -1)
        activity = np.cumsum(diff[:n_slots])

        self.n_slots = n_slots
        self.bounds = []            # level -> boundaries
        self.postings = []          # level -> bucket -> np.ndarray of targets
        self.postings_by_part = []  # level -> bucket -> {part: np.ndarray}
        self.n_l = []               # level -> np.ndarray(|Index_l[b]|)
        self.n_lj = []              # level -> bucket -> {part: count}
        self.idf_global = []        # level -> np.ndarray
        self.idf_local = []         # level -> bucket -> {part: value}

        for l, K_l in enumerate(self.K_levels):
            bnds = bucket_boundaries(activity, K_l)
            postings = [set() for _ in range(K_l)]
            for e, s in enumerate(tgt_supports):
                if len(s) == 0:
                    continue
                touched = set()
                for lo, hi in s.tolist():
                    for b in range(*_interval_bucket_range(bnds, lo, hi)):
                        touched.add(b)
                for b in touched:
                    postings[b].add(e)
            postings = [np.asarray(sorted(x), dtype=np.int64) for x in postings]

            n_l_arr = np.array([len(x) for x in postings], dtype=np.int64)
            n_lj_arr, by_part = [], []
            for b, arr in enumerate(postings):
                counts, groups = {}, {}
                for e in arr.tolist():
                    j = int(self.tgt_membership[e])
                    counts[j] = counts.get(j, 0) + 1
                    groups.setdefault(j, []).append(e)
                n_lj_arr.append(counts)
                by_part.append({j: np.asarray(v, dtype=np.int64)
                                for j, v in groups.items()})
            self.postings.append(postings)
            self.postings_by_part.append(by_part)
            self.n_l.append(n_l_arr)
            self.n_lj.append(n_lj_arr)
            self.idf_global.append(
                np.log(1.0 + self.n_tgt / np.maximum(1, n_l_arr)))
            idf_local = []
            for b in range(K_l):
                idf_local.append({
                    j: float(np.log(1.0 + self.part_sizes[j] / max(1, c)))
                    for j, c in n_lj_arr[b].items()})
            self.idf_local.append(idf_local)
            self.bounds.append(bnds)

        # IDF_med: median of IDF_global over nonempty finest-level buckets
        # (average of the two middle values for an even count, 0 if none).
        finest = self.L - 1
        ig = self.idf_global[finest]
        nonempty = ig[self.n_l[finest] > 0] if len(ig) else np.array([])
        if nonempty.size == 0:
            self.IDF_med = 0.0
        else:
            srt = np.sort(nonempty)
            if srt.size % 2 == 1:
                self.IDF_med = float(srt[srt.size // 2])
            else:
                self.IDF_med = float((srt[srt.size // 2 - 1] +
                                      srt[srt.size // 2]) / 2.0)

    # ------------------------------------------------------------------
    def idf_mix(self, level, bucket, part):
        """IDF_mix(b, l; P_j) of Eq. (6)."""
        ig = float(self.idf_global[level][bucket])
        il = float(self.idf_local[level][bucket].get(part, 0.0))
        return self.alpha * il + (1.0 - self.alpha) * ig

    def idf_mix_array(self, level, part):
        """Dense IDF_mix(., l; P_j) cache used by the cascade."""
        key = (level, part)
        if key not in self._idf_arr_cache:
            K = self.K_levels[level]
            arr = np.array(self.idf_global[level], dtype=np.float64)
            loc = np.fromiter(
                (self.idf_local[level][b].get(part, 0.0) for b in range(K)),
                dtype=np.float64, count=K)
            self._idf_arr_cache[key] = self.alpha * loc + (1.0 - self.alpha) * arr
        return self._idf_arr_cache[key]

    def postings_of(self, level, bucket, part):
        """Index_l[b] ∩ P_j (direct iteration)."""
        return self.postings_by_part[level][bucket].get(part, ())

    def part_targets(self, part):
        """Sorted target entities of partition P_j."""
        if part not in self._part_cache:
            self._part_cache[part] = np.where(
                self.tgt_membership == part)[0].astype(np.int64)
        return self._part_cache[part]

    def dense_incidence(self, level, part):
        """Dense (K_l x |P_j|) incidence of Index_l[b] ∩ P_j.

        Only used for the coarse levels where K_l is small; it turns the
        Score_l aggregation of Eq. (6) into one matrix-vector product.
        """
        key = (level, part)
        if key not in self._dense_cache:
            targets = self.part_targets(part)
            pos = {int(e): i for i, e in enumerate(targets)}
            mat = np.zeros((self.K_levels[level], len(targets)),
                           dtype=np.float32)
            for b in range(self.K_levels[level]):
                arr = self.postings_of(level, b, part)
                if len(arr):
                    cols = np.fromiter(
                        (pos[int(e)] for e in arr.tolist()),
                        dtype=np.int64, count=len(arr))
                    mat[b, cols] = 1.0
            self._dense_cache[key] = (targets, mat)
        return self._dense_cache[key]

    def buckets_of(self, support, level):
        """Buckets(e,l): nonempty Level-l buckets intersecting the support."""
        bnds = self.bounds[level]
        n_l = self.n_l[level]
        out = set()
        for lo, hi in support.tolist():
            for b in range(*_interval_bucket_range(bnds, lo, hi)):
                if b < len(n_l) and n_l[b] > 0:
                    out.add(b)
        return sorted(out)

    def bucket_ranges(self, support, level, nonempty=True):
        """Buckets(e,l) as merged half-open bucket ranges [a, b).

        Nonempty trimming uses the cumulative nonempty-bucket count, so each
        support interval costs O(log K) rather than a scan over its span.
        """
        bnds = self.bounds[level]
        _idf_pref, cnt_pref = self._level_prefixes(level)
        spans = []
        for lo, hi in support.tolist():
            a, b = _interval_bucket_range(bnds, lo, hi)
            if b <= a:
                continue
            if nonempty:
                c0, c1 = int(cnt_pref[a]), int(cnt_pref[b])
                if c1 <= c0:
                    continue
                a = int(np.searchsorted(cnt_pref, c0 + 1, side="left")) - 1
                b = int(np.searchsorted(cnt_pref, c1, side="left"))
            spans.append((max(0, a), b))
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def query_weights(self, support, level, part):
        """Bucket weight vector w[b] = IDF_mix(b,l;P_j) over Buckets(e,l)."""
        K = self.K_levels[level]
        w = np.zeros(K, dtype=np.float64)
        arr = self.idf_mix_array(level, part)
        for a, b in self.bucket_ranges(support, level):
            w[a:b] = arr[a:b]
        return w

    def tgt_bucket_ranges(self, e, level):
        """Cached Buckets(e',l) of a target entity."""
        key = (level, e)
        r = self._range_cache.get(key)
        if r is None:
            r = self.bucket_ranges(self.tgt_supports[e], level)
            self._range_cache[key] = r
        return r

    def scores_against(self, support, level, part, candidates):
        """Score_l(e, e'; P_j) for a small candidate list (Eq. 6)."""
        pref = np.concatenate(([0.0], np.cumsum(
            self.query_weights(support, level, part))))
        out = []
        for e2 in candidates:
            s = 0.0
            for a, b in self.tgt_bucket_ranges(e2, level):
                s += float(pref[b] - pref[a])
            out.append(s)
        return out

    def _level_prefixes(self, level):
        """Prefix sums of global IDF and nonempty-bucket indicators."""
        if level not in self._prefix_cache:
            nz = (self.n_l[level] > 0).astype(np.float64)
            self._prefix_cache[level] = (
                np.concatenate(([0.0], np.cumsum(
                    self.idf_global[level] * nz))),
                np.concatenate(([0], np.cumsum(nz.astype(np.int64)))),
            )
        return self._prefix_cache[level]

    def routing_stats(self, support, level):
        """(|Buckets(e,l)|, sum of IDF_global over those buckets)."""
        idf_pref, cnt_pref = self._level_prefixes(level)
        n_buckets, idf_sum = 0, 0.0
        for a, b in self.bucket_ranges(support, level):
            n_buckets += int(cnt_pref[b] - cnt_pref[a])
            idf_sum += float(idf_pref[b] - idf_pref[a])
        return n_buckets, idf_sum

# ----------------------------------------------------------------------
# Paper routine name (Algorithm 5, line 2)
# ----------------------------------------------------------------------
TargetIndex = MultiResolutionIndex
