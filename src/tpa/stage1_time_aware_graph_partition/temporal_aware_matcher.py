# -*- coding: utf-8 -*-
"""Stage I-3: Temporal-Aware Target TKG Candidate Matcher.

Report reference: Section III-B3, Eqs. (4)-(5), Algorithm 1 lines 8-10.

After independent partitioning the two partition sets need not align.  For
source partition P_i^s and target partition P_j^t:

    SO_ij = |(P_i^s x P_j^t) ∩ S^a| / min(|P_i^s|, |P_j^t|)            (4)
    TO_ij = |TP_i^s ∩ TP_j^t| / max(1, |TP_i^s ∪ TP_j^t|)
    Match(P_i^s, P_j^t) = gamma * SO_ij + (1 - gamma) * TO_ij          (5)

The target partitions are ranked by Match and the top
m = ceil(rho_m * K_t) are retained for each source partition.  The
retained index set M is fixed for all iterations.
"""
import math
from collections import defaultdict

import numpy as np

from .unified_temporal_profile import (
    partition_temporal_support,
    support_intersection_size,
    support_size,
)


def partition_temporal_supports(p_cal, members):
    """TP_k^x for every partition (Algorithm 1 line 6)."""
    return [partition_temporal_support(p_cal, m) for m in members]


def match_partitions(src_members, tgt_members, src_tsup, tgt_tsup, seeds,
                     gamma=0.5, rho_m=0.2):
    """Temporal-aware target candidate matcher (Eqs. 4-5).

    Returns ``M: dict src_index -> [tgt_index, ...]`` sorted by Match
    descending (ties by ascending target index), with exactly
    ``m = max(1, ceil(rho_m * K_t))`` entries per source partition (or all
    target partitions when K_t < m).
    """
    src_ent2part, tgt_ent2part = {}, {}
    for i, members in enumerate(src_members):
        for e in members:
            src_ent2part[e] = i
    for j, members in enumerate(tgt_members):
        for e in members:
            tgt_ent2part[e] = j

    seed_cnt = defaultdict(lambda: defaultdict(int))
    for u, v in seeds:
        i, j = src_ent2part.get(u), tgt_ent2part.get(v)
        if i is not None and j is not None:
            seed_cnt[i][j] += 1

    K_t = len(tgt_members)
    m_keep = min(K_t, max(1, math.ceil(rho_m * K_t)))
    M = {}
    for i, Pi in enumerate(src_members):
        TPi = src_tsup[i]
        scores = []
        for j, Pj in enumerate(tgt_members):
            so = seed_cnt[i][j] / max(1, min(len(Pi), len(Pj)))
            TPj = tgt_tsup[j]
            inter = support_intersection_size(TPi, TPj)
            union = support_size(TPi) + support_size(TPj) - inter
            to = inter / max(1, union) if union else 0.0
            scores.append((gamma * so + (1.0 - gamma) * to, j))
        scores.sort(key=lambda x: (-x[0], x[1]))
        M[i] = [j for _s, j in scores[:m_keep]]
    return M


def iter_partition_pairs(M):
    """Yield (i, j) over the retained partition index set M."""
    for i in sorted(M):
        for j in M[i]:
            yield i, j

# ----------------------------------------------------------------------
# Paper routine name (Algorithm 1, line 9)
# ----------------------------------------------------------------------
TopMatch = match_partitions
