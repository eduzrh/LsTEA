# -*- coding: utf-8 -*-
"""Stage I: Time-Aware Graph Partition (report Algorithm 1).

Pipeline
--------
1. Unified temporal profile space construction over both TKGs
   (Section III-B1, Eq. 1, Definition 4).
2. Dual TKG partitioner: temporal-cohesion weighted METIS on each side
   independently (Section III-B2, Eqs. 2-3).
3. Temporal-aware target TKG candidate matcher: SO/TO scoring and top-m
   retention (Section III-B3, Eqs. 4-5).

The stage exposes per-substage wall-clock timings for the runtime
runtime breakdown.
"""
import time

from .unified_temporal_profile import build_profiles, partition_temporal_support
from .dual_tkg_partitioner import run_dual_tkg_partitioner, partition_balance
from .temporal_aware_matcher import (
    match_partitions,
    partition_temporal_supports,
    iter_partition_pairs,
)


class StageOneResult:
    """Outputs of Algorithm 1: p_cal, P^s, P^t and the retained pairs M."""

    def __init__(self, src_members, tgt_members, M, src_tsup, tgt_tsup,
                 src_membership, tgt_membership):
        self.src_members = src_members          # list[list[int]]
        self.tgt_members = tgt_members
        self.M = M                              # dict src -> list[tgt]
        self.src_tsup = src_tsup                # TP_i^s
        self.tgt_tsup = tgt_tsup                # TP_j^t
        self.src_membership = src_membership
        self.tgt_membership = tgt_membership

    def pairs(self):
        return list(iter_partition_pairs(self.M))


def run_time_aware_graph_partition(data, K_s, K_t, zeta=1.0, gamma=0.5,
                                   rho_m=0.2, epsilon_bal=0.1, seed=0,
                                   verbose=True, timing=None):
    """Algorithm 1 (report Section III-B).

    Returns ``(StageOneResult, p_cal_source, p_cal_target)``.
    """
    timing = timing if timing is not None else {}

    t0 = time.time()
    p_cal_s = build_profiles(data.gs, data.N)
    p_cal_t = build_profiles(data.gt, data.N)
    timing["profile_construction_s"] = timing.get("profile_construction_s", 0.0) \
        + (time.time() - t0)

    t0 = time.time()
    (src_members, tgt_members, src_membership,
     tgt_membership) = run_dual_tkg_partitioner(
        data, p_cal_s, p_cal_t, K_s, K_t, zeta=zeta,
        epsilon_bal=epsilon_bal, seed=seed)
    timing["dual_partitioner_s"] = timing.get("dual_partitioner_s", 0.0) \
        + (time.time() - t0)

    t0 = time.time()
    src_tsup = partition_temporal_supports(p_cal_s, src_members)
    tgt_tsup = partition_temporal_supports(p_cal_t, tgt_members)
    M = match_partitions(src_members, tgt_members, src_tsup, tgt_tsup,
                         data.seeds, gamma=gamma, rho_m=rho_m)
    timing["partition_matching_s"] = timing.get("partition_matching_s", 0.0) \
        + (time.time() - t0)

    if verbose:
        n_pairs = sum(len(v) for v in M.values())
        bal_s = partition_balance(src_members, data.gs.n_ent, K_s, epsilon_bal)
        bal_t = partition_balance(tgt_members, data.gt.n_ent, K_t, epsilon_bal)
        print(f"[Stage I] K_s={K_s} K_t={K_t} |M|={n_pairs} pairs "
              f"(src parts={len(src_members)}, tgt parts={len(tgt_members)}, "
              f"balance src/tgt={bal_s:.2f}/{bal_t:.2f})", flush=True)

    return (StageOneResult(src_members, tgt_members, M, src_tsup, tgt_tsup,
                           src_membership, tgt_membership),
            p_cal_s, p_cal_t)

# ----------------------------------------------------------------------
# Paper routine name (Algorithm 1)
# ----------------------------------------------------------------------
TimeAwareGraphPartition = run_time_aware_graph_partition
