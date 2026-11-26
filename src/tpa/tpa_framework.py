# -*- coding: utf-8 -*-
"""TPA: Temporal Partitioning-and-Alignment (report Algorithm 5).

Complete workflow
-----------------
    (p_cal, P^s, P^t, M) <- Algorithm 1
    Index                <- TargetIndex(p_cal | E^t)
    p <- p_cal | E^s ; S <- S_a ; S_cum, R <- empty
    for t = 1..I:
        for (i,j) in M:  S_ij <- S ∩ (P_i x P_j)
                         (C5[i,j], Theta[i,j]) <- Algorithm 3
        (R, S_acc, S, p, Psi) <- Algorithm 4
        if S_acc = empty: break
        S_cum <- S_cum ∪ S_acc
    S_hat_b <- S_cum ∪ R

Hyperparameters follow Section IV-A3 / Appendix C-C: K by scale, h = 2,
D_sig = 13, alpha = 0.9, beta = 1.0 (xi_1 = xi_2 = 0.5), I = 5 iterations of
50 epochs for neural backbones, rho_m = 0.2, gamma = 0.5, lambda = 1.5,
(b1,b2,b3) = (100,50,20), b_struct = 20, delta_c = 0.15.
"""
import json
import os
import time
from collections import defaultdict

import numpy as np

from .stage1_time_aware_graph_partition import run_time_aware_graph_partition
from .stage2_progressive_temporal_context_refinement import (
    MultiResolutionIndex,
    ProgressiveContextRefiner,
)
from .stage3_temporal_error_aware_optimization import context_construction as CTX
from .stage3_temporal_error_aware_optimization import (
    temporal_error_aware_corrector as TEC,
)
from .backbones import build_backbone


DEFAULT_PARAMS = dict(
    K_s=None, K_t=None,               # per-side partition count (Section IV-A3)
    seed=0,                           # Stage-I partition / calibration RNG
    zeta=1.0,                         # temporal cohesion weight (Eq. 2)
    gamma=0.5,                        # seed / temporal balance (Eq. 5)
    rho_m=0.2,                        # retained target partitions fraction
    epsilon_bal=0.1,                  # partition balance tolerance (Eq. 3)
    K_ticks=None,                     # bucket counts per index level
    index_levels=3,                   # L
    alpha=0.9,                        # IDF_local / IDF_global mix (Eq. 6)
    budgets=(100, 50, 20),            # cascade budgets b1,b2,b3
    quantiles=(0.5, 0.7),             # cascade quantiles eta2 < eta3
    b_struct=20,                      # structural-vote budget
    lam=1.5,                          # routing margin lambda
    h=2,                              # ego-network depth
    tau_struct=0.5,                   # local verification threshold
    M=10000, F=50000,                 # context budgets (Eq. 8)
    delta_c=0.15,                     # acceptance margin (Section III-D2)
    delta0=5, rho_b=0.1,              # profile-update budgets (Eq. 10)
    I=5,                              # iteration limit
    epochs=50,                        # epochs per iteration (neural)
    calibrate_lambda=True,            # held-out lambda selection (Appendix C-C)
    # ---- ablations (Sec. IV-C joint removal; Appendix E-C one at a time) ----
    time_aware=True,                  # TC & TM joint switch
    tc_enabled=True,                  # w/o TC
    tm_enabled=True,                  # w/o TM
    enable_trf=True,                  # w/o TRF
    enable_lsv=True,                  # w/o LSV
    enable_tec=True,                  # w/o TEC
    route_only=None,                  # "tem"/"str": temporal/structural only
    tec_accept_all=False,             # w/o margin
    idf_mask=True,                    # w/o IDF mask
    budget_cap=True,                  # w/o budget
    loss_margin=True,                 # drop the ranking margin (contrastive only)
    # ---- bookkeeping ----
    checkpoint_dir=None,
    save_checkpoints=True,
)


class TPAFramework:
    """Three-stage TPA orchestration with iteration-level state."""

    def __init__(self, data, params=None, verbose=True):
        self.data = data
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.verbose = verbose
        self.stats = defaultdict(float)
        self.timing = defaultdict(float)
        self.contexts = []
        self.backbone = None

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------
    def _default_ticks(self, n_slots):
        finest = max(3, min(int(n_slots), 1024))
        mid = max(2, finest // 4)
        coarse = max(1, mid // 4)
        levels = int(self.params.get("index_levels", 3))
        ticks = [coarse, mid, finest][:levels]
        return tuple(max(1, min(int(n_slots + 1), k)) for k in ticks)

    @staticmethod
    def _peak_memory_gb():
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def run(self, backbone="trea", backbone_params=None):
        data, P = self.data, self.params
        if data.N <= 1:
            # RQ5: without timestamps every temporal component degenerates to
            # its structural counterpart -- structural METIS partitioning,
            # seed-overlap partition matching, and structural seed voting.
            P["route_only"] = "str"
            P["time_aware"] = False
        t_start = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        # ================= Stage I: Algorithm 1 =================
        self.log("[Stage I] Time-Aware Graph Partition "
                 "(unified temporal profiles + dual TKG partitioner + "
                 "temporal-aware candidate matcher)")
        K_s = P["K_s"] or P["K_t"]
        K_t = P["K_t"] or P["K_s"]
        tc_on = P.get("time_aware", True) and P.get("tc_enabled", True)
        tm_on = P.get("time_aware", True) and P.get("tm_enabled", True)
        stage_one, p_cal_s, p_cal_t = run_time_aware_graph_partition(
            data, K_s, K_t,
            zeta=P["zeta"] if tc_on else 0.0,
            gamma=P["gamma"] if tm_on else 1.0,
            rho_m=P["rho_m"], epsilon_bal=P["epsilon_bal"],
            seed=int(P.get("seed", 0)),
            verbose=self.verbose, timing=self.timing)

        # ============ Stage II-A: TargetIndex (Algorithm 5 line 2) ============
        self.log("[Stage II-A] Multi-Resolution Temporal Index "
                 f"(L={P['index_levels']}, alpha={P['alpha']})")
        t0 = time.time()
        if P["K_ticks"] is None:
            P["K_ticks"] = self._default_ticks(data.N)
        P["K_ticks"] = tuple(P["K_ticks"])[:max(1, int(P["index_levels"]))]
        index = MultiResolutionIndex(p_cal_t, stage_one.tgt_membership,
                                     K_ticks=P["K_ticks"], alpha=P["alpha"])
        self.timing["index_construction_s"] += time.time() - t0

        refiner = ProgressiveContextRefiner(
            data, stage_one, index, p_cal_s, p_cal_t, P)
        if P.get("calibrate_lambda", True) and data.N > 1:
            t0 = time.time()
            P["lam"] = self._calibrate_lambda(
                data, stage_one, index, p_cal_s, p_cal_t, refiner)
            # One-time model selection, reported as ``setup_s`` and kept out
            # of the eleven pipeline timers so that the stage profile stays a
            # partition of the per-iteration pipeline cost.
            self.stats["setup_s"] = time.time() - t0
            self.log(f"  [held-out] routing margin lambda={P['lam']}")
            # The calibration probe shares the refiner; discard the timers it
            # accumulated so the profile starts from the first real iteration.
            for k in refiner.timing:
                refiner.timing[k] = 0.0

        src_profiles = {e: s.copy() for e, s in enumerate(p_cal_s)}
        seeds = list(data.seeds)
        s_cum = set()
        R_last = []
        subtask_cov = defaultdict(set)
        coverage = {}
        last_contexts = []
        refiner_prev = {}

        for it in range(1, P["I"] + 1):
            t_it = time.time()
            seed_by_pair = defaultdict(list)
            for (s, t) in seeds:
                i = int(stage_one.src_membership[s])
                j = int(stage_one.tgt_membership[t])
                if j in stage_one.M.get(i, []):
                    seed_by_pair[(i, j)].append((s, t))

            # ---------- Stage II: Algorithm 3 over all retained pairs ----
            self.log(f"[Stage II] Progressive Temporal Context Refinement "
                     f"(Algorithm 3, iteration {it}/{P['I']})")
            all_c5, all_theta, all_routes = {}, {}, {}
            n_queries = n_candidates = 0
            t_stage = time.time()
            for (i, j) in stage_one.pairs():
                c5, theta, routes = refiner.refine_pair(
                    i, j, seed_by_pair.get((i, j), []), src_profiles)
                all_c5[(i, j)] = c5
                all_theta[(i, j)] = theta
                all_routes[(i, j)] = routes
                n_queries += len(c5)
                n_candidates += sum(len(v) for v in c5.values())
            # refiner.timing is cumulative over iterations; transfer only
            # the delta so the eleven-stage profile stays a partition of the
            # wall clock instead of growing quadratically with I.
            for k, v in refiner.timing.items():
                self.timing[k] += v - refiner_prev.get(k, 0.0)
                refiner_prev[k] = v
            self.log(f"  [Stage II] pairs={len(stage_one.pairs())} "
                     f"queries={n_queries} candidates={n_candidates} "
                     f"time={time.time() - t_stage:.1f}s")

            # ---------- Stage III: Algorithm 4 ----------
            self.log("[Stage III] Temporal Error-Aware Optimization "
                     "(context construction + TEA training + TEC)")
            t0 = time.time()
            contexts = []
            for (i, j) in stage_one.pairs():
                c5 = all_c5[(i, j)]
                spairs = seed_by_pair.get((i, j), [])
                if not c5 and not spairs:
                    continue
                score_of = self._make_score_fn(index, j, src_profiles)
                vote_of = (self._make_structural_score_fn(refiner)
                           if data.N <= 1
                           else self._make_vote_fn(refiner, spairs, j))
                ctx = CTX.build_context(
                    data, c5, all_routes[(i, j)], all_theta[(i, j)], spairs,
                    score_of, vote_of, P["M"], P["F"])
                if ctx is not None:
                    ctx["pair"] = (i, j)
                    contexts.append(ctx)
                    for e in ctx["Hs"]:
                        subtask_cov[e].update(ctx["Ht"])
            self.timing["context_construction_s"] += time.time() - t0
            self.log(f"  [Stage III] contexts={len(contexts)} "
                     f"entities={sum(len(c['Hs']) + len(c['Ht']) for c in contexts)} "
                     f"facts={sum(len(c['facts_s']) + len(c['facts_t']) for c in contexts)} "
                     f"time={time.time() - t0:.1f}s")
            self.contexts = contexts

            t0 = time.time()
            if self.backbone is None:
                self.backbone = build_backbone(
                    backbone, data, backbone_params or {})
            self.backbone.fit(contexts, epochs=P["epochs"],
                              src_profiles=src_profiles,
                              tgt_profiles=p_cal_t,
                              log=lambda m: self.log("    " + m))
            self.timing["training_s"] += time.time() - t0
            self.log(f"  [TEA training] backbone={backbone} "
                     f"epochs={P['epochs']} time={time.time() - t0:.1f}s")

            t0 = time.time()
            proposals = []
            coverage = {}
            for ctx in contexts:
                emb = self.backbone.embed(ctx)
                proposals += TEC.infer_proposals(ctx, emb)
                for e, cands in ctx.get("in_c5", {}).items():
                    ys = emb.get(("s", e))
                    if ys is None or not cands:
                        continue
                    ys_n = max(1e-8, float(np.linalg.norm(ys)))
                    dst = coverage.setdefault(e, {})
                    for e2 in cands:
                        yt = emb.get(("t", e2))
                        if yt is None:
                            continue
                        s = float(np.dot(ys, yt) /
                                  max(1e-8, ys_n * float(np.linalg.norm(yt))))
                        if s > dst.get(e2, float("-inf")):
                            dst[e2] = s
            R_set = TEC.resolve(proposals)
            self.timing["inference_resolution_s"] += time.time() - t0

            t0 = time.time()
            if P.get("enable_tec", True):
                s_acc = list(R_set) if P.get("tec_accept_all") \
                    else TEC.accept(R_set, P["delta_c"])
                for (e1, e2, _sp, _sm, _cf) in s_acc:
                    mask = TEC.budget_mask(
                        src_profiles[e1], p_cal_t[e2], index,
                        delta0=P["delta0"], rho_b=P["rho_b"],
                        idf_mask=P.get("idf_mask", True),
                        budget_cap=P.get("budget_cap", True))
                    src_profiles[e1] = TEC.update_profile(
                        src_profiles[e1], mask)
            else:
                # w/o TEC: standard pseudo-seed propagation
                s_acc = list(R_set)
            self.timing["corrector_s"] += time.time() - t0

            for p in s_acc:
                s_cum.add((p[0], p[1]))
            seeds = list(set(seeds) | {(p[0], p[1]) for p in s_acc})
            R_last = R_set
            last_contexts = contexts
            self._save_checkpoint(it, s_acc, s_cum, coverage)
            self.log(f"  [iter {it}] queries={n_queries} "
                     f"candidates={n_candidates} contexts={len(contexts)} "
                     f"proposals={len(R_set)} accepted={len(s_acc)} "
                     f"seeds={len(seeds)} time={time.time() - t_it:.1f}s")
            if not s_acc:
                break

        pred = s_cum | {(p[0], p[1]) for p in R_last}
        # The eleven stage timers are disjoint, so their sum is the Overall
        # row of the runtime breakdown; the elapsed wall clock is kept apart
        # because it also covers one-time lambda selection and I/O.
        self.stats["total_s"] = float(sum(self.timing.values()))
        self.stats["wall_s"] = time.time() - t_start
        for k, v in self.timing.items():
            self.stats[k] = v
        self.stats["peak_gpu_gb"] = self._peak_memory_gb()
        self.stats["n_seeds_final"] = len(seeds)
        self.stats["n_pred"] = len(pred)
        return {
            "predictions": pred,
            "s_cum": s_cum,
            "stats": dict(self.stats),
            "coverage": coverage,
            "subtask_coverage": dict(subtask_cov),
            "contexts": last_contexts,
            "backbone": self.backbone,
            "stage_one": stage_one,
            "index": index,
            "p_cal_s": p_cal_s,
            "p_cal_t": p_cal_t,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _make_score_fn(index, j, src_profiles):
        """Score_3(e,e';P_j) at the finest index level (Eq. 6)."""
        level = index.L - 1
        cache = {}

        def score_of(e, e2):
            key = (e, e2)
            if key not in cache:
                s = 0.0
                sup = (src_profiles.get(e) if isinstance(src_profiles, dict)
                       else (src_profiles[e] if e < len(src_profiles)
                             else None))
                if sup is not None:
                    for b in index.buckets_of(sup, level):
                        if e2 in index.postings_of(level, b, j):
                            s += index.idf_mix(level, b, j)
                cache[key] = s
            return cache[key]
        return score_of

    @staticmethod
    def _make_structural_score_fn(refiner):
        """Relation-profile Jaccard used to rank structural-route pairs."""
        cache = {}

        def score_of(e, e2):
            key = (e, e2)
            if key not in cache:
                a = set(refiner.src_rels.get(e, ()))
                b = set(refiner.tgt_rels.get(e2, ()))
                cache[key] = len(a & b) / max(1, len(a | b))
            return cache[key]
        return score_of

    @staticmethod
    def _make_vote_fn(refiner, seed_pairs, j):
        """vote^(t)(e,e';P_i,P_j) of Eq. (31)."""
        seed_map = dict(seed_pairs)
        part_j_set = set(refiner.so.tgt_members[j])

        def vote_of(e, e2):
            v = 0
            for u in refiner.src_adj.get(e, ()):
                counterpart = seed_map.get(u)
                if counterpart is not None \
                        and e2 in refiner.tgt_adj.get(counterpart, ()) \
                        and e2 in part_j_set:
                    v += 1
            return v
        return vote_of

    # ------------------------------------------------------------------
    def _calibrate_lambda(self, data, stage_one, index, p_cal_s, p_cal_t,
                          refiner, n_hide=60,
                          grid=(1.2, 1.5, 1.8, 2.0, 3.0)):
        """Select lambda on held-out seed pairs (report Appendix C-C).

        A sample of seed pairs is hidden from the seed set; for each lambda in
        the grid the refiner is run on the affected partition pairs and the
        candidate truth-in-C5 rate is measured.  The lambda with the best rate
        is kept.
        """
        if not data.seeds:
            return self.params["lam"]
        rng = np.random.RandomState(int(self.params.get("seed", 0)) + 7)
        n_hide = min(n_hide, max(1, len(data.seeds) // 10))
        hidden_idx = rng.choice(len(data.seeds), n_hide, replace=False)
        hidden = [data.seeds[k] for k in hidden_idx]
        hidden_set = set(hidden)
        kept = [p for p in data.seeds if p not in hidden_set]
        by_pair = defaultdict(list)
        for (u, v) in hidden:
            i = int(stage_one.src_membership[u])
            j = int(stage_one.tgt_membership[v])
            if j in stage_one.M.get(i, []):
                by_pair[(i, j)].append((u, v))
        if not by_pair:
            return self.params["lam"]
        kept_by_pair = defaultdict(list)
        for (s, t) in kept:
            i = int(stage_one.src_membership[s])
            j = int(stage_one.tgt_membership[t])
            if j in stage_one.M.get(i, []):
                kept_by_pair[(i, j)].append((s, t))
        profiles = {e: s.copy() for e, s in enumerate(p_cal_s)}
        pairs = sorted(by_pair)[:8]
        best_lam, best_rate = self.params["lam"], -1.0
        for lam in grid:
            self.params["lam"] = lam
            hit = total = 0
            for (i, j) in pairs:
                c5, _theta, _routes = refiner.refine_pair(
                    i, j, kept_by_pair.get((i, j), []), profiles)
                for (u, v) in by_pair[(i, j)]:
                    total += 1
                    hit += int(v in c5.get(u, ()))
            rate = hit / max(1, total)
            if rate > best_rate:
                best_lam, best_rate = lam, rate
        return best_lam

    # ------------------------------------------------------------------
    def _save_checkpoint(self, iteration, accepted, s_cum, coverage):
        if not self.params.get("save_checkpoints"):
            return
        base = self.params.get("checkpoint_dir")
        if not base:
            return
        d = os.path.join(base, f"iteration_{iteration:02d}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "pseudo_seeds.json"), "w") as f:
            json.dump({
                "iteration": iteration,
                "accepted": [[int(a), int(b)] for (a, b, *_r) in accepted],
                "cumulative": [[int(a), int(b)] for (a, b) in sorted(s_cum)],
                "n_accepted": len(accepted),
                "n_cumulative": len(s_cum),
                "n_queries": len(coverage),
            }, f, indent=2)
        if self.backbone is not None:
            try:
                import torch
                sd = self.backbone.state_dict()
                if sd:
                    torch.save(sd, os.path.join(d, "backbone.pt"))
            except Exception:
                pass