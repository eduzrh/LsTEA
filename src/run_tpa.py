#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TPA entry point: run the full pipeline on one dataset / backbone / seed.

Examples
--------
    python src/run_tpa.py --data_dir data/DICEWS-10K --backbone trea \
        --seed 1011 --out results/backbone_comparison/TPA/DICEWS-10K/trea_seed1011.json
    python src/run_tpa.py --data_dir data/YAGO-WIKI50K --backbone lighttea \
        --config configs/tpa_default.json --set route_only=tem

The emitted JSON is a self-contained run record: the resolved configuration,
dataset statistics, the effectiveness metrics (Hits@1, Hits@5, MRR,
counterpart recall), a per-stage wall-clock profile of the pipeline
(partitioning, index construction, refinement, context construction,
training, inference/resolution, corrector), and the peak accelerator
memory.  Reported averages are aggregated from these per-seed records.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tpa.dataload import TEAData
from tpa.tpa_framework import TPAFramework
from tpa.evaluate import evaluate, subtask_recall, hits_over_recall

# Eleven-stage wall-clock profile of the pipeline, in execution order.
STAGE_TIMERS = [
    "profile_construction_s",
    "dual_partitioner_s",
    "partition_matching_s",
    "index_construction_s",
    "routing_cascade_s",
    "batch_temporal_scan_s",
    "local_structure_verification_s",
    "context_construction_s",
    "training_s",
    "inference_resolution_s",
    "corrector_s",
]


def K_for_scale(n_ent):
    """Per-side partition count by dataset scale (Section IV-A3)."""
    if n_ent <= 20000:
        return 5
    if n_ent <= 120000:
        return 10
    if n_ent <= 1500000:
        return 100
    return 200


def _load_config(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _apply_sets(config, assignments):
    """Apply ``--set key=value`` overrides (JSON-decoded when possible)."""
    for item in assignments or []:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except Exception:
            value = raw
        config[key] = value
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--backbone", default="trea",
                    choices=["gcn_align", "tea_gnn", "rrea", "trea",
                             "dual_amn", "lighttea"])
    ap.add_argument("--config", default=None,
                    help="JSON config (defaults + experiment overrides)")
    ap.add_argument("--set", action="append", default=[],
                    help="override one config key, e.g. --set lam=1.8")
    ap.add_argument("--tag", default=None,
                    help="experiment tag written into the result record")
    ap.add_argument("--experiment", default=None,
                    help="experiment identifier, e.g. component_ablation")
    ap.add_argument("--variant", default="TPA",
                    help="variant label of this run, e.g. w/o TRF")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--K_t", type=int, default=None)
    ap.add_argument("--I", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--F", type=int, default=None)
    ap.add_argument("--delta_c", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--tau_struct", type=float, default=None)
    ap.add_argument("--rho_m", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--zeta", type=float, default=None)
    ap.add_argument("--budgets", default=None,
                    help="cascade budgets b1,b2,b3")
    ap.add_argument("--quantiles", default=None, help="eta2,eta3")
    ap.add_argument("--b_struct", type=int, default=None)
    ap.add_argument("--index_levels", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--route_only", choices=["tem", "str"], default=None)
    ap.add_argument("--no_trf", action="store_true")
    ap.add_argument("--no_lsv", action="store_true")
    ap.add_argument("--no_tec", action="store_true")
    ap.add_argument("--no_time_aware", action="store_true")
    ap.add_argument("--no_tc", action="store_true")
    ap.add_argument("--no_tm", action="store_true")
    ap.add_argument("--no_margin", action="store_true")
    ap.add_argument("--no_idf_mask", action="store_true")
    ap.add_argument("--no_budget", action="store_true")
    ap.add_argument("--strip_timestamps", action="store_true",
                    help="static protocol / TPA without timestamps (Appendix E-B)")
    ap.add_argument("--time_coverage", type=float, default=1.0)
    ap.add_argument("--time_granularity", choices=["day", "month", "year"],
                    default=None,
                    help="controlled calendar resolution (Appendix E-F)")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1011,
                    help="model / partition RNG seed")
    ap.add_argument("--split_seed", type=int, default=1011,
                    help="fixed 20/80 benchmark split seed")
    ap.add_argument("--alpha", type=float, default=None,
                    help="IDF_local / IDF_global mix (Eq. 6)")
    ap.add_argument("--h", type=int, default=None,
                    help="ego-network depth for the 13-d signature")
    ap.add_argument("--delta0", type=int, default=None,
                    help="minimum profile-update budget Delta_0")
    ap.add_argument("--rho_b", type=float, default=None,
                    help="profile-update budget ratio rho_b")
    ap.add_argument("--epsilon_bal", type=float, default=None,
                    help="METIS balance tolerance (Eq. 3)")
    ap.add_argument("--K_ticks", default=None,
                    help="comma-separated bucket counts per index level")
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = _apply_sets(_load_config(args.config), args.set)
    t0 = time.time()
    data = TEAData(args.data_dir, seed=args.split_seed,
                   time_coverage=args.time_coverage,
                   static=args.strip_timestamps,
                   time_granularity=args.time_granularity)

    K = args.K or config.get("K") or K_for_scale(
        max(data.gs.n_ent, data.gt.n_ent))
    params = {
        "K_s": K,
        "K_t": args.K_t or config.get("K_t") or K,
        "seed": args.seed,
        "alpha": args.alpha if args.alpha is not None
        else config.get("alpha", 0.9),
        "h": args.h if args.h is not None else config.get("h", 2),
        "delta0": args.delta0 if args.delta0 is not None
        else config.get("delta0", 5),
        "rho_b": args.rho_b if args.rho_b is not None
        else config.get("rho_b", 0.1),
        "epsilon_bal": args.epsilon_bal if args.epsilon_bal is not None
        else config.get("epsilon_bal", 0.1),
        "I": args.I if args.I is not None else config.get("I", 5),
        "epochs": args.epochs if args.epochs is not None
        else config.get("epochs", 50),
        "M": args.M if args.M is not None else config.get("M", 10000),
        "F": args.F if args.F is not None else config.get("F", 50000),
        "delta_c": args.delta_c if args.delta_c is not None
        else config.get("delta_c", 0.15),
        "lam": args.lam if args.lam is not None else config.get("lam", 1.5),
        "tau_struct": args.tau_struct if args.tau_struct is not None
        else config.get("tau_struct", 0.5),
        "rho_m": args.rho_m if args.rho_m is not None
        else config.get("rho_m", 0.2),
        "gamma": args.gamma if args.gamma is not None
        else config.get("gamma", 0.5),
        "zeta": args.zeta if args.zeta is not None
        else config.get("zeta", 1.0),
        "enable_trf": not args.no_trf and config.get("enable_trf", True),
        "enable_lsv": not args.no_lsv and config.get("enable_lsv", True),
        "enable_tec": not args.no_tec and config.get("enable_tec", True),
        "time_aware": not args.no_time_aware
        and config.get("time_aware", True),
        "tc_enabled": not args.no_tc and config.get("tc_enabled", True),
        "tm_enabled": not args.no_tm and config.get("tm_enabled", True),
        "route_only": args.route_only or config.get("route_only"),
        "index_levels": args.index_levels or config.get("index_levels", 3),
        "tec_accept_all": args.no_margin or config.get("tec_accept_all", False),
        "loss_margin": config.get("loss_margin", True),
        "idf_mask": not args.no_idf_mask and config.get("idf_mask", True),
        "budget_cap": not args.no_budget and config.get("budget_cap", True),
        "calibrate_lambda": config.get("calibrate_lambda", True),
        "time_coverage": args.time_coverage,
        "time_granularity": args.time_granularity,
        "strip_timestamps": args.strip_timestamps,
        "checkpoint_dir": args.checkpoint_dir,
    }
    if args.K_ticks:
        params["K_ticks"] = tuple(int(x) for x in args.K_ticks.split(","))
    elif config.get("K_ticks"):
        params["K_ticks"] = tuple(int(x) for x in config["K_ticks"])
    if args.budgets:
        params["budgets"] = tuple(int(x) for x in args.budgets.split(","))
    elif "budgets" in config:
        params["budgets"] = tuple(config["budgets"])
    if args.quantiles:
        params["quantiles"] = tuple(float(x) for x in args.quantiles.split(","))
    elif "quantiles" in config:
        params["quantiles"] = tuple(config["quantiles"])
    if args.b_struct is not None:
        params["b_struct"] = args.b_struct
    elif "b_struct" in config:
        params["b_struct"] = config["b_struct"]

    print(f"dataset={data.name} |E_s|={data.gs.n_ent} |E_t|={data.gt.n_ent} "
          f"|Q_s|={len(data.gs.triples)} |Q_t|={len(data.gt.triples)} "
          f"N={data.N} seeds={len(data.seeds)} test={len(data.test_pairs)} "
          f"K={K}", flush=True)

    bp = {"dim": args.dim, "lr": args.lr, "seed": args.seed}
    for key in ("n_neg", "margin", "temperature", "beta", "n_prop",
                "lighttea_alpha", "lighttea_dim", "lighttea_mini_dim"):
        if key in config:
            bp[key] = config[key]
    bp_run = dict(bp)
    bp_run["loss_margin"] = params["loss_margin"]
    if args.device:
        bp_run["device"] = args.device

    pipe = TPAFramework(data, params, verbose=True)
    out = pipe.run(backbone=args.backbone, backbone_params=bp_run)
    metrics = evaluate(data.test_pairs, out["predictions"], out["coverage"])
    metrics["recall"] = subtask_recall(data.test_pairs,
                                       out["subtask_coverage"])
    metrics["hit1_over_recall"] = hits_over_recall(metrics)

    # Runtime block: eleven stage timers plus the total and the peak memory
    # of the run.  Benchmarks at the 100K scale and above are reported in
    # hours, the 15K/50K benchmarks in seconds; the stage rows follow that
    # reporting unit while ``total_s`` keeps the raw seconds.
    unit = "h" if max(data.gs.n_ent, data.gt.n_ent) >= 100000 else "s"
    scale = 3600.0 if unit == "h" else 1.0
    runtime = {k: float(out["stats"][k]) / scale for k in STAGE_TIMERS
               if k in out["stats"]}
    total_s = float(out["stats"].get("total_s", 0.0))
    runtime["total_s"] = total_s
    # One-time lambda selection (Appendix C-C) is kept outside the eleven
    # pipeline rows; ``total_s`` is their sum and equals ``reported_total``.
    if "setup_s" in out["stats"]:
        runtime["setup_s"] = float(out["stats"]["setup_s"]) / scale
    runtime["peak_gpu_gb"] = float(out["stats"].get("peak_gpu_gb", 0.0))
    runtime["reported_total"] = total_s / scale
    runtime["time_unit"] = unit

    cfg = {k: (list(v) if isinstance(v, tuple) else v)
           for k, v in pipe.params.items()
           if k not in ("checkpoint_dir", "save_checkpoints", "K_ticks")}
    if args.K_ticks:
        cfg["K_ticks"] = list(pipe.params["K_ticks"])
    if cfg.get("time_granularity") is None:
        cfg.pop("time_granularity", None)
    if not cfg.get("strip_timestamps"):
        cfg.pop("strip_timestamps", None)
    cfg.update({k: v for k, v in bp.items() if v is not None})
    if args.split_seed != 1011:
        cfg["split_seed"] = args.split_seed

    experiment = (args.experiment
                  or (args.tag or "").split("/")[0] or None)
    record = {
        "experiment": args.tag or "",
        "experiment_dir": experiment,
        "dataset": data.name,
        "backbone": args.backbone,
        "method": "TPA",
        "variant": args.variant,
        "seed": args.seed,
        "config": cfg,
        "backbone_config": bp,
        "dataset_statistics": data.statistics(),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "runtime": runtime,
        "n_predictions": len(out["predictions"]),
        "n_cumulative_pseudo_seeds": len(out["s_cum"]),
        "wall_s": time.time() - t0,
    }
    print(json.dumps(record["metrics"], indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()