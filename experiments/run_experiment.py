#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one registered experiment over its five seeds.

Examples
--------
    python experiments/run_experiment.py backbone_comparison --data_root data

    python experiments/run_experiment.py partition_count_sweep --dry_run

Missing datasets are skipped and reported in ``run_manifest.json`` so that the
repository can be exercised on a machine that only hosts part of the LsTEA
benchmarks.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import registry  # noqa: E402


def _slug(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def run_setting(spec, seeds, args):
    data_root = os.path.abspath(args.data_root)
    data_dir = os.path.join(data_root, spec["data_dir"])
    if not os.path.isdir(data_dir):
        return {"status": "missing", "dataset": spec["dataset"],
                "data_dir": data_dir}

    cfg = os.path.abspath(args.config)
    experiment_dir = os.path.join(args.out, _slug(spec["variant"]),
                                  _slug(spec["dataset"]))
    log_dir = os.path.join(args.log_dir, spec["experiment"])
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    done = []
    for seed in seeds:
        out_json = os.path.join(
            experiment_dir, "%s_seed%d.json" % (_slug(spec["backbone"]), seed))
        if os.path.exists(out_json) and not args.force:
            done.append(out_json)
            continue
        cmd = [sys.executable, os.path.join(ROOT, "src", "run_tpa.py"),
               "--data_dir", data_dir,
               "--backbone", spec["backbone"],
               "--config", cfg,
               "--seed", str(seed),
               "--experiment", spec["experiment"],
               "--variant", spec["variant"],
               "--tag", "%s/%s/%s" % (spec["experiment"], spec["variant"],
                                      spec["dataset"]),
               "--out", out_json]
        if spec.get("K") and "--K" not in spec["args"]:
            cmd += ["--K", str(spec["K"])]
        cmd += spec["args"]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.device:
            cmd += ["--device", args.device]
        log_path = os.path.join(
            log_dir, "%s_%s_%s_seed%d.log" % (
                _slug(spec["dataset"]), _slug(spec["backbone"]),
                _slug(spec["variant"]), seed))
        with open(log_path, "w") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=ROOT)
        if proc.returncode != 0 or not os.path.exists(out_json):
            return {"status": "failed", "dataset": spec["dataset"],
                    "variant": spec["variant"], "backbone": spec["backbone"],
                    "log": log_path}
        done.append(out_json)
    return {"status": "ok", "runs": done}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--config", default=os.path.join(
        ROOT, "configs", "tpa_default.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--log_dir", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seed list (default: registry)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.experiment not in registry.EXPERIMENTS:
        raise SystemExit("unknown experiment: %s" % args.experiment)
    entry = registry.EXPERIMENTS[args.experiment]
    if entry["kind"] == "derived":
        raise SystemExit("%s is derived; run its source experiment instead"
                         % args.experiment)
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds
             else registry.SEEDS)
    args.out = args.out or os.path.join(ROOT, "results", args.experiment)

    specs = list(registry.iter_settings(args.experiment))
    if args.dry_run:
        for spec in specs:
            print(json.dumps(spec))
        return

    results = []
    for k, spec in enumerate(specs, 1):
        print("[%d/%d] %s | %s | %s" % (
            k, len(specs), spec["variant"], spec["dataset"],
            spec["backbone"]), flush=True)
        results.append(run_setting(spec, seeds, args))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_manifest.json"), "w") as f:
        json.dump({"experiment": args.experiment, "seeds": seeds,
                   "results": results}, f, indent=2)
    missing = [r for r in results if r["status"] != "ok"]
    print("finished: %d/%d settings ok" % (len(results) - len(missing),
                                           len(results)))


if __name__ == "__main__":
    main()