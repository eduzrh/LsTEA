#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate per-seed JSON records produced by
``experiments/run_experiment.py``.

Writes ``summary.csv`` and ``summary.md`` next to the run records.  Each row
is one (dataset, backbone, variant) configuration and reports the mean and
sample standard deviation over its seeds for Hits@1/5, MRR, counterpart
recall, Hits@1-over-recall, wall time and peak memory.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRICS = [("hit1", "H@1"), ("hit5", "H@5"), ("mrr", "MRR"),
           ("recall", "Rec."), ("hit1_over_recall", "H@1/Rec.")]
RUNTIME = [("total_s", "Time"), ("peak_gpu_gb", "Mem.")]


def _load_runs(root):
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"),
                                 recursive=True)):
        if os.path.basename(path) in ("run_manifest.json", "summary.json"):
            continue
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if "metrics" not in rec:
            continue
        key = (rec.get("dataset"), rec.get("backbone"),
               rec.get("experiment") or rec.get("method", "TPA"))
        groups[key].append(rec)
    return groups


def _stat(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def summarize(root):
    groups = _load_runs(root)
    rows = []
    for (dataset, backbone, experiment), runs in sorted(groups.items()):
        row = {"dataset": dataset, "backbone": backbone,
               "experiment": experiment, "n_runs": len(runs)}
        for field, label in METRICS:
            vals = [r["metrics"].get(field, 0.0) for r in runs]
            mean, std = _stat(vals)
            row[label] = round(mean * 100.0, 4)
            row[label + "_std"] = round(std * 100.0, 4)
        for field, label in RUNTIME:
            if field == "total_s":
                vals = [r["runtime"].get("reported_total",
                                         r["runtime"].get("total_s", 0.0))
                        for r in runs]
            else:
                vals = [r["runtime"].get(field, 0.0) for r in runs]
            mean, std = _stat(vals)
            row[label] = round(mean, 2)
            row[label + "_std"] = round(std, 2)
        row["time_unit"] = runs[0]["runtime"].get("time_unit", "s")
        rows.append(row)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows, path, title):
    if not rows:
        return
    cols = ["dataset", "backbone", "experiment", "n_runs",
            "H@1", "H@1_std", "H@5", "MRR", "Rec.", "H@1/Rec.",
            "Time", "time_unit", "Mem."]
    lines = ["# " + title, "",
             "Mean ± sample standard deviation over five seeds.", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            if c in ("H@1", "H@5", "MRR", "Rec.", "H@1/Rec."):
                cells.append("%.2f ± %.2f" % (r[c], r.get(c + "_std", 0.0)))
            else:
                cells.append(str(r.get(c, "")))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def summarize_stages(root):
    """Per-stage runtime and peak memory over the five runs."""
    groups = defaultdict(lambda: defaultdict(list))
    units = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"),
                                 recursive=True)):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if "runtime" not in rec:
            continue
        ds = rec.get("dataset")
        units[ds] = rec["runtime"].get("time_unit", "s")
        for key, val in rec["runtime"].items():
            # ``setup_s`` (one-time lambda selection) is not a pipeline stage.
            if key.endswith("_s") and key not in ("total_s", "setup_s"):
                groups[ds][key].append(float(val))
    rows = []
    for ds in sorted(groups):
        for key, vals in sorted(groups[ds].items()):
            mean, std = _stat(vals)
            rows.append({"dataset": ds, "stage": key,
                         "Time": round(mean, 3), "Time_std": round(std, 3),
                         "time_unit": units.get(ds, "s"),
                         "n_runs": len(vals)})
    if rows:
        write_csv(rows, os.path.join(root, "summary_stages.csv"))
        lines = ["# Per-stage runtime", "",
                 "Mean ± sample standard deviation over five seeds.", "",
                 "| dataset | stage | Time | time_unit | n_runs |",
                 "|---|---|---|---|---|"]
        for r in rows:
            lines.append("| %s | %s | %.2f ± %.2f | %s | %d |" % (
                r["dataset"], r["stage"], r["Time"], r["Time_std"],
                r["time_unit"], r["n_runs"]))
        with open(os.path.join(root, "summary_stages.md"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory with per-seed JSON records")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    rows = summarize(args.root)
    write_csv(rows, os.path.join(args.root, "summary.csv"))
    write_markdown(rows, os.path.join(args.root, "summary.md"),
                   args.title or os.path.basename(os.path.normpath(args.root)))
    if os.path.basename(os.path.normpath(args.root)) == "stage_runtime":
        summarize_stages(args.root)
    print("aggregated %d configurations" % len(rows))


if __name__ == "__main__":
    main()