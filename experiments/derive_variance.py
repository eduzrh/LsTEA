#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive the run-to-run standard deviations of the main comparison.

The five per-seed records of ``results/backbone_comparison`` are the
run-level evidence behind the main results.  This script recomputes the sample
standard deviation (ddof = 1) of Hits@1, MRR and counterpart recall for every
TPA configuration and writes ``results/seed_variance/summary.csv`` together
with the published standard deviations for comparison.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "seed_variance")
REFERENCE = os.path.join(ROOT, "results", "reference",
                         "reported_results.json")
DATASETS = ["DICEWS-10K", "YAGO-WIKI50K", "T-2M", "YAGO-WIKI-2M"]
BB = {"GCN-Align": "gcn_align", "TEA-GNN": "tea_gnn", "RREA": "rrea",
      "TREA": "trea", "Dual-AMN": "dual_amn", "LightTEA": "lighttea"}


def _load():
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(
            ROOT, "results", "backbone_comparison", "**", "*.json"),
            recursive=True)):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("metrics"):
            groups[(rec["dataset"], rec["backbone"])].append(rec)
    return groups


def _paper_std():
    with open(REFERENCE, encoding="utf-8") as f:
        reference = json.load(f)
    out = {}
    for row in reference["seed_variance"]:
        if row["method"] != "TPA":
            continue
        backbone = BB[row["backbone"]]
        for i, ds in enumerate(DATASETS):
            h1, mrr, rec = row["values"][i * 3:(i + 1) * 3]
            out[(ds, backbone)] = (h1, mrr, rec)
    return out


def main():
    paper = _paper_std()
    rows = []
    for (ds, backbone), runs in sorted(_load().items()):
        if len(runs) < 2:
            continue
        def stat(key):
            vals = np.asarray([r["metrics"][key] for r in runs],
                              dtype=np.float64) * 100.0
            return float(vals.mean()), float(vals.std(ddof=1))
        h1m, h1s = stat("hit1")
        mm, ms = stat("mrr")
        rm, rs = stat("recall")
        ph, pm, pr = paper.get((ds, backbone), (None, None, None))
        rows.append({
            "dataset": ds, "backbone": backbone, "method": "TPA",
            "n_runs": len(runs),
            "H@1_mean": round(h1m, 4), "H@1_std": round(h1s, 4),
            "MRR_mean": round(mm, 4), "MRR_std": round(ms, 4),
            "Rec_mean": round(rm, 4), "Rec_std": round(rs, 4),
            "paper_H@1_std": ph, "paper_MRR_std": pm,
            "paper_Rec_std": pr})
    os.makedirs(OUT, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(os.path.join(OUT, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = ["# seed_variance", "",
             "Standard deviations recomputed from the five seed records of "
             "`results/backbone_comparison` (sample standard deviation, ddof = 1).",
             "",
             "| dataset | backbone | n_runs | H@1 (mean ± std) | "
             "MRR (mean ± std) | Rec. (mean ± std) | paper std (H@1/MRR/Rec.) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %d | %.2f ± %.2f | %.2f ± %.2f | "
                     "%.2f ± %.2f | %s / %s / %s |" % (
                         r["dataset"], r["backbone"], r["n_runs"],
                         r["H@1_mean"], r["H@1_std"], r["MRR_mean"],
                         r["MRR_std"], r["Rec_mean"], r["Rec_std"],
                         r["paper_H@1_std"], r["paper_MRR_std"],
                         r["paper_Rec_std"]))
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %d TPA configurations to %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()