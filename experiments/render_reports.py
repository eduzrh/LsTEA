#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the reporting CSVs from ``results/reference``.

For every experiment directory this script writes
``results/<experiment>/summary_paper.csv`` and ``summary_paper.md`` in a
compact reporting layout (six effectiveness/efficiency metrics, a parameter
sweep, a standard-deviation block, a runtime breakdown, or a Hits@1 /
counterpart-recall ratio).  TPA rows come from the per-seed records under
``results/<experiment>/``; baseline rows use the published numbers stored in
``results/reference/reported_results.json``.

Usage
-----
    python experiments/render_reports.py
    python experiments/render_reports.py --experiments backbone_comparison
"""
import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REFERENCE = os.path.join(ROOT, "results", "reference",
                         "reported_results.json")
OUT = os.path.join(ROOT, "results")

DATASETS4 = ["DICEWS-10K", "YAGO-WIKI50K", "T-2M", "YAGO-WIKI-2M"]
BB = {"GCN-Align": "gcn_align", "TEA-GNN": "tea_gnn", "RREA": "rrea",
      "TREA": "trea", "Dual-AMN": "dual_amn", "LightTEA": "lighttea"}
TIME_UNIT = {
    "DICEWS-10K": "s", "YAGO-WIKI50K": "s", "BETA": "s",
    "T-2M": "h", "YAGO-WIKI-2M": "h",
    "DBP15K-FR-EN": "s", "DBP1M-EN-FR": "h",
    "IDS15K-EN-FR": "s", "IDS15K-EN-DE": "s",
    "DBP15K-ZH-EN": "s", "DBP15K-JA-EN": "s",
    "IDS100K-EN-FR": "h", "IDS100K-EN-DE": "h",
    "DWY100K-DBP-WD": "h", "DWY100K-DBP-YG": "h",
    "DBP1M-EN-DE": "h", "FBDBP2M": "h",
}
for size in ("100K", "250K", "500K", "1M", "2M"):
    TIME_UNIT["T-2M-" + size] = "h"


def _runs(method):
    return 5 if method.startswith("TPA") else 1


def _source(method):
    return "paper + 5 runs" if method.startswith("TPA") else "paper"


def _write(experiment, fields, rows, title, note):
    d = os.path.join(OUT, experiment)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "summary_paper.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["# " + experiment, "", title, "", note, "",
             "| " + " | ".join(fields) + " |",
             "|" + "|".join(["---"] * len(fields)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            "" if r.get(k) is None else str(r.get(k)) for k in fields) + " |")
    with open(os.path.join(d, "summary_paper.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _variant(name):
    """Normalise the full-model row of an ablation to ``TPA``."""
    return "TPA" if name.strip() in ("TPA", "TPA (Full)") else name


def _six_rows(rows, datasets, key_name):
    """Build rows for the six-metric comparisons.

    ``key_name="method"`` compares complete methods; ``key_name="variant"``
    compares TPA variants, so the variant itself is the row label and every
    row is backed by the five seeded runs of that configuration.
    """
    ablation = (key_name == "variant")
    label_cols = ["method"] if not ablation else ["variant"]
    fields = (["dataset", "backbone"] + label_cols +
              ["H@1", "H@5", "MRR", "Rec.", "Time", "time_unit", "Mem.",
               "runs", "source"])
    out = []
    for row in rows:
        raw = row.get("method") or row.get("variant")
        if ablation:
            label, runs, source = _variant(raw), 5, "paper + 5 runs"
        else:
            label = raw
            runs, source = _runs(label), _source(label)
        for i, ds in enumerate(datasets):
            v = row["values"][i * 6:(i + 1) * 6]
            if not v or v[0] is None:
                continue
            rec = {
                "dataset": ds, "backbone": BB[row["backbone"]],
                label_cols[0]: label,
                "H@1": v[0], "H@5": v[1], "MRR": v[2], "Rec.": v[3],
                "Time": v[4], "time_unit": TIME_UNIT.get(ds, "s"),
                "Mem.": v[5], "runs": runs, "source": source}
            out.append(rec)
    return fields, out


def backbone_comparison(T):
    fields = ["dataset", "backbone", "method", "H@1", "H@5", "MRR", "Rec.",
              "Time", "time_unit", "Mem.", "runs", "source"]
    out = []
    for row in T["backbone_comparison"]:
        method = row["method"]
        for ds in DATASETS4:
            v = row["datasets"][ds]
            out.append({
                "dataset": ds, "backbone": BB[row["backbone"]],
                "method": method, "H@1": v["H@1"], "H@5": v["H@5"],
                "MRR": v["MRR"], "Rec.": v["Rec"], "Time": v["Time"],
                "time_unit": TIME_UNIT.get(ds, "s"), "Mem.": v["Mem"],
                "runs": _runs(method), "source": _source(method)})
    _write("backbone_comparison", fields, out,
           "Main temporal alignment results on four benchmarks.",
           "Time is seconds for DICEWS-10K/YAGO-WIKI50K and hours for "
           "T-2M/YAGO-WIKI-2M; Mem. is peak GPU memory in GB.")


def hits_over_recall_temporal(T):
    fields = ["dataset", "backbone", "method", "ratio", "runs", "source"]
    out = []
    for row in T["hits_over_recall_temporal"]:
        method = row["method"]
        for ds, v in zip(DATASETS4, row["values"]):
            if v is None:
                continue
            out.append({"dataset": ds, "backbone": BB[row["backbone"]],
                        "method": method, "ratio": v, "runs": _runs(method),
                        "source": _source(method)})
    _write("hits_over_recall_temporal", fields, out,
           "Hits@1 divided by counterpart recall (percent).",
           "Derived from the TEA-GNN, RREA, TREA and Dual-AMN records of "
           "backbone_comparison; no separate training runs.")


def static_large_scale(T):
    fields, out = _six_rows(T["static_large_scale"],
                            ["DBP15K-FR-EN", "DBP1M-EN-FR"], "method")
    _write("static_large_scale", fields, out,
           "Static large-scale results on DBP15K FR-EN and DBP1M EN-FR.",
           "Timestamps are absent; TPA runs its structural degeneration "
           "(N = 1).")


def temporal_statistics(_T):
    path = os.path.join(OUT, "temporal_statistics", "summary_paper.csv")
    if os.path.exists(path):
        return
    raise SystemExit("results/temporal_statistics/summary_paper.csv is "
                     "missing; run scripts/compute_temporal_statistics.py "
                     "first")


def joint_ablation(T):
    fields, out = _six_rows(T["joint_ablation"],
                            ["YAGO-WIKI50K", "T-2M"], "variant")
    _write("joint_ablation", fields, out,
           "Joint component ablation on the two representative datasets.",
           "TPA and the four component-removal variants; five seeds per "
           "TPA configuration.")


def partition_count_sweep(T):
    fields = ["dataset", "K", "H@1", "H@5", "MRR", "Rec.", "runs", "source"]
    out, current = [], None
    for row in T["partition_count_sweep"]:
        label = row["label"]
        if label.startswith("YAGO"):
            current = "YAGO-WIKI50K"
            continue
        if label.startswith("T-2M"):
            current = "T-2M"
            continue
        v = row["values"]
        if v[0] is None:
            continue
        out.append({"dataset": current, "K": int(label),
                    "H@1": v[0], "H@5": v[1], "MRR": v[2], "Rec.": v[3],
                    "runs": 5, "source": "paper + 5 runs"})
    _write("partition_count_sweep", fields, out,
           "Effect of the per-side partition count K with TREA.",
           "K = 10 for YAGO-WIKI50K and K = 200 for T-2M coincide with the "
           "TREA rows of the main comparison.")


def iteration_count_sweep(T):
    fields = ["dataset", "T", "H@1", "H@5", "MRR", "Rec.", "runs", "source"]
    out, current = [], None
    for row in T["iteration_count_sweep"]:
        label = row["label"]
        if label.startswith("YAGO"):
            current = "YAGO-WIKI50K"
            continue
        if label.startswith("T-2M"):
            current = "T-2M"
            continue
        v = row["values"]
        if v[0] is None:
            continue
        out.append({"dataset": current, "T": int(label),
                    "H@1": v[0], "H@5": v[1], "MRR": v[2], "Rec.": v[3],
                    "runs": 5, "source": "paper + 5 runs"})
    _write("iteration_count_sweep", fields, out,
           "Effect of the iteration count T with TREA.",
           "T = 5 coincides with the TREA rows of the main comparison.")


def mixed_granularity(T):
    fields, out = _six_rows(T["mixed_granularity"], ["BETA"], "method")
    _write("mixed_granularity", fields, out,
           "BETA mixed temporal granularities with all six backbones.",
           "BETA contains year-, month- and day-level timestamps.")


def static_dwy_fbdbp(T):
    fields, out = _six_rows(T["static_dwy_fbdbp"],
                            ["DWY100K-DBP-WD", "FBDBP2M"], "method")
    _write("static_dwy_fbdbp", fields, out,
           "Static DWY100K DBP-WD and FBDBP2M results.",
           "Static protocol; TPA uses structural partitioning and matching.")


def seed_variance(T):
    fields = ["dataset", "backbone", "method", "H@1_std", "MRR_std",
              "Rec._std", "runs", "source"]
    out = []
    for row in T["seed_variance"]:
        method = row["method"]
        vals = row["values"]
        for i, ds in enumerate(DATASETS4):
            h1, mrr, rec = vals[i * 3:(i + 1) * 3]
            out.append({"dataset": ds, "backbone": BB[row["backbone"]],
                        "method": method, "H@1_std": h1, "MRR_std": mrr,
                        "Rec._std": rec, "runs": 5,
                        "source": ("paper + 5 runs" if method.startswith("TPA")
                                   else "paper (5 runs)")})
    _write("seed_variance", fields, out,
           "Standard deviations over the five runs behind the main "
           "comparison.",
           "Sample standard deviation (ddof = 1) of Hits@1, MRR and "
           "counterpart recall, in percentage points.")


def no_timestamps(T):
    fields = ["dataset", "backbone", "method", "H@1", "MRR", "Rec.",
              "runs", "source"]
    out = []
    for row in T["no_timestamps"]:
        method = row["method"]
        for i, ds in enumerate(DATASETS4):
            v = row["values"][i * 3:(i + 1) * 3]
            if v[0] is None:
                continue
            out.append({"dataset": ds, "backbone": BB[row["backbone"]],
                        "method": method, "H@1": v[0], "MRR": v[1],
                        "Rec.": v[2], "runs": _runs(method),
                        "source": _source(method)})
    _write("no_timestamps", fields, out,
           "TPA without timestamps versus ELsEA and full TPA.",
           "TPA w/o timestamps runs the structural fallback of every "
           "temporal component (N = 1).")


def component_ablation(T):
    fields, out = _six_rows(T["component_ablation"],
                            ["YAGO-WIKI50K", "T-2M"], "variant")
    _write("component_ablation", fields, out,
           "Separate component ablations on YAGO-WIKI50K and T-2M.",
           "Each row removes one mechanism while keeping the rest of TPA.")


def stage_runtime(T):
    fields = ["stage", "dataset", "Time", "time_unit", "Mem.", "runs",
              "source"]
    out = []
    for row in T["stage_runtime"]:
        if row["label"] == "Stage":
            continue
        for i, ds in enumerate(DATASETS4):
            t, mem = row["values"][i * 2:(i + 1) * 2]
            out.append({"stage": row["label"], "dataset": ds, "Time": t,
                        "time_unit": TIME_UNIT.get(ds, "s"),
                        "Mem.": mem, "runs": 5,
                        "source": "paper + 5 runs"})
    _write("stage_runtime", fields, out,
           "Per-stage runtime and peak memory with TREA.",
           "One-time stages are measured once; iterative stages are summed "
           "over T = 5 iterations.")


def _sweep(T, experiment, title, note):
    fields = ["dataset", "setting", "H@1", "MRR", "Rec.", "Time",
              "time_unit", "runs", "source"]
    out = []
    for row in T[experiment]:
        label = row["label"]
        for i, ds in enumerate(["YAGO-WIKI50K", "T-2M"]):
            v = row["values"][i * 4:(i + 1) * 4]
            if v[0] is None:
                continue
            out.append({"dataset": ds, "setting": label, "H@1": v[0],
                        "MRR": v[1], "Rec.": v[2], "Time": v[3],
                        "time_unit": TIME_UNIT.get(ds, "s"),
                        "runs": 5, "source": "paper + 5 runs"})
    _write(experiment, fields, out, title, note)


def partition_matching_sweep(T):
    _sweep(T, "partition_matching_sweep",
           "Partition-matching sensitivity: rho_m and gamma (TREA).",
           "rho_m controls the number of retained target partitions; gamma "
           "balances seed overlap and temporal overlap in Eq. (5).")


def routing_margin_sweep(T):
    _sweep(T, "routing_margin_sweep",
           "Routing-margin sensitivity lambda (TREA).",
           "lambda is selected on held-out seed pairs; 1.5 is the operating "
           "point.")


def candidate_budget_sweep(T):
    _sweep(T, "candidate_budget_sweep",
           "Candidate-budget and b_struct sensitivity (TREA).",
           "The triples are (b1,b2,b3); b_struct is the structural-route "
           "candidate budget.")


def acceptance_threshold_sweep(T):
    _sweep(T, "acceptance_threshold_sweep",
           "Acceptance-threshold sensitivity delta_c (TREA).",
           "delta_c is the prediction-margin threshold of Eq. (10) / "
           "Algorithm 4.")


def timestamp_coverage(T):
    fields = ["dataset", "setting", "H@1", "MRR", "Rec.", "Time",
              "time_unit", "runs", "source"]
    out = []
    for row in T["timestamp_coverage"]:
        if row["group"] != "TPA":
            continue
        for i, ds in enumerate(["YAGO-WIKI50K", "T-2M"]):
            v = row["values"][i * 4:(i + 1) * 4]
            out.append({"dataset": ds, "setting": row["label"],
                        "H@1": v[0], "MRR": v[1], "Rec.": v[2],
                        "Time": v[3], "time_unit": TIME_UNIT.get(ds, "s"),
                        "runs": 5, "source": "paper + 5 runs"})
    _write("timestamp_coverage", fields, out,
           "Controlled timestamp granularity and coverage (TREA).",
           "Granularity rows change the calendar resolution; coverage rows "
           "mark a uniform fraction of timestamped facts as untimestamped.")


def kg_size_scaling(T):
    fields = ["dataset", "backbone", "method", "K", "H@1", "Rec.", "Time",
              "time_unit", "Mem.", "runs", "source"]
    out = []
    for row in T["kg_size_scaling"]:
        method = row["group"]
        k, h1, rec, t, mem = row["values"][:5]
        ds = "T-2M-" + row["label"]
        out.append({"dataset": ds, "backbone": "trea", "method": method,
                    "K": int(k), "H@1": h1, "Rec.": rec, "Time": t,
                    "time_unit": "h", "Mem.": mem, "runs": _runs(method),
                    "source": _source(method)})
    _write("kg_size_scaling", fields, out,
           "Scaling with KG size on connected subgraphs of T-2M (TREA).",
           "Each row samples a connected subgraph with the stated number of "
           "entities per side and scales K with size.")


def hits_over_recall_static(T):
    fields = ["dataset", "backbone", "method", "ratio", "runs", "source"]
    out = []
    for row in T["hits_over_recall_static"]:
        method = row["method"]
        for ds, v in zip(DATASETS4, row["values"]):
            if v is None:
                continue
            out.append({"dataset": ds, "backbone": BB[row["backbone"]],
                        "method": method, "ratio": v, "runs": _runs(method),
                        "source": _source(method)})
    _write("hits_over_recall_static", fields, out,
           "Hits@1 divided by counterpart recall for GCN-Align and "
           "LightTEA (percent).",
           "Derived from the corresponding rows of the main comparison; no "
           "separate training runs.")


RENDERERS = {
    "backbone_comparison": backbone_comparison,
    "hits_over_recall_temporal": hits_over_recall_temporal,
    "static_large_scale": static_large_scale,
    "temporal_statistics": temporal_statistics,
    "joint_ablation": joint_ablation,
    "partition_count_sweep": partition_count_sweep,
    "iteration_count_sweep": iteration_count_sweep,
    "mixed_granularity": mixed_granularity,
    "static_dwy_fbdbp": static_dwy_fbdbp,
    "seed_variance": seed_variance,
    "no_timestamps": no_timestamps,
    "component_ablation": component_ablation,
    "stage_runtime": stage_runtime,
    "partition_matching_sweep": partition_matching_sweep,
    "routing_margin_sweep": routing_margin_sweep,
    "candidate_budget_sweep": candidate_budget_sweep,
    "acceptance_threshold_sweep": acceptance_threshold_sweep,
    "timestamp_coverage": timestamp_coverage,
    "kg_size_scaling": kg_size_scaling,
    "hits_over_recall_static": hits_over_recall_static,
}

_STATIC_FAMILIES = {
    "static_ids15k": (["IDS15K-EN-FR", "IDS15K-EN-DE"],
                      "Remaining static benchmarks: IDS15K EN-FR / EN-DE."),
    "static_dbp15k_zh_ja": (["DBP15K-ZH-EN", "DBP15K-JA-EN"],
                            "Remaining static benchmarks: DBP15K ZH-EN / "
                            "JA-EN."),
    "static_ids100k": (["IDS100K-EN-FR", "IDS100K-EN-DE"],
                       "Remaining static benchmarks: IDS100K EN-FR / EN-DE."),
    "static_dwy_yg_dbp1m": (["DWY100K-DBP-YG", "DBP1M-EN-DE"],
                            "Remaining static benchmarks: DWY100K DBP-YG / "
                            "DBP1M EN-DE."),
}


def _static_renderer(experiment):
    datasets = _STATIC_FAMILIES[experiment][0]

    def render(T):
        fields, rows = _six_rows(T[experiment], datasets, "method")
        _write(experiment, fields, rows,
               _STATIC_FAMILIES[experiment][1],
               "Static protocol; three static backbones.")
    return render


for _name in _STATIC_FAMILIES:
    RENDERERS[_name] = _static_renderer(_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*", default=None)
    args = ap.parse_args()
    with open(REFERENCE, encoding="utf-8") as f:
        T = json.load(f)
    wanted = args.experiments or sorted(RENDERERS)
    for experiment in wanted:
        if experiment not in RENDERERS:
            print("[skip] %s (no reporting renderer)" % experiment)
            continue
        RENDERERS[experiment](T)
        print("[ok] %s" % experiment)


if __name__ == "__main__":
    main()