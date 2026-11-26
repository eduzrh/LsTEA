# -*- coding: utf-8 -*-
"""Registry of the experiments run for the TPA study.

Every entry describes one experiment in terms of what it measures and the
concrete runs it needs: dataset, backbone, TPA variant switches and
hyperparameter overrides.  ``experiments/run_experiment.py`` consumes the
registry and writes one JSON record per (setting, seed) with the same schema
as ``src/run_tpa.py``.

Experiment identifiers are descriptive:

  dataset_overview            benchmark composition and split sizes
  backbone_comparison         main temporal results (6 backbones x 4 datasets)
  hits_over_recall_temporal   Hits@1 / counterpart-recall ratio
  static_large_scale          static DBP15K FR-EN and DBP1M EN-FR
  temporal_statistics         temporal statistics of the LsTEA benchmarks
  joint_ablation              joint removal of TC & TM / TRF / LSV / TEC
  partition_count_sweep       per-side partition count K
  iteration_count_sweep       refinement/optimization iterations T
  mixed_granularity           BETA year/month/day timestamps
  static_dwy_fbdbp            static DWY100K DBP-WD and FBDBP2M
  seed_variance               run-to-run standard deviations
  no_timestamps               TPA with temporal information disabled
  component_ablation          one-mechanism-at-a-time ablations
  stage_runtime               per-stage wall-clock and peak memory
  partition_matching_sweep    rho_m and gamma
  routing_margin_sweep        routing margin lambda
  candidate_budget_sweep      cascade budgets and b_struct
  acceptance_threshold_sweep  pseudo-seed acceptance threshold delta_c
  timestamp_coverage          calendar granularity and timestamp coverage
  kg_size_scaling             connected subgraphs from 100K to 2M entities
  static_ids15k               static IDS15K EN-FR / EN-DE
  static_dbp15k_zh_ja         static DBP15K ZH-EN / JA-EN
  static_ids100k              static IDS100K EN-FR / EN-DE
  static_dwy_yg_dbp1m         static DWY100K DBP-YG / DBP1M EN-DE
  hits_over_recall_static     Hits@1 / recall for the static backbones

``reference`` records where the experiment is reported in the technical
report (section or appendix heading).

Usage
-----
    python experiments/registry.py --emit-configs experiments/configs
    python experiments/registry.py --list
"""
import argparse
import json
import os

SEEDS = [1011, 2022, 3033, 4044, 5055]
TEMPORAL = ["DICEWS-10K", "YAGO-WIKI50K", "T-2M", "YAGO-WIKI-2M"]
TEMPORAL_CONVENTIONAL = ["DICEWS-10K", "YAGO-WIKI50K", "BETA"]
BACKBONES = ["gcn_align", "tea_gnn", "rrea", "trea", "dual_amn", "lighttea"]
STATIC_BACKBONES = ["gcn_align", "rrea", "dual_amn"]

DATASET_PATHS = {
    "DICEWS-10K": "DICEWS-10K",
    "YAGO-WIKI50K": "YAGO-WIKI50K",
    "BETA": "BETA",
    "T-2M": "T-2M",
    "YAGO-WIKI-2M": "YAGO-WIKI-2M",
    "IDS15K-EN-FR": "IDS15K/EN-FR",
    "IDS15K-EN-DE": "IDS15K/EN-DE",
    "DBP15K-ZH-EN": "dbp15k/DBP15K_ZH-EN",
    "DBP15K-JA-EN": "dbp15k/DBP15K_JA-EN",
    "DBP15K-FR-EN": "dbp15k/DBP15K_FR-EN",
    "IDS100K-EN-FR": "IDS100K/EN-FR",
    "IDS100K-EN-DE": "IDS100K/EN-DE",
    "DWY100K-DBP-WD": "DWY100K/DBP-WD",
    "DWY100K-DBP-YG": "DWY100K/DBP-YG",
    "DBP1M-EN-FR": "dbp15k/DBP1M_EN-FR",
    "DBP1M-EN-DE": "dbp15k/DBP1M_EN-DE",
    "FBDBP2M": "FBDBP2M",
}

DATASET_MF = {
    "DICEWS-10K": (10000, 50000),
    "YAGO-WIKI50K": (20000, 100000),
    "BETA": (20000, 100000),
    "DBP15K-FR-EN": (10000, 50000),
    "IDS15K-EN-FR": (10000, 50000),
    "IDS15K-EN-DE": (10000, 50000),
    "DBP15K-ZH-EN": (10000, 50000),
    "DBP15K-JA-EN": (10000, 50000),
    "IDS100K-EN-FR": (20000, 100000),
    "IDS100K-EN-DE": (20000, 100000),
    "DWY100K-DBP-WD": (20000, 100000),
    "DWY100K-DBP-YG": (20000, 100000),
    "DBP1M-EN-FR": (50000, 250000),
    "DBP1M-EN-DE": (50000, 250000),
    "T-2M": (50000, 250000),
    "YAGO-WIKI-2M": (50000, 250000),
    "FBDBP2M": (50000, 250000),
}

DATASET_K = {
    "DICEWS-10K": 5,
    "YAGO-WIKI50K": 10,
    "BETA": 10,
    "IDS15K-EN-FR": 5,
    "IDS15K-EN-DE": 5,
    "DBP15K-ZH-EN": 5,
    "DBP15K-JA-EN": 5,
    "DBP15K-FR-EN": 5,
    "IDS100K-EN-FR": 10,
    "IDS100K-EN-DE": 10,
    "DWY100K-DBP-WD": 10,
    "DWY100K-DBP-YG": 10,
    "DBP1M-EN-FR": 100,
    "DBP1M-EN-DE": 100,
    "T-2M": 200,
    "YAGO-WIKI-2M": 200,
    "FBDBP2M": 200,
}

# variant name -> CLI switches of src/run_tpa.py
VARIANTS = {
    "TPA": [],
    "w/o TC & TM": ["--no_tc", "--no_tm"],
    "w/o TRF": ["--no_trf"],
    "w/o LSV": ["--no_lsv"],
    "w/o TEC": ["--no_tec"],
    "w/o TC": ["--no_tc"],
    "w/o TM": ["--no_tm"],
    "temporal only": ["--route_only", "tem"],
    "structural only": ["--route_only", "str"],
    "L=1": ["--index_levels", "1"],
    "L=2": ["--index_levels", "2"],
    "w/o margin": ["--no_margin"],
    "w/o IDF mask": ["--no_idf_mask"],
    "w/o budget": ["--no_budget"],
    "TPA w/o timestamps": ["--strip_timestamps"],
}


def _settings(datasets, backbones, variants, extra=None):
    out = []
    for name, switches in variants.items():
        for dataset in datasets:
            for backbone in backbones:
                spec = {
                    "variant": name,
                    "dataset": dataset,
                    "backbone": backbone,
                    "args": list(switches),
                }
                if extra:
                    spec["args"] += list(extra)
                out.append(spec)
    return out


def _experiment(exp_id, title, settings, kind="run", notes="", reference=""):
    return {
        "id": exp_id,
        "title": title,
        "kind": kind,
        "settings": settings,
        "notes": notes,
        "reference": reference,
    }


EXPERIMENTS = {}


EXPERIMENTS["backbone_comparison"] = _experiment(
    "backbone_comparison",
    "Main temporal alignment results: Hits@1/5, MRR, counterpart recall, "
    "runtime and peak memory on four temporal benchmarks with six backbones.",
    _settings(TEMPORAL, BACKBONES, {"TPA": []}),
    notes="20%/80% seed/test split; K by scale (5/10/200); T=5 iterations.",
    reference="Sec. IV-B",
)

EXPERIMENTS["hits_over_recall_temporal"] = _experiment(
    "hits_over_recall_temporal",
    "Hits@1 / counterpart-recall ratio on the four temporal benchmarks.",
    [], kind="derived",
    notes="Derived from the per-seed records of backbone_comparison; "
          "no separate runs.",
    reference="Sec. IV-B",
)

EXPERIMENTS["temporal_statistics"] = _experiment(
    "temporal_statistics",
    "Temporal statistics of the five LsTEA benchmarks.",
    [], kind="derived",
    notes="Computed by scripts/compute_temporal_statistics.py from the "
          "packaged datasets.",
    reference="Appendix C-A",
)

EXPERIMENTS["static_large_scale"] = _experiment(
    "static_large_scale",
    "Static large-scale results on DBP15K FR-EN and DBP1M EN-FR.",
    _settings(["DBP15K-FR-EN", "DBP1M-EN-FR"], STATIC_BACKBONES,
              {"TPA": []}, extra=["--strip_timestamps"]),
    notes="Static protocol: no timestamps, N=1, structural degeneration.",
    reference="Sec. IV-F",
)

EXPERIMENTS["joint_ablation"] = _experiment(
    "joint_ablation",
    "Joint component ablation on YAGO-WIKI50K and T-2M.",
    _settings(["YAGO-WIKI50K", "T-2M"], ["trea", "lighttea"], {
        "TPA": [],
        "w/o TC & TM": VARIANTS["w/o TC & TM"],
        "w/o TRF": VARIANTS["w/o TRF"],
        "w/o LSV": VARIANTS["w/o LSV"],
        "w/o TEC": VARIANTS["w/o TEC"],
    }),
    reference="Sec. IV-C",
)

EXPERIMENTS["partition_count_sweep"] = _experiment(
    "partition_count_sweep",
    "Effect of the per-side partition count K with TREA.",
    [
        {"variant": "K=%d" % k, "dataset": ds, "backbone": "trea",
         "args": ["--K", str(k)]}
        for ds, grid in (("YAGO-WIKI50K", [3, 5, 7, 10]),
                         ("T-2M", [160, 180, 200, 220]))
        for k in grid
    ],
    reference="Sec. IV-D",
)

EXPERIMENTS["iteration_count_sweep"] = _experiment(
    "iteration_count_sweep",
    "Effect of the iteration count T with TREA.",
    [
        {"variant": "T=%d" % t, "dataset": ds, "backbone": "trea",
         "args": ["--I", str(t)]}
        for ds in ("YAGO-WIKI50K", "T-2M")
        for t in (1, 2, 3, 4, 5)
    ],
    reference="Sec. IV-D",
)

EXPERIMENTS["mixed_granularity"] = _experiment(
    "mixed_granularity",
    "BETA mixed temporal granularities with all six backbones.",
    _settings(["BETA"], BACKBONES, {"TPA": []}),
    reference="Sec. IV-E",
)

EXPERIMENTS["static_dwy_fbdbp"] = _experiment(
    "static_dwy_fbdbp",
    "Static DWY100K DBP-WD and FBDBP2M with three backbones.",
    _settings(["DWY100K-DBP-WD", "FBDBP2M"], STATIC_BACKBONES,
              {"TPA": []}, extra=["--strip_timestamps"]),
    reference="Sec. IV-F",
)

EXPERIMENTS["seed_variance"] = _experiment(
    "seed_variance",
    "Run-to-run standard deviations over the five seeds.",
    [], kind="derived",
    notes="Computed from the five seed records of backbone_comparison.",
    reference="Appendix E-A",
)

EXPERIMENTS["no_timestamps"] = _experiment(
    "no_timestamps",
    "TPA without timestamps versus ELsEA and full TPA.",
    _settings(TEMPORAL, ["trea", "lighttea"], {
        "TPA w/o timestamps": VARIANTS["TPA w/o timestamps"],
        "TPA": [],
    }),
    reference="Appendix E-B",
)

EXPERIMENTS["component_ablation"] = _experiment(
    "component_ablation",
    "Separate component ablations on YAGO-WIKI50K and T-2M.",
    _settings(["YAGO-WIKI50K", "T-2M"], ["trea", "lighttea"], {
        "TPA": [],
        "w/o TC": VARIANTS["w/o TC"],
        "w/o TM": VARIANTS["w/o TM"],
        "temporal only": VARIANTS["temporal only"],
        "structural only": VARIANTS["structural only"],
        "L=1": VARIANTS["L=1"],
        "L=2": VARIANTS["L=2"],
        "w/o margin": VARIANTS["w/o margin"],
        "w/o IDF mask": VARIANTS["w/o IDF mask"],
        "w/o budget": VARIANTS["w/o budget"],
    }),
    reference="Appendix E-C",
)

EXPERIMENTS["stage_runtime"] = _experiment(
    "stage_runtime",
    "Per-stage wall-clock and peak memory with TREA on four datasets.",
    _settings(TEMPORAL, ["trea"], {"TPA": []}),
    notes="Stage timings are emitted by TPAFramework in the runtime block.",
    reference="Appendix E-D",
)

EXPERIMENTS["partition_matching_sweep"] = _experiment(
    "partition_matching_sweep",
    "Partition-matching sensitivity: rho_m and gamma (TREA).",
    [
        {"variant": "rho_m=%s" % v, "dataset": ds, "backbone": "trea",
         "args": ["--rho_m", str(v)]}
        for ds in ("YAGO-WIKI50K", "T-2M") for v in (0.1, 0.2, 0.3)
    ] + [
        {"variant": "gamma=%s" % v, "dataset": ds, "backbone": "trea",
         "args": ["--gamma", str(v)]}
        for ds in ("YAGO-WIKI50K", "T-2M") for v in (0.3, 0.5, 0.7)
    ],
    reference="Appendix E-E",
)

EXPERIMENTS["routing_margin_sweep"] = _experiment(
    "routing_margin_sweep",
    "Routing-margin sensitivity lambda (TREA).",
    [
        {"variant": "lambda=%s" % v, "dataset": ds, "backbone": "trea",
         "args": ["--lam", str(v)]}
        for ds in ("YAGO-WIKI50K", "T-2M") for v in (1.2, 1.5, 1.8, 2.0)
    ],
    reference="Appendix E-E",
)

EXPERIMENTS["candidate_budget_sweep"] = _experiment(
    "candidate_budget_sweep",
    "Candidate-budget and b_struct sensitivity (TREA).",
    [
        {"variant": "(b1,b2,b3)=(%s)" % b, "dataset": ds,
         "backbone": "trea", "args": ["--budgets", b]}
        for ds in ("YAGO-WIKI50K", "T-2M")
        for b in ("50,25,10", "100,50,20", "150,75,30")
    ] + [
        {"variant": "b_struct=%d" % v, "dataset": ds, "backbone": "trea",
         "args": ["--b_struct", str(v)]}
        for ds in ("YAGO-WIKI50K", "T-2M") for v in (10, 20, 30)
    ],
    reference="Appendix E-E",
)

EXPERIMENTS["acceptance_threshold_sweep"] = _experiment(
    "acceptance_threshold_sweep",
    "Acceptance-threshold sensitivity delta_c (TREA).",
    [
        {"variant": "delta_c=%s" % v, "dataset": ds, "backbone": "trea",
         "args": ["--delta_c", str(v)]}
        for ds in ("YAGO-WIKI50K", "T-2M") for v in (0.10, 0.15, 0.20, 0.25)
    ],
    reference="Appendix E-E",
)

EXPERIMENTS["timestamp_coverage"] = _experiment(
    "timestamp_coverage",
    "Controlled timestamp granularity and coverage (TREA).",
    [
        {"variant": "Granularity: %s" % g, "dataset": ds, "backbone": "trea",
         "args": ["--time_granularity", g]}
        for ds in ("YAGO-WIKI50K", "T-2M")
        for g in ("day", "month", "year")
    ] + [
        {"variant": "Coverage: %d%%" % int(c * 100), "dataset": ds,
         "backbone": "trea", "args": ["--time_coverage", str(c)]}
        for ds in ("YAGO-WIKI50K", "T-2M")
        for c in (1.0, 0.75, 0.5, 0.25, 0.0)
    ],
    notes="Granularity rows use --time_granularity; coverage rows keep native "
          "granularity and use --time_coverage.",
    reference="Appendix E-F",
)

EXPERIMENTS["kg_size_scaling"] = _experiment(
    "kg_size_scaling",
    "Scaling with KG size on connected subgraphs of T-2M (TREA).",
    [
        {"variant": "TPA", "dataset": "T-2M-%s" % size,
         "backbone": "trea", "args": ["--K", str(k)]}
        for size, k in (("100K", 10), ("250K", 25), ("500K", 50),
                        ("1M", 100), ("2M", 200))
    ],
    notes="Subgraphs are produced by scripts/sample_connected_subgraph.py.",
    reference="Appendix E-G",
)

EXPERIMENTS["static_ids15k"] = _experiment(
    "static_ids15k",
    "Remaining static benchmarks: IDS15K EN-FR / EN-DE.",
    _settings(["IDS15K-EN-FR", "IDS15K-EN-DE"], STATIC_BACKBONES,
              {"TPA": []}, extra=["--strip_timestamps"]),
    reference="Appendix E-H",
)

EXPERIMENTS["static_dbp15k_zh_ja"] = _experiment(
    "static_dbp15k_zh_ja",
    "Remaining static benchmarks: DBP15K ZH-EN / JA-EN.",
    _settings(["DBP15K-ZH-EN", "DBP15K-JA-EN"], STATIC_BACKBONES,
              {"TPA": []}, extra=["--strip_timestamps"]),
    reference="Appendix E-H",
)

EXPERIMENTS["static_ids100k"] = _experiment(
    "static_ids100k",
    "Remaining static benchmarks: IDS100K EN-FR / EN-DE.",
    _settings(["IDS100K-EN-FR", "IDS100K-EN-DE"], STATIC_BACKBONES,
              {"TPA": []}, extra=["--strip_timestamps"]),
    reference="Appendix E-H",
)

EXPERIMENTS["static_dwy_yg_dbp1m"] = _experiment(
    "static_dwy_yg_dbp1m",
    "Remaining static benchmarks: DWY100K DBP-YG and DBP1M EN-DE.",
    _settings(["DWY100K-DBP-YG", "DBP1M-EN-DE"], STATIC_BACKBONES,
              {"TPA": []}, extra=["--strip_timestamps"]),
    reference="Appendix E-H",
)

EXPERIMENTS["hits_over_recall_static"] = _experiment(
    "hits_over_recall_static",
    "Hits@1 / counterpart-recall ratio for GCN-Align and LightTEA.",
    [], kind="derived",
    notes="Computed from the GCN-Align and LightTEA records of "
          "backbone_comparison.",
    reference="Appendix E-I",
)


def iter_settings(exp_id):
    """Yield resolved run specifications for one experiment."""
    entry = EXPERIMENTS[exp_id]
    for spec in entry["settings"]:
        dataset = spec["dataset"]
        resolved = {
            "experiment": exp_id,
            "variant": spec["variant"],
            "dataset": dataset,
            "backbone": spec["backbone"],
            "args": list(spec["args"]),
            "K": DATASET_K.get(dataset),
            "data_dir": DATASET_PATHS.get(dataset, dataset),
        }
        mf = DATASET_MF.get(dataset)
        if mf is None and dataset.startswith("T-2M-"):
            mf = DATASET_MF["T-2M"]
        if mf is not None:
            resolved["M"], resolved["F"] = mf
            resolved["args"] += ["--M", str(mf[0]), "--F", str(mf[1])]
        yield resolved


def emit_configs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for exp_id, entry in EXPERIMENTS.items():
        settings = []
        for spec in iter_settings(exp_id):
            settings.append({
                "variant": spec["variant"],
                "dataset": spec["dataset"],
                "backbone": spec["backbone"],
                "args": spec["args"],
                "K": spec.get("K"),
                "M": spec.get("M"),
                "F": spec.get("F"),
            })
        cfg = {
            "experiment": exp_id,
            "title": entry["title"],
            "kind": entry["kind"],
            "seeds": SEEDS,
            "settings": settings,
            "notes": entry["notes"],
            "reference": entry["reference"],
        }
        with open(os.path.join(out_dir, exp_id + ".json"), "w") as f:
            json.dump(cfg, f, indent=2)
    print("wrote %d experiment configs to %s" % (len(EXPERIMENTS), out_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-configs", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.emit_configs:
        emit_configs(args.emit_configs)
    if args.list:
        for exp_id, entry in EXPERIMENTS.items():
            print(exp_id, len(entry["settings"]), entry["title"])


if __name__ == "__main__":
    main()