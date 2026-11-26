



✨ On Temporal Entity Alignment at Scale ✨







![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)
![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)

[English](README.md) | [简体中文](./README_zh_CN.md)



**[📰 Introduction](#-introduction)** | **[🏗️ Architecture](#architecture)** | **[⚙️ Installation](#installation)** | **[🚀 Quick Start](#-quick-start)**   
**[📦 Datasets](#-datasets)** | **[📖 Usage](#-usage)** | **[🔬 Reproducibility](#-reproducibility)** | **[📜 License](#-license)** | **[📬 Contact](#-contact)**

---



## 📰 Latest News


| 🆕 Updates          | 📅 Date | 📝 Description                          |
| ------------------- | ------- | --------------------------------------- |
| 🎉 **Code Release** | -       | TPA codebase and datasets now available |


> **Before running:** After cloning, download the benchmark datasets listed in the Datasets section and extract them under `data/`.

---



## 📰 Introduction

**Large-scale Temporal Entity Alignment (LsTEA)** addresses a critical challenge in temporal knowledge graph integration. To the best of our knowledge, this is the **first work** to formally formulate and solve this problem, which we term **LsTEA**. This task presents unique challenges due to **Time-Heterogeneity Constrained Dual Graph Partitioning** (i.e., timeline discrepancies and temporally incoherent structural cuts), **Imbalanced Temporal Evidence** (i.e., highly varying temporal comparability across entities and partitions), and **Cascading Temporal Accumulation Errors** (i.e., wrong pseudo-seed pairs transferring incorrect active periods into iterative self-training) that are common in real-world scenarios.

We have officially introduced a series of complete and high-quality LsTEA benchmarks and proposed **TPA**, a new paradigm based on **temporal partitioning-and-alignment** to systematically address the unique challenges of LsTEA. TPA effectively pairs subgraphs with high temporal coherence, isolates temporally comparable candidates without heavy neural temporal encoding, and introduces a new **temporal error-aware corrector** to prevent temporal accumulation errors from cascading during iterative expansion of seed pairs.

## 🔥 Key Features


| Feature                             | Icon | Description                                                                                                                                                          |
| ----------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Time-Aware Graph Partition**      | 🔄   | Unified temporal profiles, temporally cohesive dual partitioning, and seed/temporal-overlap partition matching                                                       |
| **Multi-Resolution Temporal Index** | 🔍   | Lightweight non-neural index that retrieves candidates with overlapping active periods                                                                               |
| **Cost-Aware Routing**              | 🚦   | Per-entity routing across temporal, structural, or combined retrieval routes                                                                                         |
| **Temporal Error-Aware Corrector**  | 🔗   | Margin-based pseudo-seed acceptance with high-IDF gated profile updates                                                                                              |
| **State-of-the-Art Performance**    | 📈   | Consistently outperforming the strongest baseline ELsEA by 3.17–5.43 Hits@1 points on the 2M benchmarks, with 18.4–29.6% less runtime and 19.8–35.8% less GPU memory |


---



## 🏗️ Architecture

TPA adopts a **temporal partitioning-and-alignment** paradigm, comprising several key stages:

Stage 1: Time-Aware Graph Partition 🔄

Stage 2: Progressive Temporal Context Refinement 📐

Stage 3: Temporal Error-Aware Optimization 🔗

> 📖 For detailed architecture descriptions and theoretical foundations, refer to the accompanying paper.

---



## ⚙️ Installation



### 📋 Prerequisites

First, install dependencies:

```bash

pip install -r requirements.txt

```



### 📦 Main Dependencies


| Package        | Version   | Purpose                                  |
| -------------- | --------- | ---------------------------------------- |
| 🐍 **Python**  | >= 3.8    | Core language                            |
| 🔥 **PyTorch** | >= 2.0    | Deep learning framework                  |
| 🔢 **NumPy**   | >= 1.23   | Numerical computing                      |
| 🧮 **SciPy**   | >= 1.10   | Sparse matrices and scientific computing |
| 🪓 **PyMETIS** | >= 2022.1 | Graph partitioning (METIS)               |


> 💡 **Note:** A CUDA-enabled PyTorch build is recommended for the neural backbones; TPA also records the peak accelerator memory of every run.

---



## 📦 Datasets

For our newly proposed **LsTEA** scenario, we introduce a series of benchmark datasets, including the million-scale temporal benchmarks **T-2M** and **YAGO-WIKI-2M**.


| Dataset               | Description                                              | Scale (entities per side) |
| --------------------- | -------------------------------------------------------- | ------------------------- |
| **DICEWS-10K**        | Temporal benchmark (ICEWS-ICEWS)                         | ~10K                      |
| **YAGO-WIKI50K**      | Temporal benchmark (YAGO-WIKI)                           | ~50K                      |
| **BETA**              | Temporal benchmark with mixed year/month/day granularity | ~43K                      |
| **T-2M**              | Million-scale temporal benchmark (FB-DBP)                | 2.3M / 2.2M               |
| **YAGO-WIKI-2M**      | Million-scale temporal benchmark (YAGO-WIKI)             | 2.6M / 2.6M               |
| **IDS15K / DBP15K**   | Static benchmarks                                        | 15K / 20K                 |
| **IDS100K / DWY100K** | Static benchmarks                                        | 100K                      |
| **DBP1M / FBDBP2M**   | Static benchmarks                                        | 1M / 2M                   |




### 🔗 Download Links


[![Google Drive](https://img.shields.io/badge/Google_Drive-Download-green?style=for-the-badge)](https://drive.google.com/drive/folders/11raFe7shjMvendajb2wZ0Ab4xPB090YE?usp=sharing)



> 🔐 **Million-scale TKGA** (T-2M, YAGO-WIKI-2M) archive password: `lstea`



**Note:** The representative datasets used in experiments are derived from [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [BETA](https://github.com/DexterZeng/BETA), [OpenEA](https://github.com/nju-websoft/OpenEA), [LIME](https://github.com/DexterZeng/LIME), [ELsEA](https://github.com/wx-qzhou/ELsEA), [LargeEA](https://github.com/ZJU-DAILY/LargeEA) and related works.

---



## 🚀 Quick Start



### Step 1: Clone the Repository 📥

```bash

git clone https://github.com/eduzrh/LsTEA.git

cd LsTEA

```



### Step 2: Prepare Datasets 📦

Download and extract datasets to `./data/`

### Step 3: Run the Main Experiment ▶️

```bash

python src/run_tpa.py --data_dir data/DICEWS-10K --backbone trea

```



### Step 4: View Results 📊


| Metric     | Description                                                                       |
| ---------- | --------------------------------------------------------------------------------- |
| **Hits@1** | Proportion of correct alignments ranked first                                     |
| **Hits@5** | Proportion of correct alignments in top-5 candidates                              |
| **MRR**    | Mean Reciprocal Rank                                                              |
| **Recall** | Counterpart recall: proportion of test pairs covered by the constructed sub-tasks |


---



## 📖 Usage



### Basic Usage

**Run complete pipeline:**

```bash

python src/run_tpa.py --data_dir data/DICEWS-10K --backbone trea --out results/demo.json

```

Each run emits a self-contained JSON record: the resolved configuration, dataset statistics, the effectiveness metrics (Hits@1, Hits@5, MRR, counterpart recall), a per-stage wall-clock profile of the pipeline, and the peak GPU memory.

### Advanced Options

**Configure pipeline parameters:**

```bash

python src/run_tpa.py --data_dir data/T-2M \
    --backbone trea \
    --config configs/tpa_default.json \
    --K 200 \
    --I 5 \
    --epochs 50 \
    --device cuda:0 \
    --seed 1011 \
    --out results/backbone_comparison/TPA/T-2M/trea_seed1011.json

```

**Parameter Descriptions:**


| Parameter    | Type      | Default      | Description                                                                  |
| ------------ | --------- | ------------ | ---------------------------------------------------------------------------- |
| `--data_dir` | str       | **Required** | Path to dataset directory                                                    |
| `--backbone` | str       | trea         | TEA backbone: `gcn_align`, `tea_gnn`, `rrea`, `trea`, `dual_amn`, `lighttea` |
| `--config`   | str       | None         | JSON config file (defaults in `configs/tpa_default.json`)                    |
| `--set`      | key=value | -            | Override one config key, e.g. `--set lam=1.8`                                |
| `--K`        | int       | auto         | Per-side partition count (5/10/100/200 by dataset scale)                     |
| `--I`        | int       | 5            | Maximum refinement/optimization iterations                                   |
| `--epochs`   | int       | 50           | Training epochs per iteration (neural backbones)                             |
| `--device`   | str       | auto         | Torch device, e.g. `cuda:0`                                                  |
| `--seed`     | int       | 1011         | Model / partition RNG seed                                                   |
| `--out`      | str       | None         | Output path of the JSON run record                                           |




### Ablation Studies

Every component can be disabled from the command line for the ablation studies (Sec. IV-C / Appendix E-C):

```bash

python src/run_tpa.py --data_dir data/YAGO-WIKI50K --backbone lighttea --no_trf

```


| Flag                   | Ablated component                                |
| ---------------------- | ------------------------------------------------ |
| `--no_time_aware`      | TC & TM (temporal cohesion & temporal matching)  |
| `--no_trf`             | Temporal reachability filter (TRF)               |
| `--no_lsv`             | Local structure verification (LSV)               |
| `--no_tec`             | Temporal error-aware corrector (TEC)             |
| `--no_tc` / `--no_tm`  | Temporal cohesion / temporal matching separately |
| `--route_only tem|str` | Temporal-only or structural-only routing         |
| `--no_margin`          | Margin-free pseudo-seed acceptance               |
| `--no_idf_mask`        | High-IDF gated profile update                    |
| `--no_budget`          | Slot budget of the corrector                     |




### Static Protocol & Controlled Temporal Settings

- `--strip_timestamps`: run the static protocol / TPA without timestamps (Appendix E-B)
- `--time_granularity day|month|year`: controlled calendar resolution (Appendix E-F)
- `--time_coverage 0.0–1.0`: fraction of timestamped facts retained

---



## 🔬 Reproducibility

We are committed to ensuring full reproducibility of our results. The following resources are provided:

### 📋 Experimental Configuration

- **Hyperparameters**: All hyperparameter settings are documented in `configs/tpa_default.json` and can be configured via command-line arguments
- **Random Seeds**: Five seeds (`1011`, `2022`, `3033`, `4044`, `5055`); the 20/80 benchmark split is fixed by `--split_seed`
- **Environment**: Dependencies as specified in `requirements.txt`



### 📊 Reproducing Main Results

To reproduce the main experimental results reported in the paper:

1. **Download datasets** following the format described in the Datasets section
2. **Run the registered experiments** (missing datasets are skipped and reported in the run manifest):

```bash

python experiments/run_experiment.py backbone_comparison --data_root data

```

1. **Aggregate and render reports**:

```bash

python experiments/summarize_results.py results/backbone_comparison

python experiments/render_reports.py

```



### 🏗️ Code Organization

The codebase is organized into modular components for clarity:

- `src/run_tpa.py`: Main pipeline entry point
- `src/tpa/stage1_time_aware_graph_partition/`: Unified temporal profiles, dual TKG partitioner, and partition matcher
- `src/tpa/stage2_progressive_temporal_context_refinement/`: Multi-resolution temporal index, cost-aware routing, temporal reachability filter, and local structure verification
- `src/tpa/stage3_temporal_error_aware_optimization/`: Context construction and temporal error-aware corrector
- `src/tpa/backbones/`: Six TEA backbones (GCN-Align, TEA-GNN, RREA, TREA, Dual-AMN, LightTEA)
- `src/tpa/tpa_framework.py`: Three-stage TPA orchestrator
- `experiments/`: Experiment registry, runner, aggregation, and report rendering



### 📝 Documentation

- Comprehensive inline code comments explaining key design decisions
- Clear module structure with standardized naming conventions
- This README with step-by-step usage instructions

---



## 📊 Evaluation Metrics

We employ standard knowledge graph alignment metrics for transparency and comparability:

- **Hits@1**: Proportion of correct alignments ranked first
- **Hits@5**: Proportion of correct alignments in top-5 candidates  
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct alignments
- **Counterpart Recall**: Proportion of to-be-aligned test pairs covered by the constructed sub-tasks



## 📜 License

[GPL-3.0 License](LICENSE) - Copyright notices preserved.

## 📬 Contact

- **Email**: [runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)
- **GitHub Issues**: For technical concerns, create an Issue in the [GitHub repository](https://github.com/eduzrh/LsTEA/issues). Labels: `bug`, `enhancement`, `question`.

Responses targeted within 2-3 business days.



## 🔗 References

- [LargeEA: Aligning Entities for Large-scale Knowledge Graphs](https://doi.org/10.14778/3489496.3489504).
Chenghao Ge, Xinxin Liu, Lu Chen, Bolin Zheng, and Yan Gao.
Proceedings of the VLDB Endowment (PVLDB), 2021.
- [On Entity Alignment at Scale](https://link.springer.com/article/10.1007/s00778-022-00722-7).
Weixin Zeng, Xiang Zhao, Xinyi Li, Jiuyang Tang, and Wei Wang.
The VLDB Journal, 2022.
- [ClusterEA: Scalable Entity Alignment with Stochastic Training and Normalized Mini-batch Similarities](https://dl.acm.org/doi/10.1145/3534678.3539255).
Yan Gao, Xinxin Liu, Jia Wu, Tong Li, Peng Wang, and Lei Chen.
Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2022.
- [High-quality Task Division for Large-scale Entity Alignment](https://dl.acm.org/doi/10.1145/3511808.3557352).
Binfeng Liu, Wen Hua, Guido Zuccon, Genghao Zhao, and Xike Zhang.
Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), 2022.
- [Enhancing Large-scale Entity Alignment with Critical Structure and High-quality Context (ELsEA)](https://ieeexplore.ieee.org/document/11113003).
Qian Zhou, Wei Chen, Liang Zhang, Pengpeng Zhao, Jiajie Xu, and Lei Zhao.
IEEE International Conference on Data Engineering (ICDE), 2025.
- [METIS: A Software Package for Partitioning Unstructured Graphs, Partitioning Meshes, and Computing Fill-Reducing Orderings of Sparse Matrices](https://hdl.handle.net/11299/215346).
George Karypis and Vipin Kumar.
Technical Report 97-061, University of Minnesota, 1997.
- [Time-aware Entity Alignment using Temporal Relational Attention (TREA)](https://dl.acm.org/doi/10.1145/3485447.3511982).
Chenguang Xu, Fatima Su, Ba Xiong, and Jens Lehmann.
Proceedings of the ACM Web Conference (WWW), 2022.
- [Time-aware Graph Neural Network for Entity Alignment between Temporal Knowledge Graphs (TEA-GNN)](https://aclanthology.org/2021.emnlp-main.712/).
Chenguang Xu, Fatima Su, and Jens Lehmann.
Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021.
- [An Effective and Efficient Time-aware Entity Alignment Framework via Two-aspect Three-view Label Propagation (LightTEA)](https://www.ijcai.org/proceedings/2023/0559.pdf).
Lihua Cai, Xin Mao, Yifei Xiao, Cheng Wu, and Man Lan.
Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2023.
- [Relational Reflection Entity Alignment (RREA)](https://dl.acm.org/doi/10.1145/3340531.3412076).
Xin Mao, Wenting Wang, Hao Xu, Yuanbin Wu, and Man Lan.
Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), 2020.
- [Boosting the Speed of Entity Alignment 10×: Dual Attention Matching Network with Normalized Hard Sample Mining (Dual-AMN)](https://doi.org/10.1145/3442381.3449897).
Xin Mao, Wenting Wang, Yuanbin Wu, and Man Lan.
Proceedings of the Web Conference (WWW), 2021.
- [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks (GCN-Align)](https://doi.org/10.18653/v1/d18-1032).
Zhichun Wang, Qingsong Lv, Xiaohan Lan, and Yu Zhang.
Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.
- [Benchmarking Challenges for Temporal Knowledge Graph Alignment](https://api.semanticscholar.org/CorpusID:273501043).
Weixin Zeng, Jie Zhou, and Xiang Zhao.
Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM), 2024.



## 🙏 Acknowledgement

The following open source projects were partially referenced in this work. We sincerely appreciate their contributions:

[Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [BETA](https://github.com/DexterZeng/BETA), [LargeEA](https://github.com/ZJU-DAILY/LargeEA), [LIME](https://github.com/DexterZeng/LIME), [ELsEA](https://github.com/wx-qzhou/ELsEA), [OpenEA](https://github.com/nju-websoft/OpenEA), [METIS/PyMETIS](https://github.com/europlexo/pymetis), [PyTorch](https://github.com/pytorch/pytorch)

---

