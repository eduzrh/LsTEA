



✨ On Temporal Entity Alignment at Scale ✨







![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-blue)
![Language: Python 3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)
![Made with PyTorch](https://img.shields.io/badge/Made%20with-pytorch-orange.svg?style=flat-square)
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)

[English](README.md) | [简体中文](./README_zh_CN.md)



**[📰 简介](#-introduction)** | **[🏗️ 架构](#architecture)** | **[⚙️ 安装](#installation)** | **[🚀 快速开始](#-quick-start)**   
**[📦 数据集](#-datasets)** | **[📖 使用说明](#-usage)** | **[🔬 可复现性](#-reproducibility)** | **[📜 许可证](#-license)** | **[📬 联系方式](#-contact)**

---



## 📰 最新动态


| 🆕 更新       | 📅 日期 | 📝 描述           |
| ----------- | ----- | --------------- |
| 🎉 **代码发布** | -     | TPA 代码库和数据集现已可用 |


> **运行前请注意：** 克隆仓库后，请先下载数据集一节中列出的基准数据集，并解压到 `data/` 目录下。

---



# 📰 简介

**大规模时序实体对齐（LsTEA）** 解决了时序知识图谱集成中的一个关键挑战。据我们所知，这是**首个**正式提出并解决该问题的工作，我们将其称为 **LsTEA**。由于**时间异构约束下的双图划分**（即时间线差异和时间不连贯的结构切割）、**不均衡的时序证据**（即不同实体和分区之间的时序可比性差异巨大）和**级联式时序累积误差**（即错误的伪种子对将错误的活动时期带入迭代自训练）在真实场景中普遍存在，该任务面临着独特的挑战。

我们正式引入了一系列完整且高质量的 LsTEA 基准，并提出了 **TPA**，这是一种基于**时序划分-对齐**的新范式，以系统性地解决 LsTEA 的独特挑战。TPA 有效配对具有高度时序连贯性的子图，在不依赖重型神经时序编码的情况下隔离时序可比的候选实体，并引入了一种新的**时序误差感知校正器**，防止时序累积误差在种子对的迭代扩展过程中级联放大。

## 🔥 核心特性


| 特性            | 图标  | 描述                                                                                  |
| ------------- | --- | ----------------------------------------------------------------------------------- |
| **时序感知图划分**   | 🔄  | 统一时序画像、时序连贯的双图划分，以及基于种子对/时序重叠的分区匹配                                                  |
| **多分辨率时序索引**  | 🔍  | 轻量级非神经索引，检索活动时期重叠的候选实体                                                              |
| **代价感知路由**    | 🚦  | 为每个实体在时序、结构或组合检索路由之间自动选择                                                            |
| **时序误差感知校正器** | 🔗  | 基于间隔（margin）的伪种子对接受机制与高 IDF 门控的画像更新                                                 |
| **最先进的性能**    | 📈  | 在 2M 规模基准上持续超越最强基线 ELsEA 3.17–5.43 个 Hits@1 点，运行时间减少 18.4–29.6%，GPU 显存减少 19.8–35.8% |


---



## 🏗️ 架构

TPA 采用**时序划分-对齐**范式，包含以下几个关键阶段：

阶段 1：时序感知图划分 🔄

阶段 2：渐进式时序上下文精化 📐

阶段 3：时序误差感知优化 🔗

> 📖 有关详细的架构描述和理论基础，请参考随附的论文。

---



## ⚙️ 安装



### 📋 前置要求

首先，安装依赖项：

```bash

pip install -r requirements.txt

```



### 📦 主要依赖


| 包              | 版本        | 用途         |
| -------------- | --------- | ---------- |
| 🐍 **Python**  | >= 3.8    | 核心语言       |
| 🔥 **PyTorch** | >= 2.0    | 深度学习框架     |
| 🔢 **NumPy**   | >= 1.23   | 数值计算       |
| 🧮 **SciPy**   | >= 1.10   | 稀疏矩阵与科学计算  |
| 🪓 **PyMETIS** | >= 2022.1 | 图划分（METIS） |


> 💡 **注意：** 神经 backbone 建议使用 CUDA 版 PyTorch；TPA 还会记录每次运行的峰值显存。

---



## 📦 数据集

对于我们新提出的 **LsTEA** 场景，我们引入了一系列基准数据集，其中包括百万规模时序基准 **T-2M** 和 **YAGO-WIKI-2M**。


| 数据集                   | 描述                  | 规模（每侧实体数）   |
| --------------------- | ------------------- | ----------- |
| **DICEWS-10K**        | 时序基准（ICEWS-ICEWS）   | ~10K        |
| **YAGO-WIKI50K**      | 时序基准（YAGO-WIKI）     | ~50K        |
| **BETA**              | 年/月/日混合粒度的时序基准      | ~43K        |
| **T-2M**              | 百万规模时序基准（FB-DBP）    | 2.3M / 2.2M |
| **YAGO-WIKI-2M**      | 百万规模时序基准（YAGO-WIKI） | 2.6M / 2.6M |
| **IDS15K / DBP15K**   | 静态基准                | 15K / 20K   |
| **IDS100K / DWY100K** | 静态基准                | 100K        |
| **DBP1M / FBDBP2M**   | 静态基准                | 1M / 2M     |




### 🔗 下载链接



[![Google Drive](https://img.shields.io/badge/Google_Drive-Download-green?style=for-the-badge)](https://drive.google.com/drive/folders/11raFe7shjMvendajb2wZ0Ab4xPB090YE?usp=sharing)



> 🔐 百万规模 TKGA（T-2M、YAGO-WIKI-2M）压缩包密码：`lstea`



**注意：** 实验中使用的代表性数据集来源于 [Dual-AMN](https://github.com/MaoXinn/Dual-AMN)、[JAPE](https://github.com/nju-websoft/JAPE)、[GCN-Align](https://github.com/1049451037/GCN-Align)、[BETA](https://github.com/DexterZeng/BETA)、[OpenEA](https://github.com/nju-websoft/OpenEA)、[LIME](https://github.com/DexterZeng/LIME)、[ELsEA](https://github.com/wx-qzhou/ELsEA)、[LargeEA](https://github.com/ZJU-DAILY/LargeEA) 及相关工作。

---



## 🚀 快速开始



### 步骤 1：克隆仓库 📥

```bash

git clone https://github.com/eduzrh/LsTEA.git

cd LsTEA

```



### 步骤 2：准备数据集 📦

下载并解压数据集到 `./data/`

### 步骤 3：运行主实验 ▶️

```bash

python src/run_tpa.py --data_dir data/DICEWS-10K --backbone trea

```



### 步骤 4：查看结果 📊


| 指标         | 描述                     |
| ---------- | ---------------------- |
| **Hits@1** | 排名第一的正确对齐比例            |
| **Hits@5** | 前 5 名候选中的正确对齐比例        |
| **MRR**    | 平均倒数排名                 |
| **Recall** | 对应实体召回率：被构建子任务覆盖的测试对比例 |


---



## 📖 使用说明



### 基本用法

**运行完整流程：**

```bash

python src/run_tpa.py --data_dir data/DICEWS-10K --backbone trea --out results/demo.json

```

每次运行都会输出一条自包含的 JSON 记录：解析后的配置、数据集统计信息、有效性指标（Hits@1、Hits@5、MRR、对应实体召回率）、流程各阶段的耗时画像，以及峰值 GPU 显存。

### 高级选项

**配置流程参数：**

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

**参数说明：**


| 参数           | 类型        | 默认值    | 描述                                                                     |
| ------------ | --------- | ------ | ---------------------------------------------------------------------- |
| `--data_dir` | str       | **必需** | 数据集目录路径                                                                |
| `--backbone` | str       | trea   | TEA backbone：`gcn_align`、`tea_gnn`、`rrea`、`trea`、`dual_amn`、`lighttea` |
| `--config`   | str       | None   | JSON 配置文件（默认值见 `configs/tpa_default.json`）                             |
| `--set`      | key=value | -      | 覆盖单个配置项，例如 `--set lam=1.8`                                             |
| `--K`        | int       | auto   | 每侧分区数量（按数据集规模自动取 5/10/100/200）                                         |
| `--I`        | int       | 5      | 精化/优化的最大迭代次数                                                           |
| `--epochs`   | int       | 50     | 每次迭代的训练轮数（神经 backbone）                                                 |
| `--device`   | str       | auto   | Torch 设备，例如 `cuda:0`                                                   |
| `--seed`     | int       | 1011   | 模型/划分的随机数种子                                                            |
| `--out`      | str       | None   | JSON 运行记录的输出路径                                                         |




### 消融实验

消融实验（论文第 IV-C 节 / 附录 E-C）中的每个组件都可以通过命令行开关禁用：

```bash

python src/run_tpa.py --data_dir data/YAGO-WIKI50K --backbone lighttea --no_trf

```


| 开关                     | 被消融的组件              |
| ---------------------- | ------------------- |
| `--no_time_aware`      | TC & TM（时序连贯性与时序匹配） |
| `--no_trf`             | 时序可达性过滤（TRF）        |
| `--no_lsv`             | 局部结构验证（LSV）         |
| `--no_tec`             | 时序误差感知校正器（TEC）      |
| `--no_tc` / `--no_tm`  | 分别消融时序连贯性 / 时序匹配    |
| `--route_only tem|str` | 仅时序路由或仅结构路由         |
| `--no_margin`          | 无间隔约束的伪种子对接受        |
| `--no_idf_mask`        | 高 IDF 门控的画像更新       |
| `--no_budget`          | 校正器的时序槽预算           |




### 静态协议与受控时序设置

- `--strip_timestamps`：运行静态协议 / 不使用时间戳的 TPA（附录 E-B）
- `--time_granularity day|month|year`：受控日历粒度（附录 E-F）
- `--time_coverage 0.0–1.0`：保留带时间戳事实的比例

---



## 🔬 可复现性

我们致力于确保结果的完全可复现性。提供以下资源：

### 📋 实验配置

- **超参数**：所有超参数设置都记录在 `configs/tpa_default.json` 中，可通过命令行参数配置
- **随机种子**：五个种子（`1011`、`2022`、`3033`、`4044`、`5055`）；20/80 基准划分由 `--split_seed` 固定
- **环境**：依赖项如 `requirements.txt` 中指定



### 📊 复现主要结果

要复现论文中报告的主要实验结果：

1. **下载数据集**，按照数据集部分描述的格式
2. **运行已注册的实验**（缺失的数据集会被跳过并在运行清单中报告）：

```bash

python experiments/run_experiment.py backbone_comparison --data_root data

```

1. **汇总并生成报告**：

```bash

python experiments/summarize_results.py results/backbone_comparison

python experiments/render_reports.py

```



### 🏗️ 代码组织

代码库组织为模块化组件以便清晰：

- `src/run_tpa.py`: 主流程入口
- `src/tpa/stage1_time_aware_graph_partition/`: 统一时序画像、双 TKG 划分器与分区匹配器
- `src/tpa/stage2_progressive_temporal_context_refinement/`: 多分辨率时序索引、代价感知路由、时序可达性过滤与局部结构验证
- `src/tpa/stage3_temporal_error_aware_optimization/`: 上下文构建与时序误差感知校正器
- `src/tpa/backbones/`: 六个 TEA backbone（GCN-Align、TEA-GNN、RREA、TREA、Dual-AMN、LightTEA）
- `src/tpa/tpa_framework.py`: TPA 三阶段编排器
- `experiments/`: 实验注册表、运行器、汇总与报告渲染



### 📝 文档

- 全面的内联代码注释，解释关键设计决策
- 清晰的模块结构，采用标准化命名约定
- 本 README，包含逐步使用说明

---



## 📊 评估指标

我们采用标准的知识图谱对齐指标以确保透明度和可比性：

- **Hits@1**：排名第一的正确对齐比例
- **Hits@5**：前 5 名候选中的正确对齐比例  
- **MRR（平均倒数排名）**：正确对齐的平均倒数排名
- **对应实体召回率（Counterpart Recall）**：被构建子任务覆盖的待对齐测试对比例



## 📜 许可证

[GPL-3.0 License](LICENSE) - 保留版权声明。

## 📬 联系方式

- **邮箱**：[runhaozhao@nudt.edu.cn](mailto:runhaozhao@nudt.edu.cn)
- **GitHub Issues**：对于技术问题，请在 [GitHub 仓库](https://github.com/eduzrh/LsTEA/issues) 中创建 Issue。标签：`bug`、`enhancement`、`question`。

目标在 2-3 个工作日内回复。



## 🔗 参考文献

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



## 🙏 致谢

以下开源项目在本工作中被部分引用。我们真诚地感谢他们的贡献：

[Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [JAPE](https://github.com/nju-websoft/JAPE), [GCN-Align](https://github.com/1049451037/GCN-Align), [BETA](https://github.com/DexterZeng/BETA), [LargeEA](https://github.com/ZJU-DAILY/LargeEA), [LIME](https://github.com/DexterZeng/LIME), [ELsEA](https://github.com/wx-qzhou/ELsEA), [OpenEA](https://github.com/nju-websoft/OpenEA), [METIS/PyMETIS](https://github.com/europlexo/pymetis), [PyTorch](https://github.com/pytorch/pytorch)

---

