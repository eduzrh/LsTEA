# -*- coding: utf-8 -*-
"""TPA: Temporal Partitioning-and-Alignment for large-scale temporal
entity alignment (LsTEA).

Reference: "On Temporal Entity Alignment at Scale".

Layout
------
stage1_time_aware_graph_partition
    Algorithm 1: unified temporal profile space, dual TKG partitioner,
    temporal-aware target candidate matcher.
stage2_progressive_temporal_context_refinement
    Algorithms 2-3: multi-resolution temporal index, cost-aware routing,
    temporal reachability filter, local structure verification.
stage3_temporal_error_aware_optimization
    Algorithms 4-5: bounded context construction, TEA training, temporal
    error-aware corrector.
backbones
    GCN-Align, TEA-GNN, RREA, TREA, Dual-AMN and LightTEA.
"""
from .tpa_framework import TPAFramework, DEFAULT_PARAMS

__version__ = "1.0.0"
__all__ = ["TPAFramework", "DEFAULT_PARAMS"]