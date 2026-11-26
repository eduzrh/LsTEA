# -*- coding: utf-8 -*-
"""Stage III: Temporal Error-Aware Optimization (report Section III-D).

Submodules
----------
context_construction            : bounded context sub-tasks under M / F
                                  (Section III-D1, Eq. 8, Appendix A-B8)
tea_training (backbones/)       : TEA training objective (Eq. 9, Appendix A-B9)
temporal_error_aware_corrector  : Resolve / margin acceptance / high-IDF
                                  gated propagation (Section III-D2, Eqs. 10-11)
"""
from .context_construction import (
    build_context,
    context_seed_pairs,
    subtask_coverage,
    Context,
)
from .temporal_error_aware_corrector import (
    infer_proposals,
    resolve,
    accept,
    high_idf_mask,
    budget_mask,
    update_profile,
    correct,
    Infer,
    Resolve,
    BudgetMask,
)

__all__ = [
    "build_context", "context_seed_pairs", "subtask_coverage", "Context",
    "infer_proposals", "resolve", "accept", "high_idf_mask", "budget_mask",
    "update_profile", "correct", "Infer", "Resolve", "BudgetMask",
]