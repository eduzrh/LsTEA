# -*- coding: utf-8 -*-
"""GCN-Align backbone (Wang et al., EMNLP 2018; report Appendix C-B).

TPA adaptation.  Each bounded context sub-task is encoded by the shared
two-layer relational GCN without any timestamp encoder; temporal information
reaches the vertex input only through the unified-profile features supplied
by :class:`backbones.subtask.SubTaskGraph`.  The encoder output is scored with
the stabilized cosine of Eq. (35) and trained with the objective of Eq. (9),
so the backbone contributes structure-only message passing while TPA owns
candidate generation and pseudo-seed control.
"""
from .neural_tea import NeuralTEABackbone


class GCNAlignBackbone(NeuralTEABackbone):
    """Two-layer relational GCN with cosine scoring."""

    VARIANT = "gcn_align"

    def __init__(self, data, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(data, variant=self.VARIANT, **kwargs)