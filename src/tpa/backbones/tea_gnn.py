# -*- coding: utf-8 -*-
"""TEA-GNN backbone (Xu et al., EMNLP 2021; report Appendix C-B).

TPA adaptation.  TEA-GNN consumes the calibrated fact time through a time
embedding: each relational message is gated by a sigmoid attention over the
concatenation of the relation-aware message and the time vector, as in the
original time-aware GNN.  The shared two-layer encoder, the stabilized cosine
of Eq. (35), and the training objective of Eq. (9) are those of the TPA
framework, so the backbone only changes how timestamps enter message passing.
"""
from .neural_tea import NeuralTEABackbone


class TEAGNNBackbone(NeuralTEABackbone):
    """Temporal-attention message passing over timestamped facts."""

    VARIANT = "tea_gnn"

    def __init__(self, data, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(data, variant=self.VARIANT, **kwargs)