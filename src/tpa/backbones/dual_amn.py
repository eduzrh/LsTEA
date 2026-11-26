# -*- coding: utf-8 -*-
"""Dual-AMN backbone (Mao et al., WWW 2021; report Appendix C-B).

TPA adaptation.  Dual-AMN keeps two attention gates: a time-aware gate on the
relation message and a second relation gate over the concatenation of the
message and the time embedding (dual node/relation attention).  The loss uses
normalized hard negative mining over the sampled negatives, following the
original Dual-AMN training rule; the stabilized cosine of Eq. (35) and the
outer TPA loop are unchanged.
"""
from .neural_tea import NeuralTEABackbone


class DualAMNBackbone(NeuralTEABackbone):
    """Dual node/relation attention with hard negative mining."""

    VARIANT = "dual_amn"

    def __init__(self, data, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(data, variant=self.VARIANT, **kwargs)