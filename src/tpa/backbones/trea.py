# -*- coding: utf-8 -*-
"""TREA backbone (Xu et al., WWW 2022; report Appendix C-B).

TPA adaptation.  TREA couples both inputs of temporal relational attention:
the time embedding gates the relation-aware message, and the relation
embedding gates it again through an inner-product attention.  This is the
temporal-relational attention used by the original TREA encoder, placed in
the shared two-layer TPA encoder; scoring (Eq. 35) and training (Eq. 9)
remain those of TPA.
"""
from .neural_tea import NeuralTEABackbone


class TREABackbone(NeuralTEABackbone):
    """Temporal-relational attention over (relation, time) pairs."""

    VARIANT = "trea"

    def __init__(self, data, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(data, variant=self.VARIANT, **kwargs)