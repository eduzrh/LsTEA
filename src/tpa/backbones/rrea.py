# -*- coding: utf-8 -*-
"""RREA backbone (Mao et al., CIKM 2020; report Appendix C-B).

TPA adaptation.  RREA replaces the additive relation message by a relational
reflection: a learnable unit reflection vector per relation reflects the
entity message before aggregation.  No timestamp encoder is used; temporal
information reaches the vertex input only through the unified-profile
features.  Scoring (Eq. 35) and training (Eq. 9) are shared with the TPA
framework.
"""
from .neural_tea import NeuralTEABackbone


class RREABackbone(NeuralTEABackbone):
    """Relational reflection message passing."""

    VARIANT = "rrea"

    def __init__(self, data, **kwargs):
        kwargs.pop("variant", None)
        super().__init__(data, variant=self.VARIANT, **kwargs)