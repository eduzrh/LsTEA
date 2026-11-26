# -*- coding: utf-8 -*-
"""TEA backbones used by TPA (report Section III-D1, Appendix Appendix C-B).

Six backbones from the paper are registered here, one class each:

    gcn_align  GCN-Align  (Wang et al., EMNLP 2018)
    tea_gnn    TEA-GNN    (Xu et al., EMNLP 2021)
    rrea       RREA       (Mao et al., CIKM 2020)
    trea       TREA       (Xu et al., WWW 2022)
    dual_amn   Dual-AMN   (Mao et al., WWW 2021)
    lighttea   LightTEA   (Cai et al., IJCAI 2023)

The five neural backbones share the two-layer encoder and the objective of
Eq. (9); their modules differ in how relations and calibrated timestamps
enter message passing.  LightTEA is the non-neural three-view
label-propagation backbone.
"""
from .gcn_align import GCNAlignBackbone
from .tea_gnn import TEAGNNBackbone
from .rrea import RREABackbone
from .trea import TREABackbone
from .dual_amn import DualAMNBackbone
from .lighttea import LightTEABackbone
from .neural_tea import NeuralTEABackbone, TEAEncoder

BACKBONES = {
    "gcn_align": GCNAlignBackbone,
    "tea_gnn": TEAGNNBackbone,
    "rrea": RREABackbone,
    "trea": TREABackbone,
    "dual_amn": DualAMNBackbone,
    "lighttea": LightTEABackbone,
}
ALL_BACKBONES = list(BACKBONES)
NEURAL_BACKBONES = {k: v for k, v in BACKBONES.items() if k != "lighttea"}
# Variant tag consumed by the shared neural encoder.
NEURAL_VARIANTS = {k: cls.VARIANT for k, cls in NEURAL_BACKBONES.items()}


def build_backbone(name, data, params=None):
    """Instantiate a backbone by its paper name (snake_case alias)."""
    params = dict(params or {})
    if params.get("device") is None:
        params.pop("device", None)
    if name not in BACKBONES:
        raise ValueError("unknown backbone %r; available: %s"
                         % (name, ", ".join(ALL_BACKBONES)))
    if name == "lighttea":
        return LightTEABackbone(
            data,
            n_prop=params.get("n_prop", 2),
            alpha=params.get("lighttea_alpha", 0.6),
            ent_dim=params.get("lighttea_dim", 256),
            mini_dim=params.get("lighttea_mini_dim", 16),
            seed=params.get("seed", 0))
    return BACKBONES[name](
        data,
        dim=params.get("dim", 128),
        lr=params.get("lr", 0.005),
        margin=params.get("margin", 1.0),
        temperature=params.get("temperature", 0.1),
        n_neg=params.get("n_neg", 10),
        device=params.get("device"),
        seed=params.get("seed", 0),
        loss_margin=params.get("loss_margin", True),
        beta=params.get("beta", 1.0))