# -*- coding: utf-8 -*-
"""Stage II: Progressive Temporal Context Refinement (report Section III-C).

Submodules
----------
multi_resolution_temporal_index : equi-depth L-level posting lists and
                                  IDF_local / IDF_global / IDF_mix (Eqs. 6, 21)
cost_aware_routing               : B_temporal / B_struct scores, route rule,
                                  three-level cascade, SeedVote (Eqs. 24-31)
temporal_reachability_filter     : pair event graph, Fwd/Bwd scans, Theta
                                  (Section III-C3, Appendix A-B6)
local_structure_verification     : 13-dimensional signature, W_ij, SV (Eq. 7)
progressive_refiner               : per-pair Algorithm 3 driver
"""
from .multi_resolution_temporal_index import (
    MultiResolutionIndex,
    TargetIndex,
    bucket_boundaries,
)
from .cost_aware_routing import (
    routing_scores,
    decide_route,
    cascade,
    seed_vote,
    route_candidates,
    Route,
    Cascade,
    SeedVote,
    RouteRetrieve,
)
from .temporal_reachability_filter import (
    build_pair_facts,
    scan_pair,
    theta_vertices,
    theta_fact_indices,
    retained_slot_count,
    Events,
    Fwd,
    Bwd,
)
from .local_structure_verification import (
    D_SIG,
    ego_facts,
    full_signature,
    idf_weights,
    verify_score,
    Signatures,
)
from .progressive_refiner import ProgressiveContextRefiner

__all__ = [
    "MultiResolutionIndex", "TargetIndex", "bucket_boundaries",
    "routing_scores", "decide_route", "cascade", "seed_vote",
    "route_candidates", "Route", "Cascade", "SeedVote", "RouteRetrieve",
    "build_pair_facts", "scan_pair", "theta_vertices", "theta_fact_indices",
    "retained_slot_count", "Events", "Fwd", "Bwd", "D_SIG", "ego_facts",
    "full_signature", "idf_weights", "verify_score", "Signatures",
    "ProgressiveContextRefiner",
]