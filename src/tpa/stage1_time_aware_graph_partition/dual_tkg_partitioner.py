# -*- coding: utf-8 -*-
"""Stage I-2: Dual TKG Partitioner.

Report reference: Section III-B2, Eqs. (2)-(3).

Let X^x be the set of undirected entity-adjacency pairs induced by Q^x
(non-adjacent pairs have weight zero).  Each adjacent entity pair receives

    w~_ij^x = (1 + zeta * TC(e_i, e_j)) * sum_{r} f_x(r),               (2)

where the sum ranges over facts linking e_i and e_j in either direction,
f_x(r) is the relation-diversity weight (Eq. 19) and TC is the temporal
cohesion of Eq. (20).  A larger temporal overlap makes the edge more
costly to cut.

METIS independently divides each TKG into nonempty disjoint partitions
P^x = {P_k^x}_{k=1..K_x}, minimising the weighted cut subject to

    |P_k^x| <= (1 + epsilon_bal) |E^x| / K_x.                          (3)

Each unordered adjacency pair is counted once.
"""
import math
from collections import defaultdict

import numpy as np

from .unified_temporal_profile import temporal_cohesion
from .relation_diversity import relation_diversity_weights


def build_weighted_edges(kg, f_weights, supports, zeta):
    """Undirected adjacency pairs with the combined weight of Eq. (2).

    Returns ``(edge_list, edge_weights)`` with edge_list[i] = (u, v), u < v.
    """
    pair_rel_sum = defaultdict(float)
    for h, r, t in kg.triples:
        if h == t:
            continue
        u, v = (h, t) if h < t else (t, h)
        pair_rel_sum[(u, v)] += f_weights.get(r, 1.0)
    edges, weights = [], []
    for (u, v), rel_sum in pair_rel_sum.items():
        tc = temporal_cohesion(supports[u], supports[v])
        edges.append((u, v))
        weights.append((1.0 + zeta * tc) * rel_sum)
    return edges, np.asarray(weights, dtype=np.float64)


def metis_partition(n_ent, edges, edge_weights, n_parts, epsilon_bal=0.1,
                    seed=0):
    """Balanced weighted METIS partition (report Eq. 3).

    ``epsilon_bal`` is the entity-count tolerance of Eq. (3); METIS'
    internal balance constraint is used with the same target.  When pymetis
    is not installed the function falls back to a deterministic balanced
    hash partition so that the pipeline stays runnable on CPU-only hosts.
    """
    if n_parts <= 1:
        return np.zeros(n_ent, dtype=np.int32)
    try:
        import pymetis
        neighbours = [[] for _ in range(n_ent)]
        for (u, v), w in zip(edges, edge_weights):
            # pymetis takes integer edge weights; temporal cohesion and the
            # relation-diversity sum are scaled by 1000 to keep precision.
            wi = max(1, int(round(float(w) * 1000.0)))
            neighbours[u].append((v, wi))
            neighbours[v].append((u, wi))
        xadj = np.empty(n_ent + 1, dtype=np.int64)
        adjncy = np.empty(2 * len(edges), dtype=np.int64)
        eweights = np.empty(2 * len(edges), dtype=np.int64)
        pos = 0
        for u in range(n_ent):
            xadj[u] = pos
            for v, wi in sorted(neighbours[u]):
                adjncy[pos] = v
                eweights[pos] = wi
                pos += 1
        xadj[n_ent] = pos
        opts = pymetis.Options()
        opts.seed = seed
        try:
            # METIS balance constraint of Eq. (3): each part at most
            # (1 + epsilon_bal) times the average entity count.
            opts.ubvec = [1.0 + float(epsilon_bal)] * n_parts
        except Exception:
            pass
        _, membership = pymetis.part_graph(
            n_parts, xadj=xadj, adjncy=adjncy, eweights=eweights,
            options=opts)
        membership = np.asarray(membership, dtype=np.int32)
        # METIS may return fewer nonempty partitions than requested; keep the
        # membership and expose the number of nonempty parts to the caller.
        return membership
    except Exception:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_ent)
        membership = np.zeros(n_ent, dtype=np.int32)
        base = max(1, n_ent // n_parts)
        for p in range(n_parts):
            lo = p * base
            hi = (p + 1) * base if p < n_parts - 1 else n_ent
            if lo < hi:
                membership[perm[lo:hi]] = p
        return membership


def partition_members(membership, n_parts):
    """Entity lists of the nonempty partitions (P_k^x)."""
    members = []
    for p in range(n_parts):
        idx = np.where(membership == p)[0]
        if idx.size:
            members.append(idx.tolist())
    return members


def partition_balance(members, n_ent, n_parts, epsilon_bal):
    """Diagnostic: fraction of partitions respecting Eq. (3)."""
    if not members:
        return 0.0
    bound = (1.0 + epsilon_bal) * n_ent / max(1, n_parts)
    return sum(1 for m in members if len(m) <= bound) / len(members)


def run_dual_tkg_partitioner(data, p_cal_s, p_cal_t, K_s, K_t,
                             zeta=1.0, epsilon_bal=0.1, seed=0):
    """Partition both TKGs independently with temporal-cohesion weights.

    Returns ``(src_members, tgt_members, src_membership, tgt_membership)``.
    """
    parts = []
    for kg, supports, K in ((data.gs, p_cal_s, K_s), (data.gt, p_cal_t, K_t)):
        f_w = relation_diversity_weights(kg)
        edges, weights = build_weighted_edges(kg, f_w, supports, zeta)
        mem = metis_partition(kg.n_ent, edges, weights, K, epsilon_bal, seed)
        parts.append(mem)
    src_membership, tgt_membership = parts
    src_members = partition_members(src_membership, K_s)
    tgt_members = partition_members(tgt_membership, K_t)
    return src_members, tgt_members, src_membership, tgt_membership