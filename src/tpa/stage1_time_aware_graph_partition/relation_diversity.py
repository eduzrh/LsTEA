# -*- coding: utf-8 -*-
"""Relation-diversity weights of the dual TKG partitioner.

Report reference: Appendix A-B1, Eq. (19).

For relation r in G^x, Head^x(r) and Tail^x(r) are its unique heads and
tails and Q^x(r) is its fact set.  For Q^x(r) != empty:

    f_x(r) = (|Head^x(r)| + |Tail^x(r)|) / |Q^x(r)|              (19)

The weight sums head- and tail-diversity ratios and therefore prioritises
relations with fewer repeated head and tail entities.  Because
f_x(r) in (0, 2] for every active relation and zeta >= 0, Eq. (2) yields
positive METIS edge weights.
"""
from collections import defaultdict


def relation_diversity_weights(kg):
    """f_x(r) for every relation of ``kg`` (report Eq. 19)."""
    heads = defaultdict(set)
    tails = defaultdict(set)
    counts = defaultdict(int)
    for h, r, t in kg.triples:
        heads[r].add(h)
        tails[r].add(t)
        counts[r] += 1
    return {r: (len(heads[r]) + len(tails[r])) / counts[r] for r in counts}


def relation_diversity_map(kg):
    """Alias used by context construction (priority = descending f_x(r)).

    The map is cached on the graph object because context construction asks
    for it once per sub-task.
    """
    cached = getattr(kg, "_relation_diversity_map", None)
    if cached is None:
        cached = relation_diversity_weights(kg)
        kg._relation_diversity_map = cached
    return cached


def head_diversity(kg):
    """|Head^x(r)| / |Q^x(r)| used by the relation-summary features."""
    heads = defaultdict(set)
    counts = defaultdict(int)
    for h, r, _t in kg.triples:
        heads[r].add(h)
        counts[r] += 1
    return {r: len(heads[r]) / counts[r] for r in counts}