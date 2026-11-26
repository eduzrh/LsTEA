# -*- coding: utf-8 -*-
"""Evaluation: Hits@k, MRR, counterpart recall (report Section IV-A3).

Following the LsEA protocol, test pairs not covered by any retained sub-task
receive rank infinity and contribute zero to Hits@k and MRR; covered queries
are ranked by their sub-task candidate similarity scores produced by the
backbone Psi.  When a test entity appears in several contexts, its candidate
scores are aggregated by the maximum across sub-tasks.

Counterpart recall is the proportion of to-be-aligned test pairs covered by
the constructed sub-tasks, i.e. both endpoints belong to a common context
(P_i, P_j), not the size of an individual candidate set.
"""
from collections import defaultdict


def prediction_coverage(predictions, coverage_map, score=2.0):
    """Merge final alignments into the ranking map at rank-1 precedence."""
    for (u, v) in predictions:
        cov = coverage_map.setdefault(u, {})
        if score > cov.get(v, float("-inf")):
            cov[v] = score
    return coverage_map


def evaluate(test_pairs, predictions, coverage_map):
    """Return hit1 / hit5 / mrr / coverage-recall over ``test_pairs``."""
    n = len(test_pairs)
    if n == 0:
        return dict(hit1=0.0, hit5=0.0, mrr=0.0, recall=0.0,
                    n_test=0, n_covered=0)
    merged = prediction_coverage(predictions, coverage_map)
    hits1 = hits5 = covered = 0
    rr = 0.0
    for (u, v) in test_pairs:
        cands = merged.get(u)
        if cands is None or v not in cands:
            continue
        covered += 1
        score_v = cands[v]
        rank = 1 + sum(1 for s in cands.values() if s > score_v)
        if rank == 1:
            hits1 += 1
        if rank <= 5:
            hits5 += 1
        rr += 1.0 / rank
    return dict(hit1=hits1 / n, hit5=hits5 / n, mrr=rr / n,
                recall=covered / n, n_test=n, n_covered=covered)


def subtask_recall(test_pairs, subtask_cov):
    """Proportion of test pairs whose endpoints share a retained sub-task."""
    if not test_pairs:
        return 0.0
    covered = sum(1 for (u, v) in test_pairs
                  if v in subtask_cov.get(u, ()))
    return covered / len(test_pairs)


def hits_over_recall(metrics):
    """Hits@1 / Recall ratio reported in Sec. IV-B / Appendix E-I."""
    rec = metrics.get("recall", 0.0)
    return metrics.get("hit1", 0.0) / rec if rec > 0 else 0.0


def coverage_map_from_contexts(contexts, embed_fn, score_fn):
    """Build per-source candidate score maps from all final contexts.

    ``embed_fn(ctx)`` returns the backbones' joint embedding dict and
    ``score_fn(ys, yt)`` the stabilized cosine of Eq. (35).  Candidate scores
    from several sub-tasks aggregate by maximum.
    """
    coverage = {}
    for ctx in contexts:
        emb = embed_fn(ctx)
        for e, cands in ctx.get("in_c5", {}).items():
            ys = emb.get(("s", e))
            if ys is None or not cands:
                continue
            dst = coverage.setdefault(e, {})
            for e2 in cands:
                yt = emb.get(("t", e2))
                if yt is None:
                    continue
                s = score_fn(ys, yt)
                if s > dst.get(e2, float("-inf")):
                    dst[e2] = s
    return coverage


def subtask_coverage_map(contexts):
    cov = defaultdict(set)
    for ctx in contexts:
        for e in ctx["Hs"]:
            cov[e].update(ctx["Ht"])
    return cov