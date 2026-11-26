# -*- coding: utf-8 -*-
"""Benchmark loading and unified temporal calibration.

Report reference: Section III-B1 (Eq. 1, Definition 4), Appendix C-A.

Supported raw formats (TEA-GNN / LightTEA / BETA releases):

  * 4-column triples: ``head  relation  tail  time``
  * 5-column triples: ``head  relation  tail  t_begin  t_end``
  * 6-column BETA:    ``head  relation  tail  t_begin  t_end  (unused)``
  * ``~`` or an empty token marks an untimestamped fact.

Unified temporal calibration.  Every distinct time value observed in either
graph is mapped to its rank; the ranks form the common calibrated slot grid
a_0 < ... < a_{N-1}, and a fact interval [t_b, t_e] activates the half-open
slot range [rank(t_b), rank(t_e) + 1).  For releases whose time columns are
chronologically ordered ids (YAGO-WIKI50K, BETA), rank compression of the
ids is order-isomorphic to rank compression of the underlying dates, so the
calibrated slot structure is identical.

Static benchmarks (IDS15K, DBP15K, IDS100K, DWY100K, DBP1M, FBDBP2M) are
loaded with ``static=True``: timestamps are ignored, N = 1, and every
temporal component of TPA degenerates to its structural counterpart, which
is the RQ5 protocol.

Entity and relation ids.  Raw releases may continue target numbering after
the source side.  Both sides are remapped to dense local spaces; relation
names from ``rel_ids_1`` / ``rel_ids_2`` are merged into one shared
ontology when available.

Ground truth / splits.  ``ref_pairs`` (and ``sup_pairs`` when shipped) form
the equivalence set.  Following DivEA / ELsEA, 20% of the pairs are seed
pairs and the remaining 80% are test pairs.
"""
import os

import numpy as np


def _read_lines(fn):
    with open(fn, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def _read_rel_names(fn):
    mapping = {}
    if not os.path.exists(fn):
        return mapping
    for ln in _read_lines(fn):
        parts = ln.split("\t")
        if len(parts) >= 2:
            try:
                mapping[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return mapping


def _parse_time(s):
    """Parse a raw time token to a sortable integer.

    Handles integers/floats ("3856", "1979.0"), ISO dates ("1979-05-16",
    "1996-01", "1979") and BETA-style padded negatives ("-0001").
    """
    s = s.strip()
    if s in ("~", "", "unknown", "None"):
        return None
    try:
        return int(float(s))
    except ValueError:
        pass
    parts = s.split("-")
    if len(parts) >= 2:
        try:
            y = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 1
            d = int(parts[2]) if len(parts) > 2 else 1
            return y * 10000 + m * 100 + d
        except ValueError:
            return None
    return None


def _coarsen_time(value, granularity, salt=0):
    """Map a parsed time value onto the requested calendar granularity.

    ``day`` keeps a native day-level timestamp; ``month`` and ``year``
    collapse YYYYMMDD values to YYYYMM / YYYY.  Releases that store only a
    year id (YAGO-WIKI50K) receive deterministic month/day representatives
    keyed by the fact position, so that the controlled granularities of
    Appendix E-F remain distinguishable instead of collapsing to one slot per
    year.
    """
    if value is None or granularity in (None, "", "native"):
        return value
    if granularity not in ("day", "month", "year"):
        raise ValueError("unknown time granularity: %r" % (granularity,))
    if granularity == "day":
        if abs(value) < 1000:                  # year id -> representative day
            month = salt * 37 % 12 + 1
            day = salt * 53 % 28 + 1
            return value * 10000 + month * 100 + day
        return value
    if abs(value) < 1000:                      # year id
        if granularity == "year":
            return value
        return value * 100 + (salt * 37 % 12 + 1)
    if value >= 10000101:                      # YYYYMMDD
        year = value // 10000
        month = (value // 100) % 100
        return year if granularity == "year" else year * 100 + month
    if granularity == "year":                  # ordinary year id
        return value
    return value * 100 + (salt * 37 % 12 + 1)


def _triple_file(data_dir, side):
    """Use the canonical ``triples_{1,2}`` release.

    A few packaged copies add ``triples_{1,2}c`` where untimestamped facts
    are rewritten with time 0; that destroys the ``~`` semantics, so the
    plain file is authoritative and the ``c`` variant is only a fallback
    when the plain file is absent.
    """
    plain = os.path.join(data_dir, "triples_%d" % side)
    packed = os.path.join(data_dir, "triples_%dc" % side)
    if os.path.exists(plain):
        return plain
    return packed


def _read_time_columns(fn):
    """Yield (tb_token, te_token) for every triple line of ``fn``."""
    with open(fn, encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if len(parts) >= 5:
                yield parts[3], parts[4]
            elif len(parts) == 4:
                yield parts[3], parts[3]


def read_triples(fn, time_rank, ent_remap=None, rel_remap=None,
                 static=False, time_granularity=None):
    """Read one side; returns (triples, slot_intervals).

    ``slot_intervals[i]`` is ``(lo, hi)`` (half-open) or ``None`` for an
    untimestamped fact / static loading.
    """
    ent_remap = ent_remap or {}
    rel_remap = rel_remap or {}
    triples, intervals = [], []
    with open(fn, encoding="utf-8") as f:
        for pos, ln in enumerate(f):
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            interval = None
            if not static:
                if len(parts) >= 5:
                    tb_s, te_s = parts[3], parts[4]
                elif len(parts) == 4:
                    tb_s = te_s = parts[3]
                else:
                    tb_s = te_s = "~"
                if tb_s not in ("~", "") and te_s not in ("~", ""):
                    tb_key, te_key = _parse_time(tb_s), _parse_time(te_s)
                    tb_key = _coarsen_time(tb_key, time_granularity, pos)
                    te_key = _coarsen_time(te_key, time_granularity, pos)
                    if (tb_key is not None and te_key is not None
                            and tb_key in time_rank and te_key in time_rank):
                        tb, te = time_rank[tb_key], time_rank[te_key]
                        if tb > te:
                            tb, te = te, tb
                        interval = (tb, te + 1)
            triples.append((ent_remap.get(h, h), rel_remap.get(r, r),
                            ent_remap.get(t, t)))
            intervals.append(interval)
    return triples, intervals


def read_pairs(fn, ent_remap=None):
    ent_remap = ent_remap or {}
    pairs = []
    for ln in _read_lines(fn):
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        pairs.append((ent_remap.get(int(parts[0]), int(parts[0])),
                      ent_remap.get(int(parts[1]), int(parts[1]))))
    return pairs


class TemporalKG:
    """One side of a knowledge graph with dense local ids."""

    def __init__(self, triples, intervals, n_ent):
        self.triples = triples
        self.intervals = intervals
        self.n_ent = n_ent
        self.head2facts = {}
        self.tail2facts = {}
        for idx, (h, _r, t) in enumerate(triples):
            self.head2facts.setdefault(h, []).append(idx)
            self.tail2facts.setdefault(t, []).append(idx)

    def facts_of(self, ent):
        return self.head2facts.get(ent, []) + self.tail2facts.get(ent, [])

    @property
    def n_facts(self):
        return len(self.triples)


def _dense_remap(ids):
    return {g: i for i, g in enumerate(sorted(set(ids)))}


class TEAData:
    """A full entity-alignment benchmark with calibrated temporal slots."""

    def __init__(self, data_dir, train_ratio=0.2, seed=1011,
                 time_coverage=1.0, static=False, time_granularity=None):
        self.data_dir = data_dir
        self.name = os.path.basename(
            os.path.normpath(data_dir)).replace("_", "-")
        self.static = static
        self.time_granularity = time_granularity
        f1 = _triple_file(data_dir, 1)
        f2 = _triple_file(data_dir, 2)

        # ---- unified temporal calibration over both graphs ----
        time_vals = set()
        if not static:
            for fn in (f1, f2):
                for pos, (tb_s, te_s) in enumerate(_read_time_columns(fn)):
                    for s in (tb_s, te_s):
                        tv = _parse_time(s)
                        tv = _coarsen_time(tv, time_granularity, salt=pos)
                        if tv is not None:
                            time_vals.add(tv)
        ordered = sorted(time_vals)
        self.time_rank = {v: i for i, v in enumerate(ordered)}
        self.N = max(1, len(ordered))

        # ---- relation vocabulary ----
        rel_names_1 = _read_rel_names(os.path.join(data_dir, "rel_ids_1"))
        rel_names_2 = _read_rel_names(os.path.join(data_dir, "rel_ids_2"))
        if rel_names_1 and rel_names_2:
            shared = {}
            for name in sorted(set(rel_names_1) | set(rel_names_2)):
                shared[name] = len(shared)
            rel_remap_1 = {rid: shared[n] for n, rid in rel_names_1.items()}
            rel_remap_2 = {rid: shared[n] for n, rid in rel_names_2.items()}
        else:
            raw_r1, raw_r2 = set(), set()
            for fn, bucket in ((f1, raw_r1), (f2, raw_r2)):
                with open(fn, encoding="utf-8") as f:
                    for ln in f:
                        parts = ln.rstrip("\n").split("\t")
                        if len(parts) >= 3:
                            bucket.add(int(parts[1]))
            max_r1 = max(raw_r1) if raw_r1 else -1
            if raw_r2 and min(raw_r2) <= max_r1:
                rel_remap_1 = {}
                rel_remap_2 = {r: r + max_r1 + 1 for r in raw_r2}
            else:
                rel_remap_1, rel_remap_2 = {}, {}
        self.rel_remap_1 = rel_remap_1
        self.rel_remap_2 = rel_remap_2

        # ---- entity remaps ----
        raw_ids_1, raw_ids_2 = set(), set()
        for fn, bucket in ((f1, raw_ids_1), (f2, raw_ids_2)):
            with open(fn, encoding="utf-8") as f:
                for ln in f:
                    parts = ln.rstrip("\n").split("\t")
                    if len(parts) >= 3:
                        bucket.add(int(parts[0]))
                        bucket.add(int(parts[2]))
        ref_raw = read_pairs(os.path.join(data_dir, "ref_pairs"))
        sup_path = os.path.join(data_dir, "sup_pairs")
        sup_raw = read_pairs(sup_path) if os.path.exists(sup_path) else []
        for u, _v in ref_raw + sup_raw:
            raw_ids_1.add(u)
        for _u, v in ref_raw + sup_raw:
            raw_ids_2.add(v)
        ent_remap_1 = _dense_remap(raw_ids_1)
        ent_remap_2 = _dense_remap(raw_ids_2)

        t1, i1 = read_triples(f1, self.time_rank, ent_remap_1, rel_remap_1,
                              static=static,
                              time_granularity=time_granularity)
        t2, i2 = read_triples(f2, self.time_rank, ent_remap_2, rel_remap_2,
                              static=static,
                              time_granularity=time_granularity)
        ref = [(ent_remap_1[u], ent_remap_2[v]) for u, v in ref_raw]
        sup = [(ent_remap_1[u], ent_remap_2[v]) for u, v in sup_raw]

        self.gs = TemporalKG(t1, i1, len(ent_remap_1))
        self.gt = TemporalKG(t2, i2, len(ent_remap_2))

        gt_set = set(ref) | set(sup)
        gt_pairs = sorted(gt_set)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(gt_pairs))
        n_train = int(round(train_ratio * len(gt_pairs)))
        train_idx = set(idx[:n_train].tolist())
        self.seeds = [gt_pairs[i] for i in sorted(train_idx)]
        seed_set = set(self.seeds)
        self.test_pairs = [p for p in gt_pairs if p not in seed_set]
        self.all_pairs = gt_pairs

        if time_coverage < 1.0:
            rng_cov = np.random.RandomState(seed + 1)
            for kg in (self.gs, self.gt):
                keep = rng_cov.rand(len(kg.triples)) < time_coverage
                kg.intervals = [iv if k else None
                                for k, iv in zip(keep, kg.intervals)]

        self.n_rel = max(
            max((r for _h, r, _t in t1), default=-1),
            max((r for _h, r, _t in t2), default=-1),
        ) + 1

    # ------------------------------------------------------------------
    def statistics(self):
        def ts_ratio(kg):
            n = len(kg.triples)
            k = sum(1 for iv in kg.intervals if iv is not None)
            return 100.0 * k / max(1, n)

        # Appendix C-A reports the entities that participate in facts.  Releases
        # such as DBP1M also carry link-only vertices in ``ent_ids``; those
        # remain valid candidates in the graph but are not counted as KG
        # entities that participate in at least one fact.
        act_s, act_t = set(), set()
        for h, _r, t in self.gs.triples:
            act_s.add(h)
            act_s.add(t)
        for h, _r, t in self.gt.triples:
            act_t.add(h)
            act_t.add(t)

        return {
            "dataset": self.name,
            "n_ent_s": len(act_s),
            "n_ent_t": len(act_t),
            "n_ent_s_all": self.gs.n_ent,
            "n_ent_t_all": self.gt.n_ent,
            "facts_s": len(self.gs.triples),
            "facts_t": len(self.gt.triples),
            "relations": len({r for (_h, r, _t) in self.gs.triples}),
            "n_rel": len({r for (_h, r, _t) in self.gs.triples}),
            "relations_s": len({r for (_h, r, _t) in self.gs.triples}),
            "relations_t": len({r for (_h, r, _t) in self.gt.triples}),
            "N": self.N,
            "time_granularity": self.time_granularity or "native",
            "timestamped_ratio_s": ts_ratio(self.gs),
            "timestamped_ratio_t": ts_ratio(self.gt),
            "train_pairs": len(self.seeds),
            "test_pairs": len(self.test_pairs),
        }