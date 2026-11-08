"""Reference (oracle) implementation of the Rust slicer ``hg_slice`` (R-T4).

Written independently from the Rust code, from the specification in D-015 and
issues #11-#13: plain Python loops and dictionaries, no shared helpers. Floating
point operations follow the specified order, so PPR results match bit for bit.

Complexity is irrelevant here (oracle, small graphs only).
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from holm.retrieval.types import LocalSubgraph, SliceParams


@dataclass
class RawGraph:
    """Hypergraph as plain lists; members of every edge sorted by (role, node)."""

    n_nodes: int
    prio: list[int]
    edge_rel: list[int]
    edge_ptr: list[int]
    edge_nodes: list[int]
    edge_roles: list[int]
    types: list[tuple[int, int]] = field(default_factory=list)
    intervals: list[tuple[int, int, int]] = field(default_factory=list)
    n_classes: int = 0

    @classmethod
    def from_edges(
        cls,
        n_nodes: int,
        prio: Sequence[int],
        edges: Sequence[tuple[int, Sequence[tuple[int, int]]]],
        types: Sequence[tuple[int, int]] = (),
        intervals: Sequence[tuple[int, int, int]] = (),
        n_classes: int | None = None,
    ) -> "RawGraph":
        ptr = [0]
        nodes: list[int] = []
        roles: list[int] = []
        rels: list[int] = []
        for rel, members in edges:
            for node, role in sorted(members, key=lambda m: (m[1], m[0])):
                nodes.append(node)
                roles.append(role)
            ptr.append(len(nodes))
            rels.append(rel)
        types_l = [(int(v), int(c)) for v, c in types]
        if n_classes is None:
            n_classes = 1 + max((c for _, c in types_l), default=-1)
        return cls(
            n_nodes,
            list(prio),
            rels,
            ptr,
            nodes,
            roles,
            types_l,
            [(int(e), int(s), int(t)) for e, s, t in intervals],
            n_classes,
        )


class _Oracle:
    def __init__(self, g: RawGraph, p: SliceParams) -> None:
        self.g = g
        self.p = p
        self.members: list[list[tuple[int, int]]] = [
            list(zip(g.edge_nodes[a:b], g.edge_roles[a:b], strict=True))
            for a, b in zip(g.edge_ptr[:-1], g.edge_ptr[1:], strict=True)
        ]
        self.inc: list[list[tuple[int, int]]] = [[] for _ in range(g.n_nodes)]
        for e, ms in enumerate(self.members):
            for node, role in ms:
                self.inc[node].append((e, role))
        self.types: list[set[int]] = [set() for _ in range(g.n_nodes)]
        for v, c in g.types:
            self.types[v].add(c)
        self.hull: dict[int, tuple[int, int]] = {}
        for e, s, t in g.intervals:
            lo, hi = self.hull.get(e, (s, t))
            self.hull[e] = (min(lo, s), max(hi, t))
        self.rels = None if p.relations is None else set(p.relations)
        self.classes = None if p.class_filter is None else set(p.class_filter)

    def rel_ok(self, e: int) -> bool:
        return self.rels is None or self.g.edge_rel[e] in self.rels

    def edge_ok(self, e: int) -> bool:
        if self.p.time_window is None or e not in self.hull:
            return True
        lo, hi = self.p.time_window
        s, t = self.hull[e]
        return s <= hi and lo <= t

    def node_ok(self, v: int) -> bool:
        return self.classes is None or bool(self.types[v] & self.classes)

    def inc_prio(self, v: int, e: int) -> int:
        return max((self.g.prio[w] for w, _ in self.members[e] if w != v), default=0)

    def top(self, v: int) -> list[tuple[int, int]]:
        cand = [(e, r) for e, r in self.inc[v] if self.rel_ok(e)]
        cand.sort(key=lambda t: (-self.inc_prio(v, t[0]), t[0], t[1]))
        return [t for t in cand if self.edge_ok(t[0])][: self.p.delta]

    def theta(self, v: int) -> float:
        deg = sum(1 for e, _ in self.inc[v] if self.rel_ok(e))
        return self.p.epsilon * float(max(min(deg, self.p.delta), 1))


def _merge_seeds(
    g: RawGraph, seeds: Sequence[int], masses: Sequence[float], p: SliceParams
) -> list[tuple[int, float]]:
    if len(seeds) == 0:
        raise ValueError("no seeds")
    pairs = sorted(zip(seeds, masses, strict=True), key=lambda t: t[0])
    merged: list[tuple[int, float]] = []
    for v, m in pairs:
        if not 0 <= v < g.n_nodes:
            raise ValueError(f"seed node {v} out of range")
        if merged and merged[-1][0] == v:
            merged[-1] = (v, merged[-1][1] + m)
        else:
            merged.append((v, m))
    if len(merged) > p.max_nodes:
        raise ValueError(f"{len(merged)} seeds exceed max_nodes {p.max_nodes}")
    return merged


def _ball(o: _Oracle, seeds: list[int]) -> tuple[list[int], list[float], list[int], dict[str, Any]]:
    depth = {s: 0 for s in seeds}
    order = list(seeds)
    frontier = list(seeds)
    traversed: list[int] = []
    pushes = work = 0
    stop = False
    for d in range(o.p.hops):
        nxt: list[int] = []
        for u in frontier:
            incs = o.top(u)
            pushes += 1
            work += len(incs)
            for e, _ in incs:
                traversed.append(e)
                for w, _ in o.members[e]:
                    if w == u or w in depth or not o.node_ok(w):
                        continue
                    if len(order) >= o.p.max_nodes:
                        stop = True
                        break
                    depth[w] = d + 1
                    order.append(w)
                    nxt.append(w)
                if stop:
                    break
            if stop:
                break
        if stop or not nxt:
            break
        frontier = nxt
    scores = [1.0 / (1.0 + depth[v]) for v in order]
    return (
        order,
        scores,
        traversed,
        {"pushes": pushes, "work": work, "touched": len(order), "residual_mass": 0.0},
    )


def _ppr(
    o: _Oracle, seeds: list[tuple[int, float]]
) -> tuple[list[int], list[float], list[int], dict[str, Any]]:
    alpha = o.p.alpha
    p: dict[int, float] = {}
    r: dict[int, float] = {}
    for v, m in seeds:
        r[v] = r.get(v, 0.0) + m
    queue: deque[int] = deque()
    queued: set[int] = set()
    for v, _ in seeds:
        if r[v] >= o.theta(v) and v not in queued:
            queued.add(v)
            queue.append(v)
    traversed: list[int] = []
    pushes = work = 0
    while queue:
        u = queue.popleft()
        queued.discard(u)
        ru = r[u]
        if ru < o.theta(u):
            continue
        spans: list[list[int]] = []
        for e, _ in o.top(u):
            others = [w for w, _ in o.members[e] if w != u and o.node_ok(w)]
            if others:
                spans.append(others)
                traversed.append(e)
        pushes += 1
        if not spans:
            p[u] = p.get(u, 0.0) + ru
            r[u] = 0.0
            continue
        work += len(spans)
        p[u] = p.get(u, 0.0) + alpha * ru
        r[u] = 0.0
        share = ((1.0 - alpha) * ru) / len(spans)
        for others in spans:
            m = share / len(others)
            for w in others:
                r[w] = r.get(w, 0.0) + m
                if w not in queued and r[w] >= o.theta(w):
                    queued.add(w)
                    queue.append(w)
    seed_ids = sorted({v for v, _ in seeds})
    is_seed = set(seed_ids)
    rest = sorted(
        ((v, x) for v, x in p.items() if x != 0.0 and v not in is_seed), key=lambda t: (-t[1], t[0])
    )
    rest = rest[: max(o.p.max_nodes - len(seed_ids), 0)]
    nodes = seed_ids + [v for v, _ in rest]
    scores = [p.get(v, 0.0) for v in seed_ids] + [x for _, x in rest]
    residual = sum(r[v] for v in sorted(r) if r[v] != 0.0)
    stats = {"pushes": pushes, "work": work, "touched": len(set(p) | set(r)), "residual_mass": residual}
    return nodes, scores, traversed, stats


def slice_ref(g: RawGraph, seeds: Sequence[int], masses: Sequence[float], p: SliceParams) -> LocalSubgraph:
    """Oracle of ``hg_slice::slice``."""
    merged = _merge_seeds(g, seeds, masses, p)
    o = _Oracle(g, p)
    run: Callable[..., Any] = _ball if p.method == "ball" else _ppr
    arg: Any = [v for v, _ in merged] if p.method == "ball" else merged
    nodes, scores, traversed, stats = run(o, arg)

    local = {v: i for i, v in enumerate(nodes)}
    cand = set(traversed)
    for u in nodes:
        if len(o.inc[u]) <= p.scan_cap:
            cand.update(e for e, _ in o.inc[u])
    edges: list[int] = []
    rel: list[int] = []
    ptr = [0]
    members: list[int] = []
    roles: list[int] = []
    for e in sorted(cand):
        if not o.rel_ok(e) or not o.edge_ok(e):
            continue
        ms = o.members[e]
        if not all(w in local for w, _ in ms):
            continue
        edges.append(e)
        rel.append(g.edge_rel[e])
        members.extend(local[w] for w, _ in ms)
        roles.extend(role for _, role in ms)
        ptr.append(len(members))
    return LocalSubgraph.from_native(
        {
            "nodes": np.array(nodes, dtype=np.uint32),
            "scores": np.array(scores, dtype=np.float64),
            "edges": np.array(edges, dtype=np.uint32),
            "edge_rel": np.array(rel, dtype=np.uint16),
            "edge_ptr": np.array(ptr, dtype=np.uint32),
            "edge_members": np.array(members, dtype=np.uint32),
            "edge_roles": np.array(roles, dtype=np.uint8),
            **stats,
        }
    )
