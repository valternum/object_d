"""Interfaces between retrieval stages (R-I1).

Changing these dataclasses requires updating the contract tests and a D-xxx entry.
Shapes: N = selected nodes, E = induced edges, M = total members of induced edges.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

METHODS = ("ball", "ppr")


@dataclass(frozen=True)
class SliceParams:
    """Parameters of one per-query dynamic slice (mirrors ``hg_slice::SliceParams``, D-015).

    method: ``"ball"`` (bounded BFS) or ``"ppr"`` (personalized PageRank, local push).
    max_nodes: slice budget, seeds included. hops: BFS depth (ball). delta: incidences
    expanded per node (top-delta by priority). alpha / epsilon: PPR teleport and tolerance;
    PPR work is bounded by total seed mass / (alpha * epsilon). scan_cap: rows longer than
    this are not scanned for induced edges. relations / class_filter: allowed relation ids /
    class ids (already closed under subclasses); time_window: inclusive years.
    """

    method: str
    max_nodes: int
    hops: int
    delta: int
    alpha: float
    epsilon: float
    scan_cap: int
    relations: tuple[int, ...] | None = None
    class_filter: tuple[int, ...] | None = None
    time_window: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be >= 1")
        if min(self.hops, self.delta, self.scan_cap) < 0:
            raise ValueError("hops, delta and scan_cap must be >= 0")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not (self.epsilon > 0.0 and math.isfinite(self.epsilon)):
            raise ValueError("epsilon must be > 0")
        if self.time_window is not None and self.time_window[0] > self.time_window[1]:
            raise ValueError("time_window must satisfy lo <= hi")

    def native_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``holm_native.Graph.slice`` / ``Index.slice``."""
        return {
            "method": self.method,
            "max_nodes": self.max_nodes,
            "hops": self.hops,
            "delta": self.delta,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "scan_cap": self.scan_cap,
            "relations": None
            if self.relations is None
            else np.array(sorted(set(self.relations)), dtype=np.uint16),
            "class_filter": None
            if self.class_filter is None
            else np.array(sorted(set(self.class_filter)), dtype=np.uint32),
            "time_window": self.time_window,
        }


@dataclass(frozen=True)
class SliceStats:
    """Work counters: expansions/pushes, sum of d(u), touched nodes, residual PPR mass."""

    pushes: int
    work: int
    touched: int
    residual_mass: float


@dataclass(frozen=True)
class LocalSubgraph:
    """A slice in local ids (local id = position in ``nodes``).

    nodes: [N] uint32 global ids, seeds first. scores: [N] float64 (ball: 1/(1+depth);
    ppr: estimated mass). edges: [E] uint32 global ids, sorted. edge_rel: [E] uint16.
    edge_ptr: [E+1] uint32 offsets. edge_members: [M] uint32 local ids. edge_roles: [M] uint8.
    """

    nodes: NDArray[np.uint32]
    scores: NDArray[np.float64]
    edges: NDArray[np.uint32]
    edge_rel: NDArray[np.uint16]
    edge_ptr: NDArray[np.uint32]
    edge_members: NDArray[np.uint32]
    edge_roles: NDArray[np.uint8]
    stats: SliceStats

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @classmethod
    def from_native(cls, d: dict[str, Any]) -> "LocalSubgraph":
        """Validates shapes and dtypes of a native (or oracle) result dictionary."""
        s = cls(
            nodes=np.asarray(d["nodes"], dtype=np.uint32),
            scores=np.asarray(d["scores"], dtype=np.float64),
            edges=np.asarray(d["edges"], dtype=np.uint32),
            edge_rel=np.asarray(d["edge_rel"], dtype=np.uint16),
            edge_ptr=np.asarray(d["edge_ptr"], dtype=np.uint32),
            edge_members=np.asarray(d["edge_members"], dtype=np.uint32),
            edge_roles=np.asarray(d["edge_roles"], dtype=np.uint8),
            stats=SliceStats(
                pushes=int(d["pushes"]),
                work=int(d["work"]),
                touched=int(d["touched"]),
                residual_mass=float(d["residual_mass"]),
            ),
        )
        if s.scores.shape != s.nodes.shape:
            raise ValueError("scores and nodes differ in length")
        if s.edge_rel.shape != s.edges.shape:
            raise ValueError("edge_rel and edges differ in length")
        if (
            s.edge_ptr.shape[0] != s.n_edges + 1
            or int(s.edge_ptr[0]) != 0
            or int(s.edge_ptr[-1]) != s.edge_members.shape[0]
            or bool(np.any(np.diff(s.edge_ptr.astype(np.int64)) < 0))
        ):
            raise ValueError("edge_ptr is not a valid offset array")
        if s.edge_roles.shape != s.edge_members.shape:
            raise ValueError("edge_roles and edge_members differ in length")
        if s.edge_members.size and int(s.edge_members.max()) >= s.n_nodes:
            raise ValueError("edge_members holds a local id >= n_nodes")
        return s
