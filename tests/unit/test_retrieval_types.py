"""Contracts of holm.retrieval.types (R-I1): validation and native conversion."""

import numpy as np
import pytest

from holm.retrieval.types import LocalSubgraph, SliceParams


def ok(**kw: object) -> SliceParams:
    base: dict[str, object] = dict(
        method="ppr", max_nodes=64, hops=2, delta=16, alpha=0.15, epsilon=1e-4, scan_cap=4096
    )
    base.update(kw)
    return SliceParams(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        {"method": "dfs"},
        {"max_nodes": 0},
        {"delta": -1},
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"epsilon": 0.0},
        {"time_window": (2010, 2000)},
    ],
)
def test_invalid_params_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ok(**bad)


def test_native_kwargs_round_trip() -> None:
    p = ok(relations=(3, 1), class_filter=(7,), time_window=(1990, 2000))
    kw = p.native_kwargs()
    assert kw["method"] == "ppr"
    assert kw["relations"].dtype == np.uint16
    assert kw["relations"].tolist() == [1, 3]
    assert kw["class_filter"].dtype == np.uint32
    assert kw["time_window"] == (1990, 2000)


def test_local_subgraph_from_native_checks_shapes() -> None:
    d = {
        "nodes": np.array([4, 2], dtype=np.uint32),
        "scores": np.array([1.0, 0.5]),
        "edges": np.array([7], dtype=np.uint32),
        "edge_rel": np.array([0], dtype=np.uint16),
        "edge_ptr": np.array([0, 2], dtype=np.uint32),
        "edge_members": np.array([0, 1], dtype=np.uint32),
        "edge_roles": np.array([0, 1], dtype=np.uint8),
        "pushes": 1,
        "work": 1,
        "touched": 2,
        "residual_mass": 0.0,
    }
    s = LocalSubgraph.from_native(d)
    assert s.n_nodes == 2 and s.n_edges == 1
    assert s.stats.work == 1
    bad = dict(d, edge_ptr=np.array([0, 3], dtype=np.uint32))
    with pytest.raises(ValueError, match="edge_ptr"):
        LocalSubgraph.from_native(bad)
    bad = dict(d, edge_members=np.array([0, 5], dtype=np.uint32))
    with pytest.raises(ValueError, match="local id"):
        LocalSubgraph.from_native(bad)
