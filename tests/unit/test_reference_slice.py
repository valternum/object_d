"""NumPy/Python oracle of the slicer (holm.reference.slice) on hand-checked cases."""

import numpy as np
import pytest

from holm.reference.slice import RawGraph, slice_ref
from holm.retrieval.types import SliceParams


def params(method: str, **kw: object) -> SliceParams:
    base: dict[str, object] = dict(
        method=method,
        max_nodes=1 << 30,
        hops=2,
        delta=1 << 30,
        alpha=0.15,
        epsilon=1e-6,
        scan_cap=1 << 30,
    )
    base.update(kw)
    return SliceParams(**base)  # type: ignore[arg-type]


def path5() -> RawGraph:
    return RawGraph.from_edges(5, [0] * 5, [(0, [(i, 0), (i + 1, 1)]) for i in range(4)])


def test_ball_on_a_path() -> None:
    s = slice_ref(path5(), [0], [1.0], params("ball"))
    assert s.nodes.tolist() == [0, 1, 2]
    np.testing.assert_array_equal(s.scores, [1.0, 0.5, 1.0 / 3.0])
    assert s.edges.tolist() == [0, 1]
    assert s.edge_members.tolist() == [0, 1, 1, 2]
    assert s.edge_ptr.tolist() == [0, 2, 4]


def test_hub_fan_out_is_capped_by_priority() -> None:
    g = RawGraph.from_edges(101, list(range(101)), [(0, [(0, 0), (leaf, 1)]) for leaf in range(1, 101)])
    s = slice_ref(g, [0], [1.0], params("ball", delta=5, hops=1))
    assert s.nodes.tolist() == [0, 100, 99, 98, 97, 96]
    assert s.stats.work == 5


def test_ppr_two_node_closed_form() -> None:
    g = RawGraph.from_edges(2, [0, 0], [(0, [(0, 0), (1, 1)])])
    s = slice_ref(g, [0], [1.0], params("ppr", alpha=0.5, epsilon=1e-10))
    assert s.nodes.tolist() == [0, 1]
    assert s.scores[0] == pytest.approx(2.0 / 3.0, abs=1e-8)
    assert s.scores[1] == pytest.approx(1.0 / 3.0, abs=1e-8)


def test_ppr_isolated_seed_absorbs_its_mass() -> None:
    g = RawGraph.from_edges(3, [0, 0, 0], [(0, [(1, 0), (2, 1)])])
    s = slice_ref(g, [0], [2.0], params("ppr"))
    assert s.nodes.tolist() == [0]
    assert s.scores.tolist() == [2.0]


def test_filters_on_a_typed_star() -> None:
    # centre 0, leaves 1..4 over relations 0,0,1,1; leaf classes 0,1,0,1; edge 0 valid 1951..2002
    g = RawGraph.from_edges(
        5,
        [0] * 5,
        [(0, [(0, 0), (1, 1)]), (0, [(0, 0), (2, 1)]), (1, [(0, 0), (3, 1)]), (1, [(0, 0), (4, 1)])],
        types=[(1, 0), (2, 1), (3, 0), (4, 1)],
        intervals=[(0, 1951, 2002), (2, 2010, 2**31 - 1)],
    )
    for method in ("ball", "ppr"):
        rel = slice_ref(g, [0], [1.0], params(method, relations=(1,)))
        assert sorted(rel.nodes.tolist()) == [0, 3, 4]
        cls = slice_ref(g, [0], [1.0], params(method, class_filter=(1,)))
        assert sorted(cls.nodes.tolist()) == [0, 2, 4]
        win = slice_ref(g, [0], [1.0], params(method, time_window=(2003, 2005)))
        assert sorted(win.nodes.tolist()) == [0, 2, 4]


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="out of range"):
        slice_ref(path5(), [9], [1.0], params("ball"))
    with pytest.raises(ValueError, match="exceed max_nodes"):
        slice_ref(path5(), [0, 1], [1.0, 1.0], params("ball", max_nodes=1))
