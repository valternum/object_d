"""Contract: Rust slicer (holm.holm_native) == Python oracle (holm.reference.slice).

TESTING §5: hypothesis graphs <= 30 nodes, <= 40 edges, arity 2-5. Ball must match
exactly; PPR must select the same nodes in the same order with |diff p| <= 1e-12.
"""

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from holm.reference.slice import RawGraph, slice_ref
from holm.retrieval.types import LocalSubgraph, SliceParams

native = pytest.importorskip("holm.holm_native", reason="build with `make build-native`")
pytestmark = pytest.mark.contract

ARRAYS = ("nodes", "edges", "edge_rel", "edge_ptr", "edge_members", "edge_roles")


@st.composite
def graphs(draw: st.DrawFn) -> RawGraph:
    n = draw(st.integers(2, 30))
    prio = draw(st.lists(st.integers(0, 5), min_size=n, max_size=n))
    member = st.tuples(st.integers(0, n - 1), st.integers(0, 2))
    edges = draw(
        st.lists(st.tuples(st.integers(0, 3), st.lists(member, min_size=2, max_size=5)), max_size=40)
    )
    types = draw(st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, 3)), max_size=2 * n))
    intervals = draw(
        st.lists(
            st.tuples(
                st.integers(0, max(len(edges) - 1, 0)), st.integers(1900, 2000), st.integers(2000, 2100)
            ),
            max_size=len(edges),
        )
        if edges
        else st.just([])
    )
    return RawGraph.from_edges(n, prio, edges, types=types, intervals=intervals, n_classes=4)


@st.composite
def params(draw: st.DrawFn) -> SliceParams:
    window = draw(st.none() | st.tuples(st.integers(1900, 2050), st.integers(0, 80)))
    return SliceParams(
        method=draw(st.sampled_from(["ball", "ppr"])),
        max_nodes=draw(st.integers(1, 40)),
        hops=draw(st.integers(0, 4)),
        delta=draw(st.integers(0, 6)),
        alpha=draw(st.floats(0.05, 0.95)),
        epsilon=draw(st.sampled_from([1e-2, 1e-3, 1e-5])),
        scan_cap=draw(st.integers(0, 8)),
        relations=draw(st.none() | st.lists(st.integers(0, 3), max_size=3).map(tuple)),
        class_filter=draw(st.none() | st.lists(st.integers(0, 3), max_size=3).map(tuple)),
        time_window=None if window is None else (window[0], window[0] + window[1]),
    )


def native_graph(g: RawGraph) -> object:
    return native.Graph.from_arrays(
        g.n_nodes,
        np.asarray(g.prio, dtype=np.uint32),
        np.asarray(g.edge_rel, dtype=np.uint16),
        np.asarray(g.edge_ptr, dtype=np.uint64),
        np.asarray(g.edge_nodes, dtype=np.uint32),
        np.asarray(g.edge_roles, dtype=np.uint8),
        type_nodes=np.asarray([v for v, _ in g.types], dtype=np.uint32),
        type_classes=np.asarray([c for _, c in g.types], dtype=np.uint32),
        n_classes=g.n_classes,
        iv_edges=np.asarray([e for e, _, _ in g.intervals], dtype=np.uint32),
        iv_start=np.asarray([s for _, s, _ in g.intervals], dtype=np.int32),
        iv_end=np.asarray([t for _, _, t in g.intervals], dtype=np.int32),
    )


def run_native(ng: object, seeds: list[int], masses: list[float], p: SliceParams) -> LocalSubgraph:
    d = ng.slice(  # type: ignore[attr-defined]
        np.asarray(seeds, dtype=np.uint32), np.asarray(masses, dtype=np.float64), **p.native_kwargs()
    )
    return LocalSubgraph.from_native(d)


def assert_same(a: LocalSubgraph, b: LocalSubgraph, method: str) -> None:
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    if method == "ball":
        np.testing.assert_array_equal(a.scores, b.scores)
    else:
        np.testing.assert_allclose(a.scores, b.scores, rtol=0, atol=1e-12)
    assert a.stats.work == b.stats.work and a.stats.pushes == b.stats.pushes


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(g=graphs(), p=params(), data=st.data())
def test_rust_slice_equals_oracle(g: RawGraph, p: SliceParams, data: st.DataObject) -> None:
    k = data.draw(st.integers(1, min(3, p.max_nodes, g.n_nodes)))
    seeds = data.draw(st.lists(st.integers(0, g.n_nodes - 1), min_size=k, max_size=k, unique=True))
    masses = data.draw(st.lists(st.floats(0.1, 2.0), min_size=k, max_size=k))
    want = slice_ref(g, seeds, masses, p)
    got = run_native(native_graph(g), seeds, masses, p)
    assert_same(got, want, p.method)


def test_invalid_seed_is_value_error() -> None:
    g = RawGraph.from_edges(3, [0, 0, 0], [(0, [(0, 0), (1, 1)])])
    p = SliceParams(method="ball", max_nodes=4, hops=1, delta=4, alpha=0.15, epsilon=1e-4, scan_cap=8)
    with pytest.raises(ValueError, match="out of range"):
        run_native(native_graph(g), [7], [1.0], p)


def test_index_built_from_fixture(tmp_path: Path) -> None:
    fx = Path(__file__).resolve().parents[2] / "rust/crates/hg_index/tests/fixtures"
    out = tmp_path / "mini.holmidx"
    stats = native.build_index(
        schema=str(fx / "schema_sample.ttl"),
        taxonomy=[str(fx / "mini/taxonomy.ttl")],
        facts=[str(fx / "mini/facts.ttl")],
        meta=[str(fx / "mini/meta.ttl")],
        out=str(out),
        tmp_dir=str(tmp_path),
        chunk_bytes=64,
    )
    assert stats["nodes"] == 6 and stats["edges"] == 4
    ix = native.Index.open(str(out), verify=True)
    france = ix.node_id("yago:France")
    assert france == 0
    assert ix.node_term(france) == "yago:France"
    assert ix.node_id("nope") is None
    degrees = ix.degrees()
    assert degrees.dtype == np.uint32 and degrees.shape == (ix.n_nodes,)
    # France: capital -> Paris, Lyon location France, Paris location France.
    assert int(degrees[france]) == 3
    assert int(degrees[ix.node_id("yago:Bahraini_dinar")]) == 0
    place = ix.class_id("schema:Place")
    closure = ix.class_closure([place])
    assert sorted(ix.class_name(int(c)) for c in closure) == [
        "schema:AdministrativeArea",
        "schema:City",
        "schema:Place",
        "yago:Capital_city",
    ]
    loc = ix.relation_id("schema:location")
    p = SliceParams(
        method="ball",
        max_nodes=10,
        hops=1,
        delta=10,
        alpha=0.15,
        epsilon=1e-4,
        scan_cap=100,
        relations=(loc,),
    )
    s = LocalSubgraph.from_native(
        ix.slice(np.array([france], dtype=np.uint32), np.array([1.0]), **p.native_kwargs())
    )
    names = sorted(ix.node_term(int(v)) for v in s.nodes)
    assert names == ["yago:France", "yago:Lyon", "yago:Paris"]
    lyon_paris = p.native_kwargs() | {"time_window": (2003, 2010)}
    s2 = LocalSubgraph.from_native(
        ix.slice(np.array([france], dtype=np.uint32), np.array([1.0]), **lyon_paris)
    )
    assert sorted(ix.node_term(int(v)) for v in s2.nodes) == ["yago:France", "yago:Paris"]
