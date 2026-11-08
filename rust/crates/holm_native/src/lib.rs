//! PyO3 bindings of the HOLM Rust core (`holm.holm_native`).
//!
//! * `Graph.from_arrays(...)` — in-memory hypergraph (tests, synthetic data);
//! * `Index.open(path)` — memory-mapped `.holmidx` built by `build_index`;
//! * `.slice(seeds, masses, **params)` — per-query slice (`hg_slice`), returned
//!   as a dict of NumPy arrays; the GIL is released while slicing.
#![deny(unsafe_op_in_unsafe_fn)]

use std::path::{Path, PathBuf};

use hg_core::attrs::{ClassSet, EdgeIntervals, NodeTypes};
use hg_core::container::{Container, Verify};
use hg_core::graph::{Hypergraph, HypergraphBuilder, RelSet};
use hg_index::build::{build_index as build, BuildConfig, Predicates};
use hg_index::index::IndexView;
use hg_slice::{slice, GraphRefs, LocalSubgraph, Method, Seed, SliceParams};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn slice_of<'a, T: numpy::Element>(
    a: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("`{name}` must be a contiguous 1-D array")))
}

/// Arguments shared by `Graph.slice` and `Index.slice`.
struct SliceArgs<'py> {
    seeds: PyReadonlyArray1<'py, u32>,
    masses: PyReadonlyArray1<'py, f64>,
    method: String,
    max_nodes: usize,
    hops: u32,
    delta: usize,
    alpha: f64,
    epsilon: f64,
    scan_cap: usize,
    relations: Option<PyReadonlyArray1<'py, u16>>,
    class_filter: Option<PyReadonlyArray1<'py, u32>>,
    time_window: Option<(i32, i32)>,
}

impl SliceArgs<'_> {
    fn build(&self, n_classes: usize) -> PyResult<(Vec<Seed>, SliceParams)> {
        let seeds = slice_of(&self.seeds, "seeds")?;
        let masses = slice_of(&self.masses, "masses")?;
        if seeds.len() != masses.len() {
            return Err(value_err("seeds and masses differ in length"));
        }
        let method = match self.method.as_str() {
            "ball" => Method::Ball,
            "ppr" => Method::Ppr,
            other => return Err(value_err(format!("unknown method `{other}`"))),
        };
        let relations = match &self.relations {
            None => RelSet::all(),
            Some(r) => RelSet::only(slice_of(r, "relations")?),
        };
        let class_filter = match &self.class_filter {
            None => None,
            Some(c) => {
                let mut set = ClassSet::new(n_classes);
                for &id in slice_of(c, "class_filter")? {
                    if id as usize >= n_classes {
                        return Err(value_err(format!("class id {id} out of range {n_classes}")));
                    }
                    set.insert(id);
                }
                Some(set)
            }
        };
        let seeds = seeds
            .iter()
            .zip(masses)
            .map(|(&node, &mass)| Seed { node, mass })
            .collect();
        let params = SliceParams {
            method,
            max_nodes: self.max_nodes,
            hops: self.hops,
            delta: self.delta,
            alpha: self.alpha,
            epsilon: self.epsilon,
            scan_cap: self.scan_cap,
            relations,
            class_filter,
            time_window: self.time_window,
        };
        Ok((seeds, params))
    }
}

fn to_dict<'py>(py: Python<'py>, s: LocalSubgraph) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("nodes", s.nodes.into_pyarray(py))?;
    d.set_item("scores", s.scores.into_pyarray(py))?;
    d.set_item("edges", s.edges.into_pyarray(py))?;
    d.set_item("edge_rel", s.edge_rel.into_pyarray(py))?;
    d.set_item("edge_ptr", s.edge_ptr.into_pyarray(py))?;
    d.set_item("edge_members", s.edge_members.into_pyarray(py))?;
    d.set_item("edge_roles", s.edge_roles.into_pyarray(py))?;
    d.set_item("pushes", s.stats.pushes)?;
    d.set_item("work", s.stats.work)?;
    d.set_item("touched", s.stats.touched)?;
    d.set_item("residual_mass", s.stats.residual_mass)?;
    Ok(d)
}

fn run_slice<'py>(
    py: Python<'py>,
    refs: GraphRefs<'_>,
    args: SliceArgs<'py>,
    n_classes: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let (seeds, params) = args.build(n_classes)?;
    let res = py
        .detach(|| slice(&refs, &seeds, &params))
        .map_err(value_err)?;
    to_dict(py, res)
}

/// In-memory hypergraph with node types and edge intervals.
#[pyclass(name = "Graph", module = "holm.holm_native", frozen)]
struct PyGraph {
    g: Hypergraph<'static>,
    types: NodeTypes<'static>,
    intervals: EdgeIntervals<'static>,
    n_classes: usize,
}

#[pymethods]
impl PyGraph {
    /// Builds a graph from edge arrays (`edge_ptr`: `[n_edges + 1]` member offsets).
    #[staticmethod]
    #[pyo3(signature = (n_nodes, node_prio, edge_rel, edge_ptr, edge_nodes, edge_roles, *,
        type_nodes=None, type_classes=None, n_classes=0, iv_edges=None, iv_start=None, iv_end=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
        n_nodes: u32,
        node_prio: PyReadonlyArray1<'_, u32>,
        edge_rel: PyReadonlyArray1<'_, u16>,
        edge_ptr: PyReadonlyArray1<'_, u64>,
        edge_nodes: PyReadonlyArray1<'_, u32>,
        edge_roles: PyReadonlyArray1<'_, u8>,
        type_nodes: Option<PyReadonlyArray1<'_, u32>>,
        type_classes: Option<PyReadonlyArray1<'_, u32>>,
        n_classes: usize,
        iv_edges: Option<PyReadonlyArray1<'_, u32>>,
        iv_start: Option<PyReadonlyArray1<'_, i32>>,
        iv_end: Option<PyReadonlyArray1<'_, i32>>,
    ) -> PyResult<Self> {
        let (prio, rel, ptr) = (
            slice_of(&node_prio, "node_prio")?,
            slice_of(&edge_rel, "edge_rel")?,
            slice_of(&edge_ptr, "edge_ptr")?,
        );
        let (nodes, roles) = (
            slice_of(&edge_nodes, "edge_nodes")?,
            slice_of(&edge_roles, "edge_roles")?,
        );
        if prio.len() != n_nodes as usize {
            return Err(value_err("node_prio must have n_nodes entries"));
        }
        if ptr.len() != rel.len() + 1
            || ptr[0] != 0
            || ptr[rel.len()] as usize != nodes.len()
            || roles.len() != nodes.len()
        {
            return Err(value_err(
                "edge_ptr / edge_nodes / edge_roles are inconsistent",
            ));
        }
        let mut b = HypergraphBuilder::new(n_nodes);
        for (v, &p) in prio.iter().enumerate() {
            b.set_node_prio(v as u32, p).map_err(value_err)?;
        }
        let mut members = Vec::new();
        for (e, &r) in rel.iter().enumerate() {
            let (a, z) = (ptr[e] as usize, ptr[e + 1] as usize);
            if a > z {
                return Err(value_err("edge_ptr must be non-decreasing"));
            }
            members.clear();
            members.extend(nodes[a..z].iter().copied().zip(roles[a..z].iter().copied()));
            b.add_edge(r, &members).map_err(value_err)?;
        }
        let g = b.build();
        let pairs: Vec<(u32, u32)> = match (&type_nodes, &type_classes) {
            (Some(tn), Some(tc)) => {
                let (tn, tc) = (slice_of(tn, "type_nodes")?, slice_of(tc, "type_classes")?);
                if tn.len() != tc.len() {
                    return Err(value_err("type_nodes and type_classes differ in length"));
                }
                if let Some(&c) = tc.iter().find(|&&c| c as usize >= n_classes) {
                    return Err(value_err(format!(
                        "type class {c} >= n_classes {n_classes}"
                    )));
                }
                tn.iter().copied().zip(tc.iter().copied()).collect()
            }
            (None, None) => Vec::new(),
            _ => return Err(value_err("give both type_nodes and type_classes")),
        };
        let types = NodeTypes::from_pairs(n_nodes as usize, pairs).map_err(value_err)?;
        let triples: Vec<(u32, i32, i32)> = match (&iv_edges, &iv_start, &iv_end) {
            (Some(e), Some(s), Some(t)) => {
                let (e, s, t) = (
                    slice_of(e, "iv_edges")?,
                    slice_of(s, "iv_start")?,
                    slice_of(t, "iv_end")?,
                );
                if e.len() != s.len() || e.len() != t.len() {
                    return Err(value_err("interval arrays differ in length"));
                }
                if let Some(&bad) = e.iter().find(|&&x| x as usize >= g.n_edges()) {
                    return Err(value_err(format!("interval edge {bad} out of range")));
                }
                e.iter()
                    .zip(s)
                    .zip(t)
                    .map(|((&e, &s), &t)| (e, s, t))
                    .collect()
            }
            (None, None, None) => Vec::new(),
            _ => return Err(value_err("give all of iv_edges, iv_start, iv_end")),
        };
        Ok(Self {
            g,
            types,
            intervals: EdgeIntervals::from_triples(triples),
            n_classes,
        })
    }

    /// Number of nodes.
    #[getter]
    fn n_nodes(&self) -> usize {
        self.g.n_nodes()
    }

    /// Number of hyperedges.
    #[getter]
    fn n_edges(&self) -> usize {
        self.g.n_edges()
    }

    /// Per-query slice; see `holm.retrieval.types.SliceParams`.
    #[pyo3(signature = (seeds, masses, *, method, max_nodes, hops, delta, alpha, epsilon, scan_cap,
        relations=None, class_filter=None, time_window=None))]
    #[allow(clippy::too_many_arguments)]
    fn slice<'py>(
        &self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'py, u32>,
        masses: PyReadonlyArray1<'py, f64>,
        method: String,
        max_nodes: usize,
        hops: u32,
        delta: usize,
        alpha: f64,
        epsilon: f64,
        scan_cap: usize,
        relations: Option<PyReadonlyArray1<'py, u16>>,
        class_filter: Option<PyReadonlyArray1<'py, u32>>,
        time_window: Option<(i32, i32)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let args = SliceArgs {
            seeds,
            masses,
            method,
            max_nodes,
            hops,
            delta,
            alpha,
            epsilon,
            scan_cap,
            relations,
            class_filter,
            time_window,
        };
        let refs = GraphRefs {
            graph: &self.g,
            types: Some(&self.types),
            intervals: Some(&self.intervals),
        };
        run_slice(py, refs, args, self.n_classes)
    }
}

/// Memory-mapped index built by `build_index`.
#[pyclass(name = "Index", module = "holm.holm_native", frozen)]
struct PyIndex {
    c: Container,
}

impl PyIndex {
    fn view(&self) -> PyResult<IndexView<'_>> {
        IndexView::new(&self.c).map_err(value_err)
    }
}

#[pymethods]
impl PyIndex {
    /// Opens an index; `verify=True` recomputes every section checksum.
    #[staticmethod]
    #[pyo3(signature = (path, verify=false))]
    fn open(path: PathBuf, verify: bool) -> PyResult<Self> {
        let c = Container::open(&path, if verify { Verify::Full } else { Verify::Header })
            .map_err(value_err)?;
        IndexView::new(&c).map_err(value_err)?;
        Ok(Self { c })
    }

    /// Number of nodes.
    #[getter]
    fn n_nodes(&self) -> PyResult<usize> {
        Ok(self.view()?.graph.n_nodes())
    }

    /// Number of hyperedges.
    #[getter]
    fn n_edges(&self) -> PyResult<usize> {
        Ok(self.view()?.graph.n_edges())
    }

    /// Number of relations.
    #[getter]
    fn n_relations(&self) -> PyResult<usize> {
        Ok(self.view()?.n_relations())
    }

    /// Number of classes.
    #[getter]
    fn n_classes(&self) -> PyResult<usize> {
        Ok(self.view()?.n_classes())
    }

    /// Degree (number of incidences) of every node, `[n_nodes]` uint32.
    fn degrees<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let ix = self.view()?;
        let g = &ix.graph;
        let d: Vec<u32> = (0..g.n_nodes() as u32).map(|v| g.degree(v) as u32).collect();
        Ok(d.into_pyarray(py))
    }

    /// Node id of an entity term, or None.
    fn node_id(&self, term: &str) -> PyResult<Option<u32>> {
        Ok(self.view()?.node_id(term))
    }

    /// Entity term of a node.
    fn node_term(&self, v: u32) -> PyResult<String> {
        let ix = self.view()?;
        if v as usize >= ix.graph.n_nodes() {
            return Err(value_err(format!("node {v} out of range")));
        }
        Ok(ix.node_term(v).to_owned())
    }

    /// Relation id by name, or None.
    fn relation_id(&self, name: &str) -> PyResult<Option<u16>> {
        Ok(self.view()?.relation_id(name))
    }

    /// Relation name.
    fn relation_name(&self, r: u16) -> PyResult<String> {
        let ix = self.view()?;
        if usize::from(r) >= ix.n_relations() {
            return Err(value_err(format!("relation {r} out of range")));
        }
        Ok(ix.relation_name(r).to_owned())
    }

    /// Class id by name, or None.
    fn class_id(&self, name: &str) -> PyResult<Option<u32>> {
        Ok(self.view()?.class_id(name))
    }

    /// Class name.
    fn class_name(&self, c: u32) -> PyResult<String> {
        let ix = self.view()?;
        if c as usize >= ix.n_classes() {
            return Err(value_err(format!("class {c} out of range")));
        }
        Ok(ix.class_name(c).to_owned())
    }

    /// Roots and all their subclasses, as sorted class ids.
    fn class_closure<'py>(
        &self,
        py: Python<'py>,
        roots: Vec<u32>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let ix = self.view()?;
        if let Some(&bad) = roots.iter().find(|&&c| c as usize >= ix.n_classes()) {
            return Err(value_err(format!("class {bad} out of range")));
        }
        Ok(ix.class_closure(&roots).ids().into_pyarray(py))
    }

    /// Per-query slice; see `holm.retrieval.types.SliceParams`.
    #[pyo3(signature = (seeds, masses, *, method, max_nodes, hops, delta, alpha, epsilon, scan_cap,
        relations=None, class_filter=None, time_window=None))]
    #[allow(clippy::too_many_arguments)]
    fn slice<'py>(
        &self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'py, u32>,
        masses: PyReadonlyArray1<'py, f64>,
        method: String,
        max_nodes: usize,
        hops: u32,
        delta: usize,
        alpha: f64,
        epsilon: f64,
        scan_cap: usize,
        relations: Option<PyReadonlyArray1<'py, u16>>,
        class_filter: Option<PyReadonlyArray1<'py, u32>>,
        time_window: Option<(i32, i32)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ix = self.view()?;
        let args = SliceArgs {
            seeds,
            masses,
            method,
            max_nodes,
            hops,
            delta,
            alpha,
            epsilon,
            scan_cap,
            relations,
            class_filter,
            time_window,
        };
        let refs = GraphRefs {
            graph: &ix.graph,
            types: Some(&ix.types),
            intervals: Some(&ix.intervals),
        };
        run_slice(py, refs, args, ix.n_classes())
    }
}

/// Builds a `.holmidx` index from YAGO 4.6 Turtle files; returns build counts.
#[pyfunction]
#[pyo3(signature = (*, out, schema=None, taxonomy=Vec::new(), facts=Vec::new(), meta=Vec::new(),
    tmp_dir=None, chunk_bytes=256 << 20))]
#[allow(clippy::too_many_arguments)]
fn build_index<'py>(
    py: Python<'py>,
    out: PathBuf,
    schema: Option<PathBuf>,
    taxonomy: Vec<PathBuf>,
    facts: Vec<PathBuf>,
    meta: Vec<PathBuf>,
    tmp_dir: Option<PathBuf>,
    chunk_bytes: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = BuildConfig {
        schema,
        taxonomy,
        facts,
        meta,
        tmp_dir: tmp_dir.unwrap_or_else(|| {
            Path::new(&out)
                .parent()
                .map_or_else(std::env::temp_dir, Path::to_path_buf)
        }),
        out,
        chunk_bytes,
        predicates: Predicates::yago46(),
    };
    let s = py.detach(|| build(&cfg)).map_err(value_err)?;
    let d = PyDict::new(py);
    d.set_item("nodes", s.nodes)?;
    d.set_item("edges", s.edges)?;
    d.set_item("relations", s.relations)?;
    d.set_item("classes", s.classes)?;
    d.set_item("type_pairs", s.type_pairs)?;
    d.set_item("typed_nodes", s.typed_nodes)?;
    d.set_item("intervals", s.intervals)?;
    d.set_item("duplicate_edges", s.duplicate_edges)?;
    d.set_item("qids", s.qids)?;
    d.set_item("ignored_meta", s.ignored_meta)?;
    Ok(d)
}

/// Module `holm.holm_native`.
#[pymodule]
fn holm_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyIndex>()?;
    m.add_function(wrap_pyfunction!(build_index, m)?)?;
    Ok(())
}
