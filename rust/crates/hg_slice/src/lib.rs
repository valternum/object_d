//! Per-query dynamic slicing of the HOLM hypergraph.
//!
//! [`slice`] selects a bounded local subgraph around seed nodes:
//! * [`Method::Ball`] — BFS by layers, at most `delta` best incidences per
//!   expanded node, up to `hops` and `max_nodes`;
//! * [`Method::Ppr`] — approximate personalized PageRank by local push
//!   (Andersen–Chung–Lang), total work ≤ M / (α·ε) independent of graph size.
//!
//! Filters (relations, node classes, edge time window) apply during traversal.
//! The result lists seeds first, then the selected nodes, and the induced
//! edges in local ids. Deterministic for identical inputs.
#![deny(unsafe_op_in_unsafe_fn)]

mod ball;
mod induced;
pub mod ppr;

use hg_core::attrs::{ClassSet, EdgeIntervals, NodeTypes};
use hg_core::graph::{EdgeId, Hypergraph, NodeId, RelId, RelSet, Role};
use thiserror::Error;

/// Slicing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Bounded BFS ball with per-node fan-out cap.
    Ball,
    /// Approximate personalized PageRank by local push.
    Ppr,
}

/// Parameters of one slice.
#[derive(Debug, Clone)]
pub struct SliceParams {
    /// Method.
    pub method: Method,
    /// Maximum number of nodes in the slice (seeds included).
    pub max_nodes: usize,
    /// Maximum BFS depth (`Ball`).
    pub hops: u32,
    /// Maximum incidences expanded per node (top-Δ by priority).
    pub delta: usize,
    /// Teleport probability (`Ppr`), in (0, 1].
    pub alpha: f64,
    /// Push tolerance (`Ppr`), > 0.
    pub epsilon: f64,
    /// Rows longer than this are not scanned for induced edges.
    pub scan_cap: usize,
    /// Allowed relations.
    pub relations: RelSet,
    /// Non-seed nodes must have a type in this set (requires node types).
    pub class_filter: Option<ClassSet>,
    /// Edges whose interval is disjoint from `[lo, hi]` are skipped.
    pub time_window: Option<(i32, i32)>,
}

/// A seed node with its mass (mass is ignored by `Ball`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seed {
    /// Global node id.
    pub node: NodeId,
    /// Non-negative mass.
    pub mass: f64,
}

/// Graph and optional attributes the slicer reads.
#[derive(Debug, Clone, Copy)]
pub struct GraphRefs<'g> {
    /// The hypergraph.
    pub graph: &'g Hypergraph<'g>,
    /// Node types (needed by `class_filter`).
    pub types: Option<&'g NodeTypes<'g>>,
    /// Edge intervals (needed by `time_window`).
    pub intervals: Option<&'g EdgeIntervals<'g>>,
}

/// Work counters of a slice.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SliceStats {
    /// Nodes expanded (`Ball`) or pushes performed (`Ppr`).
    pub pushes: usize,
    /// Sum of `d(u)` over expansions / pushes.
    pub work: usize,
    /// Distinct nodes touched.
    pub touched: usize,
    /// Residual mass left (`Ppr`).
    pub residual_mass: f64,
}

/// Result: a local subgraph in local ids (`local id` = index into `nodes`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalSubgraph {
    /// Global ids of the selected nodes; seeds first.
    pub nodes: Vec<NodeId>,
    /// `Ball`: `1 / (1 + depth)`; `Ppr`: estimated PageRank mass.
    pub scores: Vec<f64>,
    /// Global ids of the induced edges, sorted.
    pub edges: Vec<EdgeId>,
    /// Relation of every induced edge.
    pub edge_rel: Vec<RelId>,
    /// Member offsets (`edges.len() + 1`).
    pub edge_ptr: Vec<u32>,
    /// Members in local ids.
    pub edge_members: Vec<u32>,
    /// Member roles.
    pub edge_roles: Vec<Role>,
    /// Work counters.
    pub stats: SliceStats,
}

/// Slicing errors.
#[derive(Debug, Error, PartialEq)]
pub enum SliceError {
    /// A seed id is out of range.
    #[error("seed node {0} out of range")]
    InvalidSeed(NodeId),
    /// Inconsistent parameters.
    #[error("invalid parameters: {0}")]
    InvalidParams(String),
}

/// Traversal filters shared by both methods.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Filters<'a, 'g> {
    types: Option<&'g NodeTypes<'g>>,
    classes: Option<&'a ClassSet>,
    intervals: Option<&'g EdgeIntervals<'g>>,
    window: Option<(i32, i32)>,
}

impl<'a, 'g> Filters<'a, 'g> {
    pub(crate) fn new(g: &GraphRefs<'g>, p: &'a SliceParams) -> Self {
        Self {
            types: g.types,
            classes: p.class_filter.as_ref(),
            intervals: g.intervals,
            window: p.time_window,
        }
    }

    /// Edge passes the time window (edges without an interval always pass).
    pub(crate) fn edge_ok(&self, e: EdgeId) -> bool {
        match (self.window, self.intervals) {
            (Some((lo, hi)), Some(iv)) => iv.overlaps(e, lo, hi),
            _ => true,
        }
    }

    /// Node passes the class filter.
    pub(crate) fn node_ok(&self, v: NodeId) -> bool {
        match (self.classes, self.types) {
            (Some(c), Some(t)) => t.any_in(v, c),
            _ => true,
        }
    }
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, SliceError> {
    Err(SliceError::InvalidParams(msg.into()))
}

/// Validates parameters and seeds; returns seeds sorted by node with merged masses.
pub(crate) fn validate(
    g: &GraphRefs<'_>,
    seeds: &[Seed],
    p: &SliceParams,
) -> Result<Vec<Seed>, SliceError> {
    if seeds.is_empty() {
        return invalid("no seeds");
    }
    if p.max_nodes == 0 {
        return invalid("max_nodes must be >= 1");
    }
    if p.class_filter.is_some() && g.types.is_none() {
        return invalid("class_filter needs node types");
    }
    if p.time_window.is_some() && g.intervals.is_none() {
        return invalid("time_window needs edge intervals");
    }
    if let Some((lo, hi)) = p.time_window {
        if lo > hi {
            return invalid("time_window lo > hi");
        }
    }
    let n = g.graph.n_nodes() as u64;
    let mut merged: Vec<Seed> = Vec::with_capacity(seeds.len());
    let mut sorted = seeds.to_vec();
    sorted.sort_by_key(|s| s.node);
    for s in sorted {
        if u64::from(s.node) >= n {
            return Err(SliceError::InvalidSeed(s.node));
        }
        if p.method == Method::Ppr && !(s.mass.is_finite() && s.mass >= 0.0) {
            return invalid(format!("seed {} has invalid mass {}", s.node, s.mass));
        }
        match merged.last_mut() {
            Some(last) if last.node == s.node => last.mass += s.mass,
            _ => merged.push(s),
        }
    }
    if merged.len() > p.max_nodes {
        return invalid(format!(
            "{} seeds exceed max_nodes {}",
            merged.len(),
            p.max_nodes
        ));
    }
    if p.method == Method::Ppr {
        if !(p.alpha > 0.0 && p.alpha <= 1.0) {
            return invalid("alpha must be in (0, 1]");
        }
        if !(p.epsilon > 0.0 && p.epsilon.is_finite()) {
            return invalid("epsilon must be > 0");
        }
        if merged.iter().map(|s| s.mass).sum::<f64>() <= 0.0 {
            return invalid("total seed mass must be > 0");
        }
    }
    Ok(merged)
}

/// Builds the slice of `g` around `seeds`.
pub fn slice(
    g: &GraphRefs<'_>,
    seeds: &[Seed],
    params: &SliceParams,
) -> Result<LocalSubgraph, SliceError> {
    let seeds = validate(g, seeds, params)?;
    let filters = Filters::new(g, params);
    let (nodes, scores, traversed, stats) = match params.method {
        Method::Ball => ball::run(g, &filters, &seeds, params),
        Method::Ppr => {
            let mut push = ppr::Push::new(g, params);
            push.run(&seeds)?;
            let (nodes, scores) = push.select(&seeds);
            let stats = push.stats();
            (nodes, scores, push.traversed().to_vec(), stats)
        }
    };
    let ind = induced::collect(g, &filters, params, &nodes, traversed);
    Ok(LocalSubgraph {
        nodes,
        scores,
        edges: ind.edges,
        edge_rel: ind.rel,
        edge_ptr: ind.ptr,
        edge_members: ind.members,
        edge_roles: ind.roles,
        stats,
    })
}

/// Upper bound `d̂(v) = min(allowed degree, delta)` used by push thresholds.
pub fn capped_degree(g: &Hypergraph<'_>, v: NodeId, rels: &RelSet, delta: usize) -> usize {
    g.allowed_degree(v, rels).min(delta)
}
