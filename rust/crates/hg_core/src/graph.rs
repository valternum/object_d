//! Typed incidence hypergraph in CSR form.
//!
//! Two CSR directions are stored: node → incident edges (with the node's role)
//! and edge → members (with roles). Each node row is sorted by
//! `(rel asc, prio(e|v) desc, edge asc, role asc)`, where `prio(e|v)` is the
//! maximum priority of the *other* members of `e`. A per-node run directory
//! `(rel, start)` delimits the blocks of equal relation, so that selecting the
//! best `Δ` incidences over a set of relations is a k-way merge of the allowed
//! runs: O((runs + Δ) · log runs), independent of the node degree (hubs).
//!
//! Space (n nodes, m edges, I incidences, R runs): offsets in Elias–Fano,
//! `5·I` bytes for incidences, `5·I` bytes for members, `6·R` bytes for runs,
//! `2·m + 4·n` bytes for relations and priorities.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ops::Range;

use crate::ef::EliasFano;
use crate::error::{CoreError, Result};

/// Node identifier.
pub type NodeId = u32;
/// Hyperedge identifier.
pub type EdgeId = u32;
/// Relation type identifier.
pub type RelId = u16;
/// Role of a member inside a hyperedge.
pub type Role = u8;

/// Set of allowed relation types.
#[derive(Debug, Clone)]
pub struct RelSet {
    all: bool,
    words: Vec<u64>,
}

impl RelSet {
    /// Every relation allowed.
    pub fn all() -> Self {
        Self {
            all: true,
            words: Vec::new(),
        }
    }

    /// Only the listed relations allowed.
    pub fn only(ids: &[RelId]) -> Self {
        let max = ids.iter().copied().max().map_or(0, usize::from);
        let mut words = vec![0u64; max / 64 + 1];
        for &r in ids {
            words[usize::from(r) / 64] |= 1u64 << (r % 64);
        }
        Self { all: false, words }
    }

    /// Membership test.
    pub fn contains(&self, rel: RelId) -> bool {
        if self.all {
            return true;
        }
        let w = usize::from(rel) / 64;
        w < self.words.len() && (self.words[w] >> (rel % 64)) & 1 == 1
    }
}

/// One incidence of a node in a hyperedge, as returned by [`Hypergraph::top_incident`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Incidence {
    /// Hyperedge.
    pub edge: EdgeId,
    /// Role of the node in the edge.
    pub role: Role,
    /// Relation type of the edge.
    pub rel: RelId,
    /// `prio(e|v)`: maximum priority of the other members (0 if none).
    pub prio: u32,
}

/// Builder of a [`Hypergraph`].
#[derive(Debug, Clone)]
pub struct HypergraphBuilder {
    n_nodes: u32,
    node_prio: Vec<u32>,
    edge_rel: Vec<RelId>,
    edge_ptr: Vec<u64>,
    members: Vec<(Role, NodeId)>,
}

fn prio_excluding(members: &[(Role, NodeId)], prio: &[u32], v: NodeId) -> u32 {
    members
        .iter()
        .filter(|(_, w)| *w != v)
        .map(|(_, w)| prio[*w as usize])
        .max()
        .unwrap_or(0)
}

impl HypergraphBuilder {
    /// Builder for `n_nodes` nodes with priority 0.
    pub fn new(n_nodes: u32) -> Self {
        Self {
            n_nodes,
            node_prio: vec![0; n_nodes as usize],
            edge_rel: Vec::new(),
            edge_ptr: vec![0],
            members: Vec::new(),
        }
    }

    /// Sets the priority of node `v`.
    pub fn set_node_prio(&mut self, v: NodeId, prio: u32) -> Result<()> {
        if v >= self.n_nodes {
            return Err(CoreError::InvalidGraph(format!(
                "node {v} >= n_nodes {}",
                self.n_nodes
            )));
        }
        self.node_prio[v as usize] = prio;
        Ok(())
    }

    /// Adds a hyperedge of relation `rel`; members are `(node, role)` pairs.
    pub fn add_edge(&mut self, rel: RelId, members: &[(NodeId, Role)]) -> Result<EdgeId> {
        if members.is_empty() {
            return Err(CoreError::InvalidGraph("hyperedge without members".into()));
        }
        if let Some(&(v, _)) = members.iter().find(|(v, _)| *v >= self.n_nodes) {
            return Err(CoreError::InvalidGraph(format!(
                "member {v} >= n_nodes {}",
                self.n_nodes
            )));
        }
        let id = u32::try_from(self.edge_rel.len())
            .map_err(|_| CoreError::InvalidGraph("more than u32::MAX hyperedges".into()))?;
        let start = self.members.len();
        self.members.extend(members.iter().map(|&(v, r)| (r, v)));
        self.members[start..].sort_unstable();
        self.edge_rel.push(rel);
        self.edge_ptr.push(self.members.len() as u64);
        Ok(id)
    }

    /// Finalizes the CSR arrays.
    pub fn build(self) -> Hypergraph<'static> {
        let n = self.n_nodes as usize;
        let total = self.members.len();
        assert!(total <= u32::MAX as usize, "more than u32::MAX incidences");

        // Counting sort of incidences by node.
        let mut node_ptr = vec![0u64; n + 1];
        for &(_, v) in &self.members {
            node_ptr[v as usize + 1] += 1;
        }
        for v in 0..n {
            node_ptr[v + 1] += node_ptr[v];
        }
        let mut fill: Vec<u64> = node_ptr[..n].to_vec();
        let mut row_key: Vec<(RelId, u32, EdgeId, Role)> = vec![(0, 0, 0, 0); total];
        for e in 0..self.edge_rel.len() {
            let (a, b) = (self.edge_ptr[e] as usize, self.edge_ptr[e + 1] as usize);
            let members = &self.members[a..b];
            for &(role, v) in members {
                let prio = prio_excluding(members, &self.node_prio, v);
                let slot = &mut fill[v as usize];
                row_key[*slot as usize] = (self.edge_rel[e], prio, e as EdgeId, role);
                *slot += 1;
            }
        }

        let mut inc_edge = Vec::with_capacity(total);
        let mut inc_role = Vec::with_capacity(total);
        let mut run_ptr = Vec::with_capacity(n + 1);
        let mut run_rel = Vec::new();
        let mut run_start = Vec::new();
        run_ptr.push(0u64);
        for v in 0..n {
            let (a, b) = (node_ptr[v] as usize, node_ptr[v + 1] as usize);
            let row = &mut row_key[a..b];
            row.sort_unstable_by(|x, y| {
                x.0.cmp(&y.0)
                    .then(y.1.cmp(&x.1))
                    .then(x.2.cmp(&y.2))
                    .then(x.3.cmp(&y.3))
            });
            let mut prev: Option<RelId> = None;
            for (k, &(rel, _, e, role)) in row.iter().enumerate() {
                if prev != Some(rel) {
                    run_rel.push(rel);
                    run_start.push((a + k) as u32);
                    prev = Some(rel);
                }
                inc_edge.push(e);
                inc_role.push(role);
            }
            run_ptr.push(run_rel.len() as u64);
        }

        let (edge_role, edge_node): (Vec<Role>, Vec<NodeId>) = self.members.into_iter().unzip();
        let parts = HypergraphParts {
            node_prio: Cow::Owned(self.node_prio),
            node_ptr: EliasFano::from_sorted(&node_ptr).expect("prefix sums are monotone"),
            inc_edge: Cow::Owned(inc_edge),
            inc_role: Cow::Owned(inc_role),
            run_ptr: EliasFano::from_sorted(&run_ptr).expect("prefix sums are monotone"),
            run_rel: Cow::Owned(run_rel),
            run_start: Cow::Owned(run_start),
            edge_ptr: EliasFano::from_sorted(&self.edge_ptr).expect("prefix sums are monotone"),
            edge_node: Cow::Owned(edge_node),
            edge_role: Cow::Owned(edge_role),
            edge_rel: Cow::Owned(self.edge_rel),
        };
        Hypergraph::from_parts(parts).expect("builder produces consistent parts")
    }
}

/// Raw arrays of a hypergraph, used for serialization.
#[derive(Debug, Clone)]
pub struct HypergraphParts<'a> {
    /// Node priorities, `[n_nodes]`.
    pub node_prio: Cow<'a, [u32]>,
    /// Row offsets into the incidence arrays, `n_nodes + 1` values.
    pub node_ptr: EliasFano<'a>,
    /// Incident edge of every incidence, row-major.
    pub inc_edge: Cow<'a, [u32]>,
    /// Role of every incidence.
    pub inc_role: Cow<'a, [u8]>,
    /// Run offsets per node, `n_nodes + 1` values over the run arrays.
    pub run_ptr: EliasFano<'a>,
    /// Relation of every run.
    pub run_rel: Cow<'a, [u16]>,
    /// Absolute start (into the incidence arrays) of every run.
    pub run_start: Cow<'a, [u32]>,
    /// Member offsets per edge, `n_edges + 1` values.
    pub edge_ptr: EliasFano<'a>,
    /// Member nodes, edge-major, sorted by `(role, node)` inside an edge.
    pub edge_node: Cow<'a, [u32]>,
    /// Member roles.
    pub edge_role: Cow<'a, [u8]>,
    /// Relation of every edge.
    pub edge_rel: Cow<'a, [u16]>,
}

/// Immutable typed incidence hypergraph.
#[derive(Debug, Clone)]
pub struct Hypergraph<'a> {
    p: HypergraphParts<'a>,
}

/// Head of one relation run during the k-way merge.
#[derive(Debug, PartialEq, Eq)]
struct Head {
    prio: u32,
    edge: EdgeId,
    role: Role,
    rel: RelId,
    cursor: usize,
    end: usize,
}

impl Ord for Head {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: higher prio first, then smaller edge id, then smaller role.
        self.prio
            .cmp(&other.prio)
            .then(other.edge.cmp(&self.edge))
            .then(other.role.cmp(&self.role))
    }
}

impl PartialOrd for Head {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn check(cond: bool, msg: impl FnOnce() -> String) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(CoreError::InvalidGraph(msg()))
    }
}

impl<'a> Hypergraph<'a> {
    /// Validates array lengths and offset ends (O(1) checks) and wraps raw parts.
    pub fn from_parts(parts: HypergraphParts<'a>) -> Result<Self> {
        let n = parts.node_prio.len();
        let m = parts.edge_rel.len();
        check(parts.node_ptr.len() == n + 1, || "node_ptr length".into())?;
        check(parts.run_ptr.len() == n + 1, || "run_ptr length".into())?;
        check(parts.edge_ptr.len() == m + 1, || "edge_ptr length".into())?;
        let incidences = parts.node_ptr.get(n) as usize;
        check(parts.node_ptr.get(0) == 0, || {
            "node_ptr must start at 0".into()
        })?;
        check(
            parts.inc_edge.len() == incidences && parts.inc_role.len() == incidences,
            || format!("incidence arrays must have {incidences} entries"),
        )?;
        let runs = parts.run_ptr.get(n) as usize;
        check(
            parts.run_rel.len() == runs && parts.run_start.len() == runs,
            || format!("run arrays must have {runs} entries"),
        )?;
        let members = parts.edge_ptr.get(m) as usize;
        check(
            parts.edge_node.len() == members && parts.edge_role.len() == members,
            || format!("member arrays must have {members} entries"),
        )?;
        check(members == incidences, || {
            "members and incidences differ".into()
        })?;
        Ok(Self { p: parts })
    }

    /// Raw parts (for serialization).
    pub fn parts(&self) -> &HypergraphParts<'a> {
        &self.p
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.p.node_prio.len()
    }

    /// Number of hyperedges.
    pub fn n_edges(&self) -> usize {
        self.p.edge_rel.len()
    }

    /// Number of incidences (= sum of degrees = sum of edge sizes).
    pub fn n_incidences(&self) -> usize {
        self.p.inc_edge.len()
    }

    /// Priority of node `v`.
    pub fn node_prio(&self, v: NodeId) -> u32 {
        self.p.node_prio[v as usize]
    }

    fn row(&self, v: NodeId) -> Range<usize> {
        let (a, b) = self.p.node_ptr.range(v as usize);
        a as usize..b as usize
    }

    /// Number of incidences of node `v`.
    pub fn degree(&self, v: NodeId) -> usize {
        self.row(v).len()
    }

    /// Incidences of `v` in row order `(rel asc, prio desc, edge asc, role asc)`.
    pub fn incidences(&self, v: NodeId) -> impl Iterator<Item = (EdgeId, Role)> + '_ {
        let r = self.row(v);
        self.p.inc_edge[r.clone()]
            .iter()
            .copied()
            .zip(self.p.inc_role[r].iter().copied())
    }

    /// Relation runs of `v`: `(rel, range into the incidence arrays)`.
    pub fn runs(&self, v: NodeId) -> Vec<(RelId, Range<usize>)> {
        let row_end = self.row(v).end;
        let (a, b) = self.p.run_ptr.range(v as usize);
        let (a, b) = (a as usize, b as usize);
        (a..b)
            .map(|k| {
                let start = self.p.run_start[k] as usize;
                let end = if k + 1 < b {
                    self.p.run_start[k + 1] as usize
                } else {
                    row_end
                };
                (self.p.run_rel[k], start..end)
            })
            .collect()
    }

    /// Relation of edge `e`.
    pub fn edge_rel(&self, e: EdgeId) -> RelId {
        self.p.edge_rel[e as usize]
    }

    /// Members `(nodes, roles)` of edge `e`, sorted by `(role, node)`.
    pub fn edge_members(&self, e: EdgeId) -> (&[NodeId], &[Role]) {
        let (a, b) = self.p.edge_ptr.range(e as usize);
        let r = a as usize..b as usize;
        (&self.p.edge_node[r.clone()], &self.p.edge_role[r])
    }

    /// `prio(e|v)`: max priority over members of `e` that are not `v` (0 if none).
    pub fn incidence_prio(&self, v: NodeId, e: EdgeId) -> u32 {
        let (nodes, _) = self.edge_members(e);
        nodes
            .iter()
            .filter(|&&w| w != v)
            .map(|&w| self.p.node_prio[w as usize])
            .max()
            .unwrap_or(0)
    }

    /// Number of incidences of `v` whose relation is in `rels`; O(runs).
    pub fn allowed_degree(&self, v: NodeId, rels: &RelSet) -> usize {
        self.runs(v)
            .into_iter()
            .filter(|(rel, _)| rels.contains(*rel))
            .map(|(_, r)| r.len())
            .sum()
    }

    fn head(&self, v: NodeId, rel: RelId, cursor: usize, end: usize) -> Head {
        let edge = self.p.inc_edge[cursor];
        Head {
            prio: self.incidence_prio(v, edge),
            edge,
            role: self.p.inc_role[cursor],
            rel,
            cursor,
            end,
        }
    }

    /// The `delta` best incidences of `v` among relations in `rels` that also
    /// satisfy `keep`, ordered by `(prio desc, edge asc, role asc)`.
    /// Cost O((runs + delta + rejected) · log runs), independent of the degree.
    pub fn top_incident_filtered<F>(
        &self,
        v: NodeId,
        rels: &RelSet,
        delta: usize,
        mut keep: F,
        out: &mut Vec<Incidence>,
    ) where
        F: FnMut(&Incidence) -> bool,
    {
        if delta == 0 {
            return;
        }
        let mut heap: BinaryHeap<Head> = self
            .runs(v)
            .into_iter()
            .filter(|(rel, r)| rels.contains(*rel) && !r.is_empty())
            .map(|(rel, r)| self.head(v, rel, r.start, r.end))
            .collect();
        let mut taken = 0usize;
        while let Some(h) = heap.pop() {
            let inc = Incidence {
                edge: h.edge,
                role: h.role,
                rel: h.rel,
                prio: h.prio,
            };
            if keep(&inc) {
                out.push(inc);
                taken += 1;
                if taken == delta {
                    return;
                }
            }
            if h.cursor + 1 < h.end {
                heap.push(self.head(v, h.rel, h.cursor + 1, h.end));
            }
        }
    }

    /// [`Self::top_incident_filtered`] without an extra predicate.
    pub fn top_incident(&self, v: NodeId, rels: &RelSet, delta: usize, out: &mut Vec<Incidence>) {
        self.top_incident_filtered(v, rels, delta, |_| true, out);
    }
}
