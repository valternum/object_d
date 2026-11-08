//! Read-only view over a built `.holmidx` container.
//!
//! [`IndexView::new`] only performs O(1) consistency checks, so a view can be
//! created per request over a long-lived [`Container`].

use hg_core::attrs::{ClassId, ClassSet, EdgeIntervals, NodeTypes};
use hg_core::container::Container;
use hg_core::graph::{EdgeId, Hypergraph, NodeId, RelId};
use hg_core::store::read_hypergraph;

use crate::build::INDEX_FORMAT;
use crate::dict::Dict;
use crate::error::{IndexError, Result};
use crate::taxonomy::closure_over;

/// Borrowed views of every index section.
#[derive(Debug)]
pub struct IndexView<'c> {
    /// The hypergraph.
    pub graph: Hypergraph<'c>,
    /// Node types.
    pub types: NodeTypes<'c>,
    /// Edge validity intervals.
    pub intervals: EdgeIntervals<'c>,
    nodes: Dict<'c>,
    of_term: &'c [u32],
    term_of: &'c [u32],
    qid: &'c [u32],
    rels: Dict<'c>,
    classes: Dict<'c>,
    cptr: &'c [u32],
    ch: &'c [u32],
    dptr: &'c [u32],
    dom: &'c [u32],
    rptr: &'c [u32],
    rng: &'c [u32],
}

fn ensure(cond: bool, what: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(IndexError::Invalid(format!("inconsistent index: {what}")))
    }
}

impl<'c> IndexView<'c> {
    /// Creates views over an opened container.
    pub fn new(c: &'c Container) -> Result<Self> {
        let info = c.u64s("info")?;
        ensure(
            info.len() >= 5 && info[0] == INDEX_FORMAT,
            "info section / format version",
        )?;
        let v = Self {
            graph: read_hypergraph(c, "g")?,
            types: NodeTypes::read(c, "types")?,
            intervals: EdgeIntervals::read(c, "iv")?,
            nodes: Dict::read(c, "nodes")?,
            of_term: c.u32s("nodes.of_term")?,
            term_of: c.u32s("nodes.term_of")?,
            qid: c.u32s("nodes.qid")?,
            rels: Dict::read(c, "rels")?,
            classes: Dict::read(c, "classes")?,
            cptr: c.u32s("tax.cptr")?,
            ch: c.u32s("tax.ch")?,
            dptr: c.u32s("sch.dptr")?,
            dom: c.u32s("sch.dom")?,
            rptr: c.u32s("sch.rptr")?,
            rng: c.u32s("sch.rng")?,
        };
        let n = v.graph.n_nodes();
        ensure(
            info[1] as usize == n && info[2] as usize == v.graph.n_edges(),
            "info counts",
        )?;
        ensure(
            v.nodes.len() == n && v.of_term.len() == n && v.term_of.len() == n,
            "node tables",
        )?;
        ensure(
            v.qid.len() == n && v.types.n_nodes() == n,
            "node attributes",
        )?;
        ensure(v.cptr.len() == v.classes.len() + 1, "taxonomy")?;
        ensure(
            v.dptr.len() == v.rels.len() + 1 && v.rptr.len() == v.rels.len() + 1,
            "schema tables",
        )?;
        Ok(v)
    }

    /// Node id of an entity term.
    pub fn node_id(&self, term: &str) -> Option<NodeId> {
        self.nodes.lookup(term).map(|i| self.of_term[i as usize])
    }

    /// Entity term of a node.
    pub fn node_term(&self, v: NodeId) -> &str {
        self.nodes.get(self.term_of[v as usize])
    }

    /// Wikidata QID number of a node (0 if none).
    pub fn qid(&self, v: NodeId) -> u32 {
        self.qid[v as usize]
    }

    /// Relation id by name.
    pub fn relation_id(&self, name: &str) -> Option<RelId> {
        self.rels.lookup(name).map(|r| r as RelId)
    }

    /// Relation name.
    pub fn relation_name(&self, r: RelId) -> &str {
        self.rels.get(u32::from(r))
    }

    /// Number of relations.
    pub fn n_relations(&self) -> usize {
        self.rels.len()
    }

    /// Class id by name.
    pub fn class_id(&self, name: &str) -> Option<ClassId> {
        self.classes.lookup(name)
    }

    /// Class name.
    pub fn class_name(&self, c: ClassId) -> &str {
        self.classes.get(c)
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    /// Roots and all their subclasses.
    pub fn class_closure(&self, roots: &[ClassId]) -> ClassSet {
        closure_over(self.n_classes(), self.cptr, self.ch, roots)
    }

    /// Domain classes of relation `r` (from the schema).
    pub fn domain(&self, r: RelId) -> &[ClassId] {
        let r = usize::from(r);
        &self.dom[self.dptr[r] as usize..self.dptr[r + 1] as usize]
    }

    /// Range classes of relation `r` (from the schema).
    pub fn range(&self, r: RelId) -> &[ClassId] {
        let r = usize::from(r);
        &self.rng[self.rptr[r] as usize..self.rptr[r + 1] as usize]
    }

    /// First edge `a --r--> b` (a in role 0, b in role 1), if any; O(run length).
    pub fn find_edge(&self, a: NodeId, r: RelId, b: NodeId) -> Option<EdgeId> {
        let (_, range) = self.graph.runs(a).into_iter().find(|(rel, _)| *rel == r)?;
        let mut found = None;
        for (e, role) in self
            .graph
            .incidences(a)
            .skip(range.start - self.row_start(a))
            .take(range.len())
        {
            let (nodes, roles) = self.graph.edge_members(e);
            if role == 0 && nodes.iter().zip(roles).any(|(&w, &q)| w == b && q == 1) {
                found = Some(found.map_or(e, |f: EdgeId| f.min(e)));
            }
        }
        found
    }

    fn row_start(&self, a: NodeId) -> usize {
        self.graph.runs(a).first().map_or(0, |(_, r)| r.start)
    }
}
