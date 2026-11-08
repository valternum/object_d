//! Node and edge attributes used by slicing filters: class sets, node types,
//! edge validity intervals.
//!
//! Space: `NodeTypes` = Elias–Fano offsets + 4 bytes per (node, class) pair;
//! `EdgeIntervals` = 12 bytes per annotated edge. Lookups: O(1) / O(log m).

use std::borrow::Cow;

use crate::container::{Container, ContainerWriter, Payload};
use crate::ef::EliasFano;
use crate::error::{CoreError, Result};
use crate::graph::{EdgeId, NodeId};
use crate::store::{read_ef, write_ef};

/// Class identifier.
pub type ClassId = u32;

/// Set of class ids (bit set).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassSet {
    words: Vec<u64>,
    n_classes: usize,
}

impl ClassSet {
    /// Empty set over `n_classes` classes.
    pub fn new(n_classes: usize) -> Self {
        Self {
            words: vec![0; n_classes.div_ceil(64)],
            n_classes,
        }
    }

    /// Inserts `c`. Panics if `c >= n_classes`.
    pub fn insert(&mut self, c: ClassId) {
        assert!(
            (c as usize) < self.n_classes,
            "class {c} out of range {}",
            self.n_classes
        );
        self.words[c as usize / 64] |= 1u64 << (c % 64);
    }

    /// Membership (false for ids outside the range).
    pub fn contains(&self, c: ClassId) -> bool {
        (c as usize) < self.n_classes && (self.words[c as usize / 64] >> (c % 64)) & 1 == 1
    }

    /// Number of members.
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Members in increasing order.
    pub fn ids(&self) -> Vec<ClassId> {
        (0..self.n_classes as u32)
            .filter(|&c| self.contains(c))
            .collect()
    }
}

/// Types of every node (CSR node → sorted class ids).
#[derive(Debug, Clone)]
pub struct NodeTypes<'a> {
    ptr: EliasFano<'a>,
    cls: Cow<'a, [u32]>,
}

impl NodeTypes<'static> {
    /// Builds from `(node, class)` pairs in any order; duplicates are removed.
    pub fn from_pairs(n_nodes: usize, mut pairs: Vec<(NodeId, ClassId)>) -> Result<Self> {
        if let Some(&(v, _)) = pairs.iter().find(|(v, _)| *v as usize >= n_nodes) {
            return Err(CoreError::InvalidParts(format!(
                "node {v} >= n_nodes {n_nodes}"
            )));
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut ptr = vec![0u64; n_nodes + 1];
        for &(v, _) in &pairs {
            ptr[v as usize + 1] += 1;
        }
        for v in 0..n_nodes {
            ptr[v + 1] += ptr[v];
        }
        Ok(Self {
            ptr: EliasFano::from_sorted(&ptr)?,
            cls: Cow::Owned(pairs.into_iter().map(|(_, c)| c).collect()),
        })
    }
}

impl<'a> NodeTypes<'a> {
    /// Number of nodes covered.
    pub fn n_nodes(&self) -> usize {
        self.ptr.len() - 1
    }

    /// Sorted classes of node `v`.
    pub fn types(&self, v: NodeId) -> &[ClassId] {
        let (a, b) = self.ptr.range(v as usize);
        &self.cls[a as usize..b as usize]
    }

    /// True if node `v` has at least one type in `set`.
    pub fn any_in(&self, v: NodeId, set: &ClassSet) -> bool {
        self.types(v).iter().any(|&c| set.contains(c))
    }

    /// Adds sections under `name` (`.ptr`, `.cls`).
    pub fn write<'w>(&'w self, w: &mut ContainerWriter<'w>, name: &str) -> Result<()> {
        write_ef(w, &format!("{name}.ptr"), &self.ptr)?;
        w.add(&format!("{name}.cls"), Payload::U32(&self.cls))
    }

    /// Reads sections written by [`NodeTypes::write`].
    pub fn read(c: &'a Container, name: &str) -> Result<Self> {
        let ptr = read_ef(c, &format!("{name}.ptr"))?;
        let cls = c.u32s(&format!("{name}.cls"))?;
        if ptr.is_empty() || ptr.get(ptr.len() - 1) as usize != cls.len() {
            return Err(CoreError::InvalidParts(format!(
                "`{name}`: offsets do not match classes"
            )));
        }
        Ok(Self {
            ptr,
            cls: Cow::Borrowed(cls),
        })
    }
}

/// Inclusive validity intervals `[start, end]` (years) of some edges.
#[derive(Debug, Clone)]
pub struct EdgeIntervals<'a> {
    edge: Cow<'a, [u32]>,
    start: Cow<'a, [u32]>,
    end: Cow<'a, [u32]>,
}

impl EdgeIntervals<'static> {
    /// Builds from `(edge, start, end)`; several intervals of one edge are
    /// merged into their hull `[min start, max end]`.
    pub fn from_triples(mut triples: Vec<(EdgeId, i32, i32)>) -> Self {
        triples.sort_unstable();
        let (mut edge, mut start, mut end) = (Vec::new(), Vec::new(), Vec::new());
        for (e, s, t) in triples {
            if edge.last() == Some(&e) {
                let k = edge.len() - 1;
                start[k] = (start[k] as i32).min(s) as u32;
                end[k] = (end[k] as i32).max(t) as u32;
            } else {
                edge.push(e);
                start.push(s as u32);
                end.push(t as u32);
            }
        }
        Self {
            edge: Cow::Owned(edge),
            start: Cow::Owned(start),
            end: Cow::Owned(end),
        }
    }
}

impl<'a> EdgeIntervals<'a> {
    /// Number of edges with an interval.
    pub fn len(&self) -> usize {
        self.edge.len()
    }

    /// True if no edge has an interval.
    pub fn is_empty(&self) -> bool {
        self.edge.is_empty()
    }

    /// Interval of edge `e`, if any.
    pub fn get(&self, e: EdgeId) -> Option<(i32, i32)> {
        let k = self.edge.binary_search(&e).ok()?;
        Some((self.start[k] as i32, self.end[k] as i32))
    }

    /// True if `e` has no interval or its interval intersects `[lo, hi]`.
    pub fn overlaps(&self, e: EdgeId, lo: i32, hi: i32) -> bool {
        match self.get(e) {
            None => true,
            Some((s, t)) => s <= hi && lo <= t,
        }
    }

    /// Adds sections under `name` (`.e`, `.s`, `.t`; years stored as `i32` bits).
    pub fn write<'w>(&'w self, w: &mut ContainerWriter<'w>, name: &str) -> Result<()> {
        w.add(&format!("{name}.e"), Payload::U32(&self.edge))?;
        w.add(&format!("{name}.s"), Payload::U32(&self.start))?;
        w.add(&format!("{name}.t"), Payload::U32(&self.end))
    }

    /// Reads sections written by [`EdgeIntervals::write`].
    pub fn read(c: &'a Container, name: &str) -> Result<Self> {
        let edge = c.u32s(&format!("{name}.e"))?;
        let start = c.u32s(&format!("{name}.s"))?;
        let end = c.u32s(&format!("{name}.t"))?;
        if start.len() != edge.len() || end.len() != edge.len() {
            return Err(CoreError::InvalidParts(format!(
                "`{name}`: interval arrays differ in length"
            )));
        }
        Ok(Self {
            edge: Cow::Borrowed(edge),
            start: Cow::Borrowed(start),
            end: Cow::Borrowed(end),
        })
    }
}
