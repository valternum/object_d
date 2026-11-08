//! Induced edges of a node selection.
//!
//! Candidates: edges used during the traversal plus all allowed incidences of
//! selected nodes whose degree is at most `scan_cap` (hub rows are not
//! scanned; their edges are found from the other endpoint or from the
//! traversal). An edge is kept if it passes the filters and all its members
//! are selected. Time O(Σ_{deg ≤ scan_cap} deg + E log E).

use std::collections::HashMap;

use hg_core::graph::{EdgeId, NodeId, RelId, Role};

use crate::{Filters, GraphRefs, SliceParams};

pub(crate) struct Induced {
    pub edges: Vec<EdgeId>,
    pub rel: Vec<RelId>,
    pub ptr: Vec<u32>,
    pub members: Vec<u32>,
    pub roles: Vec<Role>,
}

pub(crate) fn collect(
    g: &GraphRefs<'_>,
    f: &Filters<'_, '_>,
    p: &SliceParams,
    nodes: &[NodeId],
    mut cand: Vec<EdgeId>,
) -> Induced {
    let local: HashMap<NodeId, u32> = nodes
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    for &u in nodes {
        if g.graph.degree(u) <= p.scan_cap {
            cand.extend(g.graph.incidences(u).map(|(e, _)| e));
        }
    }
    cand.sort_unstable();
    cand.dedup();
    let mut out = Induced {
        edges: Vec::new(),
        rel: Vec::new(),
        ptr: vec![0],
        members: Vec::new(),
        roles: Vec::new(),
    };
    for e in cand {
        let rel = g.graph.edge_rel(e);
        if !p.relations.contains(rel) || !f.edge_ok(e) {
            continue;
        }
        let (ms, rs) = g.graph.edge_members(e);
        if !ms.iter().all(|v| local.contains_key(v)) {
            continue;
        }
        out.edges.push(e);
        out.rel.push(rel);
        out.members.extend(ms.iter().map(|v| local[v]));
        out.roles.extend_from_slice(rs);
        out.ptr.push(out.members.len() as u32);
    }
    out
}
