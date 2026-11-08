//! Bounded BFS ball.
//!
//! Layer by layer from the seeds; each expanded node contributes at most
//! `delta` incidences (best by priority, allowed relation, inside the time
//! window); members passing the class filter are added in edge order until
//! `max_nodes`. Time O(hops · |frontier| · (runs + Δ) · log runs).

use std::collections::HashMap;

use hg_core::graph::{EdgeId, NodeId};

use crate::{Filters, GraphRefs, Seed, SliceParams, SliceStats};

pub(crate) type Selection = (Vec<NodeId>, Vec<f64>, Vec<EdgeId>, SliceStats);

pub(crate) fn run(
    g: &GraphRefs<'_>,
    f: &Filters<'_, '_>,
    seeds: &[Seed],
    p: &SliceParams,
) -> Selection {
    let mut depth: HashMap<NodeId, u32> = HashMap::new();
    let mut order: Vec<NodeId> = Vec::new();
    for s in seeds {
        depth.insert(s.node, 0);
        order.push(s.node);
    }
    let mut frontier: Vec<NodeId> = order.clone();
    let mut traversed = Vec::new();
    let mut stats = SliceStats::default();
    let mut incs = Vec::new();
    'layers: for d in 0..p.hops {
        let mut next = Vec::new();
        for &u in &frontier {
            incs.clear();
            g.graph.top_incident_filtered(
                u,
                &p.relations,
                p.delta,
                |i| f.edge_ok(i.edge),
                &mut incs,
            );
            stats.pushes += 1;
            stats.work += incs.len();
            for inc in &incs {
                traversed.push(inc.edge);
                for &w in g.graph.edge_members(inc.edge).0 {
                    if w == u || depth.contains_key(&w) || !f.node_ok(w) {
                        continue;
                    }
                    if order.len() >= p.max_nodes {
                        break 'layers;
                    }
                    depth.insert(w, d + 1);
                    order.push(w);
                    next.push(w);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    stats.touched = order.len();
    let scores = order
        .iter()
        .map(|v| 1.0 / (1.0 + f64::from(depth[v])))
        .collect();
    (order, scores, traversed, stats)
}
