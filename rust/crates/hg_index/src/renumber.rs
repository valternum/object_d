//! Locality-preserving node renumbering.
//!
//! New ids follow BFS discovery order: start nodes are taken by
//! `(priority desc, id asc)`, neighbours are discovered in increasing old id.
//! Nodes that are close in the graph get close ids, so the adjacency rows read
//! during a slice touch fewer pages of the memory-mapped index.
//!
//! Time O(I·k + n log n) for I incidences and edge arity k; memory O(n).

use std::collections::VecDeque;

use hg_core::graph::{Hypergraph, HypergraphBuilder};

/// BFS discovery order as a permutation `old id -> new id`.
pub fn bfs_order(g: &Hypergraph<'_>) -> Vec<u32> {
    let n = g.n_nodes();
    let mut starts: Vec<u32> = (0..n as u32).collect();
    starts.sort_by(|&a, &b| g.node_prio(b).cmp(&g.node_prio(a)).then(a.cmp(&b)));

    let mut new_id = vec![u32::MAX; n];
    let mut next = 0u32;
    let mut queue = VecDeque::new();
    let mut neighbours = Vec::new();
    for s in starts {
        if new_id[s as usize] != u32::MAX {
            continue;
        }
        new_id[s as usize] = next;
        next += 1;
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            neighbours.clear();
            for (e, _) in g.incidences(u) {
                neighbours.extend(g.edge_members(e).0.iter().copied().filter(|&w| w != u));
            }
            neighbours.sort_unstable();
            neighbours.dedup();
            for &w in &neighbours {
                if new_id[w as usize] == u32::MAX {
                    new_id[w as usize] = next;
                    next += 1;
                    queue.push_back(w);
                }
            }
        }
    }
    new_id
}

/// Relabels nodes by `perm` (`old -> new`); edge ids and edge order are kept.
pub fn relabel(g: &Hypergraph<'_>, perm: &[u32]) -> Hypergraph<'static> {
    assert_eq!(perm.len(), g.n_nodes(), "permutation size");
    let mut b = HypergraphBuilder::new(g.n_nodes() as u32);
    for v in 0..g.n_nodes() as u32 {
        b.set_node_prio(perm[v as usize], g.node_prio(v))
            .expect("perm is a bijection");
    }
    let mut members = Vec::new();
    for e in 0..g.n_edges() as u32 {
        let (nodes, roles) = g.edge_members(e);
        members.clear();
        members.extend(
            nodes
                .iter()
                .zip(roles)
                .map(|(&v, &r)| (perm[v as usize], r)),
        );
        b.add_edge(g.edge_rel(e), &members)
            .expect("edges are valid");
    }
    b.build()
}
