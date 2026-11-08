//! Issue #9: locality-preserving renumbering (INV-PERM-1, isomorphism).

use hg_core::graph::{Hypergraph, HypergraphBuilder, NodeId, Role};
use hg_index::renumber::{bfs_order, relabel};
use proptest::prelude::*;

fn graph(n: u32, prio: &[u32], edges: &[(u16, Vec<(NodeId, Role)>)]) -> Hypergraph<'static> {
    let mut b = HypergraphBuilder::new(n);
    for (v, &p) in prio.iter().enumerate() {
        b.set_node_prio(v as u32, p).unwrap();
    }
    for (rel, m) in edges {
        b.add_edge(*rel, m).unwrap();
    }
    b.build()
}

#[test]
fn hand_checked_bfs_order() {
    // Path 0-1-2 plus 0-3 and an isolated node 4; node 3 has the highest priority.
    let g = graph(
        5,
        &[0, 0, 0, 5, 1],
        &[
            (0, vec![(0, 0), (1, 1)]),
            (0, vec![(1, 0), (2, 1)]),
            (1, vec![(3, 0), (0, 1)]),
        ],
    );
    // Start at 3 (prio 5) -> 0 -> 1 -> 2, then the next start by priority: 4.
    assert_eq!(bfs_order(&g), vec![1, 2, 3, 0, 4]);
}

#[test]
fn ties_in_priority_start_from_the_smallest_id() {
    let g = graph(4, &[0, 0, 0, 0], &[(0, vec![(2, 0), (3, 1)])]);
    // Start 0 (isolated), then 1 (isolated), then 2 -> 3.
    assert_eq!(bfs_order(&g), vec![0, 1, 2, 3]);
}

#[test]
fn neighbours_are_discovered_in_id_order() {
    // Star around node 0 with edges listed in reverse order.
    let g = graph(
        4,
        &[9, 0, 0, 0],
        &[
            (0, vec![(0, 0), (3, 1)]),
            (0, vec![(0, 0), (1, 1)]),
            (0, vec![(0, 0), (2, 1)]),
        ],
    );
    assert_eq!(bfs_order(&g), vec![0, 1, 2, 3]);
}

type Raw = (u32, Vec<u32>, Vec<(u16, Vec<(NodeId, Role)>)>);

fn raw() -> impl Strategy<Value = Raw> {
    (1u32..30).prop_flat_map(|n| {
        (
            Just(n),
            proptest::collection::vec(0u32..4, n as usize),
            proptest::collection::vec(
                (0u16..3, proptest::collection::vec((0..n, 0u8..2), 1..4)),
                0..40,
            ),
        )
    })
}

proptest! {
    /// INV-PERM-1: the order is a bijection on node ids.
    #[test]
    fn inv_perm_1_bijection((n, prio, edges) in raw()) {
        let g = graph(n, &prio, &edges);
        let mut p = bfs_order(&g);
        p.sort_unstable();
        prop_assert_eq!(p, (0..n).collect::<Vec<_>>());
    }

    /// Relabelling is an isomorphism: same edges, relations, roles and priorities.
    #[test]
    fn relabel_is_an_isomorphism((n, prio, edges) in raw()) {
        let g = graph(n, &prio, &edges);
        let perm = bfs_order(&g);
        let h = relabel(&g, &perm);
        prop_assert_eq!(h.n_nodes(), g.n_nodes());
        prop_assert_eq!(h.n_edges(), g.n_edges());
        for e in 0..g.n_edges() as u32 {
            prop_assert_eq!(h.edge_rel(e), g.edge_rel(e));
            let (gn, gr) = g.edge_members(e);
            let mut want: Vec<(u8, u32)> = gr.iter().copied().zip(gn.iter().map(|&v| perm[v as usize])).collect();
            want.sort_unstable();
            let (hn, hr) = h.edge_members(e);
            let got: Vec<(u8, u32)> = hr.iter().copied().zip(hn.iter().copied()).collect();
            prop_assert_eq!(got, want);
        }
        for v in 0..n {
            prop_assert_eq!(h.node_prio(perm[v as usize]), g.node_prio(v));
            prop_assert_eq!(h.degree(perm[v as usize]), g.degree(v));
        }
    }
}
