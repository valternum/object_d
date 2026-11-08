//! Issue #4: typed incidence hypergraph, relation runs, top-delta selection.

use hg_core::graph::{Hypergraph, HypergraphBuilder, Incidence, NodeId, RelId, RelSet, Role};
use proptest::prelude::*;

/// Raw description of a test hypergraph.
#[derive(Debug, Clone)]
struct RawGraph {
    n: u32,
    prio: Vec<u32>,
    edges: Vec<(RelId, Vec<(NodeId, Role)>)>,
}

fn build(raw: &RawGraph) -> Hypergraph<'static> {
    let mut b = HypergraphBuilder::new(raw.n);
    for (v, &p) in raw.prio.iter().enumerate() {
        b.set_node_prio(v as NodeId, p).unwrap();
    }
    for (rel, members) in &raw.edges {
        b.add_edge(*rel, members).unwrap();
    }
    b.build()
}

/// Oracle prio(e|v): max priority of members that are not v, 0 if none.
fn oracle_prio(raw: &RawGraph, e: usize, v: NodeId) -> u32 {
    raw.edges[e]
        .1
        .iter()
        .filter(|(w, _)| *w != v)
        .map(|(w, _)| raw.prio[*w as usize])
        .max()
        .unwrap_or(0)
}

/// Oracle: every allowed incidence of v, sorted by (prio desc, edge asc, role asc), first delta.
fn oracle_top(
    raw: &RawGraph,
    v: NodeId,
    allowed: &dyn Fn(RelId) -> bool,
    delta: usize,
) -> Vec<Incidence> {
    let mut all = Vec::new();
    for (e, (rel, members)) in raw.edges.iter().enumerate() {
        if !allowed(*rel) {
            continue;
        }
        for &(w, role) in members {
            if w == v {
                all.push(Incidence {
                    edge: e as u32,
                    role,
                    rel: *rel,
                    prio: oracle_prio(raw, e, v),
                });
            }
        }
    }
    all.sort_by(|a, b| {
        b.prio
            .cmp(&a.prio)
            .then(a.edge.cmp(&b.edge))
            .then(a.role.cmp(&b.role))
    });
    all.truncate(delta);
    all
}

fn raw_graph() -> impl Strategy<Value = RawGraph> {
    (1u32..30).prop_flat_map(|n| {
        let prio = proptest::collection::vec(0u32..6, n as usize);
        let member = (0..n, 0u8..3);
        let edge = (0u16..4, proptest::collection::vec(member, 1..6));
        let edges = proptest::collection::vec(edge, 0..40);
        (Just(n), prio, edges).prop_map(|(n, prio, edges)| RawGraph { n, prio, edges })
    })
}

fn hand_graph() -> RawGraph {
    // 0 = France (prio 9), 1 = Paris (5), 2 = Lyon (3), 3 = Macron (7), 4 = EU (8)
    RawGraph {
        n: 5,
        prio: vec![9, 5, 3, 7, 8],
        edges: vec![
            (1, vec![(1, 0), (0, 1)]), // e0 Paris  partOf  France
            (1, vec![(2, 0), (0, 1)]), // e1 Lyon   partOf  France
            (2, vec![(0, 0), (1, 1)]), // e2 France capital Paris
            (3, vec![(3, 0), (0, 1)]), // e3 Macron nationality France
            (4, vec![(0, 0), (4, 1)]), // e4 France memberOf EU
        ],
    }
}

#[test]
fn hand_checked_rows_and_runs() {
    let g = build(&hand_graph());
    assert_eq!(g.n_nodes(), 5);
    assert_eq!(g.n_edges(), 5);
    assert_eq!(g.n_incidences(), 10);
    assert_eq!(g.degree(0), 5);
    assert_eq!(g.degree(2), 1);
    // France's row: rel 1 run sorted by prio desc: e0 (Paris 5) then e1 (Lyon 3); then rel 2, 3, 4.
    let row: Vec<(u32, u8)> = g.incidences(0).collect();
    assert_eq!(row, vec![(0, 1), (1, 1), (2, 0), (3, 1), (4, 0)]);
    let runs: Vec<(RelId, usize)> = g
        .runs(0)
        .into_iter()
        .map(|(r, range)| (r, range.len()))
        .collect();
    assert_eq!(runs, vec![(1, 2), (2, 1), (3, 1), (4, 1)]);
    assert_eq!(g.incidence_prio(0, 3), 7);
    assert_eq!(g.edge_rel(4), 4);
    assert_eq!(g.edge_members(2), (&[0u32, 1][..], &[0u8, 1][..]));
}

#[test]
fn hand_checked_top_delta_merges_runs_by_priority() {
    let g = build(&hand_graph());
    let mut out = Vec::new();
    g.top_incident(0, &RelSet::all(), 3, &mut out);
    let edges: Vec<u32> = out.iter().map(|i| i.edge).collect();
    // prio(e|France): e4 EU 8, e3 Macron 7, e0 Paris 5, e2 Paris 5, e1 Lyon 3 -> ties by edge id.
    assert_eq!(edges, vec![4, 3, 0]);
    out.clear();
    g.top_incident(0, &RelSet::only(&[1, 2]), 10, &mut out);
    assert_eq!(
        out.iter().map(|i| i.edge).collect::<Vec<_>>(),
        vec![0, 2, 1]
    );
    assert_eq!(g.allowed_degree(0, &RelSet::only(&[1, 2])), 3);
    out.clear();
    g.top_incident(0, &RelSet::all(), 0, &mut out);
    assert!(out.is_empty());
}

#[test]
fn filtered_top_delta_skips_rejected_and_keeps_counting() {
    let g = build(&hand_graph());
    let mut out = Vec::new();
    g.top_incident_filtered(0, &RelSet::all(), 2, |i| i.edge != 4, &mut out);
    assert_eq!(out.iter().map(|i| i.edge).collect::<Vec<_>>(), vec![3, 0]);
}

#[test]
fn rel_set_membership() {
    let s = RelSet::only(&[0, 63, 64, 65535]);
    assert!(s.contains(0) && s.contains(63) && s.contains(64) && s.contains(65535));
    assert!(!s.contains(1) && !s.contains(1000));
    assert!(RelSet::all().contains(12345));
}

#[test]
fn builder_rejects_out_of_range_member() {
    let mut b = HypergraphBuilder::new(3);
    assert!(b.add_edge(0, &[(0, 0), (3, 1)]).is_err());
    assert!(b.set_node_prio(3, 1).is_err());
}

#[test]
fn builder_rejects_empty_edge() {
    let mut b = HypergraphBuilder::new(3);
    assert!(b.add_edge(0, &[]).is_err());
}

#[test]
fn graph_without_edges() {
    let g = HypergraphBuilder::new(4).build();
    assert_eq!(g.n_edges(), 0);
    assert_eq!(g.degree(3), 0);
    assert!(g.runs(2).is_empty());
}

proptest! {
    /// INV-CORE-1: H and H^T agree as multisets; degrees and edge sizes add up.
    #[test]
    fn inv_core_1_incidence_and_membership_agree(raw in raw_graph()) {
        let g = build(&raw);
        let mut from_rows: Vec<(u32, u32, u8)> = Vec::new();
        for v in 0..raw.n {
            let row: Vec<_> = g.incidences(v).collect();
            prop_assert_eq!(row.len(), g.degree(v));
            from_rows.extend(row.into_iter().map(|(e, r)| (v, e, r)));
        }
        let mut from_edges: Vec<(u32, u32, u8)> = Vec::new();
        for e in 0..g.n_edges() as u32 {
            let (nodes, roles) = g.edge_members(e);
            from_edges.extend(nodes.iter().zip(roles).map(|(&v, &r)| (v, e, r)));
        }
        from_rows.sort_unstable();
        from_edges.sort_unstable();
        prop_assert_eq!(&from_rows, &from_edges);
        let total: usize = raw.edges.iter().map(|(_, m)| m.len()).sum();
        prop_assert_eq!(g.n_incidences(), total);
    }

    /// INV-CORE-3: relation types and (role, node) members are preserved per edge.
    #[test]
    fn inv_core_3_relations_and_roles_preserved(raw in raw_graph()) {
        let g = build(&raw);
        prop_assert_eq!(g.n_edges(), raw.edges.len());
        for (e, (rel, members)) in raw.edges.iter().enumerate() {
            prop_assert_eq!(g.edge_rel(e as u32), *rel);
            let mut want: Vec<(u8, u32)> = members.iter().map(|&(v, r)| (r, v)).collect();
            want.sort_unstable();
            let (nodes, roles) = g.edge_members(e as u32);
            let got: Vec<(u8, u32)> = roles.iter().copied().zip(nodes.iter().copied()).collect();
            prop_assert_eq!(got, want);
        }
    }

    /// top-delta equals the sort-everything oracle for delta in {0, 1, 3, unbounded}.
    #[test]
    fn top_delta_matches_sort_oracle(raw in raw_graph(), allowed_mask in 0u8..16) {
        let g = build(&raw);
        let allowed_ids: Vec<RelId> = (0..4u16).filter(|r| allowed_mask & (1 << r) != 0).collect();
        let rels = RelSet::only(&allowed_ids);
        let allowed = |r: RelId| allowed_ids.contains(&r);
        for v in 0..raw.n {
            for delta in [0usize, 1, 3, usize::MAX] {
                let mut out = Vec::new();
                g.top_incident(v, &rels, delta, &mut out);
                prop_assert_eq!(&out, &oracle_top(&raw, v, &allowed, delta), "v={} delta={}", v, delta);
            }
            let want_deg = oracle_top(&raw, v, &allowed, usize::MAX).len();
            prop_assert_eq!(g.allowed_degree(v, &rels), want_deg);
        }
    }

    /// Runs partition each row into contiguous blocks with strictly increasing relations.
    #[test]
    fn runs_partition_rows(raw in raw_graph()) {
        let g = build(&raw);
        for v in 0..raw.n {
            let runs = g.runs(v);
            let covered: usize = runs.iter().map(|(_, r)| r.len()).sum();
            prop_assert_eq!(covered, g.degree(v));
            for w in runs.windows(2) {
                prop_assert!(w[0].0 < w[1].0);
                prop_assert_eq!(w[0].1.end, w[1].1.start);
            }
        }
    }
}
