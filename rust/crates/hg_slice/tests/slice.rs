//! Issues #11–#13: slice API, ball, PPR local push, filters.

use std::collections::{BTreeSet, HashSet, VecDeque};

use hg_core::attrs::{ClassSet, EdgeIntervals, NodeTypes};
use hg_core::graph::{Hypergraph, HypergraphBuilder, NodeId, RelSet, Role};
use hg_slice::{slice, GraphRefs, LocalSubgraph, Method, Seed, SliceError, SliceParams};
use proptest::prelude::*;

type Edges = Vec<(u16, Vec<(NodeId, Role)>)>;

fn graph(n: u32, prio: &[u32], edges: &Edges) -> Hypergraph<'static> {
    let mut b = HypergraphBuilder::new(n);
    for (v, &p) in prio.iter().enumerate() {
        b.set_node_prio(v as u32, p).unwrap();
    }
    for (rel, m) in edges {
        b.add_edge(*rel, m).unwrap();
    }
    b.build()
}

fn params(method: Method) -> SliceParams {
    SliceParams {
        method,
        max_nodes: usize::MAX,
        hops: 2,
        delta: usize::MAX,
        alpha: 0.15,
        epsilon: 1e-6,
        scan_cap: usize::MAX,
        relations: RelSet::all(),
        class_filter: None,
        time_window: None,
    }
}

fn refs<'g>(g: &'g Hypergraph<'g>) -> GraphRefs<'g> {
    GraphRefs {
        graph: g,
        types: None,
        intervals: None,
    }
}

fn seed(node: u32) -> Seed {
    Seed { node, mass: 1.0 }
}

fn path5() -> Hypergraph<'static> {
    graph(
        5,
        &[0; 5],
        &(0..4)
            .map(|i| (0u16, vec![(i, 0u8), (i + 1, 1u8)]))
            .collect(),
    )
}

/// INV-SLICE-3: every member of every returned edge is a selected node, ids are consistent.
fn check_well_formed(g: &Hypergraph<'_>, s: &LocalSubgraph) {
    assert_eq!(s.nodes.len(), s.scores.len());
    assert_eq!(s.edge_ptr.len(), s.edges.len() + 1);
    assert_eq!(s.edge_rel.len(), s.edges.len());
    assert!(
        s.edges.windows(2).all(|w| w[0] < w[1]),
        "edges sorted and unique"
    );
    let distinct: HashSet<u32> = s.nodes.iter().copied().collect();
    assert_eq!(distinct.len(), s.nodes.len(), "nodes unique");
    for (k, &e) in s.edges.iter().enumerate() {
        let (a, b) = (s.edge_ptr[k] as usize, s.edge_ptr[k + 1] as usize);
        let local: Vec<u32> = s.edge_members[a..b]
            .iter()
            .map(|&l| s.nodes[l as usize])
            .collect();
        let (gn, gr) = g.edge_members(e);
        assert_eq!(local, gn, "members map back to the global edge");
        assert_eq!(&s.edge_roles[a..b], gr);
        assert_eq!(s.edge_rel[k], g.edge_rel(e));
    }
}

// ----------------------------------------------------------------- ball (#11)

#[test]
fn ball_on_a_path() {
    let g = path5();
    let s = slice(&refs(&g), &[seed(0)], &params(Method::Ball)).unwrap();
    assert_eq!(s.nodes, vec![0, 1, 2]);
    assert_eq!(s.scores, vec![1.0, 0.5, 1.0 / 3.0]);
    assert_eq!(s.edges, vec![0, 1]);
    assert_eq!(s.edge_members, vec![0, 1, 1, 2]);
    check_well_formed(&g, &s);
}

#[test]
fn ball_budget_counts_seeds() {
    let g = path5();
    let mut p = params(Method::Ball);
    p.max_nodes = 2;
    p.hops = 10;
    let s = slice(&refs(&g), &[seed(2)], &p).unwrap();
    assert_eq!(s.nodes.len(), 2);
    assert_eq!(s.nodes[0], 2);
    check_well_formed(&g, &s);
}

#[test]
fn hub_fan_out_is_capped_by_priority() {
    // Star: centre 0, leaves 1..=100 with priority = leaf id.
    let prio: Vec<u32> = (0..=100).collect();
    let edges: Edges = (1..=100)
        .map(|l| (0u16, vec![(0, 0u8), (l, 1u8)]))
        .collect();
    let g = graph(101, &prio, &edges);
    let mut p = params(Method::Ball);
    p.delta = 5;
    p.hops = 1;
    let s = slice(&refs(&g), &[seed(0)], &p).unwrap();
    assert_eq!(s.nodes, vec![0, 100, 99, 98, 97, 96]);
    assert_eq!(s.stats.work, 5, "only delta incidences are read");
    check_well_formed(&g, &s);
}

#[test]
fn seeds_are_validated() {
    let g = path5();
    assert_eq!(
        slice(&refs(&g), &[seed(9)], &params(Method::Ball)),
        Err(SliceError::InvalidSeed(9))
    );
    let mut p = params(Method::Ball);
    p.max_nodes = 1;
    assert!(matches!(
        slice(&refs(&g), &[seed(0), seed(1)], &p),
        Err(SliceError::InvalidParams(_))
    ));
    let mut p = params(Method::Ppr);
    p.alpha = 0.0;
    assert!(matches!(
        slice(&refs(&g), &[seed(0)], &p),
        Err(SliceError::InvalidParams(_))
    ));
    let mut p = params(Method::Ppr);
    p.class_filter = Some(ClassSet::new(3));
    assert!(
        matches!(
            slice(&refs(&g), &[seed(0)], &p),
            Err(SliceError::InvalidParams(_))
        ),
        "class filter without types"
    );
    assert!(
        matches!(
            slice(&refs(&g), &[], &params(Method::Ppr)),
            Err(SliceError::InvalidParams(_))
        ),
        "no seeds"
    );
}

fn oracle_ball(
    n: u32,
    edges: &Edges,
    allowed: &dyn Fn(u16) -> bool,
    seeds: &[u32],
    hops: u32,
) -> BTreeSet<u32> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
    for (rel, m) in edges {
        if !allowed(*rel) {
            continue;
        }
        for &(a, _) in m {
            for &(b, _) in m {
                if a != b {
                    adj[a as usize].push(b);
                }
            }
        }
    }
    let mut dist = vec![u32::MAX; n as usize];
    let mut q = VecDeque::new();
    for &s in seeds {
        dist[s as usize] = 0;
        q.push_back(s);
    }
    while let Some(u) = q.pop_front() {
        if dist[u as usize] == hops {
            continue;
        }
        for &w in &adj[u as usize] {
            if dist[w as usize] == u32::MAX {
                dist[w as usize] = dist[u as usize] + 1;
                q.push_back(w);
            }
        }
    }
    (0..n).filter(|&v| dist[v as usize] != u32::MAX).collect()
}

type RawGraph = (u32, Vec<u32>, Edges);

fn raw_graph() -> impl Strategy<Value = RawGraph> {
    (1u32..20).prop_flat_map(|n| {
        (
            Just(n),
            proptest::collection::vec(0u32..5, n as usize),
            proptest::collection::vec(
                (0u16..3, proptest::collection::vec((0..n, 0u8..2), 1..5)),
                0..30,
            ),
        )
    })
}

// ------------------------------------------------------------------ ppr (#12)

/// Dense oracle on the same capped walk: W from top-Δ incidences (uniform), then a uniform
/// other member; nodes without usable incidences get a self-loop. pr(x) = α Σ_k x((1-α)W)^k.
fn dense_pr(g: &Hypergraph<'_>, delta: usize, alpha: f64, x: &[f64]) -> Vec<f64> {
    let n = g.n_nodes();
    let mut w = vec![vec![0.0f64; n]; n];
    for u in 0..n as u32 {
        let mut incs = Vec::new();
        g.top_incident(u, &RelSet::all(), delta, &mut incs);
        let usable: Vec<Vec<u32>> = incs
            .iter()
            .map(|i| {
                g.edge_members(i.edge)
                    .0
                    .iter()
                    .copied()
                    .filter(|&m| m != u)
                    .collect::<Vec<_>>()
            })
            .filter(|o: &Vec<u32>| !o.is_empty())
            .collect();
        if usable.is_empty() {
            w[u as usize][u as usize] = 1.0;
            continue;
        }
        for others in &usable {
            for &m in others {
                w[u as usize][m as usize] += 1.0 / (usable.len() as f64 * others.len() as f64);
            }
        }
    }
    let mut out = vec![0.0; n];
    let mut term = x.to_vec();
    for _ in 0..10_000 {
        let mut next = vec![0.0; n];
        let mut mass = 0.0;
        for i in 0..n {
            out[i] += alpha * term[i];
            for j in 0..n {
                next[j] += (1.0 - alpha) * term[i] * w[i][j];
            }
        }
        for v in &next {
            mass += v;
        }
        term = next;
        if mass < 1e-15 {
            break;
        }
    }
    out
}

fn ppr_dense_vectors(g: &Hypergraph<'_>, s: &LocalSubgraph) -> Vec<f64> {
    let mut p = vec![0.0; g.n_nodes()];
    for (&v, &score) in s.nodes.iter().zip(&s.scores) {
        p[v as usize] = score;
    }
    p
}

#[test]
fn ppr_two_node_closed_form() {
    // Single edge 0-1, alpha = 1/2: pr = (2/3, 1/3).
    let g = graph(2, &[0, 0], &vec![(0u16, vec![(0, 0u8), (1, 1u8)])]);
    let mut p = params(Method::Ppr);
    p.alpha = 0.5;
    p.epsilon = 1e-10;
    let s = slice(&refs(&g), &[seed(0)], &p).unwrap();
    assert_eq!(s.nodes, vec![0, 1]);
    assert!((s.scores[0] - 2.0 / 3.0).abs() < 1e-8, "{}", s.scores[0]);
    assert!((s.scores[1] - 1.0 / 3.0).abs() < 1e-8, "{}", s.scores[1]);
    check_well_formed(&g, &s);
}

#[test]
fn ppr_isolated_seed_absorbs_its_mass() {
    let g = graph(3, &[0; 3], &vec![(0u16, vec![(1, 0u8), (2, 1u8)])]);
    let s = slice(
        &refs(&g),
        &[Seed { node: 0, mass: 2.0 }],
        &params(Method::Ppr),
    )
    .unwrap();
    assert_eq!(s.nodes, vec![0]);
    assert_eq!(s.scores, vec![2.0]);
    assert_eq!(s.stats.residual_mass, 0.0);
}

#[test]
fn ppr_selection_puts_seeds_first_and_breaks_ties_by_id() {
    let edges: Edges = (1..=4).map(|l| (0u16, vec![(0, 0u8), (l, 1u8)])).collect();
    let g = graph(5, &[0; 5], &edges);
    let mut p = params(Method::Ppr);
    p.max_nodes = 3;
    let s = slice(&refs(&g), &[seed(3)], &p).unwrap();
    // Seed 3, then the centre 0 (highest mass), then the best leaf: leaves 1, 2, 4 tie -> 1.
    assert_eq!(s.nodes, vec![3, 0, 1]);
    check_well_formed(&g, &s);
}

// -------------------------------------------------------------- filters (#13)

fn typed_star() -> (
    Hypergraph<'static>,
    NodeTypes<'static>,
    EdgeIntervals<'static>,
) {
    // centre 0; leaves 1..=4 over relations 0,0,1,1; leaf types: 1 -> class 0, 2 -> 1, 3 -> 0, 4 -> 1
    let edges: Edges = vec![
        (0, vec![(0, 0), (1, 1)]),
        (0, vec![(0, 0), (2, 1)]),
        (1, vec![(0, 0), (3, 1)]),
        (1, vec![(0, 0), (4, 1)]),
    ];
    let g = graph(5, &[0; 5], &edges);
    let types = NodeTypes::from_pairs(5, vec![(1, 0), (2, 1), (3, 0), (4, 1)]).unwrap();
    // edge 0 (0-1) valid 1951..2002, edge 2 (0-3) valid from 2010
    let iv = EdgeIntervals::from_triples(vec![(0, 1951, 2002), (2, 2010, i32::MAX)]);
    (g, types, iv)
}

#[test]
fn inv_slice_filter_1_relations() {
    let (g, types, iv) = typed_star();
    for method in [Method::Ball, Method::Ppr] {
        let mut p = params(method);
        p.relations = RelSet::only(&[1]);
        let r = GraphRefs {
            graph: &g,
            types: Some(&types),
            intervals: Some(&iv),
        };
        let s = slice(&r, &[seed(0)], &p).unwrap();
        let set: BTreeSet<u32> = s.nodes.iter().copied().collect();
        assert_eq!(set, BTreeSet::from([0, 3, 4]), "{method:?}");
        assert!(s.edge_rel.iter().all(|&r| r == 1));
        check_well_formed(&g, &s);
    }
}

#[test]
fn inv_slice_filter_2_classes() {
    let (g, types, iv) = typed_star();
    for method in [Method::Ball, Method::Ppr] {
        let mut p = params(method);
        let mut cs = ClassSet::new(2);
        cs.insert(1);
        p.class_filter = Some(cs.clone());
        let r = GraphRefs {
            graph: &g,
            types: Some(&types),
            intervals: Some(&iv),
        };
        let s = slice(&r, &[seed(0)], &p).unwrap();
        assert_eq!(s.nodes[0], 0, "the untyped seed is kept");
        for &v in &s.nodes[1..] {
            assert!(
                types.any_in(v, &cs),
                "{method:?}: node {v} violates the class filter"
            );
        }
        let set: BTreeSet<u32> = s.nodes.iter().copied().collect();
        assert_eq!(set, BTreeSet::from([0, 2, 4]), "{method:?}");
    }
}

#[test]
fn inv_slice_filter_3_time_window() {
    let (g, types, iv) = typed_star();
    for method in [Method::Ball, Method::Ppr] {
        let mut p = params(method);
        p.time_window = Some((2003, 2005));
        let r = GraphRefs {
            graph: &g,
            types: Some(&types),
            intervals: Some(&iv),
        };
        let s = slice(&r, &[seed(0)], &p).unwrap();
        let set: BTreeSet<u32> = s.nodes.iter().copied().collect();
        // edge 0 ended 2002, edge 2 starts 2010; edges 1 and 3 have no interval.
        assert_eq!(set, BTreeSet::from([0, 2, 4]), "{method:?}");
        for &e in &s.edges {
            let (lo, hi) = (2003, 2005);
            assert!(iv.overlaps(e, lo, hi));
        }
    }
}

#[test]
fn time_window_without_intervals_is_an_error() {
    let g = path5();
    let mut p = params(Method::Ball);
    p.time_window = Some((0, 1));
    assert!(matches!(
        slice(&refs(&g), &[seed(0)], &p),
        Err(SliceError::InvalidParams(_))
    ));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// INV-SLICE-BALL-1: unbounded ball = exact BFS ball; INV-SLICE-FILTER-1 for relations.
    #[test]
    fn inv_slice_ball_1_matches_bfs((n, prio, edges) in raw_graph(), hops in 0u32..4, s0 in 0u32..20, mask in 1u8..8) {
        let g = graph(n, &prio, &edges);
        let s0 = s0 % n;
        let allowed_ids: Vec<u16> = (0..3u16).filter(|r| mask & (1 << r) != 0).collect();
        let mut p = params(Method::Ball);
        p.hops = hops;
        p.relations = RelSet::only(&allowed_ids);
        let s = slice(&refs(&g), &[seed(s0)], &p).unwrap();
        let got: BTreeSet<u32> = s.nodes.iter().copied().collect();
        let want = oracle_ball(n, &edges, &|r| allowed_ids.contains(&r), &[s0], hops);
        prop_assert_eq!(got, want);
        prop_assert!(s.edge_rel.iter().all(|r| allowed_ids.contains(r)));
        check_well_formed(&g, &s);
    }

    /// INV-SLICE-3 with budgets, INV-SLICE-1 determinism, both methods.
    #[test]
    fn inv_slice_3_budget_and_determinism((n, prio, edges) in raw_graph(), budget in 1usize..8, delta in 1usize..4, s0 in 0u32..20) {
        let g = graph(n, &prio, &edges);
        for method in [Method::Ball, Method::Ppr] {
            let mut p = params(method);
            p.max_nodes = budget;
            p.delta = delta;
            p.hops = 3;
            let a = slice(&refs(&g), &[seed(s0 % n)], &p).unwrap();
            let b = slice(&refs(&g), &[seed(s0 % n)], &p).unwrap();
            prop_assert!(a.nodes.len() <= budget);
            prop_assert_eq!(a.nodes[0], s0 % n);
            prop_assert_eq!(&a, &b);
            check_well_formed(&g, &a);
        }
    }

    /// INV-PPR-1 (termination), INV-PPR-2 (work bound), INV-PPR-3 (mass), INV-PPR-4 (ACL identity).
    #[test]
    fn inv_ppr_invariants((n, prio, edges) in raw_graph(), delta in 1usize..5, s0 in 0u32..20, alpha in 0.1f64..0.9) {
        let g = graph(n, &prio, &edges);
        let eps = 1e-4;
        let mut p = params(Method::Ppr);
        p.delta = delta;
        p.alpha = alpha;
        p.epsilon = eps;
        let s0 = s0 % n;
        let gr = refs(&g);
        let mut detailed = hg_slice::ppr::Push::new(&gr, &p);
        detailed.run(&[seed(s0)]).unwrap();
        // INV-PPR-1
        for (v, r) in detailed.residuals() {
            let dhat = hg_slice::capped_degree(&g, v, &RelSet::all(), delta).max(1) as f64;
            prop_assert!(r < eps * dhat, "node {} keeps residual {} >= eps*dhat {}", v, r, eps * dhat);
        }
        // INV-PPR-2
        prop_assert!(detailed.stats().work as f64 <= 1.0 / (alpha * eps) + 1e-9);
        // INV-PPR-3
        let total: f64 = detailed.estimates().iter().map(|(_, x)| x).sum::<f64>()
            + detailed.residuals().iter().map(|(_, x)| x).sum::<f64>();
        prop_assert!((total - 1.0).abs() <= 1e-12, "mass {}", total);
        // INV-PPR-4: p + pr(r) = pr(seed)
        let mut x_seed = vec![0.0; n as usize];
        x_seed[s0 as usize] = 1.0;
        let mut x_res = vec![0.0; n as usize];
        for (v, r) in detailed.residuals() {
            x_res[v as usize] = r;
        }
        let mut p_vec = vec![0.0; n as usize];
        for (v, x) in detailed.estimates() {
            p_vec[v as usize] = x;
        }
        let pr_s = dense_pr(&g, delta, alpha, &x_seed);
        let pr_r = dense_pr(&g, delta, alpha, &x_res);
        for v in 0..n as usize {
            prop_assert!((p_vec[v] + pr_r[v] - pr_s[v]).abs() <= 1e-9, "node {}: {} + {} != {}", v, p_vec[v], pr_r[v], pr_s[v]);
        }
        // The public slice reports the same estimates for its nodes.
        let s = slice(&refs(&g), &[seed(s0)], &p).unwrap();
        let dense = ppr_dense_vectors(&g, &s);
        for (v, x) in detailed.estimates() {
            if s.nodes.contains(&v) {
                prop_assert_eq!(dense[v as usize], x);
            }
        }
    }
}
