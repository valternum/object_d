//! Approximate personalized PageRank by local push (Andersen–Chung–Lang 2006).
//!
//! Walk: from `u` choose one of its `d(u)` usable incidences (top-Δ by
//! priority, allowed relation, inside the time window, with at least one other
//! member passing the class filter) uniformly, then a uniform such member.
//! A node without usable incidences behaves as a self-loop: it absorbs its
//! residual.
//!
//! Push rule (FIFO queue, deterministic): while `r(u) ≥ θ(u)` with
//! `θ(u) = ε · max(d̂(u), 1)`, `d̂(u) = min(allowed degree, Δ)` (an upper bound
//! of `d(u)` known from run lengths), move `α·r(u)` to `p(u)` and spread
//! `(1-α)·r(u)` over the walk. Invariants: `p + pr(α, r) = pr(α, s)`,
//! `Σp + Σr = Σs`, and work `Σ d(u) ≤ Σs / (α·ε)`.

use std::collections::{HashMap, HashSet, VecDeque};

use hg_core::graph::{EdgeId, NodeId};

use crate::{validate, Filters, GraphRefs, Seed, SliceError, SliceParams, SliceStats};

/// Push state (estimates `p`, residuals `r`).
#[derive(Debug)]
pub struct Push<'a, 'g> {
    g: &'a GraphRefs<'g>,
    params: &'a SliceParams,
    p: HashMap<NodeId, f64>,
    r: HashMap<NodeId, f64>,
    theta: HashMap<NodeId, f64>,
    traversed: Vec<EdgeId>,
    stats: SliceStats,
}

impl<'a, 'g> Push<'a, 'g> {
    /// Fresh state.
    pub fn new(g: &'a GraphRefs<'g>, params: &'a SliceParams) -> Self {
        Self {
            g,
            params,
            p: HashMap::new(),
            r: HashMap::new(),
            theta: HashMap::new(),
            traversed: Vec::new(),
            stats: SliceStats::default(),
        }
    }

    fn threshold(&mut self, v: NodeId) -> f64 {
        let (g, p) = (self.g, self.params);
        *self.theta.entry(v).or_insert_with(|| {
            let dhat = crate::capped_degree(g.graph, v, &p.relations, p.delta).max(1);
            p.epsilon * dhat as f64
        })
    }

    /// Runs the push loop from `seeds` until no node exceeds its threshold.
    pub fn run(&mut self, seeds: &[Seed]) -> Result<(), SliceError> {
        let mut params = self.params.clone();
        params.method = crate::Method::Ppr;
        let seeds = validate(self.g, seeds, &params)?;
        let f = Filters::new(self.g, self.params);
        let (alpha, delta) = (self.params.alpha, self.params.delta);
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        let mut queued: HashSet<NodeId> = HashSet::new();
        for s in &seeds {
            *self.r.entry(s.node).or_insert(0.0) += s.mass;
        }
        for s in &seeds {
            if self.r[&s.node] >= self.threshold(s.node) && queued.insert(s.node) {
                queue.push_back(s.node);
            }
        }
        let mut incs = Vec::new();
        let mut flat: Vec<NodeId> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        while let Some(u) = queue.pop_front() {
            queued.remove(&u);
            let ru = self.r[&u];
            if ru < self.threshold(u) {
                continue;
            }
            incs.clear();
            self.g.graph.top_incident_filtered(
                u,
                &self.params.relations,
                delta,
                |i| f.edge_ok(i.edge),
                &mut incs,
            );
            flat.clear();
            spans.clear();
            for inc in &incs {
                let start = flat.len();
                flat.extend(
                    self.g
                        .graph
                        .edge_members(inc.edge)
                        .0
                        .iter()
                        .copied()
                        .filter(|&w| w != u && f.node_ok(w)),
                );
                if flat.len() > start {
                    spans.push((start, flat.len()));
                    self.traversed.push(inc.edge);
                }
            }
            self.stats.pushes += 1;
            let d = spans.len();
            if d == 0 {
                *self.p.entry(u).or_insert(0.0) += ru;
                self.r.insert(u, 0.0);
                continue;
            }
            self.stats.work += d;
            *self.p.entry(u).or_insert(0.0) += alpha * ru;
            self.r.insert(u, 0.0);
            let share = ((1.0 - alpha) * ru) / d as f64;
            for &(a, b) in &spans {
                let m = share / (b - a) as f64;
                for &w in &flat[a..b] {
                    let rw = self.r.entry(w).or_insert(0.0);
                    *rw += m;
                    let rw = *rw;
                    if !queued.contains(&w) && rw >= self.threshold(w) {
                        queued.insert(w);
                        queue.push_back(w);
                    }
                }
            }
        }
        let mut touched: HashSet<NodeId> = self.p.keys().copied().collect();
        touched.extend(self.r.keys().copied());
        self.stats.touched = touched.len();
        self.stats.residual_mass = self.residuals().iter().map(|(_, x)| x).sum();
        Ok(())
    }

    /// Non-zero estimates, sorted by node id.
    pub fn estimates(&self) -> Vec<(NodeId, f64)> {
        sorted_nonzero(&self.p)
    }

    /// Non-zero residuals, sorted by node id.
    pub fn residuals(&self) -> Vec<(NodeId, f64)> {
        sorted_nonzero(&self.r)
    }

    /// Work counters.
    pub fn stats(&self) -> SliceStats {
        self.stats.clone()
    }

    /// Edges used by at least one push.
    pub fn traversed(&self) -> &[EdgeId] {
        &self.traversed
    }

    /// Seeds (by id) first, then nodes by `(p desc, id asc)` up to `max_nodes`.
    pub fn select(&self, seeds: &[Seed]) -> (Vec<NodeId>, Vec<f64>) {
        let mut seed_ids: Vec<NodeId> = seeds.iter().map(|s| s.node).collect();
        seed_ids.sort_unstable();
        seed_ids.dedup();
        let is_seed: HashSet<NodeId> = seed_ids.iter().copied().collect();
        let mut rest: Vec<(NodeId, f64)> = self
            .estimates()
            .into_iter()
            .filter(|(v, _)| !is_seed.contains(v))
            .collect();
        rest.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        rest.truncate(self.params.max_nodes.saturating_sub(seed_ids.len()));
        let mut nodes = Vec::with_capacity(seed_ids.len() + rest.len());
        let mut scores = Vec::with_capacity(nodes.capacity());
        for v in seed_ids {
            nodes.push(v);
            scores.push(self.p.get(&v).copied().unwrap_or(0.0));
        }
        for (v, x) in rest {
            nodes.push(v);
            scores.push(x);
        }
        (nodes, scores)
    }
}

fn sorted_nonzero(m: &HashMap<NodeId, f64>) -> Vec<(NodeId, f64)> {
    let mut v: Vec<(NodeId, f64)> = m
        .iter()
        .filter(|(_, &x)| x != 0.0)
        .map(|(&k, &x)| (k, x))
        .collect();
    v.sort_unstable_by_key(|(k, _)| *k);
    v
}
