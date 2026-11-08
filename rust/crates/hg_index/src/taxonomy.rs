//! Class DAG (`rdfs:subClassOf`) and closures of root sets.
//!
//! Ids are ranks of class names in byte order. Both parent and child lists are
//! stored as CSR. `closure(roots)` is a BFS over children: O(|closure| + its
//! out-edges); the result is a bit set of `n_classes` bits.

use std::collections::{BTreeSet, VecDeque};

use hg_core::attrs::{ClassId, ClassSet};

use crate::error::{IndexError, Result};

/// Class hierarchy with ids = rank of the class name in byte order.
#[derive(Debug, Clone)]
pub struct Taxonomy {
    names: Vec<String>,
    parent_ptr: Vec<u32>,
    parents: Vec<u32>,
    child_ptr: Vec<u32>,
    children: Vec<u32>,
}

fn csr(n: usize, mut pairs: Vec<(u32, u32)>) -> (Vec<u32>, Vec<u32>) {
    pairs.sort_unstable();
    pairs.dedup();
    let mut ptr = vec![0u32; n + 1];
    for &(a, _) in &pairs {
        ptr[a as usize + 1] += 1;
    }
    for i in 0..n {
        ptr[i + 1] += ptr[i];
    }
    (ptr, pairs.into_iter().map(|(_, b)| b).collect())
}

impl Taxonomy {
    /// Builds from class names and `(child, parent)` edges (names in edges are
    /// added automatically). Errors if the graph has a cycle.
    pub fn build<I, S>(names: I, edges: &[(String, String)]) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut set: BTreeSet<String> = names.into_iter().map(|s| s.as_ref().to_owned()).collect();
        for (c, p) in edges {
            set.insert(c.clone());
            set.insert(p.clone());
        }
        let names: Vec<String> = set.into_iter().collect();
        let id = |s: &str| {
            names
                .binary_search_by(|x| x.as_str().cmp(s))
                .expect("name present") as u32
        };
        let up: Vec<(u32, u32)> = edges.iter().map(|(c, p)| (id(c), id(p))).collect();
        let down: Vec<(u32, u32)> = up.iter().map(|&(c, p)| (p, c)).collect();
        let n = names.len();
        let (parent_ptr, parents) = csr(n, up);
        let (child_ptr, children) = csr(n, down);

        // Kahn: remove classes without unprocessed parents.
        let mut missing: Vec<u32> = (0..n).map(|c| parent_ptr[c + 1] - parent_ptr[c]).collect();
        let mut queue: VecDeque<u32> = (0..n as u32)
            .filter(|&c| missing[c as usize] == 0)
            .collect();
        let mut done = 0usize;
        while let Some(c) = queue.pop_front() {
            done += 1;
            for &k in &children[child_ptr[c as usize] as usize..child_ptr[c as usize + 1] as usize]
            {
                missing[k as usize] -= 1;
                if missing[k as usize] == 0 {
                    queue.push_back(k);
                }
            }
        }
        if done < n {
            let c = (0..n)
                .find(|&c| missing[c] > 0)
                .expect("some class is on a cycle");
            return Err(IndexError::Cycle(names[c].clone()));
        }
        Ok(Self {
            names,
            parent_ptr,
            parents,
            child_ptr,
            children,
        })
    }

    /// Number of classes.
    pub fn n_classes(&self) -> usize {
        self.names.len()
    }

    /// Id of a class name.
    pub fn id(&self, name: &str) -> Option<ClassId> {
        self.names
            .binary_search_by(|x| x.as_str().cmp(name))
            .ok()
            .map(|i| i as u32)
    }

    /// Name of class `c`.
    pub fn name(&self, c: ClassId) -> &str {
        &self.names[c as usize]
    }

    /// Direct parents of `c`.
    pub fn parents(&self, c: ClassId) -> &[u32] {
        &self.parents
            [self.parent_ptr[c as usize] as usize..self.parent_ptr[c as usize + 1] as usize]
    }

    /// Direct children of `c`.
    pub fn children(&self, c: ClassId) -> &[u32] {
        &self.children[self.child_ptr[c as usize] as usize..self.child_ptr[c as usize + 1] as usize]
    }

    /// Roots and all their (transitive) subclasses.
    pub fn closure(&self, roots: &[ClassId]) -> ClassSet {
        closure_over(self.n_classes(), &self.child_ptr, &self.children, roots)
    }

    /// Raw arrays `(names, parent_ptr, parents)` for serialization.
    pub fn parts(&self) -> (&[String], &[u32], &[u32]) {
        (&self.names, &self.parent_ptr, &self.parents)
    }

    /// Raw child CSR `(child_ptr, children)` for serialization.
    pub fn children_parts(&self) -> (&[u32], &[u32]) {
        (&self.child_ptr, &self.children)
    }
}

/// BFS closure of `roots` over a child CSR with `n` classes.
pub fn closure_over(n: usize, child_ptr: &[u32], children: &[u32], roots: &[ClassId]) -> ClassSet {
    let mut set = ClassSet::new(n);
    let mut queue: VecDeque<u32> = VecDeque::new();
    for &r in roots {
        if !set.contains(r) {
            set.insert(r);
            queue.push_back(r);
        }
    }
    while let Some(c) = queue.pop_front() {
        for &k in &children[child_ptr[c as usize] as usize..child_ptr[c as usize + 1] as usize] {
            if !set.contains(k) {
                set.insert(k);
                queue.push_back(k);
            }
        }
    }
    set
}
