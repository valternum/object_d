//! Index builder: YAGO 4.6 Turtle files → one `.holmidx` container.
//!
//! Passes (each streams the files once):
//! 1. taxonomy + facts: class names, `subClassOf` edges, entity terms
//!    (external-sort dictionary), relation names;
//! 2. facts: types, priorities (`yago:siteLinks`), Wikidata QIDs,
//!    entity–entity edges (sorted, deduplicated: canonical edge ids);
//! 3. meta: validity intervals of edges.
//!
//! The hypergraph is then renumbered in BFS order (locality) and written with
//! all side tables. The output is byte-identical for identical inputs
//! (INV-IDX-1). Peak memory: dictionary chunk + O(nodes + edges).

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use hg_core::attrs::{EdgeIntervals, NodeTypes};
use hg_core::container::{ContainerWriter, Payload};
use hg_core::graph::HypergraphBuilder;
use hg_core::store::write_hypergraph;

use crate::dict::{Dict, DictBuilder};
use crate::error::{IndexError, Result};
use crate::lex::{for_each_statement, Statement, Term};
use crate::renumber::{bfs_order, relabel};
use crate::schema::{parse_schema, Schema};
use crate::taxonomy::Taxonomy;

/// Index format version stored in the `info` section.
pub const INDEX_FORMAT: u64 = 1;

/// Predicate conventions of the input (YAGO 4.6 defaults in [`Predicates::yago46`]).
#[derive(Debug, Clone)]
pub struct Predicates {
    /// Class membership.
    pub type_of: String,
    /// Class hierarchy.
    pub subclass_of: String,
    /// Node priority (integer literal).
    pub priority: String,
    /// Wikidata link (`owl:sameAs wd:Q…`).
    pub same_as: String,
    /// Predicates that are never graph edges (links, labels, media).
    pub non_edges: Vec<String>,
    /// Objects of `type_of` that mark the subject as a class.
    pub class_markers: Vec<String>,
    /// Meta predicates giving an interval start / end / single point.
    pub start: Vec<String>,
    /// See `start`.
    pub end: Vec<String>,
    /// See `start`.
    pub point: Vec<String>,
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

impl Predicates {
    /// Conventions of YAGO 4.6 (checked against the 4.6 schema and samples).
    pub fn yago46() -> Self {
        Self {
            type_of: "rdf:type".into(),
            subclass_of: "rdfs:subClassOf".into(),
            priority: "yago:siteLinks".into(),
            same_as: "owl:sameAs".into(),
            non_edges: strings(&[
                "owl:sameAs",
                "schema:sameAs",
                "schema:url",
                "schema:image",
                "schema:logo",
                "schema:mainEntityOfPage",
                "rdfs:label",
                "rdfs:comment",
                "schema:alternateName",
            ]),
            class_markers: strings(&["rdfs:Class", "owl:Class", "sh:NodeShape"]),
            start: strings(&["schema:startDate"]),
            end: strings(&["schema:endDate"]),
            point: strings(&["yago:onDate"]),
        }
    }
}

/// Inputs and parameters of a build.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// SHACL schema file.
    pub schema: Option<PathBuf>,
    /// Taxonomy files.
    pub taxonomy: Vec<PathBuf>,
    /// Fact files.
    pub facts: Vec<PathBuf>,
    /// Meta-fact files.
    pub meta: Vec<PathBuf>,
    /// Output `.holmidx` path.
    pub out: PathBuf,
    /// Directory for temporary dictionary runs.
    pub tmp_dir: PathBuf,
    /// Dictionary chunk size in bytes.
    pub chunk_bytes: usize,
    /// Predicate conventions.
    pub predicates: Predicates,
}

/// Counts reported by a build.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BuildStats {
    /// Nodes (entities).
    pub nodes: usize,
    /// Distinct entity–entity edges.
    pub edges: usize,
    /// Relation types.
    pub relations: usize,
    /// Classes.
    pub classes: usize,
    /// Distinct (node, class) pairs.
    pub type_pairs: usize,
    /// Nodes with at least one type.
    pub typed_nodes: usize,
    /// Edges with a validity interval.
    pub intervals: usize,
    /// Duplicate entity–entity facts removed.
    pub duplicate_edges: usize,
    /// Nodes with a Wikidata QID.
    pub qids: usize,
    /// Meta facts that do not annotate an entity–entity edge.
    pub ignored_meta: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MetaKind {
    Start,
    End,
    Point,
}

/// Collected `(start, end, point)` years of one edge.
type YearBounds = (Option<i32>, Option<i32>, Option<i32>);

fn parse_uint(lex: &str) -> Option<u32> {
    let digits = lex.trim_start_matches('+').split('.').next()?;
    digits
        .parse::<u64>()
        .ok()
        .map(|x| x.min(u64::from(u32::MAX)) as u32)
}

fn parse_qid(iri: &str) -> Option<u32> {
    let rest = iri.strip_prefix("wd:Q").or_else(|| {
        iri.strip_prefix("<http://www.wikidata.org/entity/Q")
            .and_then(|r| r.strip_suffix('>'))
    })?;
    rest.parse().ok()
}

/// Year of an xsd date/time lexical form: optional sign, leading digits.
fn parse_year(lex: &str) -> Option<i32> {
    let (sign, rest) = match lex.as_bytes().first() {
        Some(b'-') => (-1i64, &lex[1..]),
        Some(b'+') => (1, &lex[1..]),
        _ => (1, lex),
    };
    let digits: &str = &rest[..rest.bytes().take_while(u8::is_ascii_digit).count()];
    if digits.is_empty() {
        return None;
    }
    let y: i64 = digits.parse().ok()?;
    i32::try_from(sign * y).ok()
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_paths(v: &[PathBuf]) -> String {
    let items: Vec<String> = v
        .iter()
        .map(|p| json_str(&p.display().to_string()))
        .collect();
    format!("[{}]", items.join(", "))
}

/// Builds the index described by `cfg`.
pub fn build_index(cfg: &BuildConfig) -> Result<BuildStats> {
    let pr = &cfg.predicates;
    let non_edges: HashSet<&str> = pr.non_edges.iter().map(String::as_str).collect();
    let markers: HashSet<&str> = pr.class_markers.iter().map(String::as_str).collect();
    let kind_of = |p: &str| -> Option<MetaKind> {
        if pr.start.iter().any(|x| x == p) {
            Some(MetaKind::Start)
        } else if pr.end.iter().any(|x| x == p) {
            Some(MetaKind::End)
        } else if pr.point.iter().any(|x| x == p) {
            Some(MetaKind::Point)
        } else {
            None
        }
    };

    // ---- schema and pass 1 -------------------------------------------------
    let schema: Schema = match &cfg.schema {
        Some(p) => parse_schema(&std::fs::read_to_string(p)?)?,
        None => Schema::default(),
    };
    let mut classes: BTreeSet<String> = schema.classes.keys().cloned().collect();
    let mut tax_edges: Vec<(String, String)> = Vec::new();
    for (c, shape) in &schema.classes {
        for s in &shape.supers {
            tax_edges.push((c.clone(), s.clone()));
            classes.insert(s.clone());
        }
    }
    let mut on_class_triple = |s: &str, p: &str, o: &Term<'_>| {
        if p == pr.subclass_of {
            if let Term::Iri(parent) = o {
                tax_edges.push((s.to_owned(), (*parent).to_owned()));
                classes.insert(s.to_owned());
                classes.insert((*parent).to_owned());
            }
            true
        } else if p == pr.type_of {
            if let Term::Iri(t) = o {
                if markers.contains(t) {
                    classes.insert(s.to_owned());
                } else {
                    classes.insert((*t).to_owned());
                }
            }
            false
        } else {
            false
        }
    };
    for f in &cfg.taxonomy {
        for_each_statement(f, |st| {
            if let Statement::Triple {
                s: Term::Iri(s),
                p: Term::Iri(p),
                o,
            } = st
            {
                on_class_triple(s, p, o);
            }
            Ok(())
        })?;
    }
    let mut dict = DictBuilder::new(&cfg.tmp_dir, cfg.chunk_bytes);
    let mut rel_names: BTreeSet<String> = BTreeSet::new();
    for f in &cfg.facts {
        for_each_statement(f, |st| {
            let Statement::Triple {
                s: Term::Iri(s),
                p: Term::Iri(p),
                o,
            } = st
            else {
                return Ok(());
            };
            if on_class_triple(s, p, o) {
                return Ok(());
            }
            if *p == pr.type_of {
                if let Term::Iri(t) = o {
                    if !markers.contains(t) {
                        dict.add(s)?;
                    }
                }
                return Ok(());
            }
            dict.add(s)?;
            if non_edges.contains(p) {
                return Ok(());
            }
            if let Term::Iri(obj) = o {
                dict.add(obj)?;
                if !rel_names.contains(*p) {
                    rel_names.insert((*p).to_owned());
                }
            }
            Ok(())
        })?;
    }
    let nodes = dict.finish()?.filter(|t| !classes.contains(t));
    let n = nodes.len();
    if n > u32::MAX as usize {
        return Err(IndexError::Invalid("more than u32::MAX nodes".into()));
    }
    if rel_names.len() > usize::from(u16::MAX) {
        return Err(IndexError::Invalid("more than u16::MAX relations".into()));
    }
    let rels = Dict::from_terms(&rel_names);
    let taxonomy = Taxonomy::build(classes.iter(), &tax_edges)?;

    // ---- pass 2: facts -----------------------------------------------------
    let mut prio = vec![0u32; n];
    let mut qid = vec![0u32; n];
    let mut type_pairs: Vec<(u32, u32)> = Vec::new();
    let mut edges: Vec<(u16, u32, u32)> = Vec::new();
    let mut meta_raw: Vec<((u16, u32, u32), MetaKind, i32)> = Vec::new();
    let mut ignored_meta = 0usize;
    let mut on_meta = |s: &Term<'_>, p: &Term<'_>, o: &Term<'_>, mp: &Term<'_>, mo: &Term<'_>| {
        let key = match (s, p, o) {
            (Term::Iri(s), Term::Iri(p), Term::Iri(o)) => {
                match (nodes.lookup(s), rels.lookup(p), nodes.lookup(o)) {
                    (Some(s), Some(r), Some(o)) => Some((r as u16, s, o)),
                    _ => None,
                }
            }
            _ => None,
        };
        let kind = match mp {
            Term::Iri(mp) => kind_of(mp),
            Term::Literal { .. } => None,
        };
        let year = match mo {
            Term::Literal { lex, .. } => parse_year(lex),
            Term::Iri(_) => None,
        };
        match (key, kind, year) {
            (Some(key), Some(kind), Some(year)) => meta_raw.push((key, kind, year)),
            _ => ignored_meta += 1,
        }
    };
    for f in &cfg.facts {
        for_each_statement(f, |st| {
            match st {
                Statement::Triple {
                    s: Term::Iri(s),
                    p: Term::Iri(p),
                    o,
                } => {
                    let Some(sv) = nodes.lookup(s) else {
                        return Ok(());
                    };
                    if *p == pr.type_of {
                        if let Term::Iri(t) = o {
                            if let Some(c) = taxonomy.id(t) {
                                type_pairs.push((sv, c));
                            }
                        }
                    } else if *p == pr.priority {
                        if let Term::Literal { lex, .. } = o {
                            if let Some(x) = parse_uint(lex) {
                                prio[sv as usize] = x;
                            }
                        }
                    } else if *p == pr.same_as {
                        if let Term::Iri(t) = o {
                            if let Some(q) = parse_qid(t) {
                                qid[sv as usize] = q;
                            }
                        }
                    } else if !non_edges.contains(p) {
                        if let Term::Iri(t) = o {
                            if let (Some(ov), Some(r)) = (nodes.lookup(t), rels.lookup(p)) {
                                edges.push((r as u16, sv, ov));
                            }
                        }
                    }
                }
                Statement::Meta { s, p, o, mp, mo } => on_meta(s, p, o, mp, mo),
                Statement::Triple { .. } => {}
            }
            Ok(())
        })?;
    }
    // ---- pass 3: meta ------------------------------------------------------
    for f in &cfg.meta {
        for_each_statement(f, |st| {
            if let Statement::Meta { s, p, o, mp, mo } = st {
                on_meta(s, p, o, mp, mo);
            }
            Ok(())
        })?;
    }
    edges.sort_unstable();
    let before = edges.len();
    edges.dedup();
    let duplicate_edges = before - edges.len();
    if edges.len() > u32::MAX as usize {
        return Err(IndexError::Invalid("more than u32::MAX edges".into()));
    }

    let mut per_edge: BTreeMap<u32, YearBounds> = BTreeMap::new();
    for (key, kind, year) in meta_raw {
        let Ok(e) = edges.binary_search(&key) else {
            ignored_meta += 1;
            continue;
        };
        let slot = per_edge.entry(e as u32).or_default();
        let field = match kind {
            MetaKind::Start => &mut slot.0,
            MetaKind::End => &mut slot.1,
            MetaKind::Point => &mut slot.2,
        };
        *field = Some(match *field {
            None => year,
            Some(y) if kind == MetaKind::End => y.max(year),
            Some(y) => y.min(year),
        });
    }
    let intervals = EdgeIntervals::from_triples(
        per_edge
            .into_iter()
            .map(|(e, (s, t, p))| (e, s.or(p).unwrap_or(i32::MIN), t.or(p).unwrap_or(i32::MAX)))
            .collect(),
    );

    // ---- graph, renumbering ------------------------------------------------
    let mut b = HypergraphBuilder::new(n as u32);
    for (v, &p) in prio.iter().enumerate() {
        b.set_node_prio(v as u32, p)?;
    }
    for &(r, s, o) in &edges {
        b.add_edge(r, &[(s, 0), (o, 1)])?;
    }
    let g0 = b.build();
    let perm = bfs_order(&g0);
    let graph = relabel(&g0, &perm);
    drop(g0);

    let typed_nodes = {
        let mut v: Vec<u32> = type_pairs.iter().map(|&(v, _)| v).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    let types = NodeTypes::from_pairs(
        n,
        type_pairs
            .iter()
            .map(|&(v, c)| (perm[v as usize], c))
            .collect(),
    )?;
    let mut qid_new = vec![0u32; n];
    let mut term_of = vec![0u32; n];
    for v in 0..n {
        qid_new[perm[v] as usize] = qid[v];
        term_of[perm[v] as usize] = v as u32;
    }
    let of_term = perm;

    // ---- schema tables per relation ------------------------------------------
    let (mut dptr, mut dom, mut rptr, mut rng, mut maxc) =
        (vec![0u32], Vec::new(), vec![0u32], Vec::new(), Vec::new());
    for r in 0..rels.len() as u32 {
        let info = schema.properties.get(rels.get(r));
        if let Some(info) = info {
            dom.extend(info.domain.iter().filter_map(|c| taxonomy.id(c)));
            rng.extend(info.range_classes.iter().filter_map(|c| taxonomy.id(c)));
        }
        maxc.push(info.and_then(|i| i.max_count).unwrap_or(0));
        dptr.push(dom.len() as u32);
        rptr.push(rng.len() as u32);
    }

    // ---- write ---------------------------------------------------------------
    let class_dict = Dict::from_terms(taxonomy.parts().0);
    let (_, pptr, par) = taxonomy.parts();
    let (cptr, ch) = taxonomy.children_parts();
    let stats = BuildStats {
        nodes: n,
        edges: edges.len(),
        relations: rels.len(),
        classes: taxonomy.n_classes(),
        type_pairs: type_pairs.len(),
        typed_nodes,
        intervals: intervals.len(),
        duplicate_edges,
        qids: qid_new.iter().filter(|&&q| q != 0).count(),
        ignored_meta,
    };
    {
        let mut w = ContainerWriter::new();
        w.add(
            "info",
            Payload::OwnedU64(vec![
                INDEX_FORMAT,
                n as u64,
                edges.len() as u64,
                rels.len() as u64,
                taxonomy.n_classes() as u64,
            ]),
        )?;
        write_hypergraph(&mut w, "g", &graph)?;
        nodes.write(&mut w, "nodes")?;
        w.add("nodes.of_term", Payload::U32(&of_term))?;
        w.add("nodes.term_of", Payload::U32(&term_of))?;
        w.add("nodes.qid", Payload::U32(&qid_new))?;
        rels.write(&mut w, "rels")?;
        class_dict.write(&mut w, "classes")?;
        w.add("tax.pptr", Payload::U32(pptr))?;
        w.add("tax.par", Payload::U32(par))?;
        w.add("tax.cptr", Payload::U32(cptr))?;
        w.add("tax.ch", Payload::U32(ch))?;
        types.write(&mut w, "types")?;
        intervals.write(&mut w, "iv")?;
        w.add("sch.dptr", Payload::U32(&dptr))?;
        w.add("sch.dom", Payload::U32(&dom))?;
        w.add("sch.rptr", Payload::U32(&rptr))?;
        w.add("sch.rng", Payload::U32(&rng))?;
        w.add("sch.maxc", Payload::U32(&maxc))?;
        w.write_to(&cfg.out)?;
    }

    let manifest = format!(
        "{{\n  \"format\": {INDEX_FORMAT},\n  \"index\": {},\n  \"index_bytes\": {},\n  \"nodes\": {},\n  \"edges\": {},\n  \"relations\": {},\n  \"classes\": {},\n  \"type_pairs\": {},\n  \"typed_nodes\": {},\n  \"intervals\": {},\n  \"duplicate_edges\": {},\n  \"qids\": {},\n  \"ignored_meta\": {},\n  \"chunk_bytes\": {},\n  \"inputs\": {{\n    \"schema\": {},\n    \"taxonomy\": {},\n    \"facts\": {},\n    \"meta\": {}\n  }}\n}}\n",
        json_str(&cfg.out.display().to_string()),
        std::fs::metadata(&cfg.out)?.len(),
        stats.nodes,
        stats.edges,
        stats.relations,
        stats.classes,
        stats.type_pairs,
        stats.typed_nodes,
        stats.intervals,
        stats.duplicate_edges,
        stats.qids,
        stats.ignored_meta,
        cfg.chunk_bytes,
        cfg.schema.as_ref().map_or("null".to_owned(), |p| json_str(&p.display().to_string())),
        json_paths(&cfg.taxonomy),
        json_paths(&cfg.facts),
        json_paths(&cfg.meta),
    );
    std::fs::write(cfg.out.with_extension("manifest.json"), manifest)?;
    Ok(stats)
}
