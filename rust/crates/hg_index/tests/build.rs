//! Issue #10: index builder, index view and CLI on the hand-written mini world.

use std::path::{Path, PathBuf};
use std::process::Command;

use hg_core::container::{Container, Verify};
use hg_core::graph::RelSet;
use hg_index::build::{build_index, BuildConfig, BuildStats, Predicates};
use hg_index::index::IndexView;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn out_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("holm_build_{tag}_{}.holmidx", std::process::id()))
}

fn config(out: PathBuf) -> BuildConfig {
    BuildConfig {
        schema: Some(fixture("schema_sample.ttl")),
        taxonomy: vec![fixture("mini/taxonomy.ttl")],
        facts: vec![fixture("mini/facts.ttl")],
        meta: vec![fixture("mini/meta.ttl")],
        out,
        tmp_dir: std::env::temp_dir(),
        chunk_bytes: 64,
        predicates: Predicates::yago46(),
    }
}

#[test]
fn mini_world_counts() {
    let out = out_path("counts");
    let stats = build_index(&config(out.clone())).unwrap();
    assert_eq!(
        stats,
        BuildStats {
            nodes: 6,     // France, Paris, Lyon, Daft_Punk, Around_the_World, Bahraini_dinar
            edges: 4,     // capital, 2 × location, musicBy
            relations: 3, // yago:capital, schema:location, schema:musicBy
            classes: 9,   // 7 schema classes + yago:Capital_city + yago:UnitOfMeasurement
            type_pairs: 6,
            typed_nodes: 6,
            intervals: 1,       // Lyon location France [1951, 2002]
            duplicate_edges: 1, // Paris location France listed twice
            qids: 2,
            ignored_meta: 2, // literal fact + unknown edge
        }
    );
    std::fs::remove_file(&out).unwrap();
    let _ = std::fs::remove_file(out.with_extension("manifest.json"));
}

#[test]
fn index_view_answers_from_the_mmap() {
    let out = out_path("view");
    build_index(&config(out.clone())).unwrap();
    let c = Container::open(&out, Verify::Full).unwrap();
    let ix = IndexView::new(&c).unwrap();

    for term in [
        "yago:France",
        "yago:Paris",
        "yago:Lyon",
        "yago:Daft_Punk",
        "yago:Around_the_World",
        "yago:Bahraini_dinar",
    ] {
        let v = ix.node_id(term).unwrap_or_else(|| panic!("{term} missing"));
        assert_eq!(ix.node_term(v), term);
    }
    assert_eq!(ix.node_id("wd:Q142"), None, "link objects are not nodes");
    assert_eq!(ix.node_id("schema:City"), None, "classes are not nodes");
    // France has the highest priority (siteLinks 300): BFS renumbering gives it id 0.
    assert_eq!(ix.node_id("yago:France"), Some(0));
    let france = ix.node_id("yago:France").unwrap();
    let paris = ix.node_id("yago:Paris").unwrap();
    let lyon = ix.node_id("yago:Lyon").unwrap();
    assert_eq!(ix.graph.node_prio(france), 300);
    assert_eq!(ix.graph.node_prio(paris), 250);
    assert_eq!(ix.qid(france), 142);
    assert_eq!(ix.qid(lyon), 0);

    let loc = ix.relation_id("schema:location").unwrap();
    let cap = ix.relation_id("yago:capital").unwrap();
    assert_eq!(ix.n_relations(), 3);
    assert_eq!(ix.relation_name(loc), "schema:location");
    let e = ix.find_edge(lyon, loc, france).unwrap();
    assert_eq!(ix.intervals.get(e), Some((1951, 2002)));
    let e2 = ix.find_edge(paris, loc, france).unwrap();
    assert_eq!(ix.intervals.get(e2), None);
    assert!(
        ix.find_edge(france, loc, paris).is_none(),
        "direction (roles) matters"
    );
    assert!(ix.find_edge(france, cap, paris).is_some());
    let mut out_inc = Vec::new();
    ix.graph
        .top_incident(france, &RelSet::all(), usize::MAX, &mut out_inc);
    assert_eq!(out_inc.len(), 3);

    let place = ix.class_id("schema:Place").unwrap();
    let mut closure: Vec<&str> = ix
        .class_closure(&[place])
        .ids()
        .into_iter()
        .map(|c| ix.class_name(c))
        .collect();
    closure.sort_unstable();
    assert_eq!(
        closure,
        vec![
            "schema:AdministrativeArea",
            "schema:City",
            "schema:Place",
            "yago:Capital_city"
        ]
    );
    assert_eq!(ix.n_classes(), 9);
    let capital_city = ix.class_id("yago:Capital_city").unwrap();
    assert_eq!(ix.types.types(paris), &[capital_city]);

    let city = ix.class_id("schema:City").unwrap();
    assert_eq!(ix.range(cap), &[city]);
    let music = ix.relation_id("schema:musicBy").unwrap();
    assert_eq!(
        ix.domain(music),
        &[ix.class_id("schema:CreativeWork").unwrap()]
    );
    drop(ix);
    drop(c);
    std::fs::remove_file(&out).unwrap();
    let _ = std::fs::remove_file(out.with_extension("manifest.json"));
}

#[test]
fn inv_idx_1_builds_are_byte_identical() {
    let (a, b) = (out_path("det_a"), out_path("det_b"));
    build_index(&config(a.clone())).unwrap();
    let mut cfg = config(b.clone());
    cfg.chunk_bytes = 1 << 20; // a different spill pattern must not change the result
    build_index(&cfg).unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    for p in [&a, &b] {
        std::fs::remove_file(p).unwrap();
        let _ = std::fs::remove_file(p.with_extension("manifest.json"));
    }
}

#[test]
fn cli_build_and_stats() {
    let out = out_path("cli");
    let bin = env!("CARGO_BIN_EXE_holm-index");
    let status = Command::new(bin)
        .arg("build")
        .arg("--schema")
        .arg(fixture("schema_sample.ttl"))
        .arg("--taxonomy")
        .arg(fixture("mini/taxonomy.ttl"))
        .arg("--facts")
        .arg(fixture("mini/facts.ttl"))
        .arg("--meta")
        .arg(fixture("mini/meta.ttl"))
        .arg("--out")
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    assert!(out.with_extension("manifest.json").exists());
    let stats = Command::new(bin).arg("stats").arg(&out).output().unwrap();
    assert!(stats.status.success());
    let text = String::from_utf8(stats.stdout).unwrap();
    assert!(text.contains("nodes 6"), "{text}");
    assert!(text.contains("edges 4"), "{text}");
    let bad = Command::new(bin)
        .arg("build")
        .arg("--out")
        .output()
        .unwrap();
    assert!(!bad.status.success(), "missing value must fail");
    std::fs::remove_file(&out).unwrap();
    std::fs::remove_file(out.with_extension("manifest.json")).unwrap();
}
