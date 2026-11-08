//! Issue #8: SHACL schema (INV-SHACL-1) and taxonomy closure (INV-TAX-1).

use std::collections::BTreeSet;

use hg_index::schema::parse_schema;
use hg_index::taxonomy::Taxonomy;
use hg_index::IndexError;
use proptest::prelude::*;

const FIXTURE: &str = include_str!("fixtures/schema_sample.ttl");

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

#[test]
fn inv_shacl_1_fixture_classes() {
    let sc = parse_schema(FIXTURE).unwrap();
    assert_eq!(sc.classes.len(), 7);
    assert_eq!(
        sc.classes["schema:City"].supers,
        s(&["schema:AdministrativeArea"])
    );
    assert!(sc.classes["schema:Thing"].supers.is_empty());
    assert_eq!(
        sc.classes["schema:Person"].disjoint,
        s(&["schema:CreativeWork", "schema:Organization", "schema:Place"])
    );
}

#[test]
fn inv_shacl_1_fixture_properties() {
    let sc = parse_schema(FIXTURE).unwrap();
    assert_eq!(sc.properties.len(), 3);

    let cap = &sc.properties["yago:capital"];
    assert_eq!(cap.domain, s(&["schema:AdministrativeArea"]));
    assert_eq!(cap.range_classes, s(&["schema:City"]));
    assert_eq!(cap.max_count, None);

    let music = &sc.properties["schema:musicBy"];
    assert_eq!(music.domain, s(&["schema:CreativeWork"]));
    assert_eq!(
        music.range_classes,
        s(&["schema:Organization", "schema:Person"])
    );
    assert!(music.range_datatypes.is_empty());

    let end = &sc.properties["schema:endDate"];
    assert_eq!(end.domain, s(&["schema:CreativeWork", "schema:Place"]));
    assert!(end.range_classes.is_empty());
    assert_eq!(
        end.range_datatypes,
        s(&["xsd:date", "xsd:dateTime", "xsd:gYear", "xsd:gYearMonth"])
    );
    assert_eq!(end.max_count, Some(1));
}

#[test]
fn unbalanced_brackets_are_syntax_errors() {
    let bad = "ys:p sh:path yago:x ;\n\tsh:or ([ sh:class schema:A ][ sh:class schema:B ).\n";
    assert!(matches!(
        parse_schema(bad),
        Err(IndexError::Syntax { line: 2, .. })
    ));
}

/// Real yago-4.6 schema: run with HOLM_YAGO_SCHEMA=<path to yago-schema.ttl>.
#[test]
#[ignore = "needs the YAGO 4.6 schema file (HOLM_YAGO_SCHEMA)"]
fn real_yago_schema_counts() {
    let path = std::env::var("HOLM_YAGO_SCHEMA").expect("HOLM_YAGO_SCHEMA");
    let sc = parse_schema(&std::fs::read_to_string(path).unwrap()).unwrap();
    // Measured with grep on yago-4.6 (2026-08-03): 55 lines `X a sh:NodeShape ;`
    // (2 more lines describe sh:NodeShape itself), 156 distinct `sh:path` values.
    assert_eq!(sc.classes.len(), 55);
    assert_eq!(sc.properties.len(), 156);
    assert_eq!(
        sc.properties["yago:capital"].range_classes,
        s(&["schema:City"])
    );
}

fn hand_taxonomy() -> Taxonomy {
    // Thing <- Place <- City ; Thing <- Organization <- Company ; City also <- Organization
    let edges = vec![
        ("Place".to_string(), "Thing".to_string()),
        ("City".to_string(), "Place".to_string()),
        ("Organization".to_string(), "Thing".to_string()),
        ("Company".to_string(), "Organization".to_string()),
        ("City".to_string(), "Organization".to_string()),
    ];
    Taxonomy::build(["Thing", "Lonely"], &edges).unwrap()
}

fn names(t: &Taxonomy, ids: Vec<u32>) -> Vec<String> {
    ids.into_iter().map(|c| t.name(c).to_string()).collect()
}

#[test]
fn inv_tax_1_hand_checked_closure() {
    let t = hand_taxonomy();
    assert_eq!(t.n_classes(), 6);
    // ids are byte-order ranks: City, Company, Lonely, Organization, Place, Thing
    assert_eq!(t.id("City"), Some(0));
    assert_eq!(t.id("Thing"), Some(5));
    assert_eq!(t.id("Nope"), None);
    let place = t.id("Place").unwrap();
    let org = t.id("Organization").unwrap();
    assert_eq!(names(&t, t.closure(&[place]).ids()), s(&["City", "Place"]));
    assert_eq!(
        names(&t, t.closure(&[org]).ids()),
        s(&["City", "Company", "Organization"])
    );
    assert_eq!(
        t.closure(&[t.id("Thing").unwrap()]).count(),
        5,
        "all but Lonely"
    );
    assert_eq!(t.closure(&[]).count(), 0);
    assert_eq!(
        names(&t, t.parents(0).to_vec()),
        s(&["Organization", "Place"])
    );
    assert_eq!(names(&t, t.children(org).to_vec()), s(&["City", "Company"]));
}

#[test]
fn cycle_is_reported() {
    let edges = vec![
        ("A".to_string(), "B".to_string()),
        ("B".to_string(), "C".to_string()),
        ("C".to_string(), "A".to_string()),
        ("D".to_string(), "A".to_string()),
    ];
    assert!(matches!(
        Taxonomy::build(Vec::<String>::new(), &edges),
        Err(IndexError::Cycle(_))
    ));
}

/// Fixed-point oracle: repeatedly add every class whose parent is already in the set.
fn oracle_closure(n: usize, edges: &[(usize, usize)], roots: &[usize]) -> BTreeSet<usize> {
    let mut set: BTreeSet<usize> = roots.iter().copied().collect();
    loop {
        let before = set.len();
        for &(child, parent) in edges {
            if set.contains(&parent) {
                set.insert(child);
            }
        }
        if set.len() == before {
            return set.into_iter().filter(|&c| c < n).collect();
        }
    }
}

proptest! {
    /// INV-TAX-1: closure equals the fixed-point oracle on random DAGs.
    #[test]
    fn inv_tax_1_closure_matches_oracle(
        n in 1usize..40,
        raw_edges in proptest::collection::vec((0usize..40, 0usize..40), 0..80),
        raw_roots in proptest::collection::vec(0usize..40, 0..5),
    ) {
        // Acyclic by construction: child id > parent id. Names "c00".."c39" keep byte order = numeric order.
        let edges: Vec<(usize, usize)> = raw_edges
            .into_iter()
            .map(|(a, b)| (a % n, b % n))
            .filter(|(a, b)| a > b)
            .collect();
        let roots: Vec<usize> = raw_roots.into_iter().map(|r| r % n).collect();
        let name = |i: usize| format!("c{i:02}");
        let named: Vec<(String, String)> = edges.iter().map(|&(a, b)| (name(a), name(b))).collect();
        let t = Taxonomy::build((0..n).map(name), &named).unwrap();
        let root_ids: Vec<u32> = roots.iter().map(|&r| t.id(&name(r)).unwrap()).collect();
        let got: BTreeSet<usize> = t.closure(&root_ids).ids().into_iter().map(|c| c as usize).collect();
        prop_assert_eq!(got, oracle_closure(n, &edges, &roots));
    }
}
