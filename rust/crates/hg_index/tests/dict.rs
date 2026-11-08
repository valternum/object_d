//! Issue #7: external-sort term dictionary (INV-DICT-1, INV-DICT-2).

use hg_core::container::{Container, ContainerWriter, Verify};
use hg_index::dict::{Dict, DictBuilder};
use proptest::prelude::*;

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("holm_dict_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn build(terms: &[&str], chunk_bytes: usize, tag: &str) -> Dict<'static> {
    let dir = tmp_dir(tag);
    let mut b = DictBuilder::new(&dir, chunk_bytes);
    for t in terms {
        b.add(t).unwrap();
    }
    let d = b.finish().unwrap();
    // finish() must clean up its run files.
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        0,
        "run files left behind"
    );
    std::fs::remove_dir(&dir).unwrap();
    d
}

#[test]
fn ids_are_ranks_in_byte_order() {
    let d = build(
        &["yago:b", "yago:a", "schema:Person", "yago:a", "yago:Ä"],
        1 << 20,
        "rank",
    );
    assert_eq!(d.len(), 4);
    // Byte order: "schema:Person" < "yago:a" < "yago:b" < "yago:Ä" (0xC3 > 'b').
    assert_eq!(d.get(0), "schema:Person");
    assert_eq!(d.get(1), "yago:a");
    assert_eq!(d.get(2), "yago:b");
    assert_eq!(d.get(3), "yago:Ä");
    assert_eq!(d.lookup("yago:b"), Some(2));
    assert_eq!(d.lookup("yago:c"), None);
    assert_eq!(d.lookup(""), None);
}

#[test]
fn tiny_chunks_spill_many_runs_and_merge_duplicates_once() {
    let terms = ["c", "a", "b", "a", "c", "d", "a"];
    let dir = tmp_dir("spill");
    let mut b = DictBuilder::new(&dir, 1);
    for t in terms {
        b.add(t).unwrap();
    }
    assert!(b.runs() >= 3, "expected several runs, got {}", b.runs());
    let d = b.finish().unwrap();
    std::fs::remove_dir(&dir).unwrap();
    assert_eq!(
        (0..d.len() as u32).map(|i| d.get(i)).collect::<Vec<_>>(),
        vec!["a", "b", "c", "d"]
    );
}

#[test]
fn empty_dictionary() {
    let d = build(&[], 16, "empty");
    assert!(d.is_empty());
    assert_eq!(d.lookup("x"), None);
}

#[test]
fn container_round_trip() {
    let d = Dict::from_terms(["x:1", "x:2", "y:longer term with spaces", "ä"]);
    let path = std::env::temp_dir().join(format!("holm_dict_rt_{}.holmidx", std::process::id()));
    let mut w = ContainerWriter::new();
    d.write(&mut w, "nodes").unwrap();
    w.write_to(&path).unwrap();
    let c = Container::open(&path, Verify::Full).unwrap();
    let back = Dict::read(&c, "nodes").unwrap();
    assert_eq!(back.len(), 4);
    for i in 0..4u32 {
        assert_eq!(back.get(i), d.get(i));
        assert_eq!(back.lookup(d.get(i)), Some(i));
    }
    drop(c);
    std::fs::remove_file(&path).unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// INV-DICT-1: bijection between distinct terms and ids.
    #[test]
    fn inv_dict_1_bijection(terms in proptest::collection::vec("[a-z:]{0,6}", 0..80)) {
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let d = build(&refs, 64, "bij");
        let mut want: Vec<&str> = refs.clone();
        want.sort_unstable();
        want.dedup();
        prop_assert_eq!(d.len(), want.len());
        for (i, t) in want.iter().enumerate() {
            prop_assert_eq!(d.lookup(t), Some(i as u32));
            prop_assert_eq!(d.get(i as u32), *t);
        }
    }

    /// INV-DICT-2: the result does not depend on the chunk size.
    #[test]
    fn inv_dict_2_chunk_size_invariance(terms in proptest::collection::vec("[a-c]{1,4}", 1..60)) {
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let one_run = build(&refs, 1 << 20, "big");
        let many_runs = build(&refs, 3, "small");
        prop_assert_eq!(one_run.blob(), many_runs.blob());
        prop_assert_eq!(one_run.len(), many_runs.len());
    }
}
