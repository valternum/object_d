//! Issue #5: binary container, zero-copy mmap, checksums, INV-CORE-2.

mod common;

use common::TempPath;
use hg_core::container::{Container, ContainerWriter, Payload, Verify, MAX_NAME};
use hg_core::ef::EliasFano;
use hg_core::graph::{Hypergraph, HypergraphBuilder, NodeId, RelSet, Role};
use hg_core::hash::fnv1a64;
use hg_core::store::{read_ef, read_hypergraph, write_ef, write_hypergraph};
use hg_core::CoreError;
use proptest::prelude::*;

#[test]
fn fnv1a_reference_vectors() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
}

fn write_sample(path: &std::path::Path) {
    let bytes: Vec<u8> = (0..=255u8).collect();
    let halves: Vec<u16> = vec![1, 65535, 7];
    let words: Vec<u32> = (0..1000u32).map(|i| i * 3).collect();
    let longs: Vec<u64> = vec![u64::MAX, 0, 42];
    let empty: Vec<u32> = Vec::new();
    let mut w = ContainerWriter::new();
    w.add("bytes", Payload::U8(&bytes)).unwrap();
    w.add("halves", Payload::U16(&halves)).unwrap();
    w.add("words", Payload::U32(&words)).unwrap();
    w.add("longs", Payload::U64(&longs)).unwrap();
    w.add("empty", Payload::U32(&empty)).unwrap();
    w.write_to(path).unwrap();
}

#[test]
fn inv_core_2_sections_round_trip_zero_copy() {
    let tmp = TempPath::new("roundtrip");
    write_sample(tmp.path());
    let c = Container::open(tmp.path(), Verify::Full).unwrap();
    assert_eq!(
        c.names(),
        vec!["bytes", "halves", "words", "longs", "empty"]
    );
    assert_eq!(
        c.bytes("bytes").unwrap(),
        (0..=255u8).collect::<Vec<_>>().as_slice()
    );
    assert_eq!(c.u16s("halves").unwrap(), &[1, 65535, 7]);
    let words = c.u32s("words").unwrap();
    assert_eq!(words.len(), 1000);
    assert_eq!(words[999], 2997);
    assert_eq!(words.as_ptr() as usize % 8, 0, "views must be aligned");
    assert_eq!(c.u64s("longs").unwrap(), &[u64::MAX, 0, 42]);
    assert!(c.u32s("empty").unwrap().is_empty());
}

#[test]
fn flipped_byte_is_a_checksum_error() {
    let tmp = TempPath::new("corrupt");
    write_sample(tmp.path());
    let c = Container::open(tmp.path(), Verify::Header).unwrap();
    let target =
        c.u32s("words").unwrap().as_ptr() as usize - c.bytes("bytes").unwrap().as_ptr() as usize;
    let base = {
        let raw = std::fs::read(tmp.path()).unwrap();
        // offset of the "bytes" section = file position of its first byte (0..=255 pattern)
        raw.windows(256)
            .position(|w| w == (0..=255u8).collect::<Vec<_>>().as_slice())
            .unwrap()
    };
    drop(c);
    let mut raw = std::fs::read(tmp.path()).unwrap();
    raw[base + target + 5] ^= 0x10;
    std::fs::write(tmp.path(), &raw).unwrap();
    match Container::open(tmp.path(), Verify::Full) {
        Err(CoreError::Checksum(name)) => assert_eq!(name, "words"),
        other => panic!("expected checksum error, got {other:?}"),
    }
    assert!(Container::open(tmp.path(), Verify::Header).is_ok());
}

#[test]
fn wrong_magic_version_and_truncation_are_reported() {
    let tmp = TempPath::new("header");
    write_sample(tmp.path());
    let good = std::fs::read(tmp.path()).unwrap();

    let mut bad = good.clone();
    bad[0] = b'X';
    std::fs::write(tmp.path(), &bad).unwrap();
    assert!(matches!(
        Container::open(tmp.path(), Verify::Header),
        Err(CoreError::BadMagic)
    ));

    let mut bad = good.clone();
    bad[8] = 99;
    std::fs::write(tmp.path(), &bad).unwrap();
    assert!(matches!(
        Container::open(tmp.path(), Verify::Header),
        Err(CoreError::BadVersion { found: 99, .. })
    ));

    std::fs::write(tmp.path(), &good[..good.len() - 10]).unwrap();
    assert!(matches!(
        Container::open(tmp.path(), Verify::Header),
        Err(CoreError::Truncated(_))
    ));

    std::fs::write(tmp.path(), &good[..20]).unwrap();
    assert!(matches!(
        Container::open(tmp.path(), Verify::Header),
        Err(CoreError::Truncated(_))
    ));
}

#[test]
fn missing_section_and_type_mismatch() {
    let tmp = TempPath::new("missing");
    write_sample(tmp.path());
    let c = Container::open(tmp.path(), Verify::Header).unwrap();
    assert!(matches!(c.u32s("nope"), Err(CoreError::MissingSection(_))));
    // 3 u16 = 6 bytes: not a whole number of u32.
    assert!(c.u32s("halves").is_err());
}

#[test]
fn writer_rejects_duplicate_and_long_names() {
    let data = [1u32];
    let mut w = ContainerWriter::new();
    w.add("a", Payload::U32(&data)).unwrap();
    assert!(w.add("a", Payload::U32(&data)).is_err());
    assert!(w
        .add(&"x".repeat(MAX_NAME + 1), Payload::U32(&data))
        .is_err());
}

#[test]
fn elias_fano_round_trip() {
    let tmp = TempPath::new("ef");
    let values: Vec<u64> = (0..5000u64).map(|i| i * 13 + i % 5).collect();
    let ef = EliasFano::from_sorted(&values).unwrap();
    let mut w = ContainerWriter::new();
    write_ef(&mut w, "seq", &ef).unwrap();
    w.write_to(tmp.path()).unwrap();
    let c = Container::open(tmp.path(), Verify::Full).unwrap();
    let back = read_ef(&c, "seq").unwrap();
    assert_eq!(back.iter().collect::<Vec<_>>(), values);
}

/// `(n_nodes, node priorities, edges as (relation, members))`.
type RandomGraph = (u32, Vec<u32>, Vec<(u16, Vec<(NodeId, Role)>)>);

fn random_graph() -> impl Strategy<Value = RandomGraph> {
    (1u32..25).prop_flat_map(|n| {
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

fn same_answers(a: &Hypergraph<'_>, b: &Hypergraph<'_>) {
    assert_eq!(a.n_nodes(), b.n_nodes());
    assert_eq!(a.n_edges(), b.n_edges());
    for v in 0..a.n_nodes() as u32 {
        assert_eq!(
            a.incidences(v).collect::<Vec<_>>(),
            b.incidences(v).collect::<Vec<_>>()
        );
        assert_eq!(a.runs(v), b.runs(v));
        assert_eq!(a.node_prio(v), b.node_prio(v));
        for delta in [1usize, 2, usize::MAX] {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            a.top_incident(v, &RelSet::all(), delta, &mut x);
            b.top_incident(v, &RelSet::all(), delta, &mut y);
            assert_eq!(x, y);
        }
    }
    for e in 0..a.n_edges() as u32 {
        assert_eq!(a.edge_members(e), b.edge_members(e));
        assert_eq!(a.edge_rel(e), b.edge_rel(e));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    /// INV-CORE-2: a hypergraph written and reopened through mmap answers identically.
    #[test]
    fn inv_core_2_hypergraph_round_trip((n, prio, edges) in random_graph()) {
        let mut b = HypergraphBuilder::new(n);
        for (v, p) in prio.iter().enumerate() {
            b.set_node_prio(v as u32, *p).unwrap();
        }
        for (rel, m) in &edges {
            b.add_edge(*rel, m).unwrap();
        }
        let g = b.build();
        let tmp = TempPath::new("graph");
        let mut w = ContainerWriter::new();
        write_hypergraph(&mut w, "g", &g).unwrap();
        w.write_to(tmp.path()).unwrap();
        let c = Container::open(tmp.path(), Verify::Full).unwrap();
        let back = read_hypergraph(&c, "g").unwrap();
        same_answers(&g, &back);
    }
}
