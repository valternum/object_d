//! Attributes used by slice filters (issues #8, #13): class sets, node types,
//! edge validity intervals.

mod common;

use common::TempPath;
use hg_core::attrs::{ClassSet, EdgeIntervals, NodeTypes};
use hg_core::container::{Container, ContainerWriter, Verify};

#[test]
fn class_set_membership_and_ids() {
    let mut s = ClassSet::new(200);
    for c in [0u32, 63, 64, 199] {
        s.insert(c);
    }
    s.insert(64);
    assert_eq!(s.count(), 4);
    assert!(s.contains(63) && s.contains(199));
    assert!(!s.contains(1) && !s.contains(198));
    assert_eq!(s.ids(), vec![0, 63, 64, 199]);
}

#[test]
fn node_types_sorted_unique_and_any_in() {
    let t = NodeTypes::from_pairs(4, vec![(2, 7), (0, 3), (2, 1), (2, 7), (0, 3)]).unwrap();
    assert_eq!(t.n_nodes(), 4);
    assert_eq!(t.types(0), &[3]);
    assert_eq!(t.types(1), &[] as &[u32]);
    assert_eq!(t.types(2), &[1, 7]);
    assert_eq!(t.types(3), &[] as &[u32]);
    let mut set = ClassSet::new(10);
    set.insert(7);
    assert!(t.any_in(2, &set));
    assert!(!t.any_in(0, &set));
    assert!(!t.any_in(1, &set));
}

#[test]
fn node_types_reject_out_of_range_node() {
    assert!(NodeTypes::from_pairs(2, vec![(2, 0)]).is_err());
}

#[test]
fn intervals_overlap_semantics() {
    let iv =
        EdgeIntervals::from_triples(vec![(5, 1951, 2002), (9, 2005, i32::MAX), (5, 1940, 1945)]);
    assert_eq!(iv.len(), 2);
    // Two intervals of edge 5 merge into their hull.
    assert_eq!(iv.get(5), Some((1940, 2002)));
    assert_eq!(iv.get(9), Some((2005, i32::MAX)));
    assert_eq!(iv.get(6), None);
    assert!(iv.overlaps(5, 1990, 2000));
    assert!(iv.overlaps(5, 2002, 2010), "touching end is inclusive");
    assert!(!iv.overlaps(5, 2003, 2010));
    assert!(!iv.overlaps(9, 1990, 2004));
    assert!(iv.overlaps(9, 3000, 3001), "open end");
    assert!(iv.overlaps(6, 0, 1), "edges without interval always pass");
}

#[test]
fn attributes_round_trip_through_container() {
    let t = NodeTypes::from_pairs(3, vec![(0, 2), (2, 0), (2, 5)]).unwrap();
    let iv = EdgeIntervals::from_triples(vec![(1, -500, -400), (4, 1990, 1995)]);
    let tmp = TempPath::new("attrs");
    let mut w = ContainerWriter::new();
    t.write(&mut w, "types").unwrap();
    iv.write(&mut w, "iv").unwrap();
    w.write_to(tmp.path()).unwrap();
    let c = Container::open(tmp.path(), Verify::Full).unwrap();
    let t2 = NodeTypes::read(&c, "types").unwrap();
    let iv2 = EdgeIntervals::read(&c, "iv").unwrap();
    for v in 0..3 {
        assert_eq!(t2.types(v), t.types(v));
    }
    assert_eq!(iv2.get(1), Some((-500, -400)), "negative years survive");
    assert_eq!(iv2.get(4), Some((1990, 1995)));
    assert_eq!(iv2.get(2), None);
}
