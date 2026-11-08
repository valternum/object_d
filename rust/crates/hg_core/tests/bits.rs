//! Issue #2: rank/select bit vector.

use std::borrow::Cow;

use hg_core::bits::{BitVecBuilder, RankSelect};
use proptest::prelude::*;

fn build(bits: &[bool]) -> RankSelect<'static> {
    let mut b = BitVecBuilder::new();
    for &x in bits {
        b.push(x);
    }
    b.freeze()
}

#[test]
fn rank_and_select_on_hand_checked_vector() {
    // ones at positions 0, 2, 3, 7
    let rs = build(&[true, false, true, true, false, false, false, true]);
    assert_eq!(rs.len(), 8);
    assert_eq!(rs.count_ones(), 4);
    let want_rank = [0, 1, 1, 2, 3, 3, 3, 3, 4];
    for (i, &w) in want_rank.iter().enumerate() {
        assert_eq!(rs.rank1(i), w, "rank1({i})");
    }
    assert_eq!(rs.select1(0), Some(0));
    assert_eq!(rs.select1(1), Some(2));
    assert_eq!(rs.select1(2), Some(3));
    assert_eq!(rs.select1(3), Some(7));
    assert_eq!(rs.select1(4), None);
    assert!(rs.get(3));
    assert!(!rs.get(4));
}

#[test]
fn empty_vector_has_no_ones() {
    let rs = build(&[]);
    assert_eq!(rs.len(), 0);
    assert_eq!(rs.rank1(0), 0);
    assert_eq!(rs.select1(0), None);
}

#[test]
fn all_ones_at_word_and_block_boundaries() {
    for &n in &[1usize, 63, 64, 65, 511, 512, 513, 1025] {
        let rs = build(&vec![true; n]);
        assert_eq!(rs.count_ones(), n, "n={n}");
        for i in [0, n / 2, n] {
            assert_eq!(rs.rank1(i), i, "n={n} i={i}");
        }
        assert_eq!(rs.select1(n - 1), Some(n - 1), "n={n}");
        assert_eq!(rs.select1(n), None, "n={n}");
    }
}

#[test]
fn sparse_ones_across_superblocks() {
    let mut b = BitVecBuilder::with_len(5000);
    for p in [5usize, 600, 1500, 4096] {
        b.set(p);
    }
    let rs = b.freeze();
    assert_eq!(rs.count_ones(), 4);
    assert_eq!(rs.rank1(601), 2);
    assert_eq!(rs.rank1(4096), 3);
    assert_eq!(rs.rank1(4097), 4);
    assert_eq!(rs.select1(2), Some(1500));
    assert_eq!(rs.select1(3), Some(4096));
}

#[test]
fn from_parts_rejects_wrong_directory_size() {
    let rs = build(&[true; 1000]);
    let words = rs.words().to_vec();
    let err = RankSelect::from_parts(Cow::Owned(words), Cow::Owned(vec![0]), 1000);
    assert!(err.is_err());
}

#[test]
fn from_parts_rejects_too_few_words() {
    let rs = build(&[true; 1000]);
    let err = RankSelect::from_parts(Cow::Owned(vec![u64::MAX]), Cow::Borrowed(rs.supers()), 1000);
    assert!(err.is_err());
}

#[test]
fn borrowed_parts_answer_like_the_original() {
    let bits: Vec<bool> = (0..2000).map(|i| i % 3 == 0 || i % 7 == 0).collect();
    let rs = build(&bits);
    let view = RankSelect::from_parts(
        Cow::Borrowed(rs.words()),
        Cow::Borrowed(rs.supers()),
        rs.len(),
    )
    .unwrap();
    for i in (0..=2000).step_by(37) {
        assert_eq!(view.rank1(i), rs.rank1(i));
    }
    for k in 0..rs.count_ones() {
        assert_eq!(view.select1(k), rs.select1(k));
    }
}

proptest! {
    #[test]
    fn rank_matches_naive_prefix_count(bits in proptest::collection::vec(any::<bool>(), 0..3000)) {
        let rs = build(&bits);
        let mut acc = 0usize;
        for i in 0..=bits.len() {
            prop_assert_eq!(rs.rank1(i), acc);
            if i < bits.len() && bits[i] {
                acc += 1;
            }
        }
        prop_assert_eq!(rs.count_ones(), acc);
    }

    #[test]
    fn select_returns_every_one_in_order(bits in proptest::collection::vec(any::<bool>(), 0..3000)) {
        let rs = build(&bits);
        let ones: Vec<usize> = bits.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
        for (k, &p) in ones.iter().enumerate() {
            prop_assert_eq!(rs.select1(k), Some(p));
        }
        prop_assert_eq!(rs.select1(ones.len()), None);
    }

    #[test]
    fn get_matches_input(bits in proptest::collection::vec(any::<bool>(), 1..1500)) {
        let rs = build(&bits);
        for (i, &b) in bits.iter().enumerate() {
            prop_assert_eq!(rs.get(i), b);
        }
    }
}
