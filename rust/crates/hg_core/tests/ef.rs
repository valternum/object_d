//! Issue #3: Elias–Fano monotone sequences.

use std::borrow::Cow;

use hg_core::bits::RankSelect;
use hg_core::ef::EliasFano;
use hg_core::CoreError;
use proptest::prelude::*;

/// Space bound of Elias–Fano (Vigna 2013): n·(2 + ceil(log2(U/n))) + 1 bits, U = last + 1.
/// `ceil(log2(U/n))` = smallest c >= 0 with n·2^c >= U.
fn space_bound(values: &[u64]) -> usize {
    let n = values.len();
    if n == 0 {
        return 0;
    }
    let u = u128::from(*values.last().unwrap()) + 1;
    let mut c = 0usize;
    while (n as u128) << c < u {
        c += 1;
    }
    n * (2 + c) + 1
}

#[test]
fn hand_checked_sequence() {
    let v = [2u64, 3, 5, 7, 11, 13, 24];
    let ef = EliasFano::from_sorted(&v).unwrap();
    assert_eq!(ef.len(), 7);
    for (i, &x) in v.iter().enumerate() {
        assert_eq!(ef.get(i), x, "get({i})");
    }
    assert_eq!(ef.range(2), (5, 7));
    assert_eq!(ef.range(5), (13, 24));
    assert_eq!(ef.iter().collect::<Vec<_>>(), v);
}

#[test]
fn empty_sequence() {
    let ef = EliasFano::from_sorted(&[]).unwrap();
    assert!(ef.is_empty());
    assert_eq!(ef.iter().count(), 0);
    assert_eq!(ef.payload_bits(), 0);
}

#[test]
fn duplicates_and_zeros() {
    let v = [0u64, 0, 0, 5, 5, 9, 9, 9];
    let ef = EliasFano::from_sorted(&v).unwrap();
    assert_eq!(ef.iter().collect::<Vec<_>>(), v);
}

#[test]
fn huge_universe() {
    let v = [0u64, 1 << 40, (1 << 62) + 5, u64::MAX - 1, u64::MAX];
    let ef = EliasFano::from_sorted(&v).unwrap();
    assert_eq!(ef.iter().collect::<Vec<_>>(), v);
}

#[test]
fn decreasing_input_is_rejected_with_its_index() {
    match EliasFano::from_sorted(&[1, 4, 3, 8]) {
        Err(CoreError::NotMonotone { index }) => assert_eq!(index, 2),
        other => panic!("expected NotMonotone, got {other:?}"),
    }
}

#[test]
fn csr_offsets_space_is_within_bound() {
    // CSR-like offsets: 10_000 rows, average degree 7.
    let v: Vec<u64> = (0..=10_000u64).map(|i| i * 7 + (i % 3)).collect();
    let ef = EliasFano::from_sorted(&v).unwrap();
    assert!(
        ef.payload_bits() <= space_bound(&v),
        "{} > {}",
        ef.payload_bits(),
        space_bound(&v)
    );
    // Far below 64 bits per value.
    assert!(ef.payload_bits() < v.len() * 8);
}

#[test]
fn borrowed_parts_round_trip() {
    let v: Vec<u64> = (0..500u64).map(|i| i * i).collect();
    let ef = EliasFano::from_sorted(&v).unwrap();
    let high = RankSelect::from_parts(
        Cow::Borrowed(ef.high().words()),
        Cow::Borrowed(ef.high().supers()),
        ef.high().len(),
    )
    .unwrap();
    let view = EliasFano::from_parts(ef.len(), ef.low_bits(), Cow::Borrowed(ef.low_words()), high)
        .unwrap();
    assert_eq!(view.iter().collect::<Vec<_>>(), v);
}

#[test]
fn from_parts_rejects_short_lower_bits() {
    let v: Vec<u64> = (0..500u64).map(|i| i * 1000).collect();
    let ef = EliasFano::from_sorted(&v).unwrap();
    assert!(ef.low_bits() > 0);
    let high = ef.high().clone();
    assert!(EliasFano::from_parts(ef.len(), ef.low_bits(), Cow::Owned(vec![]), high).is_err());
}

proptest! {
    #[test]
    fn get_matches_dense_sequences(mut v in proptest::collection::vec(0u64..200, 0..400)) {
        v.sort_unstable();
        let ef = EliasFano::from_sorted(&v).unwrap();
        prop_assert_eq!(ef.len(), v.len());
        for (i, &x) in v.iter().enumerate() {
            prop_assert_eq!(ef.get(i), x);
        }
        prop_assert!(ef.payload_bits() <= space_bound(&v));
    }

    #[test]
    fn get_matches_sparse_sequences(mut v in proptest::collection::vec(any::<u64>(), 1..300)) {
        v.sort_unstable();
        let ef = EliasFano::from_sorted(&v).unwrap();
        prop_assert_eq!(ef.iter().collect::<Vec<_>>(), v.clone());
        prop_assert!(ef.payload_bits() <= space_bound(&v));
    }
}
