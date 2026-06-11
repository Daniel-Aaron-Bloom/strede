//! Borrow-family flatten fixtures.

extern crate std;
mod helpers;
use helpers::*;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    match block_on(T::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[derive(Debug, PartialEq, Deserialize)]
struct Inner {
    a: u32,
    b: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Outer {
    id: u32,
    #[strede(flatten)]
    inner: Inner,
}

// Field before AND after the flatten — exercises the before/after arm split.
#[derive(Debug, PartialEq, Deserialize)]
struct OuterWithSuffix {
    prefix: u32,
    #[strede(flatten)]
    inner: Inner,
    suffix: u32,
}

#[test]
fn outer_then_inner() {
    let msg = build_map(&[
        (tstr("id").as_slice(), &[uint_small(7)]),
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("b").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse::<Outer>(&msg), Some(Outer { id: 7, inner: Inner { a: 1, b: 2 } }));
}

#[test]
fn interleaved_order() {
    let msg = build_map(&[
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("id").as_slice(), &[uint_small(7)]),
        (tstr("b").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse::<Outer>(&msg), Some(Outer { id: 7, inner: Inner { a: 1, b: 2 } }));
}

#[test]
fn inner_then_outer() {
    let msg = build_map(&[
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("b").as_slice(), &[uint_small(2)]),
        (tstr("id").as_slice(), &[uint_small(7)]),
    ]);
    assert_eq!(parse::<Outer>(&msg), Some(Outer { id: 7, inner: Inner { a: 1, b: 2 } }));
}

#[test]
fn missing_outer_field_misses() {
    let msg = build_map(&[
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("b").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse::<Outer>(&msg), None);
}

#[test]
fn missing_inner_field_misses() {
    let msg = build_map(&[
        (tstr("id").as_slice(), &[uint_small(7)]),
        (tstr("a").as_slice(), &[uint_small(1)]),
    ]);
    assert_eq!(parse::<Outer>(&msg), None);
}

#[test]
fn suffix_outer_then_inner_then_suffix() {
    let msg = build_map(&[
        (tstr("prefix").as_slice(), &[uint_small(1)]),
        (tstr("a").as_slice(), &[uint_small(2)]),
        (tstr("b").as_slice(), &[uint_small(3)]),
        (tstr("suffix").as_slice(), &[uint_small(4)]),
    ]);
    assert_eq!(
        parse::<OuterWithSuffix>(&msg),
        Some(OuterWithSuffix { prefix: 1, inner: Inner { a: 2, b: 3 }, suffix: 4 })
    );
}

#[test]
fn suffix_interleaved() {
    let msg = build_map(&[
        (tstr("a").as_slice(), &[uint_small(2)]),
        (tstr("suffix").as_slice(), &[uint_small(4)]),
        (tstr("prefix").as_slice(), &[uint_small(1)]),
        (tstr("b").as_slice(), &[uint_small(3)]),
    ]);
    assert_eq!(
        parse::<OuterWithSuffix>(&msg),
        Some(OuterWithSuffix { prefix: 1, inner: Inner { a: 2, b: 3 }, suffix: 4 })
    );
}

#[test]
fn suffix_missing_prefix_misses() {
    let msg = build_map(&[
        (tstr("a").as_slice(), &[uint_small(2)]),
        (tstr("b").as_slice(), &[uint_small(3)]),
        (tstr("suffix").as_slice(), &[uint_small(4)]),
    ]);
    assert_eq!(parse::<OuterWithSuffix>(&msg), None);
}

#[test]
fn suffix_missing_suffix_misses() {
    let msg = build_map(&[
        (tstr("prefix").as_slice(), &[uint_small(1)]),
        (tstr("a").as_slice(), &[uint_small(2)]),
        (tstr("b").as_slice(), &[uint_small(3)]),
    ]);
    assert_eq!(parse::<OuterWithSuffix>(&msg), None);
}
