//! Borrow-family flatten fixtures.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, MsgpackDeserializer<'de>, Extra = ()>,
{
    let de = MsgpackDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
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
        (fixstr("id").as_slice(), &[7u8]),
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
    ]);
    assert_eq!(
        parse::<Outer>(&msg),
        Some(Outer {
            id: 7,
            inner: Inner { a: 1, b: 2 }
        })
    );
}

#[test]
fn interleaved_order() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("id").as_slice(), &[7u8]),
        (fixstr("b").as_slice(), &[2u8]),
    ]);
    assert_eq!(
        parse::<Outer>(&msg),
        Some(Outer {
            id: 7,
            inner: Inner { a: 1, b: 2 }
        })
    );
}

#[test]
fn inner_then_outer() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
        (fixstr("id").as_slice(), &[7u8]),
    ]);
    assert_eq!(
        parse::<Outer>(&msg),
        Some(Outer {
            id: 7,
            inner: Inner { a: 1, b: 2 }
        })
    );
}

#[test]
fn missing_outer_field_misses() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
    ]);
    assert_eq!(parse::<Outer>(&msg), None);
}

#[test]
fn missing_inner_field_misses() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[7u8]),
        (fixstr("a").as_slice(), &[1u8]),
    ]);
    assert_eq!(parse::<Outer>(&msg), None);
}

#[test]
fn suffix_outer_then_inner_then_suffix() {
    let msg = build_map(&[
        (fixstr("prefix").as_slice(), &[1u8]),
        (fixstr("a").as_slice(), &[2u8]),
        (fixstr("b").as_slice(), &[3u8]),
        (fixstr("suffix").as_slice(), &[4u8]),
    ]);
    assert_eq!(
        parse::<OuterWithSuffix>(&msg),
        Some(OuterWithSuffix { prefix: 1, inner: Inner { a: 2, b: 3 }, suffix: 4 })
    );
}

#[test]
fn suffix_interleaved() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[2u8]),
        (fixstr("suffix").as_slice(), &[4u8]),
        (fixstr("prefix").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[3u8]),
    ]);
    assert_eq!(
        parse::<OuterWithSuffix>(&msg),
        Some(OuterWithSuffix { prefix: 1, inner: Inner { a: 2, b: 3 }, suffix: 4 })
    );
}

#[test]
fn suffix_missing_prefix_misses() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[2u8]),
        (fixstr("b").as_slice(), &[3u8]),
        (fixstr("suffix").as_slice(), &[4u8]),
    ]);
    assert_eq!(parse::<OuterWithSuffix>(&msg), None);
}

#[test]
fn suffix_missing_suffix_misses() {
    let msg = build_map(&[
        (fixstr("prefix").as_slice(), &[1u8]),
        (fixstr("a").as_slice(), &[2u8]),
        (fixstr("b").as_slice(), &[3u8]),
    ]);
    assert_eq!(parse::<OuterWithSuffix>(&msg), None);
}
