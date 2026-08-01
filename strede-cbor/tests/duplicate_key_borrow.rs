//! `DetectDuplicates` (duplicate wire key detection) — borrow family.
//!
//! Covers a plain duplicate field and a duplicate that lands on either side of
//! a `#[strede(flatten)]` boundary, since flatten composition shifts the
//! flattened child's wire-name indices and that offset math is exactly what
//! `DetectDuplicates` relies on to report the right field name.

extern crate std;
mod helpers;
use helpers::*;

use strede::Probe;
use strede_cbor::{CborDeserializer, CborError};
use strede_derive::Deserialize;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Result<Option<T>, CborError>
where
    T: strede::Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    Ok(match block_on(T::deserialize(de, ()))? {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    })
}

#[derive(Debug, PartialEq, Deserialize)]
struct Point {
    a: u32,
    b: u32,
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

#[test]
fn plain_struct_duplicate_first_field_errors() {
    let msg = build_map(&[
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("a").as_slice(), &[uint_small(2)]),
        (tstr("b").as_slice(), &[uint_small(3)]),
    ]);
    let err = parse::<Point>(&msg).unwrap_err();
    assert_eq!(err, CborError::DuplicateField("a"));
}

#[test]
fn plain_struct_duplicate_second_field_errors() {
    let msg = build_map(&[
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("b").as_slice(), &[uint_small(2)]),
        (tstr("b").as_slice(), &[uint_small(3)]),
    ]);
    let err = parse::<Point>(&msg).unwrap_err();
    assert_eq!(err, CborError::DuplicateField("b"));
}

#[test]
fn flatten_duplicate_outer_field_errors() {
    let msg = build_map(&[
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("id").as_slice(), &[uint_small(2)]),
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("b").as_slice(), &[uint_small(2)]),
    ]);
    let err = parse::<Outer>(&msg).unwrap_err();
    assert_eq!(err, CborError::DuplicateField("id"));
}

#[test]
fn flatten_duplicate_inner_field_errors() {
    let msg = build_map(&[
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("a").as_slice(), &[uint_small(1)]),
        (tstr("b").as_slice(), &[uint_small(2)]),
        (tstr("a").as_slice(), &[uint_small(3)]),
    ]);
    let err = parse::<Outer>(&msg).unwrap_err();
    assert_eq!(err, CborError::DuplicateField("a"));
}
