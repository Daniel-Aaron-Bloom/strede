//! `DetectDuplicates` (duplicate wire key detection) — borrow family.
//!
//! Covers a plain duplicate field and a duplicate that lands on either side of
//! a `#[strede(flatten)]` boundary, since flatten composition shifts the
//! flattened child's wire-name indices and that offset math is exactly what
//! `DetectDuplicates` relies on to report the right field name.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::{MsgpackDeserializer, MsgpackError};
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Result<Option<T>, MsgpackError>
where
    T: strede::Deserialize<'de, MsgpackDeserializer<'de>, Extra = ()>,
{
    let de = MsgpackDeserializer::new(input);
    Ok(
        match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ()))? {
            Probe::Hit((_, v)) => Some(v),
            Probe::Miss => None,
        },
    )
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
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("a").as_slice(), &[2u8]),
        (fixstr("b").as_slice(), &[3u8]),
    ]);
    let err = parse::<Point>(&msg).unwrap_err();
    assert_eq!(err, MsgpackError::DuplicateField("a"));
}

#[test]
fn plain_struct_duplicate_second_field_errors() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
        (fixstr("b").as_slice(), &[3u8]),
    ]);
    let err = parse::<Point>(&msg).unwrap_err();
    assert_eq!(err, MsgpackError::DuplicateField("b"));
}

#[test]
fn flatten_duplicate_outer_field_errors() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("id").as_slice(), &[2u8]),
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
    ]);
    let err = parse::<Outer>(&msg).unwrap_err();
    assert_eq!(err, MsgpackError::DuplicateField("id"));
}

#[test]
fn flatten_duplicate_inner_field_errors() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
        (fixstr("a").as_slice(), &[3u8]),
    ]);
    let err = parse::<Outer>(&msg).unwrap_err();
    assert_eq!(err, MsgpackError::DuplicateField("a"));
}
