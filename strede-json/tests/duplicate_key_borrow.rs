//! `DetectDuplicates` (duplicate wire key detection) — borrow family.
//!
//! Covers a plain duplicate field and a duplicate that lands on either side of
//! a `#[strede(flatten)]` boundary, since flatten composition shifts the
//! flattened child's wire-name indices and that offset math is exactly what
//! `DetectDuplicates` relies on to report the right field name.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::{JsonDeserializer, JsonError};
use strede_test_util::block_on;

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

fn parse<'de, T>(input: &'de str) -> Result<Option<T>, JsonError>
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    Ok(
        match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ()))? {
            Probe::Hit((_, v)) => Some(v),
            Probe::Miss => None,
        },
    )
}

#[test]
fn plain_struct_duplicate_first_field_errors() {
    let err = parse::<Point>(r#"{"a": 1, "a": 2, "b": 3}"#).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("a"));
}

#[test]
fn plain_struct_duplicate_second_field_errors() {
    let err = parse::<Point>(r#"{"a": 1, "b": 2, "b": 3}"#).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("b"));
}

#[test]
fn flatten_duplicate_outer_field_errors() {
    let err = parse::<Outer>(r#"{"id": 1, "id": 2, "a": 1, "b": 2}"#).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("id"));
}

#[test]
fn flatten_duplicate_inner_field_errors() {
    let err = parse::<Outer>(r#"{"id": 1, "a": 1, "b": 2, "a": 3}"#).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("a"));
}
