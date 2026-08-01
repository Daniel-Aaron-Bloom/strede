//! `#[strede(allow_unknown_fields)]` — borrow family.
//!
//! Positive-tests the attribute: unknown map keys are skipped regardless of
//! their position relative to known fields, or the unknown value's own
//! shape. Also confirms the two documented exceptions still hold with the
//! attribute present: a missing required field is still `Miss`, and a
//! duplicate known field is still `Err`.

extern crate std;
mod helpers;
use helpers::*;

use strede::Probe;
use strede_cbor::{CborDeserializer, CborError};
use strede_derive::Deserialize;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(allow_unknown_fields)]
struct Point {
    x: u32,
    y: u32,
}

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

#[test]
fn extra_field_after_known_is_skipped() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("y").as_slice(), &[uint_small(2)]),
        (tstr("z").as_slice(), &[uint_small(3)]),
    ]);
    assert_eq!(parse::<Point>(&msg).unwrap(), Some(Point { x: 1, y: 2 }));
}

#[test]
fn extra_field_before_known_is_skipped() {
    let msg = build_map(&[
        (tstr("z").as_slice(), &[uint_small(3)]),
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("y").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse::<Point>(&msg).unwrap(), Some(Point { x: 1, y: 2 }));
}

#[test]
fn extra_field_between_known_is_skipped() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("z").as_slice(), &[uint_small(3)]),
        (tstr("y").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse::<Point>(&msg).unwrap(), Some(Point { x: 1, y: 2 }));
}

#[test]
fn extra_field_with_nested_value_is_skipped() {
    let mut nested = array(2);
    nested.extend_from_slice(&[uint_small(1)]);
    nested.extend_from_slice(&build_map(&[(tstr("b").as_slice(), &[uint_small(2)])]));
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("extra").as_slice(), nested.as_slice()),
        (tstr("y").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse::<Point>(&msg).unwrap(), Some(Point { x: 1, y: 2 }));
}

#[test]
fn missing_required_field_still_misses() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("z").as_slice(), &[uint_small(3)]),
    ]);
    assert_eq!(parse::<Point>(&msg).unwrap(), None);
}

#[test]
fn duplicate_known_field_still_errors() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("x").as_slice(), &[uint_small(2)]),
        (tstr("y").as_slice(), &[uint_small(3)]),
    ]);
    let err = parse::<Point>(&msg).unwrap_err();
    assert_eq!(err, CborError::DuplicateField("x"));
}
