//! `#[strede(allow_unknown_fields)]` — borrow family.
//!
//! Positive-tests the attribute: unknown map keys are skipped regardless of
//! their position relative to known fields, or the unknown value's own
//! shape. Also confirms the two documented exceptions still hold with the
//! attribute present: a missing required field is still `Miss`, and a
//! duplicate known field is still `Err`.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::{JsonDeserializer, JsonError};
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(allow_unknown_fields)]
struct Point {
    x: u32,
    y: u32,
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
fn extra_field_after_known_is_skipped() {
    assert_eq!(
        parse::<Point>(r#"{"x": 1, "y": 2, "z": 3}"#).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn extra_field_before_known_is_skipped() {
    assert_eq!(
        parse::<Point>(r#"{"z": 3, "x": 1, "y": 2}"#).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn extra_field_between_known_is_skipped() {
    assert_eq!(
        parse::<Point>(r#"{"x": 1, "z": 3, "y": 2}"#).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn multiple_extra_fields_are_skipped() {
    assert_eq!(
        parse::<Point>(r#"{"a": 1, "x": 1, "b": [1, 2, 3], "y": 2, "c": null}"#).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn extra_field_with_nested_object_value_is_skipped() {
    assert_eq!(
        parse::<Point>(r#"{"x": 1, "extra": {"a": [1, {"b": 2}], "c": "hi"}, "y": 2}"#).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn missing_required_field_still_misses() {
    assert_eq!(parse::<Point>(r#"{"x": 1, "z": 3}"#).unwrap(), None);
}

#[test]
fn duplicate_known_field_still_errors() {
    let err = parse::<Point>(r#"{"x": 1, "x": 2, "y": 3}"#).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("x"));
}
