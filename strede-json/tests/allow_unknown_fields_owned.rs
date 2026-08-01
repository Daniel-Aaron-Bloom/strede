//! `#[strede(allow_unknown_fields)]` — owned family.
//!
//! Mirrors `allow_unknown_fields_borrow.rs`.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::JsonError;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(allow_unknown_fields)]
struct Point {
    x: u32,
    y: u32,
}

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                Ok(
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ()).await? {
                        Probe::Hit((_, v)) => Some(v),
                        Probe::Miss => None,
                    },
                )
            },
        )) as Result<Option<$ty>, JsonError>
    }};
}

#[test]
fn extra_field_after_known_is_skipped() {
    assert_eq!(
        parse!(Point, &br#"{"x": 1, "y": 2, "z": 3}"#[..]).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn extra_field_before_known_is_skipped() {
    assert_eq!(
        parse!(Point, &br#"{"z": 3, "x": 1, "y": 2}"#[..]).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn extra_field_between_known_is_skipped() {
    assert_eq!(
        parse!(Point, &br#"{"x": 1, "z": 3, "y": 2}"#[..]).unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn multiple_extra_fields_are_skipped() {
    assert_eq!(
        parse!(
            Point,
            &br#"{"a": 1, "x": 1, "b": [1, 2, 3], "y": 2, "c": null}"#[..]
        )
        .unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn extra_field_with_nested_object_value_is_skipped() {
    assert_eq!(
        parse!(
            Point,
            &br#"{"x": 1, "extra": {"a": [1, {"b": 2}], "c": "hi"}, "y": 2}"#[..]
        )
        .unwrap(),
        Some(Point { x: 1, y: 2 })
    );
}

#[test]
fn missing_required_field_still_misses() {
    assert_eq!(parse!(Point, &br#"{"x": 1, "z": 3}"#[..]).unwrap(), None);
}

#[test]
fn duplicate_known_field_still_errors() {
    let err = parse!(Point, &br#"{"x": 1, "x": 2, "y": 3}"#[..]).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("x"));
}
