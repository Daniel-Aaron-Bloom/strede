//! `#[strede(allow_unknown_fields)]` — owned family.
//!
//! Mirrors `allow_unknown_fields_borrow.rs`.

extern crate std;
mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::{CborError, ChunkedCborDeserializer};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
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
                let de = ChunkedCborDeserializer::new(shared);
                Ok(
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ()).await? {
                        Probe::Hit((_, v)) => Some(v),
                        Probe::Miss => None,
                    },
                )
            },
        )) as Result<Option<$ty>, CborError>
    }};
}

#[test]
fn extra_field_after_known_is_skipped() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("y").as_slice(), &[uint_small(2)]),
        (tstr("z").as_slice(), &[uint_small(3)]),
    ]);
    assert_eq!(parse!(Point, &msg).unwrap(), Some(Point { x: 1, y: 2 }));
}

#[test]
fn extra_field_before_known_is_skipped() {
    let msg = build_map(&[
        (tstr("z").as_slice(), &[uint_small(3)]),
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("y").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse!(Point, &msg).unwrap(), Some(Point { x: 1, y: 2 }));
}

#[test]
fn extra_field_between_known_is_skipped() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("z").as_slice(), &[uint_small(3)]),
        (tstr("y").as_slice(), &[uint_small(2)]),
    ]);
    assert_eq!(parse!(Point, &msg).unwrap(), Some(Point { x: 1, y: 2 }));
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
    assert_eq!(parse!(Point, &msg).unwrap(), Some(Point { x: 1, y: 2 }));
}

#[test]
fn missing_required_field_still_misses() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("z").as_slice(), &[uint_small(3)]),
    ]);
    assert_eq!(parse!(Point, &msg).unwrap(), None);
}

#[test]
fn duplicate_known_field_still_errors() {
    let msg = build_map(&[
        (tstr("x").as_slice(), &[uint_small(1)]),
        (tstr("x").as_slice(), &[uint_small(2)]),
        (tstr("y").as_slice(), &[uint_small(3)]),
    ]);
    let err = parse!(Point, &msg).unwrap_err();
    assert_eq!(err, CborError::DuplicateField("x"));
}
