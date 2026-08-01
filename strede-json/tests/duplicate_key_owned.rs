//! `DetectDuplicates` (duplicate wire key detection) — owned family.
//!
//! Mirrors `duplicate_key_borrow.rs`: a plain duplicate field, and a duplicate
//! on either side of a `#[strede(flatten)]` boundary to exercise the
//! offset-shifted wire-name table the owned-family flatten codegen builds.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::JsonError;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Point {
    a: u32,
    b: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Inner {
    a: u32,
    b: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Outer {
    id: u32,
    #[strede(flatten)]
    inner: Inner,
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
fn plain_struct_duplicate_first_field_errors() {
    let err = parse!(Point, &br#"{"a": 1, "a": 2, "b": 3}"#[..]).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("a"));
}

#[test]
fn plain_struct_duplicate_second_field_errors() {
    let err = parse!(Point, &br#"{"a": 1, "b": 2, "b": 3}"#[..]).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("b"));
}

#[test]
fn flatten_duplicate_outer_field_errors() {
    let err = parse!(Outer, &br#"{"id": 1, "id": 2, "a": 1, "b": 2}"#[..]).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("id"));
}

#[test]
fn flatten_duplicate_inner_field_errors() {
    let err = parse!(Outer, &br#"{"id": 1, "a": 1, "b": 2, "a": 3}"#[..]).unwrap_err();
    assert_eq!(err, JsonError::DuplicateField("a"));
}
