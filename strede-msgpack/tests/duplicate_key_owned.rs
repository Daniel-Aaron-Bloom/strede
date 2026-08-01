//! `DetectDuplicates` (duplicate wire key detection) — owned family.
//!
//! Mirrors `duplicate_key_borrow.rs`: a plain duplicate field, and a duplicate
//! on either side of a `#[strede(flatten)]` boundary to exercise the
//! offset-shifted wire-name table the owned-family flatten codegen builds.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::MsgpackError;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
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
                let de = ChunkedMsgpackDeserializer::new(shared);
                Ok(
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ()).await? {
                        Probe::Hit((_, v)) => Some(v),
                        Probe::Miss => None,
                    },
                )
            },
        )) as Result<Option<$ty>, MsgpackError>
    }};
}

#[test]
fn plain_struct_duplicate_first_field_errors() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("a").as_slice(), &[2u8]),
        (fixstr("b").as_slice(), &[3u8]),
    ]);
    let err = parse!(Point, &msg).unwrap_err();
    assert_eq!(err, MsgpackError::DuplicateField("a"));
}

#[test]
fn plain_struct_duplicate_second_field_errors() {
    let msg = build_map(&[
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
        (fixstr("b").as_slice(), &[3u8]),
    ]);
    let err = parse!(Point, &msg).unwrap_err();
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
    let err = parse!(Outer, &msg).unwrap_err();
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
    let err = parse!(Outer, &msg).unwrap_err();
    assert_eq!(err, MsgpackError::DuplicateField("a"));
}
