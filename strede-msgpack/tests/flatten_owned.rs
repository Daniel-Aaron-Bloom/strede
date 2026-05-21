//! Owned-family flatten fixtures.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

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
                match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                {
                    Probe::Hit((_, v)) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
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

#[test]
fn outer_then_inner() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[7u8]),
        (fixstr("a").as_slice(), &[1u8]),
        (fixstr("b").as_slice(), &[2u8]),
    ]);
    assert_eq!(
        parse!(Outer, &msg),
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
        parse!(Outer, &msg),
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
        parse!(Outer, &msg),
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
    assert_eq!(parse!(Outer, &msg), None);
}

#[test]
fn missing_inner_field_misses() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[7u8]),
        (fixstr("a").as_slice(), &[1u8]),
    ]);
    assert_eq!(parse!(Outer, &msg), None);
}
