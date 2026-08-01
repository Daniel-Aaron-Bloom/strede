//! Owned-family multi-flatten fixtures (2 and 3 flatten fields).

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
struct A {
    a1: u32,
    a2: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct B {
    b1: u32,
    b2: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct C {
    c1: u32,
    c2: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Two {
    id: u32,
    #[strede(flatten)]
    a: A,
    #[strede(flatten)]
    b: B,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Three {
    id: u32,
    #[strede(flatten)]
    a: A,
    #[strede(flatten)]
    b: B,
    #[strede(flatten)]
    c: C,
}

#[test]
fn two_in_order() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("a2").as_slice(), &[3u8]),
        (fixstr("b1").as_slice(), &[4u8]),
        (fixstr("b2").as_slice(), &[5u8]),
    ]);
    assert_eq!(
        parse!(Two, &msg),
        Some(Two {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
        })
    );
}

#[test]
fn two_interleaved() {
    let msg = build_map(&[
        (fixstr("b1").as_slice(), &[4u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("b2").as_slice(), &[5u8]),
        (fixstr("a2").as_slice(), &[3u8]),
    ]);
    assert_eq!(
        parse!(Two, &msg),
        Some(Two {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
        })
    );
}

#[test]
fn two_missing_b_field_misses() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("a2").as_slice(), &[3u8]),
        (fixstr("b1").as_slice(), &[4u8]),
    ]);
    assert_eq!(parse!(Two, &msg), None);
}

#[test]
fn three_in_order() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("a2").as_slice(), &[3u8]),
        (fixstr("b1").as_slice(), &[4u8]),
        (fixstr("b2").as_slice(), &[5u8]),
        (fixstr("c1").as_slice(), &[6u8]),
        (fixstr("c2").as_slice(), &[7u8]),
    ]);
    assert_eq!(
        parse!(Three, &msg),
        Some(Three {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
            c: C { c1: 6, c2: 7 },
        })
    );
}

#[test]
fn three_fully_interleaved() {
    let msg = build_map(&[
        (fixstr("c2").as_slice(), &[7u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("b1").as_slice(), &[4u8]),
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("c1").as_slice(), &[6u8]),
        (fixstr("a2").as_slice(), &[3u8]),
        (fixstr("b2").as_slice(), &[5u8]),
    ]);
    assert_eq!(
        parse!(Three, &msg),
        Some(Three {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
            c: C { c1: 6, c2: 7 },
        })
    );
}

#[test]
fn three_missing_c_field_misses() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("a2").as_slice(), &[3u8]),
        (fixstr("b1").as_slice(), &[4u8]),
        (fixstr("b2").as_slice(), &[5u8]),
        (fixstr("c1").as_slice(), &[6u8]),
    ]);
    assert_eq!(parse!(Three, &msg), None);
}
