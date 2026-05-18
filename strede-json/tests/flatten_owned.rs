//! Owned-family single-flatten fixtures exercising the v3 `FlattenMapAccessOwned` path.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

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
    ($input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                match <Outer as DeserializeOwned<_>>::deserialize_owned(de, ())
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

#[test]
fn outer_then_inner() {
    let o: Outer = parse!(&br#"{"id": 7, "a": 1, "b": 2}"#[..]).unwrap();
    assert_eq!(
        o,
        Outer {
            id: 7,
            inner: Inner { a: 1, b: 2 }
        }
    );
}

#[test]
fn interleaved_order() {
    let o: Outer = parse!(&br#"{"a": 1, "id": 7, "b": 2}"#[..]).unwrap();
    assert_eq!(
        o,
        Outer {
            id: 7,
            inner: Inner { a: 1, b: 2 }
        }
    );
}

#[test]
fn inner_then_outer() {
    let o: Outer = parse!(&br#"{"a": 1, "b": 2, "id": 7}"#[..]).unwrap();
    assert_eq!(
        o,
        Outer {
            id: 7,
            inner: Inner { a: 1, b: 2 }
        }
    );
}

#[test]
fn missing_outer_field_misses() {
    let v: Option<Outer> = parse!(&br#"{"a": 1, "b": 2}"#[..]);
    assert!(v.is_none());
}

#[test]
fn missing_inner_field_misses() {
    let v: Option<Outer> = parse!(&br#"{"id": 7, "a": 1}"#[..]);
    assert!(v.is_none());
}
