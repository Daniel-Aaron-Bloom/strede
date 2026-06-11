//! Owned-family single-flatten fixtures exercising the derive's `MapFieldProviderOwned` codegen.

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
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
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

// Field before AND after the flatten — exercises the before/after arm split.
#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct OuterWithSuffix {
    prefix: u32,
    #[strede(flatten)]
    inner: Inner,
    suffix: u32,
}

#[test]
fn outer_then_inner() {
    let o: Outer = parse!(Outer, &br#"{"id": 7, "a": 1, "b": 2}"#[..]).unwrap();
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
    let o: Outer = parse!(Outer, &br#"{"a": 1, "id": 7, "b": 2}"#[..]).unwrap();
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
    let o: Outer = parse!(Outer, &br#"{"a": 1, "b": 2, "id": 7}"#[..]).unwrap();
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
    let v: Option<Outer> = parse!(Outer, &br#"{"a": 1, "b": 2}"#[..]);
    assert!(v.is_none());
}

#[test]
fn missing_inner_field_misses() {
    let v: Option<Outer> = parse!(Outer, &br#"{"id": 7, "a": 1}"#[..]);
    assert!(v.is_none());
}

#[test]
fn suffix_outer_then_inner_then_suffix() {
    let o: OuterWithSuffix = parse!(OuterWithSuffix, &br#"{"prefix": 1, "a": 2, "b": 3, "suffix": 4}"#[..]).unwrap();
    assert_eq!(o, OuterWithSuffix { prefix: 1, inner: Inner { a: 2, b: 3 }, suffix: 4 });
}

#[test]
fn suffix_interleaved() {
    let o: OuterWithSuffix = parse!(OuterWithSuffix, &br#"{"a": 2, "suffix": 4, "prefix": 1, "b": 3}"#[..]).unwrap();
    assert_eq!(o, OuterWithSuffix { prefix: 1, inner: Inner { a: 2, b: 3 }, suffix: 4 });
}

#[test]
fn suffix_missing_prefix_misses() {
    let v: Option<OuterWithSuffix> = parse!(OuterWithSuffix, &br#"{"a": 2, "b": 3, "suffix": 4}"#[..]);
    assert!(v.is_none());
}

#[test]
fn suffix_missing_suffix_misses() {
    let v: Option<OuterWithSuffix> = parse!(OuterWithSuffix, &br#"{"prefix": 1, "a": 2, "b": 3}"#[..]);
    assert!(v.is_none());
}
