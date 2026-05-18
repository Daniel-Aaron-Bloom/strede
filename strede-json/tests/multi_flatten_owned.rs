//! Owned-family multi-flatten fixtures (2 and 3 flatten fields).

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

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

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct ThreeBoxed {
    id: u32,
    #[strede(flatten(boxed))]
    a: A,
    #[strede(flatten)]
    b: B,
    #[strede(flatten)]
    c: C,
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

#[test]
fn two_in_order() {
    let v: Two = parse!(
        Two,
        &br#"{"id": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5}"#[..]
    )
    .unwrap();
    assert_eq!(
        v,
        Two {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
        }
    );
}

#[test]
fn two_interleaved() {
    let v: Two = parse!(
        Two,
        &br#"{"b1": 4, "a1": 2, "id": 1, "b2": 5, "a2": 3}"#[..]
    )
    .unwrap();
    assert_eq!(
        v,
        Two {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
        }
    );
}

#[test]
fn two_missing_b_field_misses() {
    let v: Option<Two> = parse!(Two, &br#"{"id": 1, "a1": 2, "a2": 3, "b1": 4}"#[..]);
    assert!(v.is_none());
}

#[test]
fn three_in_order() {
    let v: Three = parse!(
        Three,
        &br#"{"id": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5, "c1": 6, "c2": 7}"#[..]
    )
    .unwrap();
    assert_eq!(
        v,
        Three {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
            c: C { c1: 6, c2: 7 },
        }
    );
}

#[test]
fn three_fully_interleaved() {
    let v: Three = parse!(
        Three,
        &br#"{"c2": 7, "a1": 2, "b1": 4, "id": 1, "c1": 6, "a2": 3, "b2": 5}"#[..]
    )
    .unwrap();
    assert_eq!(
        v,
        Three {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
            c: C { c1: 6, c2: 7 },
        }
    );
}

#[test]
fn three_missing_c_field_misses() {
    let v: Option<Three> = parse!(
        Three,
        &br#"{"id": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5, "c1": 6}"#[..]
    );
    assert!(v.is_none());
}

#[test]
fn three_boxed_in_order() {
    let v: ThreeBoxed = parse!(
        ThreeBoxed,
        &br#"{"id": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5, "c1": 6, "c2": 7}"#[..]
    )
    .unwrap();
    assert_eq!(
        v,
        ThreeBoxed {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
            c: C { c1: 6, c2: 7 },
        }
    );
}

#[test]
fn three_boxed_interleaved() {
    let v: ThreeBoxed = parse!(
        ThreeBoxed,
        &br#"{"c2": 7, "a1": 2, "b1": 4, "id": 1, "c1": 6, "a2": 3, "b2": 5}"#[..]
    )
    .unwrap();
    assert_eq!(
        v,
        ThreeBoxed {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
            c: C { c1: 6, c2: 7 },
        }
    );
}
