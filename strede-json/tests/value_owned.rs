//! Owned-family `ValueOwned` and `RawValueOwned` fixtures.

#![cfg(all(feature = "alloc", not(feature = "arbitrary_precision")))]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_json::{RawValueOwned, ValueOwned};
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
fn value_null() {
    let v: ValueOwned = parse!(ValueOwned, b"null").unwrap();
    assert_eq!(v, ValueOwned::Null);
}

#[test]
fn value_bool() {
    let v: ValueOwned = parse!(ValueOwned, b"true").unwrap();
    assert_eq!(v, ValueOwned::Bool(true));
}

#[test]
fn value_number() {
    let v: ValueOwned = parse!(ValueOwned, b"42").unwrap();
    match v {
        ValueOwned::Number(n) => assert_eq!(n.as_u64(), Some(42)),
        _ => panic!("expected Number"),
    }
}

#[test]
fn value_string() {
    let v: ValueOwned = parse!(ValueOwned, br#""hello""#).unwrap();
    assert_eq!(v, ValueOwned::String("hello".into()));
}

#[test]
fn value_array() {
    let v: ValueOwned = parse!(ValueOwned, br#"[1, 2, 3]"#).unwrap();
    match v {
        ValueOwned::Array(items) => assert_eq!(items.len(), 3),
        _ => panic!("expected Array"),
    }
}

#[test]
fn value_object_nested() {
    let v: ValueOwned = parse!(ValueOwned, br#"{"a": {"b": [1, 2]}, "c": null}"#).unwrap();
    match v {
        ValueOwned::Object(pairs) => {
            assert_eq!(pairs.len(), 2);
            assert_eq!(pairs[0].0, "a");
            assert_eq!(pairs[1].0, "c");
            assert_eq!(pairs[1].1, ValueOwned::Null);
        }
        _ => panic!("expected Object"),
    }
}

#[test]
fn raw_value_owned() {
    use strede_derive::DeserializeOwned as DeriveDeserializeOwned;

    #[derive(DeriveDeserializeOwned)]
    struct Wrap {
        id: u32,
        raw: RawValueOwned,
    }

    let parsed: Wrap = parse!(Wrap, br#"{"id": 7, "raw": {"x": [1, 2, 3]}}"#).unwrap();
    assert_eq!(parsed.id, 7);
    // RawValueOwned reconstructs from the token stream, so the captured bytes
    // are canonical/minified — interior whitespace is dropped.
    assert_eq!(parsed.raw.as_str(), r#"{"x":[1,2,3]}"#);
}
