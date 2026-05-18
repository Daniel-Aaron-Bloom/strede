//! Borrow-family single-flatten fixtures exercising the v3 `FlattenMapAccess` path.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Inner {
    a: u32,
    b: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Outer {
    id: u32,
    #[strede(flatten)]
    inner: Inner,
}

fn parse<'de, T>(input: &'de str) -> Option<T>
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn outer_then_inner() {
    let o: Outer = parse(r#"{"id": 7, "a": 1, "b": 2}"#).unwrap();
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
    let o: Outer = parse(r#"{"a": 1, "id": 7, "b": 2}"#).unwrap();
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
    let o: Outer = parse(r#"{"a": 1, "b": 2, "id": 7}"#).unwrap();
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
    let v: Option<Outer> = parse(r#"{"a": 1, "b": 2}"#);
    assert!(v.is_none());
}

#[test]
fn missing_inner_field_misses() {
    let v: Option<Outer> = parse(r#"{"id": 7, "a": 1}"#);
    assert!(v.is_none());
}
