//! Borrow-family multi-flatten fixtures (2 and 3 flatten fields).

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct A {
    a1: u32,
    a2: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct B {
    b1: u32,
    b2: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct C {
    c1: u32,
    c2: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Two {
    id: u32,
    #[strede(flatten)]
    a: A,
    #[strede(flatten)]
    b: B,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Three {
    id: u32,
    #[strede(flatten)]
    a: A,
    #[strede(flatten)]
    b: B,
    #[strede(flatten)]
    c: C,
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
fn two_in_order() {
    let v: Two = parse(r#"{"id": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5}"#).unwrap();
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
    let v: Two = parse(r#"{"b1": 4, "a1": 2, "id": 1, "b2": 5, "a2": 3}"#).unwrap();
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
fn two_missing_a_field_misses() {
    let v: Option<Two> = parse(r#"{"id": 1, "a1": 2, "b1": 4, "b2": 5}"#);
    assert!(v.is_none());
}

#[test]
fn two_missing_b_field_misses() {
    let v: Option<Two> = parse(r#"{"id": 1, "a1": 2, "a2": 3, "b1": 4}"#);
    assert!(v.is_none());
}

#[test]
fn three_in_order() {
    let v: Three =
        parse(r#"{"id": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5, "c1": 6, "c2": 7}"#).unwrap();
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
    let v: Three =
        parse(r#"{"c2": 7, "a1": 2, "b1": 4, "id": 1, "c1": 6, "a2": 3, "b2": 5}"#).unwrap();
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
    let v: Option<Three> = parse(r#"{"id": 1, "a1": 2, "a2": 3, "b1": 4, "b2": 5, "c1": 6}"#);
    assert!(v.is_none());
}

