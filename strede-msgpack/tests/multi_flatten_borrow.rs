//! Borrow-family multi-flatten fixtures (2 and 3 flatten fields).

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, MsgpackDeserializer<'de>, Extra = ()>,
{
    let de = MsgpackDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

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
        parse::<Two>(&msg),
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
        parse::<Two>(&msg),
        Some(Two {
            id: 1,
            a: A { a1: 2, a2: 3 },
            b: B { b1: 4, b2: 5 },
        })
    );
}

#[test]
fn two_missing_a_field_misses() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("b1").as_slice(), &[4u8]),
        (fixstr("b2").as_slice(), &[5u8]),
    ]);
    assert_eq!(parse::<Two>(&msg), None);
}

#[test]
fn two_missing_b_field_misses() {
    let msg = build_map(&[
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("a1").as_slice(), &[2u8]),
        (fixstr("a2").as_slice(), &[3u8]),
        (fixstr("b1").as_slice(), &[4u8]),
    ]);
    assert_eq!(parse::<Two>(&msg), None);
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
        parse::<Three>(&msg),
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
        parse::<Three>(&msg),
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
    assert_eq!(parse::<Three>(&msg), None);
}
