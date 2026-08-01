//! Borrow-family multi-flatten fixtures (2 and 3 flatten fields).

extern crate std;
mod helpers;
use helpers::*;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    match block_on(T::deserialize(de, ())).unwrap() {
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
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("a1").as_slice(), &[uint_small(2)]),
        (tstr("a2").as_slice(), &[uint_small(3)]),
        (tstr("b1").as_slice(), &[uint_small(4)]),
        (tstr("b2").as_slice(), &[uint_small(5)]),
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
        (tstr("b1").as_slice(), &[uint_small(4)]),
        (tstr("a1").as_slice(), &[uint_small(2)]),
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("b2").as_slice(), &[uint_small(5)]),
        (tstr("a2").as_slice(), &[uint_small(3)]),
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
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("a1").as_slice(), &[uint_small(2)]),
        (tstr("b1").as_slice(), &[uint_small(4)]),
        (tstr("b2").as_slice(), &[uint_small(5)]),
    ]);
    assert_eq!(parse::<Two>(&msg), None);
}

#[test]
fn two_missing_b_field_misses() {
    let msg = build_map(&[
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("a1").as_slice(), &[uint_small(2)]),
        (tstr("a2").as_slice(), &[uint_small(3)]),
        (tstr("b1").as_slice(), &[uint_small(4)]),
    ]);
    assert_eq!(parse::<Two>(&msg), None);
}

#[test]
fn three_in_order() {
    let msg = build_map(&[
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("a1").as_slice(), &[uint_small(2)]),
        (tstr("a2").as_slice(), &[uint_small(3)]),
        (tstr("b1").as_slice(), &[uint_small(4)]),
        (tstr("b2").as_slice(), &[uint_small(5)]),
        (tstr("c1").as_slice(), &[uint_small(6)]),
        (tstr("c2").as_slice(), &[uint_small(7)]),
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
        (tstr("c2").as_slice(), &[uint_small(7)]),
        (tstr("a1").as_slice(), &[uint_small(2)]),
        (tstr("b1").as_slice(), &[uint_small(4)]),
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("c1").as_slice(), &[uint_small(6)]),
        (tstr("a2").as_slice(), &[uint_small(3)]),
        (tstr("b2").as_slice(), &[uint_small(5)]),
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
        (tstr("id").as_slice(), &[uint_small(1)]),
        (tstr("a1").as_slice(), &[uint_small(2)]),
        (tstr("a2").as_slice(), &[uint_small(3)]),
        (tstr("b1").as_slice(), &[uint_small(4)]),
        (tstr("b2").as_slice(), &[uint_small(5)]),
        (tstr("c1").as_slice(), &[uint_small(6)]),
    ]);
    assert_eq!(parse::<Three>(&msg), None);
}
