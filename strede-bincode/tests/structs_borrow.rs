//! Named struct, tuple struct, and nested struct deserialization.
//!
//! Bincode encodes structs as positional fields in declaration order, same
//! as postcard — no count prefix, no field names on the wire. Tested under
//! the default `Standard` config only; the positional-matching machinery
//! itself is config-independent (see `strede-postcard`'s identical stance).

mod helpers;
use helpers::*;

use strede_bincode::{BincodeError, Standard};
use strede_derive::Deserialize;

const E: Enc = Enc::STANDARD;

// --- Type definitions ---

#[derive(Debug, PartialEq, Deserialize)]
struct Point {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Named<'de> {
    id: u32,
    label: &'de str,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Wrapper(u32);

#[derive(Debug, PartialEq, Deserialize)]
struct Pair(u32, u32);

#[derive(Debug, PartialEq, Deserialize)]
struct UnitStruct;

#[derive(Debug, PartialEq, Deserialize)]
struct Nested {
    a: u32,
    inner: Point,
    b: bool,
}

// --- Unit struct ---

#[test]
fn unit_struct_zero_bytes() {
    assert_eq!(parse::<UnitStruct, Standard>(&[]), Ok(Some(UnitStruct)));
}

#[test]
fn unit_struct_trailing_bytes_errors() {
    assert_eq!(
        parse_err::<UnitStruct, Standard>(&[0x01]),
        BincodeError::ExpectedEnd
    );
}

// --- Named structs ---

#[test]
fn point_two_fields() {
    let mut data = E.u32(1);
    data.extend_from_slice(&E.u32(2));
    assert_eq!(
        parse::<Point, Standard>(&data),
        Ok(Some(Point { x: 1, y: 2 }))
    );
}

#[test]
fn point_larger_values() {
    let mut data = E.u32(300);
    data.extend_from_slice(&E.u32(400));
    assert_eq!(
        parse::<Point, Standard>(&data),
        Ok(Some(Point { x: 300, y: 400 }))
    );
}

#[test]
fn named_with_str() {
    let mut data = E.u32(42);
    data.extend_from_slice(&E.str("hello"));
    assert_eq!(
        parse::<Named<'_>, Standard>(&data),
        Ok(Some(Named {
            id: 42,
            label: "hello"
        }))
    );
}

#[test]
fn named_truncated_errors() {
    let data = E.u32(1);
    assert_eq!(
        parse_err::<Point, Standard>(&data),
        BincodeError::UnexpectedEnd
    );
}

// --- Tuple structs ---

#[test]
fn newtype_u32() {
    let data = E.u32(7);
    assert_eq!(parse::<Wrapper, Standard>(&data), Ok(Some(Wrapper(7))));
}

#[test]
fn pair_two_fields() {
    let mut data = E.u32(10);
    data.extend_from_slice(&E.u32(20));
    assert_eq!(parse::<Pair, Standard>(&data), Ok(Some(Pair(10, 20))));
}

// --- Flatten ---

#[derive(Debug, PartialEq, Deserialize)]
struct Inner {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct OuterFlat {
    a: u32,
    #[strede(flatten)]
    inner: Inner,
    b: u32,
}

#[test]
fn flatten_positional_order() {
    // Wire: a=1, x=2, y=3, b=4 — declaration order with flatten inlined.
    let mut data = E.u32(1);
    data.extend_from_slice(&E.u32(2));
    data.extend_from_slice(&E.u32(3));
    data.extend_from_slice(&E.u32(4));
    assert_eq!(
        parse::<OuterFlat, Standard>(&data),
        Ok(Some(OuterFlat {
            a: 1,
            inner: Inner { x: 2, y: 3 },
            b: 4
        }))
    );
}

// --- Raw tuples ---

#[test]
fn tuple_two() {
    let mut data = E.u32(10);
    data.extend_from_slice(&E.u32(20));
    assert_eq!(
        parse::<(u32, u32), Standard>(&data),
        Ok(Some((10u32, 20u32)))
    );
}

#[test]
fn tuple_three() {
    let mut data = E.u32(1);
    data.extend_from_slice(&E.str("hi"));
    data.extend_from_slice(&E.bool(true));
    assert_eq!(
        parse::<(u32, &str, bool), Standard>(&data),
        Ok(Some((1u32, "hi", true)))
    );
}

// --- Nested structs ---

#[test]
fn nested_struct() {
    let mut data = E.u32(5); // a
    data.extend_from_slice(&E.u32(1)); // inner.x
    data.extend_from_slice(&E.u32(2)); // inner.y
    data.extend_from_slice(&E.bool(true)); // b
    assert_eq!(
        parse::<Nested, Standard>(&data),
        Ok(Some(Nested {
            a: 5,
            inner: Point { x: 1, y: 2 },
            b: true
        }))
    );
}
