//! Named struct, tuple struct, and nested struct deserialization via the
//! owned/chunked family. Mirrors `structs_borrow.rs`; `&'de str` fields
//! become `String` (no zero-copy borrow in the owned family). Tested under
//! the default `Standard` config only.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_bincode::{BincodeError, Standard};
use strede_derive::DeserializeOwned;

const E: Enc = Enc::STANDARD;

// --- Type definitions ---

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Point {
    x: u32,
    y: u32,
}

#[cfg(feature = "alloc")]
#[derive(Debug, PartialEq, DeserializeOwned)]
struct Named {
    id: u32,
    label: String,
}

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Wrapper(u32);

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Pair(u32, u32);

#[derive(Debug, PartialEq, DeserializeOwned)]
struct UnitStruct;

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Nested {
    a: u32,
    inner: Point,
    b: bool,
}

// --- Unit struct ---

#[test]
fn unit_struct_zero_bytes() {
    assert_eq!(parse_owned!(UnitStruct, Standard, &[]), Ok(Some(UnitStruct)));
}

#[test]
fn unit_struct_trailing_bytes_errors() {
    assert_eq!(
        parse_owned!(UnitStruct, Standard, &[0x01]).unwrap_err(),
        BincodeError::ExpectedEnd
    );
}

// --- Named structs ---

#[test]
fn point_two_fields() {
    let mut data = E.u32(1);
    data.extend_from_slice(&E.u32(2));
    assert_eq!(
        parse_owned!(Point, Standard, &data),
        Ok(Some(Point { x: 1, y: 2 }))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn named_with_str() {
    let mut data = E.u32(42);
    data.extend_from_slice(&E.str("hello"));
    assert_eq!(
        parse_owned!(Named, Standard, &data),
        Ok(Some(Named {
            id: 42,
            label: "hello".to_string()
        }))
    );
}

#[test]
fn named_truncated_errors() {
    let data = E.u32(1);
    assert_eq!(
        parse_owned!(Point, Standard, &data).unwrap_err(),
        BincodeError::UnexpectedEnd
    );
}

// --- Tuple structs ---

#[test]
fn newtype_u32() {
    let data = E.u32(7);
    assert_eq!(parse_owned!(Wrapper, Standard, &data), Ok(Some(Wrapper(7))));
}

#[test]
fn pair_two_fields() {
    let mut data = E.u32(10);
    data.extend_from_slice(&E.u32(20));
    assert_eq!(
        parse_owned!(Pair, Standard, &data),
        Ok(Some(Pair(10, 20)))
    );
}

// --- Flatten ---

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Inner {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, DeserializeOwned)]
struct OuterFlat {
    a: u32,
    #[strede(flatten)]
    inner: Inner,
    b: u32,
}

#[test]
fn flatten_positional_order() {
    let mut data = E.u32(1);
    data.extend_from_slice(&E.u32(2));
    data.extend_from_slice(&E.u32(3));
    data.extend_from_slice(&E.u32(4));
    assert_eq!(
        parse_owned!(OuterFlat, Standard, &data),
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
        parse_owned!((u32, u32), Standard, &data),
        Ok(Some((10u32, 20u32)))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn tuple_three() {
    let mut data = E.u32(1);
    data.extend_from_slice(&E.str("hi"));
    data.extend_from_slice(&E.bool(true));
    assert_eq!(
        parse_owned!((u32, String, bool), Standard, &data),
        Ok(Some((1u32, "hi".to_string(), true)))
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
        parse_owned!(Nested, Standard, &data),
        Ok(Some(Nested {
            a: 5,
            inner: Point { x: 1, y: 2 },
            b: true
        }))
    );
}
