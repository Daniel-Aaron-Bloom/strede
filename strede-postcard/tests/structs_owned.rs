//! Named struct, tuple struct, and nested struct deserialization via the
//! owned/chunked family. Mirrors `structs_borrow.rs`; `&'de str` fields
//! become `String` (no zero-copy borrow in the owned family).

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

// --- Type definitions ---

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Point {
    x: u32,
    y: u32,
}

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

#[derive(Debug, PartialEq, DeserializeOwned)]
struct WithDefault {
    x: u32,
    #[strede(default)]
    y: u32,
}

// --- Unit struct ---

#[test]
fn unit_struct_zero_bytes() {
    assert_eq!(parse_owned!(UnitStruct, &[]), Ok(Some(UnitStruct)));
}

#[test]
fn unit_struct_trailing_bytes_errors() {
    assert_eq!(
        parse_owned!(UnitStruct, &[0x01]).unwrap_err(),
        PostcardError::ExpectedEnd
    );
}

// --- Named structs ---

#[test]
fn point_two_fields() {
    let mut data = varint(1);
    data.extend_from_slice(&varint(2));
    assert_eq!(parse_owned!(Point, &data), Ok(Some(Point { x: 1, y: 2 })));
}

#[test]
fn point_larger_values() {
    let mut data = varint(300);
    data.extend_from_slice(&varint(400));
    assert_eq!(
        parse_owned!(Point, &data),
        Ok(Some(Point { x: 300, y: 400 }))
    );
}

#[test]
fn named_with_str() {
    let mut data = varint(42);
    data.extend_from_slice(&pstr("hello"));
    assert_eq!(
        parse_owned!(Named, &data),
        Ok(Some(Named {
            id: 42,
            label: "hello".to_string()
        }))
    );
}

#[test]
fn named_truncated_errors() {
    // only first field present
    let data = varint(1);
    assert_eq!(
        parse_owned!(Point, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}

#[test]
fn named_with_default_absent_errors() {
    // Postcard is positional: missing y causes UnexpectedEnd, not a graceful default.
    let data = varint(5);
    assert_eq!(
        parse_owned!(WithDefault, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}

#[test]
fn named_with_default_present() {
    let mut data = varint(5);
    data.extend_from_slice(&varint(10));
    assert_eq!(
        parse_owned!(WithDefault, &data),
        Ok(Some(WithDefault { x: 5, y: 10 }))
    );
}

// --- Tuple structs ---

#[test]
fn newtype_u32() {
    let data = varint(7);
    assert_eq!(parse_owned!(Wrapper, &data), Ok(Some(Wrapper(7))));
}

#[test]
fn pair_two_fields() {
    let mut data = varint(10);
    data.extend_from_slice(&varint(20));
    assert_eq!(parse_owned!(Pair, &data), Ok(Some(Pair(10, 20))));
}

// --- Flatten ---

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Inner {
    x: u32,
    y: u32,
}

// Wire layout: a=0, x=1, y=2, b=3 — declaration order with flatten inlined.
#[derive(Debug, PartialEq, DeserializeOwned)]
struct OuterFlat {
    a: u32,
    #[strede(flatten)]
    inner: Inner,
    b: u32,
}

#[test]
fn flatten_positional_order() {
    let mut data = varint(1);
    data.extend_from_slice(&varint(2));
    data.extend_from_slice(&varint(3));
    data.extend_from_slice(&varint(4));
    assert_eq!(
        parse_owned!(OuterFlat, &data),
        Ok(Some(OuterFlat {
            a: 1,
            inner: Inner { x: 2, y: 3 },
            b: 4
        }))
    );
}

// --- Raw tuples ---

#[test]
fn tuple_unit() {
    assert_eq!(parse_owned!((), &[]), Ok(Some(())));
}

#[test]
fn tuple_one() {
    assert_eq!(parse_owned!((u32,), &varint(42)), Ok(Some((42u32,))));
}

#[test]
fn tuple_two() {
    let mut data = varint(10);
    data.extend_from_slice(&varint(20));
    assert_eq!(parse_owned!((u32, u32), &data), Ok(Some((10u32, 20u32))));
}

#[test]
fn tuple_three() {
    let mut data = varint(1);
    data.extend_from_slice(&pstr("hi"));
    data.push(0x01); // bool true
    assert_eq!(
        parse_owned!((u32, String, bool), &data),
        Ok(Some((1u32, "hi".to_string(), true)))
    );
}

// --- Nested structs ---

#[test]
fn nested_struct() {
    let mut data = varint(5); // a
    data.extend_from_slice(&varint(1)); // inner.x
    data.extend_from_slice(&varint(2)); // inner.y
    data.push(0x01); // b = true
    assert_eq!(
        parse_owned!(Nested, &data),
        Ok(Some(Nested {
            a: 5,
            inner: Point { x: 1, y: 2 },
            b: true
        }))
    );
}
