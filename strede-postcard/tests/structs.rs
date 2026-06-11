//! Named struct, tuple struct, and nested struct deserialization.
//!
//! Postcard encodes structs as positional fields in declaration order.
//! Both named and tuple structs use the map path with `deserialize_key_by_index`.
//! No count prefix is emitted — fields are bare consecutive wire values.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Result<Option<T>, PostcardError>
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Ok(Some(v)),
        Probe::Miss => Ok(None),
    }
}

fn parse_err<'de, T>(input: &'de [u8]) -> PostcardError
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())) {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    }
}

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

#[derive(Debug, PartialEq, Deserialize)]
struct WithDefault {
    x: u32,
    #[strede(default)]
    y: u32,
}

// NOTE: postcard is positional — fields cannot be absent from the wire data even
// if they have `#[strede(default)]`. The default only fires when the arm stack
// exhausts (unsatisfied_count > 0 at Done). In practice, a missing positional
// field causes UnexpectedEnd when reading the next value, not a graceful default.

// --- Unit struct ---

#[test]
fn unit_struct_zero_bytes() {
    assert_eq!(parse::<UnitStruct>(&[]), Ok(Some(UnitStruct)));
}

#[test]
fn unit_struct_trailing_bytes_errors() {
    assert_eq!(parse_err::<UnitStruct>(&[0x01]), PostcardError::ExpectedEnd);
}

// --- Named structs ---

#[test]
fn point_two_fields() {
    // Postcard: fields in declaration order, no names on wire.
    // x=1 as varint(1) = [0x01], y=2 as varint(2) = [0x02]
    let mut data = varint(1);
    data.extend_from_slice(&varint(2));
    assert_eq!(parse::<Point>(&data), Ok(Some(Point { x: 1, y: 2 })));
}

#[test]
fn point_larger_values() {
    let mut data = varint(300);
    data.extend_from_slice(&varint(400));
    assert_eq!(parse::<Point>(&data), Ok(Some(Point { x: 300, y: 400 })));
}

#[test]
fn named_with_str() {
    let mut data = varint(42);
    data.extend_from_slice(&pstr("hello"));
    assert_eq!(
        parse::<Named<'_>>(&data),
        Ok(Some(Named { id: 42, label: "hello" }))
    );
}

#[test]
fn named_truncated_errors() {
    // only first field present
    let data = varint(1);
    assert_eq!(parse_err::<Point>(&data), PostcardError::UnexpectedEnd);
}

#[test]
fn named_with_default_absent_errors() {
    // Postcard is positional: missing y causes UnexpectedEnd, not a graceful default.
    let data = varint(5);
    assert_eq!(parse_err::<WithDefault>(&data), PostcardError::UnexpectedEnd);
}

#[test]
fn named_with_default_present() {
    let mut data = varint(5);
    data.extend_from_slice(&varint(10));
    assert_eq!(parse::<WithDefault>(&data), Ok(Some(WithDefault { x: 5, y: 10 })));
}

// --- Tuple structs ---

#[test]
fn newtype_u32() {
    // Wrapper(u32): bare field, no count prefix
    let data = varint(7);
    assert_eq!(parse::<Wrapper>(&data), Ok(Some(Wrapper(7))));
}

#[test]
fn pair_two_fields() {
    // Pair(u32, u32): two bare fields, no count prefix
    let mut data = varint(10);
    data.extend_from_slice(&varint(20));
    assert_eq!(parse::<Pair>(&data), Ok(Some(Pair(10, 20))));
}

// --- Flatten ---

#[derive(Debug, PartialEq, Deserialize)]
struct Inner {
    x: u32,
    y: u32,
}

// Wire layout: a=0, x=1, y=2, b=3 — declaration order with flatten inlined.
// The derive splits outer arms into before/after the flatten field so that
// `b` gets field index 3 (after x=1, y=2), not index 1.
#[derive(Debug, PartialEq, Deserialize)]
struct OuterFlat {
    a: u32,
    #[strede(flatten)]
    inner: Inner,
    b: u32,
}

#[test]
fn flatten_positional_order() {
    // Wire: a=1, x=2, y=3, b=4 — declaration order with flatten inlined
    let mut data = varint(1);
    data.extend_from_slice(&varint(2));
    data.extend_from_slice(&varint(3));
    data.extend_from_slice(&varint(4));
    assert_eq!(
        parse::<OuterFlat>(&data),
        Ok(Some(OuterFlat { a: 1, inner: Inner { x: 2, y: 3 }, b: 4 }))
    );
}

// --- Raw tuples ---

#[test]
fn tuple_unit() {
    assert_eq!(parse::<()>(&[]), Ok(Some(())));
}

#[test]
fn tuple_one() {
    assert_eq!(parse::<(u32,)>(&varint(42)), Ok(Some((42u32,))));
}

#[test]
fn tuple_two() {
    let mut data = varint(10);
    data.extend_from_slice(&varint(20));
    assert_eq!(parse::<(u32, u32)>(&data), Ok(Some((10u32, 20u32))));
}

#[test]
fn tuple_three() {
    let mut data = varint(1);
    data.extend_from_slice(&pstr("hi"));
    data.push(0x01); // bool true
    assert_eq!(parse::<(u32, &str, bool)>(&data), Ok(Some((1u32, "hi", true))));
}

// --- Nested structs ---

#[test]
fn nested_struct() {
    // Nested { a: 5, inner: Point { x: 1, y: 2 }, b: true }
    let mut data = varint(5);     // a
    data.extend_from_slice(&varint(1));  // inner.x
    data.extend_from_slice(&varint(2));  // inner.y
    data.push(0x01);              // b = true
    assert_eq!(
        parse::<Nested>(&data),
        Ok(Some(Nested { a: 5, inner: Point { x: 1, y: 2 }, b: true }))
    );
}
