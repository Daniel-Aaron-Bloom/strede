//! Borrow-family struct deserialization.

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
struct Point {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Unit;

#[derive(Debug, PartialEq, Deserialize)]
struct Wrapper(u32);

#[derive(Debug, PartialEq, Deserialize)]
struct Named<'de> {
    id: u32,
    label: &'de str,
}

#[derive(Debug, PartialEq, Deserialize)]
struct WithDefault {
    x: u32,
    #[strede(default)]
    y: u32,
}

// --- Unit struct (nil) ---

#[test]
fn unit_struct_nil() {
    assert_eq!(parse::<Unit>(&[0xc0]), Some(Unit));
}

#[test]
fn unit_struct_miss() {
    assert_eq!(parse::<Unit>(&[42]), None);
}

// --- Transparent newtype ---

#[test]
fn newtype_u32() {
    // Wrapper(u32) is a tuple struct → deserializes from a 1-element array.
    // fixarray(1) = 0x91, then fixint 7.
    assert_eq!(parse::<Wrapper>(&[0x91, 7]), Some(Wrapper(7)));
}

// --- Named struct from map ---

#[test]
fn point_in_order() {
    let x_key = fixstr("x");
    let y_key = fixstr("y");
    let x_val = [1u8]; // positive fixint
    let y_val = [2u8];
    let msg = build_map(&[(x_key.as_slice(), &x_val), (y_key.as_slice(), &y_val)]);
    assert_eq!(parse::<Point>(&msg), Some(Point { x: 1, y: 2 }));
}

#[test]
fn point_reversed_keys() {
    let x_key = fixstr("x");
    let y_key = fixstr("y");
    let msg = build_map(&[(y_key.as_slice(), &[9u8]), (x_key.as_slice(), &[3u8])]);
    assert_eq!(parse::<Point>(&msg), Some(Point { x: 3, y: 9 }));
}

#[test]
fn point_missing_field_misses() {
    let x_key = fixstr("x");
    let msg = build_map(&[(x_key.as_slice(), &[1u8])]);
    assert_eq!(parse::<Point>(&msg), None);
}

#[test]
fn point_extra_field_misses() {
    // without allow_unknown_fields, extra keys cause Miss
    let x_key = fixstr("x");
    let y_key = fixstr("y");
    let z_key = fixstr("z");
    let msg = build_map(&[
        (x_key.as_slice(), &[1u8]),
        (y_key.as_slice(), &[2u8]),
        (z_key.as_slice(), &[3u8]),
    ]);
    assert_eq!(parse::<Point>(&msg), None);
}

#[test]
fn named_struct_with_str() {
    let id_key = fixstr("id");
    let label_key = fixstr("label");
    let id_val = [42u8];
    let label_val = fixstr("hello");
    let msg = build_map(&[
        (id_key.as_slice(), &id_val),
        (label_key.as_slice(), label_val.as_slice()),
    ]);
    assert_eq!(
        parse::<Named<'_>>(&msg),
        Some(Named {
            id: 42,
            label: "hello"
        })
    );
}

#[test]
fn default_field_absent() {
    let x_key = fixstr("x");
    let msg = build_map(&[(x_key.as_slice(), &[5u8])]);
    assert_eq!(parse::<WithDefault>(&msg), Some(WithDefault { x: 5, y: 0 }));
}

#[test]
fn default_field_present() {
    let x_key = fixstr("x");
    let y_key = fixstr("y");
    let msg = build_map(&[(x_key.as_slice(), &[5u8]), (y_key.as_slice(), &[10u8])]);
    assert_eq!(
        parse::<WithDefault>(&msg),
        Some(WithDefault { x: 5, y: 10 })
    );
}
