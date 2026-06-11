//! Borrow-family externally-tagged enum deserialization.
#![recursion_limit = "256"]

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
enum Signal {
    Ping,
    Pong,
}

#[derive(Debug, PartialEq, Deserialize)]
enum Event {
    Ping,
    Move { x: f64, y: f64 },
    Wrap(u32),
}

#[derive(Debug, PartialEq, Deserialize)]
enum Tupled {
    Point(f64, f64),
}

// --- Unit variants ---

#[test]
fn unit_ping() {
    let msg = fixstr("Ping");
    assert_eq!(parse::<Signal>(&msg), Some(Signal::Ping));
}

#[test]
fn unit_pong() {
    let msg = fixstr("Pong");
    assert_eq!(parse::<Signal>(&msg), Some(Signal::Pong));
}

#[test]
fn unit_unknown_misses() {
    let msg = fixstr("Other");
    assert_eq!(parse::<Signal>(&msg), None);
}

#[test]
fn wrong_token_type_misses() {
    // fixint 0x05 is not a string or map
    assert_eq!(parse::<Signal>(&[0x05u8]), None);
}

// --- Struct variant ---

fn make_move(x: f64, y: f64, key_first: bool) -> Vec<u8> {
    let outer_key = fixstr("Move");
    let x_key = fixstr("x");
    let y_key = fixstr("y");
    let x_val = float64(x);
    let y_val = float64(y);
    let payload = if key_first {
        build_map(&[
            (x_key.as_slice(), x_val.as_slice()),
            (y_key.as_slice(), y_val.as_slice()),
        ])
    } else {
        build_map(&[
            (y_key.as_slice(), y_val.as_slice()),
            (x_key.as_slice(), x_val.as_slice()),
        ])
    };
    // outer single-key map: {"Move": <payload>}
    build_map(&[(outer_key.as_slice(), payload.as_slice())])
}

#[test]
fn struct_variant_fields_in_order() {
    let msg = make_move(1.5, 2.5, true);
    assert_eq!(parse::<Event>(&msg), Some(Event::Move { x: 1.5, y: 2.5 }));
}

#[test]
fn struct_variant_fields_reversed() {
    let msg = make_move(1.5, 2.5, false);
    assert_eq!(parse::<Event>(&msg), Some(Event::Move { x: 1.5, y: 2.5 }));
}

// --- Newtype variant ---

#[test]
fn newtype_variant() {
    let outer_key = fixstr("Wrap");
    let value = [7u8]; // fixint 7
    let msg = build_map(&[(outer_key.as_slice(), value.as_slice())]);
    assert_eq!(parse::<Event>(&msg), Some(Event::Wrap(7)));
}

// --- Tuple variant ---

#[test]
fn tuple_variant() {
    let outer_key = fixstr("Point");
    let x_val = float64(3.0);
    let y_val = float64(4.0);
    // value is a fixarray([x, y])
    let mut payload = vec![fixarray(2)];
    payload.extend_from_slice(&x_val);
    payload.extend_from_slice(&y_val);
    let msg = build_map(&[(outer_key.as_slice(), payload.as_slice())]);
    assert_eq!(parse::<Tupled>(&msg), Some(Tupled::Point(3.0, 4.0)));
}

// --- Edge cases ---

#[test]
fn empty_map_misses() {
    // fixmap with 0 pairs
    let msg = vec![fixmap(0)];
    assert_eq!(parse::<Event>(&msg), None);
}

#[test]
fn extra_outer_pairs_misses() {
    // map with 2 pairs — externally-tagged requires exactly 1
    let k1 = fixstr("Ping");
    let k2 = fixstr("extra");
    let nil = [0xc0u8];
    let v2 = [0x01u8];
    let msg = build_map(&[
        (k1.as_slice(), nil.as_slice()),
        (k2.as_slice(), v2.as_slice()),
    ]);
    assert_eq!(parse::<Event>(&msg), None);
}

#[test]
fn unit_variant_in_mixed_enum() {
    let msg = fixstr("Ping");
    assert_eq!(parse::<Event>(&msg), Some(Event::Ping));
}
