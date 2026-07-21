//! Owned-family externally-tagged enum deserialization.
#![recursion_limit = "512"]

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
                match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                {
                    Probe::Hit((_, v)) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
enum Signal {
    Ping,
    Pong,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
enum Event {
    Ping,
    Move { x: f64, y: f64 },
    Wrap(u32),
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
enum Tupled {
    Point(f64, f64),
}

// --- Unit variants ---

#[test]
fn unit_ping() {
    let msg = fixstr("Ping");
    assert_eq!(parse!(Signal, &msg), Some(Signal::Ping));
}

#[test]
fn unit_pong() {
    let msg = fixstr("Pong");
    assert_eq!(parse!(Signal, &msg), Some(Signal::Pong));
}

#[test]
fn unit_unknown_misses() {
    let msg = fixstr("Other");
    assert_eq!(parse!(Signal, &msg), None);
}

#[test]
fn wrong_token_type_misses() {
    assert_eq!(parse!(Signal, &[0x05u8]), None);
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
    build_map(&[(outer_key.as_slice(), payload.as_slice())])
}

#[test]
fn struct_variant_fields_in_order() {
    let msg = make_move(1.5, 2.5, true);
    assert_eq!(parse!(Event, &msg), Some(Event::Move { x: 1.5, y: 2.5 }));
}

#[test]
fn struct_variant_fields_reversed() {
    let msg = make_move(1.5, 2.5, false);
    assert_eq!(parse!(Event, &msg), Some(Event::Move { x: 1.5, y: 2.5 }));
}

// --- Newtype variant ---

#[test]
fn newtype_variant() {
    let outer_key = fixstr("Wrap");
    let value = [7u8]; // fixint 7
    let msg = build_map(&[(outer_key.as_slice(), value.as_slice())]);
    assert_eq!(parse!(Event, &msg), Some(Event::Wrap(7)));
}

// --- Tuple variant ---

#[test]
fn tuple_variant() {
    let outer_key = fixstr("Point");
    let x_val = float64(3.0);
    let y_val = float64(4.0);
    let mut payload = vec![fixarray(2)];
    payload.extend_from_slice(&x_val);
    payload.extend_from_slice(&y_val);
    let msg = build_map(&[(outer_key.as_slice(), payload.as_slice())]);
    assert_eq!(parse!(Tupled, &msg), Some(Tupled::Point(3.0, 4.0)));
}

// --- Edge cases ---

#[test]
fn empty_map_misses() {
    let msg = vec![fixmap(0)];
    assert_eq!(parse!(Event, &msg), None);
}

#[test]
fn extra_outer_pairs_misses() {
    let k1 = fixstr("Ping");
    let k2 = fixstr("extra");
    let nil = [0xc0u8];
    let v2 = [0x01u8];
    let msg = build_map(&[
        (k1.as_slice(), nil.as_slice()),
        (k2.as_slice(), v2.as_slice()),
    ]);
    assert_eq!(parse!(Event, &msg), None);
}

#[test]
fn unit_variant_in_mixed_enum() {
    let msg = fixstr("Ping");
    assert_eq!(parse!(Event, &msg), Some(Event::Ping));
}
