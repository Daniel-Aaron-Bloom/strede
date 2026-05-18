//! Owned-family adjacently-tagged enum fixtures.
#![recursion_limit = "512"]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "t", content = "c")]
enum Signal {
    Ping,
    Pong,
    Reset,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Point {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "t", content = "c")]
enum Msg {
    Move { x: u32, y: u32 },
    Tup(u32, u32),
    New(Point),
    Reset,
}

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
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

#[test]
fn unit_only_ping() {
    let s: Signal = parse!(Signal, br#"{"t": "Ping"}"#).unwrap();
    assert_eq!(s, Signal::Ping);
}

#[test]
fn unit_only_unknown_tag_misses() {
    let v: Option<Signal> = parse!(Signal, br#"{"t": "Other"}"#);
    assert!(v.is_none());
}

#[test]
fn struct_variant() {
    let m: Msg = parse!(Msg, br#"{"t": "Move", "c": {"x": 1, "y": 2}}"#).unwrap();
    assert_eq!(m, Msg::Move { x: 1, y: 2 });
}

#[test]
fn struct_variant_keys_reversed() {
    let m: Msg = parse!(Msg, br#"{"c": {"x": 3, "y": 4}, "t": "Move"}"#).unwrap();
    assert_eq!(m, Msg::Move { x: 3, y: 4 });
}

#[test]
fn tuple_variant() {
    let m: Msg = parse!(Msg, br#"{"t": "Tup", "c": [9, 10]}"#).unwrap();
    assert_eq!(m, Msg::Tup(9, 10));
}

#[test]
fn newtype_variant() {
    let m: Msg = parse!(Msg, br#"{"t": "New", "c": {"x": 5, "y": 6}}"#).unwrap();
    assert_eq!(m, Msg::New(Point { x: 5, y: 6 }));
}

#[test]
fn unit_in_mixed_enum() {
    let m: Msg = parse!(Msg, br#"{"t": "Reset"}"#).unwrap();
    assert_eq!(m, Msg::Reset);
}

#[test]
fn unknown_variant_misses() {
    let v: Option<Msg> = parse!(Msg, br#"{"t": "Nope", "c": {}}"#);
    assert!(v.is_none());
}
