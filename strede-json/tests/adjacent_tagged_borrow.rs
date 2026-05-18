//! Borrow-family adjacently-tagged enum fixtures.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "t", content = "c")]
enum Signal {
    Ping,
    Pong,
    Reset,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Point {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "t", content = "c")]
enum Msg {
    Move { x: u32, y: u32 },
    Tup(u32, u32),
    New(Point),
    Reset,
}

fn parse<'de, T>(input: &'de str) -> Option<T>
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn unit_only_ping() {
    let s: Signal = parse(r#"{"t": "Ping"}"#).unwrap();
    assert_eq!(s, Signal::Ping);
}

#[test]
fn unit_only_pong() {
    let s: Signal = parse(r#"{"t": "Pong"}"#).unwrap();
    assert_eq!(s, Signal::Pong);
}

#[test]
fn unit_only_unknown_tag_misses() {
    let v: Option<Signal> = parse(r#"{"t": "Other"}"#);
    assert!(v.is_none());
}

#[test]
fn struct_variant() {
    let m: Msg = parse(r#"{"t": "Move", "c": {"x": 1, "y": 2}}"#).unwrap();
    assert_eq!(m, Msg::Move { x: 1, y: 2 });
}

#[test]
fn struct_variant_keys_reversed() {
    let m: Msg = parse(r#"{"c": {"x": 3, "y": 4}, "t": "Move"}"#).unwrap();
    assert_eq!(m, Msg::Move { x: 3, y: 4 });
}

#[test]
fn tuple_variant() {
    let m: Msg = parse(r#"{"t": "Tup", "c": [9, 10]}"#).unwrap();
    assert_eq!(m, Msg::Tup(9, 10));
}

#[test]
fn newtype_variant() {
    let m: Msg = parse(r#"{"t": "New", "c": {"x": 5, "y": 6}}"#).unwrap();
    assert_eq!(m, Msg::New(Point { x: 5, y: 6 }));
}

#[test]
fn unit_in_mixed_enum() {
    let m: Msg = parse(r#"{"t": "Reset"}"#).unwrap();
    assert_eq!(m, Msg::Reset);
}

#[test]
fn unknown_variant_misses() {
    let v: Option<Msg> = parse(r#"{"t": "Nope", "c": {}}"#);
    assert!(v.is_none());
}
