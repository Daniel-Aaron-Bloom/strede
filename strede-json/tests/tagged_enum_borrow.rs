//! Borrow-family tagged-enum fixtures exercising the v3 `TagAwareMap` path.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "type")]
enum Shape {
    Circle { radius: u32 },
    Rect { w: u32, h: u32 },
}

#[derive(Debug, PartialEq, Deserialize)]
struct Point {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "kind")]
enum WithNewtype {
    PtV(Point),
    Other { id: u32 },
}

fn parse<'de, T>(input: &'de str) -> T
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    let probe = block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap();
    match probe {
        Probe::Hit((_, v)) => v,
        Probe::Miss => panic!("Miss"),
    }
}

#[test]
fn circle_in_order() {
    let s: Shape = parse(r#"{"type": "Circle", "radius": 7}"#);
    assert_eq!(s, Shape::Circle { radius: 7 });
}

#[test]
fn circle_tag_after_field() {
    let s: Shape = parse(r#"{"radius": 7, "type": "Circle"}"#);
    assert_eq!(s, Shape::Circle { radius: 7 });
}

#[test]
fn rect_in_order() {
    let s: Shape = parse(r#"{"type": "Rect", "w": 4, "h": 8}"#);
    assert_eq!(s, Shape::Rect { w: 4, h: 8 });
}

#[test]
fn rect_tag_between_fields() {
    let s: Shape = parse(r#"{"w": 4, "type": "Rect", "h": 8}"#);
    assert_eq!(s, Shape::Rect { w: 4, h: 8 });
}

#[test]
fn unknown_tag_misses() {
    let de = JsonDeserializer::new(br#"{"type": "Triangle"}"#);
    let probe = block_on(<Shape as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap();
    assert!(matches!(probe, Probe::Miss));
}

#[test]
fn missing_field_misses() {
    let de = JsonDeserializer::new(br#"{"type": "Rect", "w": 4}"#);
    let probe = block_on(<Shape as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap();
    assert!(matches!(probe, Probe::Miss));
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "kind")]
enum Signal {
    Ping,
    Pong,
    Reset,
}

#[test]
fn unit_only_ping() {
    let s: Signal = parse(r#"{"kind": "Ping"}"#);
    assert_eq!(s, Signal::Ping);
}

#[test]
fn unit_only_reset_with_unknown_field() {
    let s: Signal = parse(r#"{"extra": 1, "kind": "Reset", "more": null}"#);
    assert_eq!(s, Signal::Reset);
}

#[test]
fn unit_only_unknown_tag_misses() {
    let de = JsonDeserializer::new(br#"{"kind": "Other"}"#);
    let probe = block_on(<Signal as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap();
    assert!(matches!(probe, Probe::Miss));
}

#[test]
fn newtype_variant() {
    let v: WithNewtype = parse(r#"{"kind": "PtV", "x": 1, "y": 2}"#);
    assert_eq!(v, WithNewtype::PtV(Point { x: 1, y: 2 }));
}

#[test]
fn newtype_variant_tag_last() {
    let v: WithNewtype = parse(r#"{"x": 7, "y": 8, "kind": "PtV"}"#);
    assert_eq!(v, WithNewtype::PtV(Point { x: 7, y: 8 }));
}

#[test]
fn newtype_alongside_struct_variant() {
    let v: WithNewtype = parse(r#"{"kind": "Other", "id": 42}"#);
    assert_eq!(v, WithNewtype::Other { id: 42 });
}
