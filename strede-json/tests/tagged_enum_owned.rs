//! Owned-family tagged-enum fixtures exercising the v3 `TagAwareMapOwned` path.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "type")]
enum Shape {
    Circle { radius: u32 },
    Rect { w: u32, h: u32 },
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Point {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "kind")]
enum WithNewtype {
    PtV(Point),
    Other { id: u32 },
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
fn circle_in_order() {
    let s: Shape = parse!(Shape, br#"{"type": "Circle", "radius": 7}"#).unwrap();
    assert_eq!(s, Shape::Circle { radius: 7 });
}

#[test]
fn circle_tag_after_field() {
    let s: Shape = parse!(Shape, br#"{"radius": 7, "type": "Circle"}"#).unwrap();
    assert_eq!(s, Shape::Circle { radius: 7 });
}

#[test]
fn rect_in_order() {
    let s: Shape = parse!(Shape, br#"{"type": "Rect", "w": 4, "h": 8}"#).unwrap();
    assert_eq!(s, Shape::Rect { w: 4, h: 8 });
}

#[test]
fn rect_tag_between_fields() {
    let s: Shape = parse!(Shape, br#"{"w": 4, "type": "Rect", "h": 8}"#).unwrap();
    assert_eq!(s, Shape::Rect { w: 4, h: 8 });
}

#[test]
fn unknown_tag_misses() {
    let v: Option<Shape> = parse!(Shape, br#"{"type": "Triangle"}"#);
    assert!(v.is_none());
}

#[test]
fn missing_field_misses() {
    let v: Option<Shape> = parse!(Shape, br#"{"type": "Rect", "w": 4}"#);
    assert!(v.is_none());
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "kind")]
enum Signal {
    Ping,
    Pong,
    Reset,
}

#[test]
fn unit_only_ping() {
    let s: Signal = parse!(Signal, br#"{"kind": "Ping"}"#).unwrap();
    assert_eq!(s, Signal::Ping);
}

#[test]
fn unit_only_reset_with_unknown_field() {
    let s: Signal = parse!(Signal, br#"{"extra": 1, "kind": "Reset", "more": null}"#).unwrap();
    assert_eq!(s, Signal::Reset);
}

#[test]
fn unit_only_unknown_tag_misses() {
    let v: Option<Signal> = parse!(Signal, br#"{"kind": "Other"}"#);
    assert!(v.is_none());
}

#[test]
fn newtype_variant() {
    let v: WithNewtype = parse!(WithNewtype, br#"{"kind": "PtV", "x": 1, "y": 2}"#).unwrap();
    assert_eq!(v, WithNewtype::PtV(Point { x: 1, y: 2 }));
}

#[test]
fn newtype_variant_tag_last() {
    let v: WithNewtype = parse!(WithNewtype, br#"{"x": 7, "y": 8, "kind": "PtV"}"#).unwrap();
    assert_eq!(v, WithNewtype::PtV(Point { x: 7, y: 8 }));
}

#[test]
fn newtype_alongside_struct_variant() {
    let v: WithNewtype = parse!(WithNewtype, br#"{"kind": "Other", "id": 42}"#).unwrap();
    assert_eq!(v, WithNewtype::Other { id: 42 });
}
