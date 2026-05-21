//! Borrow-family internally-tagged enum fixtures.

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
#[strede(tag = "type")]
enum Shape {
    Circle { radius: u32 },
    Rect { w: u32, h: u32 },
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "kind")]
enum Signal {
    Ping,
    Pong,
    Reset,
}

fn shape_circle(radius: u8, tag_first: bool) -> Vec<u8> {
    let type_key = fixstr("type");
    let type_val = fixstr("Circle");
    let radius_key = fixstr("radius");
    let radius_val = [radius];
    if tag_first {
        build_map(&[
            (type_key.as_slice(), type_val.as_slice()),
            (radius_key.as_slice(), &radius_val),
        ])
    } else {
        build_map(&[
            (radius_key.as_slice(), &radius_val),
            (type_key.as_slice(), type_val.as_slice()),
        ])
    }
}

#[test]
fn circle_tag_first() {
    let msg = shape_circle(7, true);
    assert_eq!(parse::<Shape>(&msg), Some(Shape::Circle { radius: 7 }));
}

#[test]
fn circle_tag_last() {
    let msg = shape_circle(7, false);
    assert_eq!(parse::<Shape>(&msg), Some(Shape::Circle { radius: 7 }));
}

#[test]
fn rect_in_order() {
    let type_key = fixstr("type");
    let type_val = fixstr("Rect");
    let w_key = fixstr("w");
    let h_key = fixstr("h");
    let msg = build_map(&[
        (type_key.as_slice(), type_val.as_slice()),
        (w_key.as_slice(), &[4u8]),
        (h_key.as_slice(), &[8u8]),
    ]);
    assert_eq!(parse::<Shape>(&msg), Some(Shape::Rect { w: 4, h: 8 }));
}

#[test]
fn rect_tag_between_fields() {
    let type_key = fixstr("type");
    let type_val = fixstr("Rect");
    let w_key = fixstr("w");
    let h_key = fixstr("h");
    let msg = build_map(&[
        (w_key.as_slice(), &[4u8]),
        (type_key.as_slice(), type_val.as_slice()),
        (h_key.as_slice(), &[8u8]),
    ]);
    assert_eq!(parse::<Shape>(&msg), Some(Shape::Rect { w: 4, h: 8 }));
}

#[test]
fn unknown_tag_misses() {
    let type_key = fixstr("type");
    let type_val = fixstr("Triangle");
    let msg = build_map(&[(type_key.as_slice(), type_val.as_slice())]);
    assert_eq!(parse::<Shape>(&msg), None);
}

#[test]
fn missing_required_field_misses() {
    let type_key = fixstr("type");
    let type_val = fixstr("Rect");
    let w_key = fixstr("w");
    // h is missing
    let msg = build_map(&[
        (type_key.as_slice(), type_val.as_slice()),
        (w_key.as_slice(), &[4u8]),
    ]);
    assert_eq!(parse::<Shape>(&msg), None);
}

#[test]
fn unit_only_ping() {
    let kind_key = fixstr("kind");
    let kind_val = fixstr("Ping");
    let msg = build_map(&[(kind_key.as_slice(), kind_val.as_slice())]);
    assert_eq!(parse::<Signal>(&msg), Some(Signal::Ping));
}

#[test]
fn unit_only_reset_with_unknown_field() {
    let kind_key = fixstr("kind");
    let kind_val = fixstr("Reset");
    let extra_key = fixstr("extra");
    let msg = build_map(&[
        (extra_key.as_slice(), &[1u8]),
        (kind_key.as_slice(), kind_val.as_slice()),
    ]);
    assert_eq!(parse::<Signal>(&msg), Some(Signal::Reset));
}

#[test]
fn unit_unknown_tag_misses() {
    let kind_key = fixstr("kind");
    let kind_val = fixstr("Other");
    let msg = build_map(&[(kind_key.as_slice(), kind_val.as_slice())]);
    assert_eq!(parse::<Signal>(&msg), None);
}
