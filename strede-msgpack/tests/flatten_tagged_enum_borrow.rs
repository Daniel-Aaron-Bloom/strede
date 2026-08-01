//! `#[strede(flatten)]` on a tagged/adjacently-tagged enum's struct-variant
//! field — borrow family. Regression coverage for a bug where the struct-
//! variant helper ignored `cf.flatten` entirely and treated a flatten field
//! as an ordinary nested-map field, silently breaking the merge-into-parent
//! semantics `flatten` has everywhere else.
#![recursion_limit = "512"]

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
struct Extra {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "type")]
enum Shape {
    Circle {
        radius: u32,
        #[strede(flatten)]
        extra: Extra,
    },
    Square {
        side: u32,
    },
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "type", content = "c")]
enum AdjShape {
    Circle {
        radius: u32,
        #[strede(flatten)]
        extra: Extra,
    },
    Square {
        side: u32,
    },
}

#[test]
fn internally_tagged_flatten_in_order() {
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("x").as_slice(), &[1u8]),
        (fixstr("y").as_slice(), &[2u8]),
    ]);
    assert_eq!(
        parse::<Shape>(&msg),
        Some(Shape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        })
    );
}

#[test]
fn internally_tagged_flatten_tag_in_middle() {
    let msg = build_map(&[
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("x").as_slice(), &[1u8]),
        (fixstr("y").as_slice(), &[2u8]),
    ]);
    assert_eq!(
        parse::<Shape>(&msg),
        Some(Shape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        })
    );
}

#[test]
fn internally_tagged_flatten_interleaved() {
    let msg = build_map(&[
        (fixstr("x").as_slice(), &[1u8]),
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("y").as_slice(), &[2u8]),
    ]);
    assert_eq!(
        parse::<Shape>(&msg),
        Some(Shape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        })
    );
}

#[test]
fn internally_tagged_flatten_missing_flattened_field_misses() {
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("x").as_slice(), &[1u8]),
    ]);
    assert_eq!(parse::<Shape>(&msg), None);
}

#[test]
fn internally_tagged_non_flatten_variant_unaffected() {
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Square").as_slice()),
        (fixstr("side").as_slice(), &[3u8]),
    ]);
    assert_eq!(parse::<Shape>(&msg), Some(Shape::Square { side: 3 }));
}

#[test]
fn adjacently_tagged_flatten() {
    let content = build_map(&[
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("x").as_slice(), &[1u8]),
        (fixstr("y").as_slice(), &[2u8]),
    ]);
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("c").as_slice(), content.as_slice()),
    ]);
    assert_eq!(
        parse::<AdjShape>(&msg),
        Some(AdjShape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        })
    );
}

#[test]
fn adjacently_tagged_flatten_interleaved_content() {
    let content = build_map(&[
        (fixstr("x").as_slice(), &[1u8]),
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("y").as_slice(), &[2u8]),
    ]);
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("c").as_slice(), content.as_slice()),
    ]);
    assert_eq!(
        parse::<AdjShape>(&msg),
        Some(AdjShape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        })
    );
}

#[test]
fn adjacently_tagged_flatten_missing_flattened_field_misses() {
    let content = build_map(&[
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("x").as_slice(), &[1u8]),
    ]);
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("c").as_slice(), content.as_slice()),
    ]);
    assert_eq!(parse::<AdjShape>(&msg), None);
}

#[test]
fn adjacently_tagged_non_flatten_variant_unaffected() {
    let content = build_map(&[(fixstr("side").as_slice(), &[3u8])]);
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Square").as_slice()),
        (fixstr("c").as_slice(), content.as_slice()),
    ]);
    assert_eq!(parse::<AdjShape>(&msg), Some(AdjShape::Square { side: 3 }));
}
