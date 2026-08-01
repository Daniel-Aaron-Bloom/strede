//! `#[strede(flatten)]` on a tagged/adjacently-tagged enum's struct-variant
//! field — borrow family. Regression coverage for a bug where the struct-
//! variant helper ignored `cf.flatten` entirely and treated a flatten field
//! as an ordinary nested-map field, silently breaking the merge-into-parent
//! semantics `flatten` has everywhere else.
#![recursion_limit = "512"]

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

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
fn internally_tagged_flatten_in_order() {
    let s: Shape = parse(r#"{"type": "Circle", "radius": 5, "x": 1, "y": 2}"#).unwrap();
    assert_eq!(
        s,
        Shape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        }
    );
}

#[test]
fn internally_tagged_flatten_tag_in_middle() {
    let s: Shape = parse(r#"{"radius": 5, "type": "Circle", "x": 1, "y": 2}"#).unwrap();
    assert_eq!(
        s,
        Shape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        }
    );
}

#[test]
fn internally_tagged_flatten_interleaved() {
    let s: Shape = parse(r#"{"x": 1, "type": "Circle", "radius": 5, "y": 2}"#).unwrap();
    assert_eq!(
        s,
        Shape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        }
    );
}

#[test]
fn internally_tagged_flatten_missing_flattened_field_misses() {
    let v: Option<Shape> = parse(r#"{"type": "Circle", "radius": 5, "x": 1}"#);
    assert!(v.is_none());
}

#[test]
fn internally_tagged_non_flatten_variant_unaffected() {
    let s: Shape = parse(r#"{"type": "Square", "side": 3}"#).unwrap();
    assert_eq!(s, Shape::Square { side: 3 });
}

#[test]
fn adjacently_tagged_flatten() {
    let s: AdjShape = parse(r#"{"type": "Circle", "c": {"radius": 5, "x": 1, "y": 2}}"#).unwrap();
    assert_eq!(
        s,
        AdjShape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        }
    );
}

#[test]
fn adjacently_tagged_flatten_interleaved_content() {
    let s: AdjShape = parse(r#"{"type": "Circle", "c": {"x": 1, "radius": 5, "y": 2}}"#).unwrap();
    assert_eq!(
        s,
        AdjShape::Circle {
            radius: 5,
            extra: Extra { x: 1, y: 2 }
        }
    );
}

#[test]
fn adjacently_tagged_flatten_missing_flattened_field_misses() {
    let v: Option<AdjShape> = parse(r#"{"type": "Circle", "c": {"radius": 5, "x": 1}}"#);
    assert!(v.is_none());
}

#[test]
fn adjacently_tagged_non_flatten_variant_unaffected() {
    let s: AdjShape = parse(r#"{"type": "Square", "c": {"side": 3}}"#).unwrap();
    assert_eq!(s, AdjShape::Square { side: 3 });
}
