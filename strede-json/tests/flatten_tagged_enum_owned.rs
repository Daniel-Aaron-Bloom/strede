//! `#[strede(flatten)]` on a tagged/adjacently-tagged enum's struct-variant
//! field — owned family. Regression coverage for a bug where the struct-
//! variant helper ignored `cf.flatten` entirely and treated a flatten field
//! as an ordinary nested-map field, silently breaking the merge-into-parent
//! semantics `flatten` has everywhere else.
#![recursion_limit = "512"]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Extra {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
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

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
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
fn internally_tagged_flatten_in_order() {
    let s: Shape = parse!(Shape, br#"{"type": "Circle", "radius": 5, "x": 1, "y": 2}"#).unwrap();
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
    let s: Shape = parse!(Shape, br#"{"radius": 5, "type": "Circle", "x": 1, "y": 2}"#).unwrap();
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
    let s: Shape = parse!(Shape, br#"{"x": 1, "type": "Circle", "radius": 5, "y": 2}"#).unwrap();
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
    let v: Option<Shape> = parse!(Shape, br#"{"type": "Circle", "radius": 5, "x": 1}"#);
    assert!(v.is_none());
}

#[test]
fn internally_tagged_non_flatten_variant_unaffected() {
    let s: Shape = parse!(Shape, br#"{"type": "Square", "side": 3}"#).unwrap();
    assert_eq!(s, Shape::Square { side: 3 });
}

#[test]
fn adjacently_tagged_flatten() {
    let s: AdjShape = parse!(
        AdjShape,
        br#"{"type": "Circle", "c": {"radius": 5, "x": 1, "y": 2}}"#
    )
    .unwrap();
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
    let s: AdjShape = parse!(
        AdjShape,
        br#"{"type": "Circle", "c": {"x": 1, "radius": 5, "y": 2}}"#
    )
    .unwrap();
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
    let v: Option<AdjShape> = parse!(
        AdjShape,
        br#"{"type": "Circle", "c": {"radius": 5, "x": 1}}"#
    );
    assert!(v.is_none());
}

#[test]
fn adjacently_tagged_non_flatten_variant_unaffected() {
    let s: AdjShape = parse!(AdjShape, br#"{"type": "Square", "c": {"side": 3}}"#).unwrap();
    assert_eq!(s, AdjShape::Square { side: 3 });
}
