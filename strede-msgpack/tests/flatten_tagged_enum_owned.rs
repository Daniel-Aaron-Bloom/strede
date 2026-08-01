//! `#[strede(flatten)]` on a tagged/adjacently-tagged enum's struct-variant
//! field — owned family. Regression coverage for a bug where the struct-
//! variant helper ignored `cf.flatten` entirely and treated a flatten field
//! as an ordinary nested-map field, silently breaking the merge-into-parent
//! semantics `flatten` has everywhere else.
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

#[test]
fn internally_tagged_flatten_in_order() {
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Circle").as_slice()),
        (fixstr("radius").as_slice(), &[5u8]),
        (fixstr("x").as_slice(), &[1u8]),
        (fixstr("y").as_slice(), &[2u8]),
    ]);
    assert_eq!(
        parse!(Shape, &msg),
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
        parse!(Shape, &msg),
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
        parse!(Shape, &msg),
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
    assert_eq!(parse!(Shape, &msg), None);
}

#[test]
fn internally_tagged_non_flatten_variant_unaffected() {
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Square").as_slice()),
        (fixstr("side").as_slice(), &[3u8]),
    ]);
    assert_eq!(parse!(Shape, &msg), Some(Shape::Square { side: 3 }));
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
        parse!(AdjShape, &msg),
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
        parse!(AdjShape, &msg),
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
    assert_eq!(parse!(AdjShape, &msg), None);
}

#[test]
fn adjacently_tagged_non_flatten_variant_unaffected() {
    let content = build_map(&[(fixstr("side").as_slice(), &[3u8])]);
    let msg = build_map(&[
        (fixstr("type").as_slice(), fixstr("Square").as_slice()),
        (fixstr("c").as_slice(), content.as_slice()),
    ]);
    assert_eq!(parse!(AdjShape, &msg), Some(AdjShape::Square { side: 3 }));
}
