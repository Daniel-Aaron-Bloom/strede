//! `#[strede(flatten)]` on a purely untagged enum field (`#[strede(untagged)]`,
//! no `tag`) — MessagePack, owned family. See
//! `strede-json/tests/flatten_untagged_candidate_enum_owned.rs` for the full
//! design rationale (`NoTagCandidateArmStack`, TESTING_GAPS.md item #3(B-2)).
//!
//! Also exercises the owned-family concurrency contract directly: the
//! staggered/byte-at-a-time test below forces genuine concurrent forked-reader
//! progress across multiple live candidates before soft elimination resolves
//! which one survives.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::{block_on_loop, block_on_loop_bounded};

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

macro_rules! parse_chunked {
    ($ty:ty, $input:expr, $chunk_size:expr) => {{
        let input: &[u8] = $input;
        let chunk_size: usize = $chunk_size;
        let pos = ::core::cell::Cell::new(chunk_size.min(input.len()));
        block_on_loop_bounded(
            SharedBuf::with_async(
                &input[..chunk_size.min(input.len())],
                async |buf: &mut &[u8]| {
                    let start = pos.get();
                    let end = (start + chunk_size).min(input.len());
                    pos.set(end);
                    *buf = &input[start..end];
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
            ),
            20_000,
        )
    }};
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Circle {
    radius: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(untagged)]
enum Shape {
    Circle(Circle),
    Rectangle { width: u32, height: u32 },
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(allow_unknown_fields)]
struct Canvas {
    name: String,
    #[strede(flatten)]
    shape: Shape,
}

#[test]
fn flatten_untagged_newtype_candidate() {
    let bytes = build_map(&[
        (fixstr("name").as_slice(), fixstr("c1").as_slice()),
        (fixstr("radius").as_slice(), &[5u8]),
    ]);
    let c: Canvas = parse!(Canvas, &bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: "c1".to_string(),
            shape: Shape::Circle(Circle { radius: 5 }),
        }
    );
}

#[test]
fn flatten_untagged_struct_candidate() {
    let bytes = build_map(&[
        (fixstr("name").as_slice(), fixstr("c2").as_slice()),
        (fixstr("width").as_slice(), &[3u8]),
        (fixstr("height").as_slice(), &[4u8]),
    ]);
    let c: Canvas = parse!(Canvas, &bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: "c2".to_string(),
            shape: Shape::Rectangle {
                width: 3,
                height: 4,
            },
        }
    );
}

#[test]
fn flatten_untagged_struct_candidate_sibling_interleaved() {
    let bytes = build_map(&[
        (fixstr("width").as_slice(), &[3u8]),
        (fixstr("name").as_slice(), fixstr("c3").as_slice()),
        (fixstr("height").as_slice(), &[4u8]),
    ]);
    let c: Canvas = parse!(Canvas, &bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: "c3".to_string(),
            shape: Shape::Rectangle {
                width: 3,
                height: 4,
            },
        }
    );
}

#[test]
fn flatten_untagged_extra_field_eliminates_other_candidate() {
    let bytes = build_map(&[
        (fixstr("name").as_slice(), fixstr("c4").as_slice()),
        (fixstr("radius").as_slice(), &[2u8]),
    ]);
    let c: Canvas = parse!(Canvas, &bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: "c4".to_string(),
            shape: Shape::Circle(Circle { radius: 2 }),
        }
    );
}

#[test]
fn flatten_untagged_missing_required_field_misses() {
    let bytes = build_map(&[
        (fixstr("name").as_slice(), fixstr("c5").as_slice()),
        (fixstr("width").as_slice(), &[3u8]),
    ]);
    let v: Option<Canvas> = parse!(Canvas, &bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_untagged_unrecognized_field_alone_does_not_eliminate_sole_survivor() {
    let bytes = build_map(&[
        (fixstr("name").as_slice(), fixstr("c6").as_slice()),
        (fixstr("height").as_slice(), &[4u8]),
        (fixstr("bogus").as_slice(), &[1u8]),
        (fixstr("width").as_slice(), &[3u8]),
    ]);
    let c: Canvas = parse!(Canvas, &bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: "c6".to_string(),
            shape: Shape::Rectangle {
                width: 3,
                height: 4,
            },
        }
    );
}

#[test]
fn flatten_untagged_candidates_race_concurrently_byte_at_a_time() {
    // Circle and Rectangle both have fields that could plausibly be raced
    // concurrently against the shared key stream. Feeding the input a few
    // bytes at a time forces genuine concurrent forked-reader progress
    // across both candidates' own arm stacks until enough keys arrive to
    // settle which one survives.
    let bytes = build_map(&[
        (fixstr("name").as_slice(), fixstr("c7").as_slice()),
        (fixstr("width").as_slice(), &[3u8]),
        (fixstr("height").as_slice(), &[4u8]),
    ]);
    let input: &[u8] = &bytes;
    for chunk_size in 1..=4 {
        let c: Canvas = parse_chunked!(Canvas, input, chunk_size).unwrap();
        assert_eq!(
            c,
            Canvas {
                name: "c7".to_string(),
                shape: Shape::Rectangle {
                    width: 3,
                    height: 4,
                },
            },
            "chunk_size={chunk_size}"
        );
    }
}

// --- wire-name collision: declaration-order tie-break (accepted limitation) ---

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct First {
    value: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Second {
    value: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(untagged)]
enum Coll {
    First(First),
    Second(Second),
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct CollWrap {
    #[strede(flatten)]
    inner: Coll,
}

#[test]
fn flatten_untagged_wire_name_collision_prefers_declaration_order() {
    let bytes = build_map(&[(fixstr("value").as_slice(), &[9u8])]);
    let w: CollWrap = parse!(CollWrap, &bytes).unwrap();
    assert_eq!(
        w,
        CollWrap {
            inner: Coll::First(First { value: 9 }),
        }
    );
}
