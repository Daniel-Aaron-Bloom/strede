//! `#[strede(flatten)]` on a purely untagged enum field (`#[strede(untagged)]`,
//! no `tag`) — owned family. See `flatten_untagged_candidate_enum_borrow.rs`
//! for the full rationale (`NoTagCandidateArmStack`, TESTING_GAPS.md item
//! #3(B-2), the final deferred case).
//!
//! Also exercises the owned-family concurrency contract directly: the
//! staggered/byte-at-a-time test below forces genuine concurrent forked-reader
//! progress across multiple live candidates before soft elimination resolves
//! which one survives. If `NoTagCandidateArmStack`'s settle-then-eliminate
//! logic ever left an eliminated candidate's forked reader un-dropped, or
//! re-drove an already-resolved candidate's race, the shared buffer would
//! stall or panic - `block_on_loop_bounded` turns a hang into a fast, clear
//! panic instead of hanging CI (see `owned_staggered_concurrency.rs` for the
//! same discipline applied to plain struct fields).
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::{block_on_loop, block_on_loop_bounded};

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
                    let de = ChunkedJsonDeserializer::new(shared);
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

#[test]
fn flatten_untagged_newtype_candidate() {
    let c: Canvas = parse!(Canvas, br#"{"name": "c1", "radius": 5}"#).unwrap();
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
    let c: Canvas = parse!(Canvas, br#"{"name": "c2", "width": 3, "height": 4}"#).unwrap();
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
    let c: Canvas = parse!(
        Canvas,
        br#"{"width": 3, "name": "c3", "height": 4}"#
    )
    .unwrap();
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
    let c: Canvas = parse!(Canvas, br#"{"name": "c4", "radius": 2}"#).unwrap();
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
    let v: Option<Canvas> = parse!(Canvas, br#"{"name": "c5", "width": 3}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_untagged_unrecognized_field_alone_does_not_eliminate_sole_survivor() {
    let c: Canvas = parse!(
        Canvas,
        br#"{"name": "c6", "height": 4, "bogus": 1, "width": 3}"#
    )
    .unwrap();
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
    let input: &[u8] = br#"{"name": "c7", "width": 3, "height": 4}"#;
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
    let w: CollWrap = parse!(CollWrap, br#"{"value": 9}"#).unwrap();
    assert_eq!(
        w,
        CollWrap {
            inner: Coll::First(First { value: 9 }),
        }
    );
}
