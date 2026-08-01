//! `#[strede(flatten)]` on a purely untagged enum field (`#[strede(untagged)]`,
//! no `tag`) — CBOR, borrow family. See
//! `strede-json/tests/flatten_untagged_candidate_enum_borrow.rs` for the full
//! design rationale (`NoTagCandidateArmStack`, TESTING_GAPS.md item #3(B-2),
//! the final deferred case); this file confirms the same derive-generated
//! codegen works over a binary, non-JSON format.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]
extern crate std;
mod helpers;
use helpers::*;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    match block_on(T::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[derive(Debug, PartialEq, Deserialize)]
struct Circle {
    radius: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum Shape {
    Circle(Circle),
    Rectangle { width: u32, height: u32 },
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(allow_unknown_fields)]
struct Canvas {
    name: std::string::String,
    #[strede(flatten)]
    shape: Shape,
}

#[test]
fn flatten_untagged_newtype_candidate() {
    let bytes = build_map(&[
        (tstr("name").as_slice(), tstr("c1").as_slice()),
        (tstr("radius").as_slice(), &[5u8]),
    ]);
    let c: Canvas = parse(&bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: std::string::String::from("c1"),
            shape: Shape::Circle(Circle { radius: 5 }),
        }
    );
}

#[test]
fn flatten_untagged_struct_candidate() {
    let bytes = build_map(&[
        (tstr("name").as_slice(), tstr("c2").as_slice()),
        (tstr("width").as_slice(), &[3u8]),
        (tstr("height").as_slice(), &[4u8]),
    ]);
    let c: Canvas = parse(&bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: std::string::String::from("c2"),
            shape: Shape::Rectangle {
                width: 3,
                height: 4,
            },
        }
    );
}

#[test]
fn flatten_untagged_struct_candidate_sibling_interleaved() {
    // The sibling ("name") and both of Rectangle's own fields arrive
    // interleaved, and "width" arrives before "name" - Circle must be
    // eliminated the moment "width" is seen (it has no such field), while
    // the parent's own "name" arm and Rectangle's remaining "height" arm
    // both still resolve correctly afterward.
    let bytes = build_map(&[
        (tstr("width").as_slice(), &[3u8]),
        (tstr("name").as_slice(), tstr("c3").as_slice()),
        (tstr("height").as_slice(), &[4u8]),
    ]);
    let c: Canvas = parse(&bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: std::string::String::from("c3"),
            shape: Shape::Rectangle {
                width: 3,
                height: 4,
            },
        }
    );
}

#[test]
fn flatten_untagged_extra_field_eliminates_other_candidate() {
    // "radius" is unique to Circle. The moment it's seen, Rectangle (still
    // live, recognizes neither "radius" nor anything else yet) must be
    // eliminated so Circle can win outright despite Rectangle never having
    // been ruled out by name alone.
    let bytes = build_map(&[
        (tstr("name").as_slice(), tstr("c4").as_slice()),
        (tstr("radius").as_slice(), &[2u8]),
    ]);
    let c: Canvas = parse(&bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: std::string::String::from("c4"),
            shape: Shape::Circle(Circle { radius: 2 }),
        }
    );
}

#[test]
fn flatten_untagged_missing_required_field_misses() {
    // Rectangle's own "width" arrives but "height" never does, Circle is
    // eliminated by "width", and no live candidate ever becomes fully
    // satisfied - the whole struct must miss, not silently pick a partial
    // Rectangle.
    let bytes = build_map(&[
        (tstr("name").as_slice(), tstr("c5").as_slice()),
        (tstr("width").as_slice(), &[3u8]),
    ]);
    let v: Option<Canvas> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_untagged_unrecognized_field_alone_does_not_eliminate_sole_survivor() {
    // "height" is Rectangle's own field, so Circle is eliminated by it.
    // "bogus" is recognized by *nobody* still live (just Rectangle) - an
    // unrecognized-by-everyone key must not eliminate the sole remaining
    // candidate (that's "unknown field", not "wrong candidate"); `Canvas`'s
    // own `allow_unknown_fields` is what lets "bogus" itself be skipped
    // rather than missing the whole struct (unrelated to the flatten
    // sub-stack's own elimination logic, which is what this test targets).
    let bytes = build_map(&[
        (tstr("name").as_slice(), tstr("c6").as_slice()),
        (tstr("height").as_slice(), &[4u8]),
        (tstr("bogus").as_slice(), &[1u8]),
        (tstr("width").as_slice(), &[3u8]),
    ]);
    let c: Canvas = parse(&bytes).unwrap();
    assert_eq!(
        c,
        Canvas {
            name: std::string::String::from("c6"),
            shape: Shape::Rectangle {
                width: 3,
                height: 4,
            },
        }
    );
}

// --- wire-name collision: declaration-order tie-break (accepted limitation) ---

#[derive(Debug, PartialEq, Deserialize)]
struct First {
    value: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
struct Second {
    value: u32,
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum Coll {
    First(First),
    Second(Second),
}

#[derive(Debug, PartialEq, Deserialize)]
struct CollWrap {
    #[strede(flatten)]
    inner: Coll,
}

#[test]
fn flatten_untagged_wire_name_collision_prefers_declaration_order() {
    // Both candidates declare "value" - neither is eliminated by it (every
    // live candidate recognizes it), and the first-declared one's arm wins
    // the dispatch, consistent with the same accepted, undetected-collision
    // limitation documented for internally/adjacently-tagged flatten.
    let bytes = build_map(&[(tstr("value").as_slice(), &[9u8])]);
    let w: CollWrap = parse(&bytes).unwrap();
    assert_eq!(
        w,
        CollWrap {
            inner: Coll::First(First { value: 9 }),
        }
    );
}
