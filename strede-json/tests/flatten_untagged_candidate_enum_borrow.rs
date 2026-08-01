//! `#[strede(flatten)]` on a purely untagged enum field (`#[strede(untagged)]`,
//! no `tag`) — borrow family.
//!
//! Unlike `flatten_internal_candidate_enum_borrow.rs` (internally-tagged),
//! there is no discriminant key at all: every candidate variant's own fields
//! race directly against the parent's shared key stream from round one, via
//! the `NoTagCandidateArmStack` runtime primitive (TESTING_GAPS.md item
//! #3(B-2), the final deferred case). A candidate is permanently eliminated
//! the first round some *other* live candidate's arms recognize a key that
//! this candidate's own arms do not.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

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
    name: String,
    #[strede(flatten)]
    shape: Shape,
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
fn flatten_untagged_newtype_candidate() {
    let c: Canvas = parse(r#"{"name": "c1", "radius": 5}"#).unwrap();
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
    let c: Canvas = parse(r#"{"name": "c2", "width": 3, "height": 4}"#).unwrap();
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
    // The sibling ("name") and both of Rectangle's own fields arrive
    // interleaved, and "width" arrives before "name" - Circle must be
    // eliminated the moment "width" is seen (it has no such field), while
    // the parent's own "name" arm and Rectangle's remaining "height" arm
    // both still resolve correctly afterward.
    let c: Canvas = parse(r#"{"width": 3, "name": "c3", "height": 4}"#).unwrap();
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
    // "radius" is unique to Circle. The moment it's seen, Rectangle (still
    // live, recognizes neither "radius" nor anything else yet) must be
    // eliminated so Circle can win outright despite Rectangle never having
    // been ruled out by name alone.
    let c: Canvas = parse(r#"{"name": "c4", "radius": 2}"#).unwrap();
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
    // Rectangle's own "width" arrives but "height" never does, Circle is
    // eliminated by "width", and no live candidate ever becomes fully
    // satisfied - the whole struct must miss, not silently pick a partial
    // Rectangle.
    let v: Option<Canvas> = parse(r#"{"name": "c5", "width": 3}"#);
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
    let c: Canvas = parse(r#"{"name": "c6", "height": 4, "bogus": 1, "width": 3}"#).unwrap();
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
    let w: CollWrap = parse(r#"{"value": 9}"#).unwrap();
    assert_eq!(
        w,
        CollWrap {
            inner: Coll::First(First { value: 9 }),
        }
    );
}
