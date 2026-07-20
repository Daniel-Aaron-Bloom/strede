#![recursion_limit = "256"]

extern crate std;
mod helpers;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
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
                let de = ChunkedCborDeserializer::new(shared);
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

fn concat(parts: &[&[u8]]) -> std::vec::Vec<u8> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

// ---------------------------------------------------------------------------
// Externally-tagged enum
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
enum Signal {
    Ping,
    Pong,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
enum Event {
    Stop,
    Move { x: u32, y: u32 },
    Wrap(u32),
}

#[test]
fn unit_variant_ping() {
    let enc = helpers::tstr("Ping");
    assert_eq!(parse!(Signal, &enc), Some(Signal::Ping));
}

#[test]
fn unit_variant_pong() {
    let enc = helpers::tstr("Pong");
    assert_eq!(parse!(Signal, &enc), Some(Signal::Pong));
}

#[test]
fn unit_variant_unknown_misses() {
    let enc = helpers::tstr("Other");
    assert_eq!(parse!(Signal, &enc), None);
}

#[test]
fn unit_variant_in_externally_tagged_enum() {
    let enc = helpers::tstr("Stop");
    assert_eq!(parse!(Event, &enc), Some(Event::Stop));
}

#[test]
fn struct_variant_move() {
    let inner = helpers::build_map(&[
        (&helpers::tstr("x"), &[helpers::uint_small(3)]),
        (&helpers::tstr("y"), &[helpers::uint_small(7)]),
    ]);
    let outer = concat(&[&helpers::map(1), &helpers::tstr("Move"), &inner]);
    assert_eq!(parse!(Event, &outer), Some(Event::Move { x: 3, y: 7 }));
}

#[test]
fn newtype_variant_wrap() {
    let outer = concat(&[
        &helpers::map(1),
        &helpers::tstr("Wrap"),
        &helpers::uint8(42),
    ]);
    assert_eq!(parse!(Event, &outer), Some(Event::Wrap(42)));
}

#[test]
fn struct_variant_move_indefinite_inner_map() {
    // Inner map (struct fields) is indefinite-length; outer single-key map stays definite.
    let inner = helpers::build_map_indef(&[
        (&helpers::tstr("x"), &[helpers::uint_small(3)]),
        (&helpers::tstr("y"), &[helpers::uint_small(7)]),
    ]);
    let outer = concat(&[&helpers::map(1), &helpers::tstr("Move"), &inner]);
    assert_eq!(parse!(Event, &outer), Some(Event::Move { x: 3, y: 7 }));
}

#[test]
fn struct_variant_move_indefinite_outer_map() {
    // Outer single-key map (variant dispatch via `deserialize_payload_by_name`)
    // is indefinite-length; exercises the Break-terminated enum-payload path.
    let inner = helpers::build_map(&[
        (&helpers::tstr("x"), &[helpers::uint_small(3)]),
        (&helpers::tstr("y"), &[helpers::uint_small(7)]),
    ]);
    let outer = concat(&[&[0xbf], &helpers::tstr("Move"), &inner, &[0xff]]);
    assert_eq!(parse!(Event, &outer), Some(Event::Move { x: 3, y: 7 }));
}

#[test]
fn struct_variant_move_indefinite_both_maps() {
    let inner = helpers::build_map_indef(&[
        (&helpers::tstr("x"), &[helpers::uint_small(3)]),
        (&helpers::tstr("y"), &[helpers::uint_small(7)]),
    ]);
    let outer = concat(&[&[0xbf], &helpers::tstr("Move"), &inner, &[0xff]]);
    assert_eq!(parse!(Event, &outer), Some(Event::Move { x: 3, y: 7 }));
}

#[test]
fn empty_indefinite_map_misses_required_fields() {
    // An indefinite map with zero pairs (immediate break) must not satisfy
    // Move's required x/y fields.
    let inner = helpers::build_map_indef(&[]);
    let outer = concat(&[&helpers::map(1), &helpers::tstr("Move"), &inner]);
    assert_eq!(parse!(Event, &outer), None);
}

#[test]
fn unknown_variant_misses() {
    let outer = concat(&[
        &helpers::map(1),
        &helpers::tstr("Unknown"),
        &[helpers::uint_small(0)],
    ]);
    assert_eq!(parse!(Event, &outer), None);
}

// ---------------------------------------------------------------------------
// Internally tagged enum
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "type")]
enum Shape {
    Circle { radius: u32 },
    Rect { w: u32, h: u32 },
}

#[test]
fn internally_tagged_circle_tag_first() {
    let enc = helpers::build_map(&[
        (&helpers::tstr("type"), &helpers::tstr("Circle")),
        (&helpers::tstr("radius"), &[helpers::uint_small(5)]),
    ]);
    assert_eq!(parse!(Shape, &enc), Some(Shape::Circle { radius: 5 }));
}

#[test]
fn internally_tagged_circle_tag_last() {
    let enc = helpers::build_map(&[
        (&helpers::tstr("radius"), &[helpers::uint_small(9)]),
        (&helpers::tstr("type"), &helpers::tstr("Circle")),
    ]);
    assert_eq!(parse!(Shape, &enc), Some(Shape::Circle { radius: 9 }));
}

#[test]
fn internally_tagged_rect() {
    let enc = helpers::build_map(&[
        (&helpers::tstr("type"), &helpers::tstr("Rect")),
        (&helpers::tstr("w"), &[helpers::uint_small(4)]),
        (&helpers::tstr("h"), &[helpers::uint_small(8)]),
    ]);
    assert_eq!(parse!(Shape, &enc), Some(Shape::Rect { w: 4, h: 8 }));
}

#[test]
fn internally_tagged_unknown_misses() {
    let enc = helpers::build_map(&[(&helpers::tstr("type"), &helpers::tstr("Triangle"))]);
    assert_eq!(parse!(Shape, &enc), None);
}

// ---------------------------------------------------------------------------
// Adjacently tagged enum
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "t", content = "c")]
enum Tagged {
    Unit,
    Payload { n: u32 },
}

#[test]
fn adjacently_tagged_unit() {
    let enc = helpers::build_map(&[(&helpers::tstr("t"), &helpers::tstr("Unit"))]);
    assert_eq!(parse!(Tagged, &enc), Some(Tagged::Unit));
}

#[test]
fn adjacently_tagged_payload() {
    let content = helpers::build_map(&[(&helpers::tstr("n"), &[helpers::uint_small(7)])]);
    let enc = helpers::build_map(&[
        (&helpers::tstr("t"), &helpers::tstr("Payload")),
        (&helpers::tstr("c"), &content),
    ]);
    assert_eq!(parse!(Tagged, &enc), Some(Tagged::Payload { n: 7 }));
}

#[test]
fn adjacently_tagged_content_before_tag() {
    let content = helpers::build_map(&[(&helpers::tstr("n"), &[helpers::uint_small(3)])]);
    let enc = helpers::build_map(&[
        (&helpers::tstr("c"), &content),
        (&helpers::tstr("t"), &helpers::tstr("Payload")),
    ]);
    assert_eq!(parse!(Tagged, &enc), Some(Tagged::Payload { n: 3 }));
}

// ---------------------------------------------------------------------------
// Rename / rename_all
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(rename_all = "snake_case")]
enum Status {
    IsActive,
    NotFound,
}

#[test]
fn rename_all_snake_case() {
    assert_eq!(
        parse!(Status, &helpers::tstr("is_active")),
        Some(Status::IsActive)
    );
    assert_eq!(
        parse!(Status, &helpers::tstr("not_found")),
        Some(Status::NotFound)
    );
    assert_eq!(parse!(Status, &helpers::tstr("IsActive")), None);
}

// ---------------------------------------------------------------------------
// #[strede(other)]
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
enum WithOther {
    Known,
    #[strede(other)]
    Unknown,
}

#[test]
fn other_variant_known() {
    assert_eq!(
        parse!(WithOther, &helpers::tstr("Known")),
        Some(WithOther::Known)
    );
}

#[test]
fn other_variant_catches_unknown() {
    assert_eq!(
        parse!(WithOther, &helpers::tstr("Anything")),
        Some(WithOther::Unknown)
    );
}
