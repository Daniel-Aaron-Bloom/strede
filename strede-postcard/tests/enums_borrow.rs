//! Enum deserialization: unit variants, newtype variants, struct variants.
//!
//! Postcard enums use a varint discriminant (declaration order, 0-indexed).
//! No variant names appear on the wire.

#![recursion_limit = "256"]

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Result<Option<T>, PostcardError>
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Ok(Some(v)),
        Probe::Miss => Ok(None),
    }
}

fn parse_err<'de, T>(input: &'de [u8]) -> PostcardError
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())) {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    }
}

// --- Type definitions ---

#[derive(Debug, PartialEq, Deserialize)]
enum Signal {
    Ping, // discriminant 0
    Pong, // discriminant 1
}

#[derive(Debug, PartialEq, Deserialize)]
enum Event {
    Ping,                    // discriminant 0
    Move { x: u32, y: u32 }, // discriminant 1
    Wrap(u32),               // discriminant 2
}

#[derive(Debug, PartialEq, Deserialize)]
enum Tagged {
    A,     // 0
    B(u8), // 1
    C(u8), // 2
}

// --- Unit variants ---

#[test]
fn unit_variant_ping() {
    // Ping = discriminant 0, zero bytes payload
    assert_eq!(parse::<Signal>(&varint(0)), Ok(Some(Signal::Ping)));
}

#[test]
fn unit_variant_pong() {
    assert_eq!(parse::<Signal>(&varint(1)), Ok(Some(Signal::Pong)));
}

#[test]
fn unit_variant_out_of_range_misses() {
    assert_eq!(parse::<Signal>(&varint(99)), Ok(None));
}

#[test]
fn unit_variant_truncated_errors() {
    // varint with continuation bit set but no next byte
    assert_eq!(parse_err::<Signal>(&[0x80]), PostcardError::UnexpectedEnd);
}

// --- Struct variant ---

#[test]
fn struct_variant_move() {
    // discriminant 1, then x=10, y=20
    let mut data = varint(1);
    data.extend_from_slice(&varint(10));
    data.extend_from_slice(&varint(20));
    assert_eq!(
        parse::<Event>(&data),
        Ok(Some(Event::Move { x: 10, y: 20 }))
    );
}

// --- Newtype variant ---

#[test]
fn newtype_variant_wrap() {
    // discriminant 2, then u32=42
    let mut data = varint(2);
    data.extend_from_slice(&varint(42));
    assert_eq!(parse::<Event>(&data), Ok(Some(Event::Wrap(42))));
}

// --- Unit variant in mixed enum ---

#[test]
fn unit_variant_in_mixed_enum() {
    assert_eq!(parse::<Event>(&varint(0)), Ok(Some(Event::Ping)));
}

// --- Multiple discriminants ---

#[test]
fn tagged_a() {
    assert_eq!(parse::<Tagged>(&varint(0)), Ok(Some(Tagged::A)));
}

#[test]
fn tagged_b() {
    let mut data = varint(1);
    data.extend_from_slice(&varint(7));
    assert_eq!(parse::<Tagged>(&data), Ok(Some(Tagged::B(7))));
}

#[test]
fn tagged_c() {
    let mut data = varint(2);
    data.extend_from_slice(&varint(255));
    assert_eq!(parse::<Tagged>(&data), Ok(Some(Tagged::C(255))));
}

#[test]
fn unknown_discriminant_misses() {
    assert_eq!(parse::<Tagged>(&varint(3)), Ok(None));
}

// --- Large discriminant (multi-byte varint) ---
//
// Discriminants 0..=127 fit in a single byte; 128+ require two bytes.
// varint(127) = [0x7F], varint(128) = [0x80, 0x01].

#[derive(Debug, PartialEq, Deserialize)]
enum Wide {
    V000,
    V001,
    V002,
    V003,
    V004,
    V005,
    V006,
    V007,
    V008,
    V009,
    V010,
    V011,
    V012,
    V013,
    V014,
    V015,
    V016,
    V017,
    V018,
    V019,
    V020,
    V021,
    V022,
    V023,
    V024,
    V025,
    V026,
    V027,
    V028,
    V029,
    V030,
    V031,
    V032,
    V033,
    V034,
    V035,
    V036,
    V037,
    V038,
    V039,
    V040,
    V041,
    V042,
    V043,
    V044,
    V045,
    V046,
    V047,
    V048,
    V049,
    V050,
    V051,
    V052,
    V053,
    V054,
    V055,
    V056,
    V057,
    V058,
    V059,
    V060,
    V061,
    V062,
    V063,
    V064,
    V065,
    V066,
    V067,
    V068,
    V069,
    V070,
    V071,
    V072,
    V073,
    V074,
    V075,
    V076,
    V077,
    V078,
    V079,
    V080,
    V081,
    V082,
    V083,
    V084,
    V085,
    V086,
    V087,
    V088,
    V089,
    V090,
    V091,
    V092,
    V093,
    V094,
    V095,
    V096,
    V097,
    V098,
    V099,
    V100,
    V101,
    V102,
    V103,
    V104,
    V105,
    V106,
    V107,
    V108,
    V109,
    V110,
    V111,
    V112,
    V113,
    V114,
    V115,
    V116,
    V117,
    V118,
    V119,
    V120,
    V121,
    V122,
    V123,
    V124,
    V125,
    V126,
    V127,
    V128,
    V129,
}

#[test]
fn large_discriminant_boundary_single_byte() {
    // Index 127 is the last that fits in one varint byte.
    assert_eq!(parse::<Wide>(&[0x7F]), Ok(Some(Wide::V127)));
}

#[test]
fn large_discriminant_two_bytes() {
    // Index 128 requires two bytes: [0x80, 0x01].
    assert_eq!(parse::<Wide>(&[0x80, 0x01]), Ok(Some(Wide::V128)));
}

#[test]
fn large_discriminant_129() {
    assert_eq!(parse::<Wide>(&varint(129)), Ok(Some(Wide::V129)));
}

// --- `#[strede(other)]` catch-all ---
//
// `other` only ever targets a unit variant, so the fallback never needs to
// skip a payload: an unrecognized discriminant is treated as carrying no
// payload at all, matching upstream postcard/serde's own `#[serde(other)]`
// convention (verified against the real `postcard` + `serde` crates).

#[derive(Debug, PartialEq, Deserialize)]
enum WithOther {
    A,     // 0
    B(u8), // 1
    #[strede(other)]
    Unknown,
}

#[test]
fn other_catches_unrecognized_discriminant() {
    assert_eq!(
        parse::<WithOther>(&varint(2)),
        Ok(Some(WithOther::Unknown))
    );
    assert_eq!(
        parse::<WithOther>(&varint(99)),
        Ok(Some(WithOther::Unknown))
    );
}

#[test]
fn other_does_not_shadow_known_variants() {
    assert_eq!(parse::<WithOther>(&varint(0)), Ok(Some(WithOther::A)));
    let mut data = varint(1);
    data.extend_from_slice(&varint(7));
    assert_eq!(parse::<WithOther>(&data), Ok(Some(WithOther::B(7))));
}

#[test]
fn other_leaves_no_trailing_bytes_for_unit_fallback() {
    // Nothing follows the discriminant for the `other` case, so there's no
    // trailing-bytes error.
    assert_eq!(
        parse::<WithOther>(&varint(2)),
        Ok(Some(WithOther::Unknown))
    );
}

#[test]
fn other_with_unexpected_trailing_payload_errors() {
    // If the real (unrecognized) variant actually carried a payload on the
    // wire, `other`'s zero-payload assumption leaves those bytes unconsumed —
    // surfacing as a top-level trailing-bytes error rather than silently
    // discarding them. This is the schema-evolution caveat documented on
    // `PostcardEntry::skip_other`.
    let mut data = varint(2);
    data.extend_from_slice(&varint(123));
    assert_eq!(parse_err::<WithOther>(&data), PostcardError::ExpectedEnd);
}
