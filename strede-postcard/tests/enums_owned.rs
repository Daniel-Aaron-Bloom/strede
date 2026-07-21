//! Enum deserialization via the owned/chunked family: unit variants, newtype
//! variants, struct variants, `#[strede(other)]`. Mirrors `enums_borrow.rs`.
//!
//! Postcard enums use a varint discriminant (declaration order, 0-indexed).
//! No variant names appear on the wire.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

// --- Type definitions ---

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Signal {
    Ping, // discriminant 0
    Pong, // discriminant 1
}

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Event {
    Ping,                    // discriminant 0
    Move { x: u32, y: u32 }, // discriminant 1
    Wrap(u32),               // discriminant 2
}

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Tagged {
    A,     // 0
    B(u8), // 1
    C(u8), // 2
}

// --- Unit variants ---

#[test]
fn unit_variant_ping() {
    // Ping = discriminant 0, zero bytes payload
    assert_eq!(parse_owned!(Signal, &varint(0)), Ok(Some(Signal::Ping)));
}

#[test]
fn unit_variant_pong() {
    assert_eq!(parse_owned!(Signal, &varint(1)), Ok(Some(Signal::Pong)));
}

#[test]
fn unit_variant_out_of_range_misses() {
    assert_eq!(parse_owned!(Signal, &varint(99)), Ok(None));
}

#[test]
fn unit_variant_truncated_errors() {
    // varint with continuation bit set but no next byte
    assert_eq!(parse_owned!(Signal, &[0x80]).unwrap_err(), PostcardError::UnexpectedEnd);
}

// --- Struct variant ---

#[test]
fn struct_variant_move() {
    // discriminant 1, then x=10, y=20
    let mut data = varint(1);
    data.extend_from_slice(&varint(10));
    data.extend_from_slice(&varint(20));
    assert_eq!(
        parse_owned!(Event, &data),
        Ok(Some(Event::Move { x: 10, y: 20 }))
    );
}

// --- Newtype variant ---

#[test]
fn newtype_variant_wrap() {
    // discriminant 2, then u32=42
    let mut data = varint(2);
    data.extend_from_slice(&varint(42));
    assert_eq!(parse_owned!(Event, &data), Ok(Some(Event::Wrap(42))));
}

// --- Unit variant in mixed enum ---

#[test]
fn unit_variant_in_mixed_enum() {
    assert_eq!(parse_owned!(Event, &varint(0)), Ok(Some(Event::Ping)));
}

// --- Multiple discriminants ---

#[test]
fn tagged_a() {
    assert_eq!(parse_owned!(Tagged, &varint(0)), Ok(Some(Tagged::A)));
}

#[test]
fn tagged_b() {
    let mut data = varint(1);
    data.extend_from_slice(&varint(7));
    assert_eq!(parse_owned!(Tagged, &data), Ok(Some(Tagged::B(7))));
}

#[test]
fn tagged_c() {
    let mut data = varint(2);
    data.extend_from_slice(&varint(255));
    assert_eq!(parse_owned!(Tagged, &data), Ok(Some(Tagged::C(255))));
}

#[test]
fn unknown_discriminant_misses() {
    assert_eq!(parse_owned!(Tagged, &varint(3)), Ok(None));
}

// --- Large discriminant (multi-byte varint) ---
//
// Discriminants 0..=127 fit in a single byte; 128+ require two bytes.
// `enums_borrow.rs` covers the "hits a real 128+ variant" case with a
// 130-variant `Wide` enum, which works fine in the borrow family. The owned
// family's `EnumArmStackOwned::race` recurses one level of nested generic
// future per arm (see `strede/src/enum_arm/owned.rs`), and each level's
// future is larger than the borrow family's (it carries a `Handle<'s, B,
// F>`) — empirically, an owned-family enum derived with somewhere between 60
// and 90 variants overflows the stack (reproduced independent of which
// discriminant is queried, in both debug and release builds), well below
// where the borrow family's equivalent 130-variant enum is comfortable. This
// is a pre-existing characteristic of the shared `enum_arm/owned.rs`
// infrastructure (not specific to postcard - any owned-family format with a
// wide enum would hit the same ceiling), so it isn't fixed here. The
// multi-byte-varint-*discriminant-reading* path itself (as opposed to
// racing 100+ arms against it) is still exercised below via a small enum
// fed an out-of-range two-byte discriminant.

#[test]
fn large_discriminant_two_bytes_misses_when_out_of_range() {
    // Index 128 requires two bytes: [0x80, 0x01]. `Tagged` only declares
    // indices 0..=2, so this proves the resumable varint discriminant read
    // (not just single-byte reads) completes correctly and misses cleanly,
    // without needing a wide enough enum to hit the arm-count ceiling above.
    assert_eq!(parse_owned!(Tagged, &[0x80, 0x01]), Ok(None));
}

// --- `#[strede(other)]` catch-all ---
//
// `other` only ever targets a unit variant, so the fallback never needs to
// skip a payload: an unrecognized discriminant is treated as carrying no
// payload at all, matching upstream postcard/serde's own `#[serde(other)]`
// convention (verified against the real `postcard` + `serde` crates).

#[derive(Debug, PartialEq, DeserializeOwned)]
enum WithOther {
    A,     // 0
    B(u8), // 1
    #[strede(other)]
    Unknown,
}

#[test]
fn other_catches_unrecognized_discriminant() {
    assert_eq!(
        parse_owned!(WithOther, &varint(2)),
        Ok(Some(WithOther::Unknown))
    );
    assert_eq!(
        parse_owned!(WithOther, &varint(99)),
        Ok(Some(WithOther::Unknown))
    );
}

#[test]
fn other_does_not_shadow_known_variants() {
    assert_eq!(parse_owned!(WithOther, &varint(0)), Ok(Some(WithOther::A)));
    let mut data = varint(1);
    data.extend_from_slice(&varint(7));
    assert_eq!(parse_owned!(WithOther, &data), Ok(Some(WithOther::B(7))));
}

#[test]
fn other_leaves_no_trailing_bytes_for_unit_fallback() {
    // Nothing follows the discriminant for the `other` case, so there's no
    // trailing-bytes error.
    assert_eq!(
        parse_owned!(WithOther, &varint(2)),
        Ok(Some(WithOther::Unknown))
    );
}

#[test]
fn other_with_unexpected_trailing_payload_errors() {
    // If the real (unrecognized) variant actually carried a payload on the
    // wire, `other`'s zero-payload assumption leaves those bytes unconsumed —
    // surfacing as a top-level trailing-bytes error rather than silently
    // discarding them. This is the schema-evolution caveat documented on
    // `PostcardEntry::skip_other`/`ChunkedPostcardEntry::skip_other`.
    let mut data = varint(2);
    data.extend_from_slice(&varint(123));
    assert_eq!(parse_owned!(WithOther, &data).unwrap_err(), PostcardError::ExpectedEnd);
}
