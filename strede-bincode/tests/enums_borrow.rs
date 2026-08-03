//! Enum deserialization: unit variants, newtype variants, struct variants,
//! `#[strede(other)]`, and `#[strede(untagged)]`.
//!
//! Bincode enums use a `u32` discriminant (declaration order, 0-indexed,
//! subject to the config's int encoding). No variant names appear on the
//! wire. Tested under the default `Standard` config; the discriminant's
//! width sensitivity to config is exercised directly in
//! `primitives_borrow.rs::enum_discriminant_width` (per config) — this file
//! focuses on the derive-generated dispatch logic, which is
//! config-independent.

#![recursion_limit = "512"]

mod helpers;
use helpers::*;

use strede_bincode::{BincodeError, Standard};
use strede_derive::Deserialize;

const E: Enc = Enc::STANDARD;

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
    assert_eq!(
        parse::<Signal, Standard>(&E.discriminant(0)),
        Ok(Some(Signal::Ping))
    );
}

#[test]
fn unit_variant_pong() {
    assert_eq!(
        parse::<Signal, Standard>(&E.discriminant(1)),
        Ok(Some(Signal::Pong))
    );
}

#[test]
fn unit_variant_out_of_range_misses() {
    assert_eq!(parse::<Signal, Standard>(&E.discriminant(99)), Ok(None));
}

// --- Struct variant ---

#[test]
fn struct_variant_move() {
    let mut data = E.discriminant(1);
    data.extend_from_slice(&E.u32(10));
    data.extend_from_slice(&E.u32(20));
    assert_eq!(
        parse::<Event, Standard>(&data),
        Ok(Some(Event::Move { x: 10, y: 20 }))
    );
}

// --- Newtype variant ---

#[test]
fn newtype_variant_wrap() {
    let mut data = E.discriminant(2);
    data.extend_from_slice(&E.u32(42));
    assert_eq!(parse::<Event, Standard>(&data), Ok(Some(Event::Wrap(42))));
}

#[test]
fn unit_variant_in_mixed_enum() {
    assert_eq!(
        parse::<Event, Standard>(&E.discriminant(0)),
        Ok(Some(Event::Ping))
    );
}

// --- Multiple discriminants ---

#[test]
fn tagged_a() {
    assert_eq!(
        parse::<Tagged, Standard>(&E.discriminant(0)),
        Ok(Some(Tagged::A))
    );
}

#[test]
fn tagged_b() {
    let mut data = E.discriminant(1);
    data.extend_from_slice(&E.u8(7));
    assert_eq!(parse::<Tagged, Standard>(&data), Ok(Some(Tagged::B(7))));
}

#[test]
fn tagged_c() {
    let mut data = E.discriminant(2);
    data.extend_from_slice(&E.u8(255));
    assert_eq!(parse::<Tagged, Standard>(&data), Ok(Some(Tagged::C(255))));
}

#[test]
fn unknown_discriminant_misses() {
    assert_eq!(parse::<Tagged, Standard>(&E.discriminant(3)), Ok(None));
}

#[test]
fn discriminant_non_canonical_prefix_errors() {
    // Discriminant 1 legitimately fits a single byte; encoded here via the
    // 16-byte u128-tail prefix (254). `decode_discriminant` has no "try
    // another type" fallback, so a non-canonical prefix is a hard error —
    // this is also what closes off the previous silent-truncation hole for
    // an out-of-range discriminant (the wide-tail path this exercises is
    // exactly the path a corrupted `> u32::MAX` discriminant would take).
    let data = E.varint_with_prefix(254, 1);
    assert_eq!(
        parse_err::<Tagged, Standard>(&data),
        BincodeError::NonCanonicalVarint
    );
}

#[test]
fn wide_discriminant_past_varint_boundary_misses() {
    // 1000 requires the varint u16 tail (prefix 251); no `Tagged` variant
    // matches it, but this exercises the full discriminant-decode path
    // through the derive (not just the raw encoder), past the 250-value
    // single-byte-inline boundary.
    assert_eq!(parse::<Tagged, Standard>(&E.discriminant(1000)), Ok(None));
}

// --- `#[strede(other)]` catch-all ---

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
        parse::<WithOther, Standard>(&E.discriminant(2)),
        Ok(Some(WithOther::Unknown))
    );
    assert_eq!(
        parse::<WithOther, Standard>(&E.discriminant(99)),
        Ok(Some(WithOther::Unknown))
    );
}

#[test]
fn other_does_not_shadow_known_variants() {
    assert_eq!(
        parse::<WithOther, Standard>(&E.discriminant(0)),
        Ok(Some(WithOther::A))
    );
    let mut data = E.discriminant(1);
    data.extend_from_slice(&E.u8(7));
    assert_eq!(
        parse::<WithOther, Standard>(&data),
        Ok(Some(WithOther::B(7)))
    );
}

#[test]
fn other_with_unexpected_trailing_payload_errors() {
    // If the real (unrecognized) variant actually carried a payload on the
    // wire, `other`'s zero-payload assumption leaves those bytes unconsumed
    // — surfacing as a top-level trailing-bytes error, the same
    // schema-evolution caveat `strede-postcard` documents.
    let mut data = E.discriminant(2);
    data.extend_from_slice(&E.u32(123));
    assert_eq!(
        parse_err::<WithOther, Standard>(&data),
        BincodeError::ExpectedEnd
    );
}

// --- `#[strede(untagged)]` ---
//
// Bincode has no wire type tags, so untagged dispatch is "try each variant
// in declaration order, first structural success wins" — the same
// accepted trade-off `strede-postcard` already carries (see
// `BincodeEnumVariantProbe::deserialize_value_by_shape`'s doc comment).
// This is NOT the same as real bincode+serde, which refuses to support
// untagged enums at all; strede's dispatch model doesn't have the
// limitation that forces that refusal (see the crate's module docs).

#[cfg(feature = "alloc")]
#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum Untagged {
    Num(u32),
    Text(String),
}

#[cfg(feature = "alloc")]
#[test]
fn untagged_num() {
    assert_eq!(
        parse::<Untagged, Standard>(&E.u32(42)),
        Ok(Some(Untagged::Num(42)))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn untagged_ambiguity_first_declared_variant_wins() {
    // Known, accepted limitation of untagged dispatch for schema-driven
    // formats (matches `strede-postcard`'s identical trade-off): `Num(u32)`
    // structurally succeeds on almost any leading byte — it's just a
    // varint read with no distinguishing wire marker — so it silently
    // shadows the later-declared `Text(String)` variant here. `Text`'s own
    // wire bytes (a length-prefixed string, `[2, b'h', b'i']`) get misread
    // as `Num` (reading just the leading length byte, `2`, as a u32),
    // leaving `"hi"` as unconsumed trailing garbage — a top-level
    // trailing-bytes error, not a graceful `Text("hi")` result. This is the
    // real, structural cost of "no wire type tags": declaration order
    // decides, not correctness.
    let data = E.str("hi");
    assert_eq!(
        parse_err::<Untagged, Standard>(&data),
        BincodeError::ExpectedEnd
    );
}
