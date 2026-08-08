//! Enum deserialization via the owned/chunked family. Mirrors
//! `enums_borrow.rs`.

#![recursion_limit = "512"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_bincode::{BincodeError, Standard};
use strede_derive::DeserializeOwned;

const E: Enc = Enc::STANDARD;

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Signal {
    Ping,
    Pong,
}

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Event {
    Ping,
    Move { x: u32, y: u32 },
    Wrap(u32),
}

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Tagged {
    A,
    B(u8),
    C(u8),
}

#[test]
fn unit_variant_ping() {
    assert_eq!(
        parse_owned!(Signal, Standard, &E.discriminant(0)),
        Ok(Some(Signal::Ping))
    );
}

#[test]
fn unit_variant_out_of_range_misses() {
    assert_eq!(
        parse_owned!(Signal, Standard, &E.discriminant(99)),
        Ok(None)
    );
}

#[test]
fn struct_variant_move() {
    let mut data = E.discriminant(1);
    data.extend_from_slice(&E.u32(10));
    data.extend_from_slice(&E.u32(20));
    assert_eq!(
        parse_owned!(Event, Standard, &data),
        Ok(Some(Event::Move { x: 10, y: 20 }))
    );
}

#[test]
fn newtype_variant_wrap() {
    let mut data = E.discriminant(2);
    data.extend_from_slice(&E.u32(42));
    assert_eq!(
        parse_owned!(Event, Standard, &data),
        Ok(Some(Event::Wrap(42)))
    );
}

#[test]
fn tagged_b() {
    let mut data = E.discriminant(1);
    data.extend_from_slice(&E.u8(7));
    assert_eq!(
        parse_owned!(Tagged, Standard, &data),
        Ok(Some(Tagged::B(7)))
    );
}

#[test]
fn unknown_discriminant_misses() {
    assert_eq!(parse_owned!(Tagged, Standard, &E.discriminant(3)), Ok(None));
}

#[test]
fn discriminant_non_canonical_prefix_errors() {
    // Mirrors `enums_borrow.rs`'s identical test: discriminant 1 fits a
    // single byte, but is encoded here via the 16-byte u128-tail prefix
    // (254) — `decode_discriminant` has no "try another type" fallback, so
    // this is a hard error.
    let data = E.varint_with_prefix(254, 1);
    assert_eq!(
        parse_owned!(Tagged, Standard, &data).unwrap_err(),
        BincodeError::NonCanonicalVarint
    );
}

// --- `#[strede(other)]` catch-all ---

#[derive(Debug, PartialEq, DeserializeOwned)]
enum WithOther {
    A,
    B(u8),
    #[strede(other)]
    Unknown,
}

#[test]
fn other_catches_unrecognized_discriminant() {
    assert_eq!(
        parse_owned!(WithOther, Standard, &E.discriminant(2)),
        Ok(Some(WithOther::Unknown))
    );
}

#[test]
fn other_with_unexpected_trailing_payload_errors() {
    let mut data = E.discriminant(2);
    data.extend_from_slice(&E.u32(123));
    assert_eq!(
        parse_owned!(WithOther, Standard, &data).unwrap_err(),
        BincodeError::ExpectedEnd
    );
}

// --- `#[strede(untagged)]` — mirrors `enums_borrow.rs`'s identical
// ambiguity documentation ---

#[cfg(feature = "alloc")]
#[derive(Debug, PartialEq, DeserializeOwned)]
#[strede(untagged)]
enum Untagged {
    Num(u32),
    Text(String),
}

#[cfg(feature = "alloc")]
#[test]
fn untagged_num() {
    assert_eq!(
        parse_owned!(Untagged, Standard, &E.u32(42)),
        Ok(Some(Untagged::Num(42)))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn untagged_ambiguity_first_declared_variant_wins() {
    // See `enums_borrow.rs`'s identical test for the full explanation.
    let data = E.str("hi");
    assert_eq!(
        parse_owned!(Untagged, Standard, &data).unwrap_err(),
        BincodeError::ExpectedEnd
    );
}
