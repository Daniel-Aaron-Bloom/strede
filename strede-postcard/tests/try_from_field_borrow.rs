//! Borrow-family `#[strede(try_from = "FromType")]` fixture (field level).
//!
//! `port` is `u16` on the wire-facing struct but deserializes as `i32` and
//! converts via `u16::try_from(i32)`, missing (not erroring) on failure.
//! Postcard encodes `i32` as zigzag varint, so a negative value decodes
//! cleanly — the `TryFrom` failure is a genuine semantic mismatch that
//! postcard can detect *after* a successful decode, unlike a wire-shape
//! mismatch.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Server<'de> {
    name: &'de str,
    #[strede(try_from = "i32")]
    port: u16,
}

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

#[test]
fn try_from_i32_hit() {
    let mut data = pstr("web");
    data.extend_from_slice(&zigzag(8080));
    assert_eq!(
        parse::<Server<'_>>(&data),
        Ok(Some(Server {
            name: "web",
            port: 8080,
        }))
    );
}

#[test]
fn try_from_i32_negative_misses() {
    // -1 decodes fine as an i32 (zigzag), but u16::try_from(-1) fails ->
    // Miss, not an error. This is the meaningful analogue of the JSON
    // reference's negative-value case: the wire value decodes, the semantic
    // conversion rejects it.
    let mut data = pstr("web");
    data.extend_from_slice(&zigzag(-1));
    assert_eq!(parse::<Server<'_>>(&data), Ok(None));
}

// "Wrong-type Miss" (e.g. a string on the wire where `i32` is expected) is
// omitted: postcard has no wire tag to detect the mismatch — see
// from_field_borrow.rs for the same reasoning.

// Postcard has no way to observe a "missing field" gracefully — truncated
// positional input surfaces as `UnexpectedEnd`, not `Probe::Miss` (see
// structs_borrow.rs `named_truncated_errors`). Replaces the JSON reference's
// `try_from_i32_missing_field_misses` test.
#[test]
fn try_from_i32_truncated_before_port_field_errors() {
    let data = pstr("web");
    assert_eq!(parse_err::<Server<'_>>(&data), PostcardError::UnexpectedEnd);
}
