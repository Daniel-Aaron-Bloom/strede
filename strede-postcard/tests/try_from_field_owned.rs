//! Owned-family `#[strede(try_from = "FromType")]` fixture (field level).
//!
//! Mirrors `try_from_field_borrow.rs`; `&'de str` becomes `String`.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Server {
    name: String,
    #[strede(try_from = "i32")]
    port: u16,
}

#[test]
fn try_from_i32_hit() {
    let mut data = pstr("web");
    data.extend_from_slice(&zigzag(8080));
    assert_eq!(
        parse_owned!(Server, &data),
        Ok(Some(Server {
            name: "web".into(),
            port: 8080,
        }))
    );
}

#[test]
fn try_from_i32_negative_misses() {
    // -1 decodes fine as an i32 (zigzag), but u16::try_from(-1) fails ->
    // Miss, not an error.
    let mut data = pstr("web");
    data.extend_from_slice(&zigzag(-1));
    assert_eq!(parse_owned!(Server, &data), Ok(None));
}

// "Wrong-type Miss" is omitted: postcard has no wire tag to detect a
// mismatch between a string and an `i32` varint — see from_field_borrow.rs.

// Truncated positional input surfaces as `UnexpectedEnd`, not `Probe::Miss`
// (see structs_owned.rs `named_truncated_errors`). Replaces the JSON
// reference's `try_from_i32_missing_field_misses` test.
#[test]
fn try_from_i32_truncated_before_port_field_errors() {
    let data = pstr("web");
    assert_eq!(
        parse_owned!(Server, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}

#[test]
fn try_from_i32_negative_misses_chunked() {
    let mut data = pstr("web");
    data.extend_from_slice(&zigzag(-1));
    assert_eq!(parse_owned_chunked!(Server, &data, 2), Ok(None));
}
