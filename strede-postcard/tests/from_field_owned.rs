//! Owned-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! Mirrors `from_field_borrow.rs`; `&'de str` becomes `String`.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Item {
    name: String,
    #[strede(from = "u16")]
    count: u32,
}

#[test]
fn from_u16_hit() {
    let mut data = pstr("widget");
    data.extend_from_slice(&varint(42));
    assert_eq!(
        parse_owned!(Item, &data),
        Ok(Some(Item {
            name: "widget".into(),
            count: 42,
        }))
    );
}

// "Wrong-type Miss" is omitted: postcard has no wire tag to detect a mismatch
// between a string and a `u16` varint — see from_field_borrow.rs for the same
// reasoning.

#[test]
fn from_u16_out_of_range_misses() {
    let mut data = pstr("widget");
    data.extend_from_slice(&varint(70000));
    assert_eq!(parse_owned!(Item, &data), Ok(None));
}

// Postcard has no way to observe a "missing field" gracefully — truncated
// positional input surfaces as `UnexpectedEnd`, not `Probe::Miss` (see
// structs_owned.rs `named_truncated_errors`). Replaces the JSON reference's
// `from_u16_missing_field_misses` test.
#[test]
fn from_u16_truncated_before_count_field_errors() {
    let data = pstr("widget");
    assert_eq!(
        parse_owned!(Item, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}

#[test]
fn from_u16_hit_chunked() {
    let mut data = pstr("widget");
    data.extend_from_slice(&varint(42));
    assert_eq!(
        parse_owned_chunked!(Item, &data, 2),
        Ok(Some(Item {
            name: "widget".into(),
            count: 42,
        }))
    );
}
