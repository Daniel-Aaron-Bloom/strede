//! Owned-family `#[strede(deserialize_owned_with = "path")]` fixture.
//!
//! Mirrors `deserialize_with_borrow.rs`: hex color string "#rrggbb" -> `u32`,
//! but streamed through `deserialize_str_chunks` since the owned family has
//! no zero-copy `&str` borrow.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede::{Chunk, DeserializerOwned, EntryOwned, Probe, StrAccessOwned, hit, or_miss};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_postcard::PostcardError;

async fn deserialize_owned_hex_color<D: DeserializerOwned>(
    d: D,
    _extra: (),
) -> Result<Probe<(D::Claim, u32)>, D::Error> {
    d.entry(|[e]| async move {
        let mut chunks = hit!(e.deserialize_str_chunks().await);
        let mut out = String::new();
        let claim = loop {
            match chunks.next_str(|s| out.push_str(s)).await? {
                Chunk::Data((new, ())) => chunks = new,
                Chunk::Done(claim) => break claim,
            }
        };
        let hex = or_miss!(out.strip_prefix('#'));
        let value = or_miss!(u32::from_str_radix(hex, 16).ok());
        Ok(Probe::Hit((claim, value)))
    })
    .await
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Swatch {
    name: String,
    #[strede(deserialize_owned_with = "deserialize_owned_hex_color")]
    color: u32,
}

#[test]
fn hex_color_hit() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#ff0000"));
    let s: Swatch = parse_owned!(Swatch, &data).unwrap().unwrap();
    assert_eq!(
        s,
        Swatch {
            name: "red".into(),
            color: 0xff0000,
        }
    );
}

#[test]
fn hex_color_missing_hash_prefix_misses() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("ff0000"));
    assert_eq!(parse_owned!(Swatch, &data), Ok(None));
}

#[test]
fn hex_color_invalid_hex_digits_misses() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#zzzzzz"));
    assert_eq!(parse_owned!(Swatch, &data), Ok(None));
}

// Postcard has no way to observe a "missing field" — running out of bytes
// before the positionally-expected `color` field is a hard `UnexpectedEnd`
// error, not a `Probe::Miss` (see structs_owned.rs `named_truncated_errors`
// for the established convention). This replaces the JSON reference's
// `hex_color_missing_field_misses` test.
#[test]
fn hex_color_truncated_before_color_field_errors() {
    let data = pstr("red");
    assert_eq!(
        parse_owned!(Swatch, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}

#[test]
fn hex_color_hit_chunked() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#ff0000"));
    let s: Swatch = parse_owned_chunked!(Swatch, &data, 3).unwrap().unwrap();
    assert_eq!(
        s,
        Swatch {
            name: "red".into(),
            color: 0xff0000,
        }
    );
}
