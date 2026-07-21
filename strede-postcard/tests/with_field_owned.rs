//! Owned-family `#[strede(with = "module")]` fixture (field level).
//!
//! `with` is shorthand for setting `deserialize_owned_with` to
//! `module::deserialize_owned`. Mirrors `with_field_borrow.rs`.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede::{Chunk, DeserializerOwned, EntryOwned, Probe, StrAccessOwned};
use strede::{hit, or_miss};
use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

mod hex_color {
    use super::*;

    pub async fn deserialize_owned<D: DeserializerOwned>(
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
}

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Swatch {
    name: String,
    #[strede(with = "hex_color")]
    color: u32,
}

#[test]
fn with_hex_color_hit() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#ff0000"));
    assert_eq!(
        parse_owned!(Swatch, &data),
        Ok(Some(Swatch {
            name: "red".into(),
            color: 0xff0000,
        }))
    );
}

#[test]
fn with_hex_color_missing_hash_prefix_misses() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("ff0000"));
    assert_eq!(parse_owned!(Swatch, &data), Ok(None));
}

// Truncated positional input surfaces as `UnexpectedEnd`, not `Probe::Miss`
// (see structs_owned.rs `named_truncated_errors`). Replaces the JSON
// reference's `with_hex_color_missing_field_misses` test.
#[test]
fn with_hex_color_truncated_before_color_field_errors() {
    let data = pstr("red");
    assert_eq!(
        parse_owned!(Swatch, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}

#[test]
fn with_hex_color_hit_chunked() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#ff0000"));
    assert_eq!(
        parse_owned_chunked!(Swatch, &data, 3),
        Ok(Some(Swatch {
            name: "red".into(),
            color: 0xff0000,
        }))
    );
}
