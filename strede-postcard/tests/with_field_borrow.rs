//! Borrow-family `#[strede(with = "module")]` fixture (field level).
//!
//! `with` is shorthand for setting `deserialize_with` to `module::deserialize`.
//! Same hex-color scenario as `deserialize_with_borrow.rs`, wrapped in a
//! module to exercise the path-resolution difference between the two
//! attributes.

mod helpers;
use helpers::*;

use strede::{Deserializer, Entry, Probe};
use strede::{hit, or_miss};
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

mod hex_color {
    use super::*;

    pub async fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, u32)>, D::Error> {
        d.entry(|[e]| async move {
            let (claim, s) = hit!(e.deserialize_str().await);
            let hex = or_miss!(s.strip_prefix('#'));
            let value = or_miss!(u32::from_str_radix(hex, 16).ok());
            Ok(Probe::Hit((claim, value)))
        })
        .await
    }
}

#[derive(Debug, PartialEq, Deserialize)]
struct Swatch<'de> {
    name: &'de str,
    #[strede(with = "hex_color")]
    color: u32,
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
fn with_hex_color_hit() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#ff0000"));
    assert_eq!(
        parse::<Swatch<'_>>(&data),
        Ok(Some(Swatch {
            name: "red",
            color: 0xff0000,
        }))
    );
}

#[test]
fn with_hex_color_missing_hash_prefix_misses() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("ff0000"));
    assert_eq!(parse::<Swatch<'_>>(&data), Ok(None));
}

// Postcard has no way to observe a "missing field" gracefully — truncated
// positional input surfaces as `UnexpectedEnd`, not `Probe::Miss` (see
// structs_borrow.rs `named_truncated_errors`). Replaces the JSON reference's
// `with_hex_color_missing_field_misses` test.
#[test]
fn with_hex_color_truncated_before_color_field_errors() {
    let data = pstr("red");
    assert_eq!(parse_err::<Swatch<'_>>(&data), PostcardError::UnexpectedEnd);
}
