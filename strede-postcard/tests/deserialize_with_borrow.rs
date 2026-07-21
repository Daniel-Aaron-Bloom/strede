//! Borrow-family `#[strede(deserialize_with = "path")]` fixture.
//!
//! Exercises a hand-written `deserialize` function (hex color string
//! "#rrggbb" -> `u32`) plugged into a derived struct field. Postcard encodes
//! the field as a normal length-prefixed string (`pstr`), so the custom
//! function's `deserialize_str` probe works exactly as it does for JSON —
//! only the wire encoding of the fixture data differs from the JSON reference.

mod helpers;
use helpers::*;

use strede::{Deserializer, Entry, Probe};
use strede::{hit, or_miss};
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

async fn deserialize_hex_color<'de, D: Deserializer<'de>>(
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

#[derive(Debug, PartialEq, Deserialize)]
struct Swatch<'de> {
    name: &'de str,
    #[strede(deserialize_with = "deserialize_hex_color")]
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
fn hex_color_hit() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#ff0000"));
    let s: Swatch<'_> = parse(&data).unwrap().unwrap();
    assert_eq!(
        s,
        Swatch {
            name: "red",
            color: 0xff0000,
        }
    );
}

#[test]
fn hex_color_missing_hash_prefix_misses() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("ff0000"));
    assert_eq!(parse::<Swatch<'_>>(&data), Ok(None));
}

#[test]
fn hex_color_invalid_hex_digits_misses() {
    let mut data = pstr("red");
    data.extend_from_slice(&pstr("#zzzzzz"));
    assert_eq!(parse::<Swatch<'_>>(&data), Ok(None));
}

// Postcard has no way to observe a "missing field" — running out of bytes
// before the positionally-expected `color` field is a hard `UnexpectedEnd`
// error, not a `Probe::Miss` (see structs_borrow.rs `named_truncated_errors`
// for the established convention). This replaces the JSON reference's
// `hex_color_missing_field_misses` test.
#[test]
fn hex_color_truncated_before_color_field_errors() {
    let data = pstr("red");
    assert_eq!(parse_err::<Swatch<'_>>(&data), PostcardError::UnexpectedEnd);
}
