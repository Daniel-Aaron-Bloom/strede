//! Borrow-family `#[strede(deserialize_with = "path")]` fixture.
//!
//! Exercises a hand-written `deserialize` function (CBOR text string
//! `"#rrggbb"` -> `u32`) plugged into a derived struct field.

extern crate std;
mod helpers;

use strede::{Deserializer, Entry, Probe};
use strede::{hit, or_miss};
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
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
struct Swatch {
    name: std::string::String,
    #[strede(deserialize_with = "deserialize_hex_color")]
    color: u32,
}

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    match block_on(T::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

fn concat(parts: &[&[u8]]) -> std::vec::Vec<u8> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

#[test]
fn hex_color_hit() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("red")),
        (&helpers::tstr("color"), &helpers::tstr("#ff0000")),
    ]);
    let s: Swatch = parse(&msg).unwrap();
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
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("red")),
        (&helpers::tstr("color"), &helpers::tstr("ff0000")),
    ]);
    let v: Option<Swatch> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_invalid_hex_digits_misses() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("red")),
        (&helpers::tstr("color"), &helpers::tstr("#zzzzzz")),
    ]);
    let v: Option<Swatch> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_wrong_type_misses() {
    // color is a number, not a text string
    let msg = concat(&[
        &helpers::map(2),
        &helpers::tstr("name"),
        &helpers::tstr("red"),
        &helpers::tstr("color"),
        &[helpers::uint_small(1)],
    ]);
    let v: Option<Swatch> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_missing_field_misses() {
    let msg = helpers::build_map(&[(&helpers::tstr("name"), &helpers::tstr("red"))]);
    let v: Option<Swatch> = parse(&msg);
    assert!(v.is_none());
}
