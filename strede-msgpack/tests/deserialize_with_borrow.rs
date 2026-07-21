//! Borrow-family `#[strede(deserialize_with = "path")]` fixture.
//!
//! Exercises a hand-written `deserialize` function (hex color string
//! `"#rrggbb"` -> `u32`) plugged into a derived struct field.

mod helpers;
use helpers::*;

use strede::{Deserializer, Entry, Probe};
use strede::{hit, or_miss};
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
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

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, MsgpackDeserializer<'de>, Extra = ()>,
{
    let de = MsgpackDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

fn swatch_msg(name: &str, color: &[u8]) -> Vec<u8> {
    let name_key = fixstr("name");
    let color_key = fixstr("color");
    let name_val = fixstr(name);
    build_map(&[
        (name_key.as_slice(), name_val.as_slice()),
        (color_key.as_slice(), color),
    ])
}

#[test]
fn hex_color_hit() {
    let color_val = fixstr("#ff0000");
    let msg = swatch_msg("red", &color_val);
    let s: Swatch<'_> = parse(&msg).unwrap();
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
    let color_val = fixstr("ff0000");
    let msg = swatch_msg("red", &color_val);
    let v: Option<Swatch<'_>> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_wrong_type_misses() {
    // color is a number, not a string
    let name_key = fixstr("name");
    let color_key = fixstr("color");
    let name_val = fixstr("red");
    let msg = build_map(&[
        (name_key.as_slice(), name_val.as_slice()),
        (color_key.as_slice(), &[7u8]),
    ]);
    let v: Option<Swatch<'_>> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_missing_field_misses() {
    let name_key = fixstr("name");
    let name_val = fixstr("red");
    let msg = build_map(&[(name_key.as_slice(), name_val.as_slice())]);
    let v: Option<Swatch<'_>> = parse(&msg);
    assert!(v.is_none());
}
