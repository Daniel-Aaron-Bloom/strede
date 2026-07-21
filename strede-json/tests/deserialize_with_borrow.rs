//! Borrow-family `#[strede(deserialize_with = "path")]` fixture.
//!
//! Exercises a hand-written `deserialize` function (hex color string
//! `"#rrggbb"` -> `u32`) plugged into a derived struct field.

use strede::{Deserializer, Entry, Probe};
use strede::{hit, or_miss};
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
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
    name: String,
    #[strede(deserialize_with = "deserialize_hex_color")]
    color: u32,
}

fn parse<'de, T>(input: &'de str) -> Option<T>
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn hex_color_hit() {
    let s: Swatch = parse(r##"{"name": "red", "color": "#ff0000"}"##).unwrap();
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
    let v: Option<Swatch> = parse(r#"{"name": "red", "color": "ff0000"}"#);
    assert!(v.is_none());
}

#[test]
fn hex_color_invalid_hex_digits_misses() {
    let v: Option<Swatch> = parse(r##"{"name": "red", "color": "#zzzzzz"}"##);
    assert!(v.is_none());
}

#[test]
fn hex_color_missing_field_misses() {
    let v: Option<Swatch> = parse(r#"{"name": "red"}"#);
    assert!(v.is_none());
}
