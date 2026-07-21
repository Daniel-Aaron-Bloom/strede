//! Owned-family `#[strede(deserialize_owned_with = "path")]` fixture.
//!
//! Exercises a hand-written `deserialize_owned` function (CBOR text string
//! `"#rrggbb"` -> `u32`) plugged into a derived struct field.

extern crate std;
mod helpers;

use strede::DeserializeOwned;
use strede::{
    Chunk, DeserializerOwned, EntryOwned, Probe, SharedBuf, StrAccessOwned, hit, or_miss,
};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_test_util::block_on_loop;

async fn deserialize_owned_hex_color<D: DeserializerOwned>(
    d: D,
    _extra: (),
) -> Result<Probe<(D::Claim, u32)>, D::Error> {
    d.entry(|[e]| async move {
        let mut chunks = hit!(e.deserialize_str_chunks().await);
        let mut out = std::string::String::new();
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
    name: std::string::String,
    #[strede(deserialize_owned_with = "deserialize_owned_hex_color")]
    color: u32,
}

fn concat(parts: &[&[u8]]) -> std::vec::Vec<u8> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedCborDeserializer::new(shared);
                match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                {
                    Probe::Hit((_, v)) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
}

#[test]
fn hex_color_hit() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("red")),
        (&helpers::tstr("color"), &helpers::tstr("#ff0000")),
    ]);
    let s: Swatch = parse!(Swatch, &msg).unwrap();
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
    let v: Option<Swatch> = parse!(Swatch, &msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_invalid_hex_digits_misses() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("red")),
        (&helpers::tstr("color"), &helpers::tstr("#zzzzzz")),
    ]);
    let v: Option<Swatch> = parse!(Swatch, &msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_wrong_type_misses() {
    let msg = concat(&[
        &helpers::map(2),
        &helpers::tstr("name"),
        &helpers::tstr("red"),
        &helpers::tstr("color"),
        &[helpers::uint_small(1)],
    ]);
    let v: Option<Swatch> = parse!(Swatch, &msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_missing_field_misses() {
    let msg = helpers::build_map(&[(&helpers::tstr("name"), &helpers::tstr("red"))]);
    let v: Option<Swatch> = parse!(Swatch, &msg);
    assert!(v.is_none());
}
