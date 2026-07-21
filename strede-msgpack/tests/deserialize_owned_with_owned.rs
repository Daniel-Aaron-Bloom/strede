//! Owned-family `#[strede(deserialize_owned_with = "path")]` fixture.
//!
//! Exercises a hand-written `deserialize_owned` function (hex color string
//! `"#rrggbb"` -> `u32`) plugged into a derived struct field.

mod helpers;
use helpers::*;

use strede::DeserializeOwned;
use strede::{
    Chunk, DeserializerOwned, EntryOwned, Probe, SharedBuf, StrAccessOwned, hit, or_miss,
};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

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

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
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
    let color_val = fixstr("ff0000");
    let msg = swatch_msg("red", &color_val);
    let v: Option<Swatch> = parse!(Swatch, &msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_wrong_type_misses() {
    let name_key = fixstr("name");
    let color_key = fixstr("color");
    let name_val = fixstr("red");
    let msg = build_map(&[
        (name_key.as_slice(), name_val.as_slice()),
        (color_key.as_slice(), &[7u8]),
    ]);
    let v: Option<Swatch> = parse!(Swatch, &msg);
    assert!(v.is_none());
}

#[test]
fn hex_color_missing_field_misses() {
    let name_key = fixstr("name");
    let name_val = fixstr("red");
    let msg = build_map(&[(name_key.as_slice(), name_val.as_slice())]);
    let v: Option<Swatch> = parse!(Swatch, &msg);
    assert!(v.is_none());
}
