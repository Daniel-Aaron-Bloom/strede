//! Owned-family `#[strede(with = "module")]` fixture (field level).
//!
//! `with` is shorthand for setting `deserialize_owned_with` to
//! `module::deserialize_owned`.

use strede::{
    Chunk, DeserializeOwned, DeserializerOwned, EntryOwned, Probe, SharedBuf, StrAccessOwned,
};
use strede::{hit, or_miss};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

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

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Swatch {
    name: String,
    #[strede(with = "hex_color")]
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
                let de = ChunkedJsonDeserializer::new(shared);
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
fn with_hex_color_hit() {
    let s: Swatch = parse!(Swatch, br##"{"name": "red", "color": "#ff0000"}"##).unwrap();
    assert_eq!(
        s,
        Swatch {
            name: "red".into(),
            color: 0xff0000,
        }
    );
}

#[test]
fn with_hex_color_missing_hash_prefix_misses() {
    let v: Option<Swatch> = parse!(Swatch, br#"{"name": "red", "color": "ff0000"}"#);
    assert!(v.is_none());
}

#[test]
fn with_hex_color_missing_field_misses() {
    let v: Option<Swatch> = parse!(Swatch, br#"{"name": "red"}"#);
    assert!(v.is_none());
}
