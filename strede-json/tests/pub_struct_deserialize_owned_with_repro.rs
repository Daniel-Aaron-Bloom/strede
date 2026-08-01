//! Regression coverage for a private-type-in-public-interface leak
//! (`E0446: private type "__DeOwnedWith_<field>" in public interface`).
//!
//! The derive emits a private wrapper newtype for any field using
//! `deserialize_owned_with` / `deserialize_with` / `from` / `try_from`, to run
//! the custom conversion. That wrapper must never end up inside the
//! `MapFieldProviderOwned`/`MapFieldProvider` `Outputs` associated type, since
//! that trait impl is emitted for every derived named struct (flatten or not)
//! and is therefore part of the struct's public interface whenever the struct
//! itself is `pub`.
//!
//! This fixture goes one layer deeper than a single flat struct: `Signature`
//! deliberately does NOT implement `DeserializeOwned` itself (it can only be
//! produced through the custom function below), and `Pallet` nests `Swatch`
//! inside a `Vec`, so the fix also has to hold up through `Vec<T>`'s own
//! `DeserializeOwned` bound on `T`. The `Pallet` layer needs strede-json's
//! `Vec<T>: DeserializeOwned` impl (`src/vec.rs`), which only exists behind
//! the `alloc` feature — see `value_owned.rs` for the same gating convention.
#![cfg_attr(feature = "alloc", recursion_limit = "256")]

use strede::{
    Chunk, DeserializeOwned, DeserializerOwned, EntryOwned, Probe, SharedBuf, StrAccessOwned, hit,
    or_miss,
};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

/// Intentionally does not derive/implement `DeserializeOwned` — the only way
/// to produce one is through `deserialize_owned_signature` below.
#[derive(Debug, PartialEq)]
pub struct Signature(pub u32);

async fn deserialize_owned_signature<D: DeserializerOwned>(
    d: D,
    _extra: (),
) -> Result<Probe<(D::Claim, Signature)>, D::Error> {
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
        Ok(Probe::Hit((claim, Signature(value))))
    })
    .await
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
pub struct Swatch {
    pub id: u32,
    #[strede(deserialize_owned_with = "deserialize_owned_signature", bound = "")]
    pub signature: Signature,
}

/// One layer up: a `pub` struct holding a `Vec` of the struct with the custom
/// field, so `Swatch: DeserializeOwned` must actually hold (not just compile
/// via the wrapper trick) for `Vec<Swatch>: DeserializeOwned` to resolve.
#[cfg(feature = "alloc")]
#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
pub struct Pallet {
    pub swatches: Vec<Swatch>,
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
fn swatch_hit() {
    let s: Swatch = parse!(Swatch, br##"{"id": 1, "signature": "#ff0000"}"##).unwrap();
    assert_eq!(
        s,
        Swatch {
            id: 1,
            signature: Signature(0xff0000),
        }
    );
}

#[cfg(feature = "alloc")]
#[test]
fn pallet_of_swatches_hit() {
    let p: Pallet = parse!(
        Pallet,
        br##"{"swatches": [
            {"id": 1, "signature": "#ff0000"},
            {"id": 2, "signature": "#00ff00"}
        ]}"##
    )
    .unwrap();
    assert_eq!(
        p,
        Pallet {
            swatches: vec![
                Swatch {
                    id: 1,
                    signature: Signature(0xff0000),
                },
                Swatch {
                    id: 2,
                    signature: Signature(0x00ff00),
                },
            ],
        }
    );
}

#[cfg(feature = "alloc")]
#[test]
fn pallet_bad_signature_in_nested_swatch_misses() {
    let p: Option<Pallet> = parse!(
        Pallet,
        br##"{"swatches": [{"id": 1, "signature": "ff0000"}]}"##
    );
    assert!(p.is_none());
}
