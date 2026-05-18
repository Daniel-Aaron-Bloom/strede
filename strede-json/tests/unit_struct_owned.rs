//! Owned-family unit struct and untagged unit variant fixtures.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Unit;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(untagged)]
enum MaybeUnit {
    Null,
    Num(u32),
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
fn unit_struct_null() {
    assert_eq!(parse!(Unit, b"null"), Some(Unit));
}

#[test]
fn unit_struct_miss_on_non_null() {
    assert_eq!(parse!(Unit, b"42"), None);
    assert_eq!(parse!(Unit, b"\"hello\""), None);
}

#[test]
fn untagged_unit_variant_null() {
    assert_eq!(parse!(MaybeUnit, b"null"), Some(MaybeUnit::Null));
}

#[test]
fn untagged_unit_variant_falls_through() {
    assert_eq!(parse!(MaybeUnit, b"7"), Some(MaybeUnit::Num(7)));
}
