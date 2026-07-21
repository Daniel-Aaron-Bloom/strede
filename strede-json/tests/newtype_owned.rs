//! Owned-family newtype (transparent tuple-struct) fixtures.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Wrapper(u32);

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(transparent)]
struct TransparentWrapper(u32);

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
fn newtype_u32() {
    assert_eq!(parse!(Wrapper, b"[42]"), Some(Wrapper(42)));
}

#[test]
fn newtype_zero() {
    assert_eq!(parse!(Wrapper, b"[0]"), Some(Wrapper(0)));
}

#[test]
fn newtype_miss_on_string() {
    assert_eq!(parse!(Wrapper, b"\"hello\""), None);
}

#[test]
fn newtype_miss_on_null() {
    assert_eq!(parse!(Wrapper, b"null"), None);
}

#[test]
fn transparent_u32() {
    assert_eq!(
        parse!(TransparentWrapper, b"42"),
        Some(TransparentWrapper(42))
    );
}

#[test]
fn transparent_miss_on_string() {
    assert_eq!(parse!(TransparentWrapper, b"\"hello\""), None);
}
