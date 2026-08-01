//! Owned-family `#[strede(transparent)]` fixture.
//!
//! Mirrors `transparent_borrow.rs`; see that file for context.

extern crate std;
mod helpers;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_test_util::block_on_loop;

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
fn transparent_u32() {
    // Bare small uint, no array wrapper.
    assert_eq!(
        parse!(TransparentWrapper, &[helpers::uint_small(7)]),
        Some(TransparentWrapper(7))
    );
}

#[test]
fn transparent_miss_on_string() {
    let msg = helpers::tstr("hello");
    assert_eq!(parse!(TransparentWrapper, &msg), None);
}
