//! Owned-family `#[strede(transparent)]` fixture.
//!
//! Mirrors `transparent_borrow.rs`; see that file for context.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
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

#[test]
fn transparent_u32() {
    // Bare positive fixint, no fixarray wrapper.
    assert_eq!(
        parse!(TransparentWrapper, &[7]),
        Some(TransparentWrapper(7))
    );
}

#[test]
fn transparent_miss_on_string() {
    let msg = fixstr("hello");
    assert_eq!(parse!(TransparentWrapper, &msg), None);
}
