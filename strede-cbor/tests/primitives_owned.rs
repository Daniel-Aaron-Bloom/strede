//! Owned-family primitive deserialization: char.

extern crate std;
mod helpers;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_test_util::block_on_loop;

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

// --- char ---

#[test]
fn char_from_single_char_tstr() {
    let enc = helpers::tstr("x");
    assert_eq!(parse!(char, &enc), Some('x'));
}

#[test]
fn char_misses_multi_char_tstr() {
    let enc = helpers::tstr("xy");
    assert_eq!(parse!(char, &enc), None);
}

#[test]
fn char_misses_empty_tstr() {
    let enc = helpers::tstr("");
    assert_eq!(parse!(char, &enc), None);
}

#[test]
fn char_misses_non_string() {
    assert_eq!(parse!(char, &[0x00]), None);
    assert_eq!(parse!(char, &[helpers::cbor_null()]), None);
}
