//! Owned-family: integers out of range for the target type must miss, not error.
//!
//! Also exercises the `#[strede(untagged)]` `deserialize_value` shape-based
//! fallback path (see TESTING_GAPS.md item #8), previously only tested in
//! strede-json, strede-cbor, and strede-msgpack. See
//! `number_range_borrow.rs` for why there is no postcard equivalent of
//! `strede-json`'s null-vs-number untagged-unit-variant test.
//!
//! Uses `block_on_loop_bounded` (not `block_on_loop`) so that if the owned
//! untagged-dispatch deadlock hazard described in TESTING_GAPS.md (item #3:
//! candidate handles awaited sequentially instead of raced via
//! `select_probe!`) is ever reintroduced, this fails fast with a clear panic
//! instead of hanging the test process - mirroring
//! `strede-cbor/tests/number_range_owned.rs` and
//! `strede-json/tests/number_range_owned.rs`, where that bug was originally
//! caught.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_postcard::{ChunkedPostcardDeserializer, PostcardError};
use strede_test_util::block_on_loop_bounded;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(untagged)]
enum MaybeU8 {
    Small(u8),
    Big(u32),
}

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop_bounded(
            SharedBuf::with_async(
                input,
                async |buf: &mut &[u8]| {
                    *buf = &[];
                },
                async |shared| {
                    let de = ChunkedPostcardDeserializer::new(shared);
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ()).await {
                        Ok(Probe::Hit((_, v))) => Ok(Some(v)),
                        Ok(Probe::Miss) => Ok(None),
                        Err(e) => Err(e),
                    }
                },
            ),
            20_000,
        )
    }};
}

#[test]
fn out_of_range_misses_instead_of_erroring() {
    let result: Result<Option<u8>, PostcardError> = parse!(u8, &varint(300));
    assert_eq!(result, Ok(None));
}

#[test]
fn in_range_still_hits() {
    let result: Result<Option<u8>, PostcardError> = parse!(u8, &varint(200));
    assert_eq!(result, Ok(Some(200)));
}

#[test]
fn untagged_falls_through_to_wider_type_on_overflow() {
    let big: Result<Option<MaybeU8>, PostcardError> = parse!(MaybeU8, &varint(300));
    assert_eq!(big, Ok(Some(MaybeU8::Big(300))));
    let small: Result<Option<MaybeU8>, PostcardError> = parse!(MaybeU8, &varint(7));
    assert_eq!(small, Ok(Some(MaybeU8::Small(7))));
}
