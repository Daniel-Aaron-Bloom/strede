//! Owned-family: integers out of range for the target type must miss, not error.
//!
//! CBOR's `ParseNum` (see `strede-cbor/src/impls.rs`) already used
//! `try_from`/`checked_*` conversions throughout, so this is a confirming
//! test rather than a regression fix.
//!
//! Uses `block_on_loop_bounded` (not `block_on_loop`) so that if
//! `untagged_falls_through_to_wider_type_on_overflow` ever regresses into the
//! owned untagged-dispatch deadlock described in TESTING_GAPS.md (item #3:
//! candidate handles awaited sequentially instead of raced via
//! `select_probe!`), it fails fast with a clear panic instead of hanging the
//! test process - mirroring the equivalent JSON test
//! (`strede-json/tests/number_range_owned.rs`), which is where that bug was
//! actually caught.

extern crate std;
mod helpers;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
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
                    let de = ChunkedCborDeserializer::new(shared);
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ())
                        .await
                        .unwrap()
                    {
                        Probe::Hit((_, v)) => Some(v),
                        Probe::Miss => None,
                    }
                },
            ),
            20_000,
        )
    }};
}

#[test]
fn out_of_range_misses_instead_of_erroring() {
    assert_eq!(parse!(u8, &helpers::uint16(300)), None);
    assert_eq!(parse!(i8, &helpers::negint16(200)), None);
}

#[test]
fn in_range_still_hits() {
    assert_eq!(parse!(u8, &helpers::uint8(200)), Some(200));
}

#[test]
fn untagged_falls_through_to_wider_type_on_overflow() {
    assert_eq!(
        parse!(MaybeU8, &helpers::uint16(300)),
        Some(MaybeU8::Big(300))
    );
    assert_eq!(
        parse!(MaybeU8, &[helpers::uint_small(7)]),
        Some(MaybeU8::Small(7))
    );
}
