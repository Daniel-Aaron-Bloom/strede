//! Owned-family: integers out of range for the target type must miss, not error.
//!
//! `untagged_falls_through_to_wider_type_on_overflow` previously deadlocked
//! (hung the test process indefinitely) rather than failing fast: untagged
//! dispatch forked one live handle per candidate variant but awaited them
//! sequentially to completion instead of racing them via `select_probe!`,
//! violating CLAUDE.md's "owned family — parallel scanning and deadlock
//! hazard" contract. It stayed hidden as long as `NumberAccess::next_chunk`
//! could self-terminate a number without ever calling `Handle::next()` -
//! once that was fixed (see TESTING_GAPS.md item #3), the `u8` candidate's
//! overflow check on `"300"` needed a real refill confirmation, which then
//! blocked forever on its untouched `u32` sibling. Fixed in
//! `strede-derive/src/owned/enum_.rs` by racing all candidates concurrently.
//! Uses `block_on_loop_bounded` instead of `block_on_loop` so a regression
//! here fails fast with a clear panic instead of hanging the test process.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
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
                    let de = ChunkedJsonDeserializer::new(shared);
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
    assert_eq!(parse!(u8, b"300"), None);
    assert_eq!(parse!(i8, b"-200"), None);
}

#[test]
fn in_range_still_hits() {
    assert_eq!(parse!(u8, b"200"), Some(200));
}

#[test]
fn untagged_falls_through_to_wider_type_on_overflow() {
    assert_eq!(parse!(MaybeU8, b"300"), Some(MaybeU8::Big(300)));
    assert_eq!(parse!(MaybeU8, b"7"), Some(MaybeU8::Small(7)));
}
