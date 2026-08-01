//! Borrow-family: integers out of range for the target type must miss, not error.
//!
//! Also exercises the `#[strede(untagged)]` `deserialize_value` shape-based
//! fallback path (see TESTING_GAPS.md item #8), previously only tested in
//! strede-json, strede-cbor, and strede-msgpack. Postcard's `ParseNum` impls
//! already use `try_from` (see `strede-postcard/src/impls.rs`), so a bare
//! `u8`/`u16` overflow miss is already covered by `primitives_borrow.rs`
//! (`u8_out_of_range_misses` etc.) - what's new here is exercising the
//! *derive-generated* untagged-enum dispatch itself (`Small(u8)` missing so
//! the probe chain falls through and re-tries `Big(u32)` against the same
//! bytes via `fork`), not just the underlying numeric parse.
//!
//! There is no postcard-side equivalent of `strede-json`'s
//! `MaybeUnit { Null, Num(u32) }` untagged-unit-variant test: postcard has
//! no distinct wire representation for "null" - `ParseNum for ()` always
//! hits, unconditionally, consuming zero bytes (see `impls.rs`) - so a
//! `Null` variant declared before `Num` would win regardless of the wire
//! content, and one declared after `Num` would never be reachable. Either
//! way the test would assert something about declaration-order tie-break,
//! not about shape-based fallback, so it's omitted as not meaningful for
//! this format.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum MaybeU8 {
    Small(u8),
    Big(u32),
}

fn parse<'de, T>(input: &'de [u8]) -> Result<Option<T>, PostcardError>
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Ok(Some(v)),
        Probe::Miss => Ok(None),
    }
}

#[test]
fn out_of_range_misses_instead_of_erroring() {
    assert_eq!(parse::<u8>(&varint(300)), Ok(None));
}

#[test]
fn in_range_still_hits() {
    assert_eq!(parse::<u8>(&varint(200)), Ok(Some(200)));
}

#[test]
fn untagged_falls_through_to_wider_type_on_overflow() {
    assert_eq!(parse::<MaybeU8>(&varint(300)), Ok(Some(MaybeU8::Big(300))));
    assert_eq!(parse::<MaybeU8>(&varint(7)), Ok(Some(MaybeU8::Small(7))));
}
