//! Borrow-family: integers out of range for the target type must miss, not error.
//!
//! CBOR's `ParseNum` (see `strede-cbor/src/impls.rs`) already used
//! `try_from`/`checked_*` conversions throughout, so this is a confirming
//! test rather than a regression fix (unlike strede-json, which had a real
//! bug here — see strede-json/tests/number_range_borrow.rs).

extern crate std;
mod helpers;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum MaybeU8 {
    Small(u8),
    Big(u32),
}

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    match block_on(T::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn out_of_range_misses_instead_of_erroring() {
    // uint16(300) doesn't fit u8
    assert_eq!(parse::<u8>(&helpers::uint16(300)), None);
    // negint16(200) -> actual -201, doesn't fit i8
    assert_eq!(parse::<i8>(&helpers::negint16(200)), None);
}

#[test]
fn in_range_still_hits() {
    assert_eq!(parse::<u8>(&helpers::uint8(200)), Some(200));
}

#[test]
fn untagged_falls_through_to_wider_type_on_overflow() {
    assert_eq!(
        parse::<MaybeU8>(&helpers::uint16(300)),
        Some(MaybeU8::Big(300))
    );
    assert_eq!(
        parse::<MaybeU8>(&[helpers::uint_small(7)]),
        Some(MaybeU8::Small(7))
    );
}
