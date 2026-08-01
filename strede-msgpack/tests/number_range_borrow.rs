//! Borrow-family: integers out of range for the target type must miss, not error.
//!
//! Also exercises the `#[strede(untagged)]` `deserialize_value` shape-based
//! fallback path (see TESTING_GAPS.md item #8), previously only tested in
//! strede-json and strede-cbor.

mod helpers;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum MaybeU8 {
    Small(u8),
    Big(u32),
}

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, MsgpackDeserializer<'de>, Extra = ()>,
{
    let de = MsgpackDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn out_of_range_misses_instead_of_erroring() {
    // uint16(300) doesn't fit u8
    assert_eq!(parse::<u8>(&helpers::uint16(300)), None);
    // int16(-200) doesn't fit i8
    assert_eq!(parse::<i8>(&helpers::int16(-200)), None);
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
    // 7 fits the positive-fixint range: a single raw byte.
    assert_eq!(parse::<MaybeU8>(&[7]), Some(MaybeU8::Small(7)));
}
