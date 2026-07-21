//! Borrow-family: integers out of range for the target type must miss, not error.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum MaybeU8 {
    Small(u8),
    Big(u32),
}

fn parse<'de, T>(input: &'de str) -> Option<T>
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn out_of_range_misses_instead_of_erroring() {
    assert_eq!(parse::<u8>("300"), None);
    assert_eq!(parse::<i8>("-200"), None);
}

#[test]
fn in_range_still_hits() {
    assert_eq!(parse::<u8>("200"), Some(200));
}

#[test]
fn untagged_falls_through_to_wider_type_on_overflow() {
    assert_eq!(parse::<MaybeU8>("300"), Some(MaybeU8::Big(300)));
    assert_eq!(parse::<MaybeU8>("7"), Some(MaybeU8::Small(7)));
}
