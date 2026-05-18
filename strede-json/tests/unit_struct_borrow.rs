//! Borrow-family unit struct and untagged unit variant fixtures.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Unit;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(untagged)]
enum MaybeUnit {
    Null,
    Num(u32),
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
fn unit_struct_null() {
    assert_eq!(parse::<Unit>("null"), Some(Unit));
}

#[test]
fn unit_struct_miss_on_non_null() {
    assert_eq!(parse::<Unit>("42"), None);
    assert_eq!(parse::<Unit>(r#""hello""#), None);
}

#[test]
fn untagged_unit_variant_null() {
    assert_eq!(parse::<MaybeUnit>("null"), Some(MaybeUnit::Null));
}

#[test]
fn untagged_unit_variant_falls_through() {
    assert_eq!(parse::<MaybeUnit>("7"), Some(MaybeUnit::Num(7)));
}
