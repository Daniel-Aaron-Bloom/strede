//! Borrow-family newtype (transparent tuple-struct) fixtures.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Wrapper(u32);

#[derive(Debug, PartialEq, Deserialize)]
#[strede(transparent)]
struct TransparentWrapper(u32);

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
fn newtype_u32() {
    assert_eq!(parse::<Wrapper>("[42]"), Some(Wrapper(42)));
}

#[test]
fn newtype_zero() {
    assert_eq!(parse::<Wrapper>("[0]"), Some(Wrapper(0)));
}

#[test]
fn newtype_miss_on_string() {
    assert_eq!(parse::<Wrapper>(r#""hello""#), None);
}

#[test]
fn newtype_miss_on_null() {
    assert_eq!(parse::<Wrapper>("null"), None);
}

#[test]
fn transparent_u32() {
    assert_eq!(
        parse::<TransparentWrapper>("42"),
        Some(TransparentWrapper(42))
    );
}

#[test]
fn transparent_miss_on_string() {
    assert_eq!(parse::<TransparentWrapper>(r#""hello""#), None);
}
