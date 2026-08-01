//! Borrow-family `#[strede(transparent)]` fixture.
//!
//! Unlike a plain tuple struct (which deserializes from a 1-element CBOR
//! array), a transparent struct deserializes as its inner field directly
//! with no wrapper token at all.

extern crate std;
mod helpers;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(transparent)]
struct TransparentWrapper(u32);

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
fn transparent_u32() {
    // Bare small uint, no array wrapper.
    assert_eq!(
        parse::<TransparentWrapper>(&[helpers::uint_small(7)]),
        Some(TransparentWrapper(7))
    );
}

#[test]
fn transparent_miss_on_string() {
    let msg = helpers::tstr("hello");
    assert_eq!(parse::<TransparentWrapper>(&msg), None);
}
