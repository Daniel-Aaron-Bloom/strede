//! Borrow-family `#[strede(transparent)]` fixture.
//!
//! Mirrors `strede-json/tests/newtype_borrow.rs`'s `TransparentWrapper` case:
//! unlike a plain tuple struct (see `struct_borrow.rs`'s `Wrapper`, which
//! wraps in a 1-element array), a transparent struct deserializes as its
//! inner field directly with no wrapper token at all.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(transparent)]
struct TransparentWrapper(u32);

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
fn transparent_u32() {
    // Bare positive fixint, no fixarray wrapper.
    assert_eq!(
        parse::<TransparentWrapper>(&[7]),
        Some(TransparentWrapper(7))
    );
}

#[test]
fn transparent_miss_on_string() {
    assert_eq!(parse::<TransparentWrapper>(&fixstr("hello")), None);
}
