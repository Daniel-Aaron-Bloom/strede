//! Borrow-family `#[strede(transparent)]` fixture.
//!
//! Unlike a plain tuple struct (`structs_borrow.rs`'s `Wrapper`, which
//! already reads as a bare positional value with no count prefix — see that
//! file's header comment), a transparent struct skips the map/seq facade
//! entirely and deserializes as its inner field directly via
//! `deserialize_value`.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(transparent)]
struct TransparentWrapper(u32);

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

fn parse_err<'de, T>(input: &'de [u8]) -> PostcardError
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())) {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn transparent_u32() {
    let data = varint(7);
    assert_eq!(
        parse::<TransparentWrapper>(&data),
        Ok(Some(TransparentWrapper(7)))
    );
}

// "Wrong-type Miss" is omitted: postcard has no wire tag to detect a
// mismatch (see from_field_borrow.rs for the same reasoning). A truncated
// (empty) input is the closest observable failure.
#[test]
fn transparent_truncated_errors() {
    assert_eq!(
        parse_err::<TransparentWrapper>(&[]),
        PostcardError::UnexpectedEnd
    );
}
