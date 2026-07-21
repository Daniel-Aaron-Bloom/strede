//! Internally-tagged and adjacently-tagged enums (not supported by postcard).
//!
//! Postcard only supports externally-tagged enums (varint discriminant + payload).
//! Named tag fields cannot be represented on the wire. Both representations return
//! Probe::Miss regardless of input, matching serde-postcard's "WontImplement" stance
//! (postcard issue #125).
#![recursion_limit = "256"]

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

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

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "type")]
enum InternallyTagged {
    Foo { x: u32 },
    Bar,
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(tag = "t", content = "c")]
enum AdjacentlyTagged {
    Foo { x: u32 },
    Bar,
}

#[test]
fn internally_tagged_misses() {
    // Varint 0 would be valid for an externally-tagged variant, but internally-tagged
    // enums can never match because postcard has no wire key names; the tag arm always
    // returns Probe::Miss.
    assert_eq!(parse::<InternallyTagged>(&varint(0)), Ok(None));
}

#[test]
fn adjacently_tagged_misses() {
    assert_eq!(parse::<AdjacentlyTagged>(&varint(0)), Ok(None));
}
