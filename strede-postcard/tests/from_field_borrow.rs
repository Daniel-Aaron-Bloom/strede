//! Borrow-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! `count` is `u32` on the wire-facing struct but deserializes as `u16` and
//! converts via `u32::from(u16)`. Wire bytes are a `u16` varint regardless
//! of the target field type.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Item<'de> {
    name: &'de str,
    #[strede(from = "u16")]
    count: u32,
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
fn from_u16_hit() {
    let mut data = pstr("widget");
    data.extend_from_slice(&varint(42));
    assert_eq!(
        parse::<Item<'_>>(&data),
        Ok(Some(Item {
            name: "widget",
            count: 42,
        }))
    );
}

// "Wrong-type Miss" (e.g. a string on the wire where `u16` is expected) is
// omitted: postcard has no wire tag to detect the mismatch — the string's
// length-prefix byte would just be misdecoded as a varint rather than missing.

#[test]
fn from_u16_out_of_range_misses() {
    // 70000 overflows u16 -> Miss (ParseNum's try_from), not an error.
    let mut data = pstr("widget");
    data.extend_from_slice(&varint(70000));
    assert_eq!(parse::<Item<'_>>(&data), Ok(None));
}

// Postcard has no way to observe a "missing field" gracefully — running out
// of bytes before the positionally-expected `count` field is a hard
// `UnexpectedEnd` error, not a `Probe::Miss` (see structs_borrow.rs
// `named_truncated_errors`). This replaces the JSON reference's
// `from_u16_missing_field_misses` test.
#[test]
fn from_u16_truncated_before_count_field_errors() {
    let data = pstr("widget");
    assert_eq!(parse_err::<Item<'_>>(&data), PostcardError::UnexpectedEnd);
}
