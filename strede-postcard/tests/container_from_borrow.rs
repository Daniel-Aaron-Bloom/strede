//! Borrow-family `#[strede(from = "FromType")]` / `#[strede(try_from = "FromType")]`
//! fixtures (container level) — entirely replaces field-by-field
//! deserialization. `Scale`/`Port` decode a single primitive off the wire
//! (u8 varint / i32 zigzag varint respectively) and convert via `From`/`TryFrom`.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(from = "u8")]
struct Scale(u32);

impl From<u8> for Scale {
    fn from(v: u8) -> Self {
        Scale(v as u32 * 10)
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(try_from = "i32")]
struct Port(u16);

impl TryFrom<i32> for Port {
    type Error = ();
    fn try_from(v: i32) -> Result<Self, ()> {
        u16::try_from(v).map(Port).map_err(|_| ())
    }
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

#[test]
fn container_from_hit() {
    assert_eq!(parse::<Scale>(&varint(7)), Ok(Some(Scale(70))));
}

#[test]
fn container_from_wrong_type_misses() {
    // 300 overflows u8 -> Miss via ParseNum's try_from, standing in for the
    // JSON reference's "wrong wire type" case: postcard cannot distinguish
    // "wrong type" from "in-range but out-of-width value" since there is no
    // wire tag, so an out-of-range value is the only observable mismatch.
    assert_eq!(parse::<Scale>(&varint(300)), Ok(None));
}

#[test]
fn container_try_from_hit() {
    assert_eq!(parse::<Port>(&zigzag(8080)), Ok(Some(Port(8080))));
}

#[test]
fn container_try_from_out_of_range_misses() {
    assert_eq!(parse::<Port>(&zigzag(-1)), Ok(None));
}
