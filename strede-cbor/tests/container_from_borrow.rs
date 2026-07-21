//! Borrow-family `#[strede(from = "FromType")]` / `#[strede(try_from = "FromType")]`
//! fixtures (container level) — entirely replaces field-by-field deserialization.

extern crate std;
mod helpers;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
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
fn container_from_hit() {
    assert_eq!(parse::<Scale>(&[helpers::uint_small(7)]), Some(Scale(70)));
}

#[test]
fn container_from_wrong_type_misses() {
    let enc = helpers::tstr("7");
    assert_eq!(parse::<Scale>(&enc), None);
}

#[test]
fn container_try_from_hit() {
    let enc = helpers::uint16(8080);
    assert_eq!(parse::<Port>(&enc), Some(Port(8080)));
}

#[test]
fn container_try_from_out_of_range_misses() {
    // negint_small(0) -> actual -1, doesn't fit u16
    assert_eq!(parse::<Port>(&[helpers::negint_small(0)]), None);
}
