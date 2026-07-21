//! Borrow-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! `count` is `u32` on the wire-facing struct but deserializes as `u16` and
//! converts via `u32::from(u16)`.

extern crate std;
mod helpers;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Item {
    name: std::string::String,
    #[strede(from = "u16")]
    count: u32,
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
fn from_u16_hit() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("widget")),
        (&helpers::tstr("count"), &helpers::uint8(42)),
    ]);
    let item: Item = parse(&msg).unwrap();
    assert_eq!(
        item,
        Item {
            name: "widget".into(),
            count: 42,
        }
    );
}

#[test]
fn from_u16_wrong_type_misses() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("widget")),
        (&helpers::tstr("count"), &helpers::tstr("42")),
    ]);
    let v: Option<Item> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_out_of_range_misses() {
    // 70000 doesn't fit u16 (max 65535)
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("widget")),
        (&helpers::tstr("count"), &helpers::uint32(70000)),
    ]);
    let v: Option<Item> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_missing_field_misses() {
    let msg = helpers::build_map(&[(&helpers::tstr("name"), &helpers::tstr("widget"))]);
    let v: Option<Item> = parse(&msg);
    assert!(v.is_none());
}
