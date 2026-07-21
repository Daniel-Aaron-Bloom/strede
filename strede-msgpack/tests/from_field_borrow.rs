//! Borrow-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! `count` is `u32` on the wire-facing struct but deserializes as `u16` and
//! converts via `u32::from(u16)`.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Item<'de> {
    name: &'de str,
    #[strede(from = "u16")]
    count: u32,
}

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

fn item_msg(name: &str, count: &[u8]) -> Vec<u8> {
    let name_key = fixstr("name");
    let count_key = fixstr("count");
    let name_val = fixstr(name);
    build_map(&[
        (name_key.as_slice(), name_val.as_slice()),
        (count_key.as_slice(), count),
    ])
}

#[test]
fn from_u16_hit() {
    let msg = item_msg("widget", &uint16(42));
    let item: Item<'_> = parse(&msg).unwrap();
    assert_eq!(
        item,
        Item {
            name: "widget",
            count: 42,
        }
    );
}

#[test]
fn from_u16_wrong_type_misses() {
    let count_val = fixstr("42");
    let msg = item_msg("widget", &count_val);
    let v: Option<Item<'_>> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_out_of_range_misses() {
    // 70000 doesn't fit in u16 (encoded as uint32)
    let msg = item_msg("widget", &uint32(70000));
    let v: Option<Item<'_>> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_missing_field_misses() {
    let name_key = fixstr("name");
    let name_val = fixstr("widget");
    let msg = build_map(&[(name_key.as_slice(), name_val.as_slice())]);
    let v: Option<Item<'_>> = parse(&msg);
    assert!(v.is_none());
}
