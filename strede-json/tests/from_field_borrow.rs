//! Borrow-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! `count` is `u32` on the wire-facing struct but deserializes as `u16` and
//! converts via `u32::from(u16)`.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Item {
    name: String,
    #[strede(from = "u16")]
    count: u32,
}

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
fn from_u16_hit() {
    let item: Item = parse(r#"{"name": "widget", "count": 42}"#).unwrap();
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
    let v: Option<Item> = parse(r#"{"name": "widget", "count": "42"}"#);
    assert!(v.is_none());
}

#[test]
fn from_u16_out_of_range_misses() {
    let v: Option<Item> = parse(r#"{"name": "widget", "count": 70000}"#);
    assert!(v.is_none());
}

#[test]
fn from_u16_missing_field_misses() {
    let v: Option<Item> = parse(r#"{"name": "widget"}"#);
    assert!(v.is_none());
}
