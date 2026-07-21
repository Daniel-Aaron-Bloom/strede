//! Owned-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! `count` is `u32` on the wire-facing struct but deserializes as `u16` and
//! converts via `u32::from(u16)`.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Item {
    name: String,
    #[strede(from = "u16")]
    count: u32,
}

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
                match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                {
                    Probe::Hit((_, v)) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
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
    let item: Item = parse!(Item, &msg).unwrap();
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
    let count_val = fixstr("42");
    let msg = item_msg("widget", &count_val);
    let v: Option<Item> = parse!(Item, &msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_out_of_range_misses() {
    let msg = item_msg("widget", &uint32(70000));
    let v: Option<Item> = parse!(Item, &msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_missing_field_misses() {
    let name_key = fixstr("name");
    let name_val = fixstr("widget");
    let msg = build_map(&[(name_key.as_slice(), name_val.as_slice())]);
    let v: Option<Item> = parse!(Item, &msg);
    assert!(v.is_none());
}
