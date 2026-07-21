//! Owned-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! `count` is `u32` on the wire-facing struct but deserializes as `u16` and
//! converts via `u32::from(u16)`.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
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
                let de = ChunkedJsonDeserializer::new(shared);
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

#[test]
fn from_u16_hit() {
    let item: Item = parse!(Item, br#"{"name": "widget", "count": 42}"#).unwrap();
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
    let v: Option<Item> = parse!(Item, br#"{"name": "widget", "count": "42"}"#);
    assert!(v.is_none());
}

#[test]
fn from_u16_out_of_range_misses() {
    let v: Option<Item> = parse!(Item, br#"{"name": "widget", "count": 70000}"#);
    assert!(v.is_none());
}

#[test]
fn from_u16_missing_field_misses() {
    let v: Option<Item> = parse!(Item, br#"{"name": "widget"}"#);
    assert!(v.is_none());
}
