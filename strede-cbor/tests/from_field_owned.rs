//! Owned-family `#[strede(from = "FromType")]` fixture (field level).
//!
//! `count` is `u32` on the wire-facing struct but deserializes as `u16` and
//! converts via `u32::from(u16)`.

extern crate std;
mod helpers;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Item {
    name: std::string::String,
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
                let de = ChunkedCborDeserializer::new(shared);
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
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("widget")),
        (&helpers::tstr("count"), &helpers::uint8(42)),
    ]);
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
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("widget")),
        (&helpers::tstr("count"), &helpers::tstr("42")),
    ]);
    let v: Option<Item> = parse!(Item, &msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_out_of_range_misses() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("widget")),
        (&helpers::tstr("count"), &helpers::uint32(70000)),
    ]);
    let v: Option<Item> = parse!(Item, &msg);
    assert!(v.is_none());
}

#[test]
fn from_u16_missing_field_misses() {
    let msg = helpers::build_map(&[(&helpers::tstr("name"), &helpers::tstr("widget"))]);
    let v: Option<Item> = parse!(Item, &msg);
    assert!(v.is_none());
}
