//! Owned-family adjacently-tagged enum fixtures.
#![recursion_limit = "512"]

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

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

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "t", content = "c")]
enum Signal {
    Ping,
    Pong,
    Reset,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Point {
    x: u32,
    y: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "t", content = "c")]
enum Msg {
    Move { x: u32, y: u32 },
    New(Point),
    Reset,
}

fn point_map(x: u8, y: u8) -> Vec<u8> {
    let x_key = fixstr("x");
    let y_key = fixstr("y");
    build_map(&[(x_key.as_slice(), &[x]), (y_key.as_slice(), &[y])])
}

#[test]
fn unit_ping() {
    let t_key = fixstr("t");
    let t_val = fixstr("Ping");
    let msg = build_map(&[(t_key.as_slice(), t_val.as_slice())]);
    assert_eq!(parse!(Signal, &msg), Some(Signal::Ping));
}

#[test]
fn unit_pong() {
    let t_key = fixstr("t");
    let t_val = fixstr("Pong");
    let msg = build_map(&[(t_key.as_slice(), t_val.as_slice())]);
    assert_eq!(parse!(Signal, &msg), Some(Signal::Pong));
}

#[test]
fn unit_unknown_misses() {
    let t_key = fixstr("t");
    let t_val = fixstr("Other");
    let msg = build_map(&[(t_key.as_slice(), t_val.as_slice())]);
    assert_eq!(parse!(Signal, &msg), None);
}

#[test]
fn struct_variant_tag_first() {
    let t_key = fixstr("t");
    let t_val = fixstr("Move");
    let c_key = fixstr("c");
    let c_val = point_map(1, 2);
    let msg = build_map(&[
        (t_key.as_slice(), t_val.as_slice()),
        (c_key.as_slice(), c_val.as_slice()),
    ]);
    assert_eq!(parse!(Msg, &msg), Some(Msg::Move { x: 1, y: 2 }));
}

#[test]
fn struct_variant_content_first() {
    let t_key = fixstr("t");
    let t_val = fixstr("Move");
    let c_key = fixstr("c");
    let c_val = point_map(3, 4);
    let msg = build_map(&[
        (c_key.as_slice(), c_val.as_slice()),
        (t_key.as_slice(), t_val.as_slice()),
    ]);
    assert_eq!(parse!(Msg, &msg), Some(Msg::Move { x: 3, y: 4 }));
}

#[test]
fn newtype_variant() {
    let t_key = fixstr("t");
    let t_val = fixstr("New");
    let c_key = fixstr("c");
    let c_val = point_map(5, 6);
    let msg = build_map(&[
        (t_key.as_slice(), t_val.as_slice()),
        (c_key.as_slice(), c_val.as_slice()),
    ]);
    assert_eq!(parse!(Msg, &msg), Some(Msg::New(Point { x: 5, y: 6 })));
}

#[test]
fn unit_variant_in_mixed_enum() {
    let t_key = fixstr("t");
    let t_val = fixstr("Reset");
    let msg = build_map(&[(t_key.as_slice(), t_val.as_slice())]);
    assert_eq!(parse!(Msg, &msg), Some(Msg::Reset));
}

#[test]
fn unknown_variant_misses() {
    let t_key = fixstr("t");
    let t_val = fixstr("Nope");
    let c_key = fixstr("c");
    let c_val = point_map(1, 2);
    let msg = build_map(&[
        (t_key.as_slice(), t_val.as_slice()),
        (c_key.as_slice(), c_val.as_slice()),
    ]);
    assert_eq!(parse!(Msg, &msg), None);
}
