//! Owned-family `#[strede(try_from = "FromType")]` fixture (field level).
//!
//! `port` is `u16` on the wire-facing struct but deserializes as `i32` and
//! converts via `u16::try_from(i32)`, missing (not erroring) on failure.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Server {
    name: String,
    #[strede(try_from = "i32")]
    port: u16,
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

fn server_msg(name: &str, port: &[u8]) -> Vec<u8> {
    let name_key = fixstr("name");
    let port_key = fixstr("port");
    let name_val = fixstr(name);
    build_map(&[
        (name_key.as_slice(), name_val.as_slice()),
        (port_key.as_slice(), port),
    ])
}

#[test]
fn try_from_i32_hit() {
    let msg = server_msg("web", &uint16(8080));
    let server: Server = parse!(Server, &msg).unwrap();
    assert_eq!(
        server,
        Server {
            name: "web".into(),
            port: 8080,
        }
    );
}

#[test]
fn try_from_i32_negative_misses() {
    let msg = server_msg("web", &int8(-1));
    let v: Option<Server> = parse!(Server, &msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_wrong_type_misses() {
    let port_val = fixstr("8080");
    let msg = server_msg("web", &port_val);
    let v: Option<Server> = parse!(Server, &msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_missing_field_misses() {
    let name_key = fixstr("name");
    let name_val = fixstr("web");
    let msg = build_map(&[(name_key.as_slice(), name_val.as_slice())]);
    let v: Option<Server> = parse!(Server, &msg);
    assert!(v.is_none());
}
