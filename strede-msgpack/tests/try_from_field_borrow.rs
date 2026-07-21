//! Borrow-family `#[strede(try_from = "FromType")]` fixture (field level).
//!
//! `port` is `u16` on the wire-facing struct but deserializes as `i32` and
//! converts via `u16::try_from(i32)`, missing (not erroring) on failure.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Server<'de> {
    name: &'de str,
    #[strede(try_from = "i32")]
    port: u16,
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
    let server: Server<'_> = parse(&msg).unwrap();
    assert_eq!(
        server,
        Server {
            name: "web",
            port: 8080,
        }
    );
}

#[test]
fn try_from_i32_negative_misses() {
    let msg = server_msg("web", &int8(-1));
    let v: Option<Server<'_>> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_wrong_type_misses() {
    let port_val = fixstr("8080");
    let msg = server_msg("web", &port_val);
    let v: Option<Server<'_>> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_missing_field_misses() {
    let name_key = fixstr("name");
    let name_val = fixstr("web");
    let msg = build_map(&[(name_key.as_slice(), name_val.as_slice())]);
    let v: Option<Server<'_>> = parse(&msg);
    assert!(v.is_none());
}
