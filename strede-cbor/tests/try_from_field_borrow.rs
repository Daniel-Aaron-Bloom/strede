//! Borrow-family `#[strede(try_from = "FromType")]` fixture (field level).
//!
//! `port` is `u16` on the wire-facing struct but deserializes as `i32` and
//! converts via `u16::try_from(i32)`, missing (not erroring) on failure.

extern crate std;
mod helpers;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Server {
    name: std::string::String,
    #[strede(try_from = "i32")]
    port: u16,
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
fn try_from_i32_hit() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("web")),
        (&helpers::tstr("port"), &helpers::uint16(8080)),
    ]);
    let server: Server = parse(&msg).unwrap();
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
    // negint_small(0) -> actual -1, doesn't fit u16
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("web")),
        (&helpers::tstr("port"), &[helpers::negint_small(0)]),
    ]);
    let v: Option<Server> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_wrong_type_misses() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("web")),
        (&helpers::tstr("port"), &helpers::tstr("8080")),
    ]);
    let v: Option<Server> = parse(&msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_missing_field_misses() {
    let msg = helpers::build_map(&[(&helpers::tstr("name"), &helpers::tstr("web"))]);
    let v: Option<Server> = parse(&msg);
    assert!(v.is_none());
}
