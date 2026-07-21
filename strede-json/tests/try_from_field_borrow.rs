//! Borrow-family `#[strede(try_from = "FromType")]` fixture (field level).
//!
//! `port` is `u16` on the wire-facing struct but deserializes as `i32` and
//! converts via `u16::try_from(i32)`, missing (not erroring) on failure.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Server {
    name: String,
    #[strede(try_from = "i32")]
    port: u16,
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
fn try_from_i32_hit() {
    let server: Server = parse(r#"{"name": "web", "port": 8080}"#).unwrap();
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
    let v: Option<Server> = parse(r#"{"name": "web", "port": -1}"#);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_wrong_type_misses() {
    let v: Option<Server> = parse(r#"{"name": "web", "port": "8080"}"#);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_missing_field_misses() {
    let v: Option<Server> = parse(r#"{"name": "web"}"#);
    assert!(v.is_none());
}
