//! Owned-family `#[strede(try_from = "FromType")]` fixture (field level).
//!
//! `port` is `u16` on the wire-facing struct but deserializes as `i32` and
//! converts via `u16::try_from(i32)`, missing (not erroring) on failure.

extern crate std;
mod helpers;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Server {
    name: std::string::String,
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
fn try_from_i32_hit() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("web")),
        (&helpers::tstr("port"), &helpers::uint16(8080)),
    ]);
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
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("web")),
        (&helpers::tstr("port"), &[helpers::negint_small(0)]),
    ]);
    let v: Option<Server> = parse!(Server, &msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_wrong_type_misses() {
    let msg = helpers::build_map(&[
        (&helpers::tstr("name"), &helpers::tstr("web")),
        (&helpers::tstr("port"), &helpers::tstr("8080")),
    ]);
    let v: Option<Server> = parse!(Server, &msg);
    assert!(v.is_none());
}

#[test]
fn try_from_i32_missing_field_misses() {
    let msg = helpers::build_map(&[(&helpers::tstr("name"), &helpers::tstr("web"))]);
    let v: Option<Server> = parse!(Server, &msg);
    assert!(v.is_none());
}
