//! Owned-family `#[strede(from = "FromType")]` / `#[strede(try_from = "FromType")]`
//! fixtures (container level) — entirely replaces field-by-field deserialization.

extern crate std;
mod helpers;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(from = "u8")]
struct Scale(u32);

impl From<u8> for Scale {
    fn from(v: u8) -> Self {
        Scale(v as u32 * 10)
    }
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(try_from = "i32")]
struct Port(u16);

impl TryFrom<i32> for Port {
    type Error = ();
    fn try_from(v: i32) -> Result<Self, ()> {
        u16::try_from(v).map(Port).map_err(|_| ())
    }
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
fn container_from_hit() {
    assert_eq!(parse!(Scale, &[helpers::uint_small(7)]), Some(Scale(70)));
}

#[test]
fn container_from_wrong_type_misses() {
    let enc = helpers::tstr("7");
    assert_eq!(parse!(Scale, &enc), None);
}

#[test]
fn container_try_from_hit() {
    let enc = helpers::uint16(8080);
    assert_eq!(parse!(Port, &enc), Some(Port(8080)));
}

#[test]
fn container_try_from_out_of_range_misses() {
    assert_eq!(parse!(Port, &[helpers::negint_small(0)]), None);
}
