//! Owned-family `#[strede(from = "FromType")]` / `#[strede(try_from = "FromType")]`
//! fixtures (container level). Mirrors `container_from_borrow.rs`.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;

#[derive(Debug, PartialEq, DeserializeOwned)]
#[strede(from = "u8")]
struct Scale(u32);

impl From<u8> for Scale {
    fn from(v: u8) -> Self {
        Scale(v as u32 * 10)
    }
}

#[derive(Debug, PartialEq, DeserializeOwned)]
#[strede(try_from = "i32")]
struct Port(u16);

impl TryFrom<i32> for Port {
    type Error = ();
    fn try_from(v: i32) -> Result<Self, ()> {
        u16::try_from(v).map(Port).map_err(|_| ())
    }
}

#[test]
fn container_from_hit() {
    assert_eq!(parse_owned!(Scale, &varint(7)), Ok(Some(Scale(70))));
}

#[test]
fn container_from_wrong_type_misses() {
    // 300 overflows u8 -> Miss; see container_from_borrow.rs for why this
    // stands in for the JSON reference's "wrong wire type" case.
    assert_eq!(parse_owned!(Scale, &varint(300)), Ok(None));
}

#[test]
fn container_try_from_hit() {
    assert_eq!(parse_owned!(Port, &zigzag(8080)), Ok(Some(Port(8080))));
}

#[test]
fn container_try_from_out_of_range_misses() {
    assert_eq!(parse_owned!(Port, &zigzag(-1)), Ok(None));
}

#[test]
fn container_try_from_out_of_range_misses_chunked() {
    assert_eq!(parse_owned_chunked!(Port, &zigzag(-1), 1), Ok(None));
}
