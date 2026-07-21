//! Internally-tagged and adjacently-tagged enums via the owned/chunked
//! family (not supported by postcard). Mirrors `tag_miss_borrow.rs`.
//!
//! Postcard only supports externally-tagged enums (varint discriminant + payload).
//! Named tag fields cannot be represented on the wire. Both representations return
//! Probe::Miss regardless of input, matching serde-postcard's "WontImplement" stance
//! (postcard issue #125).

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;

#[derive(Debug, PartialEq, DeserializeOwned)]
#[strede(tag = "type")]
enum InternallyTagged {
    Foo { x: u32 },
    Bar,
}

#[derive(Debug, PartialEq, DeserializeOwned)]
#[strede(tag = "t", content = "c")]
enum AdjacentlyTagged {
    Foo { x: u32 },
    Bar,
}

#[test]
fn internally_tagged_misses() {
    // Varint 0 would be valid for an externally-tagged variant, but internally-tagged
    // enums can never match because postcard has no wire key names; the tag arm always
    // returns Probe::Miss.
    assert_eq!(parse_owned!(InternallyTagged, &varint(0)), Ok(None));
}

#[test]
fn adjacently_tagged_misses() {
    assert_eq!(parse_owned!(AdjacentlyTagged, &varint(0)), Ok(None));
}
