//! Owned-family `#[strede(transparent)]` fixture.
//!
//! Mirrors `transparent_borrow.rs`; see that file for context.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

#[derive(Debug, PartialEq, DeserializeOwned)]
#[strede(transparent)]
struct TransparentWrapper(u32);

#[test]
fn transparent_u32() {
    let data = varint(7);
    assert_eq!(
        parse_owned!(TransparentWrapper, &data),
        Ok(Some(TransparentWrapper(7)))
    );
}

// "Wrong-type Miss" is omitted: postcard has no wire tag to detect a
// mismatch (see from_field_owned.rs for the same reasoning). A truncated
// (empty) input is the closest observable failure.
#[test]
fn transparent_truncated_errors() {
    assert_eq!(
        parse_owned!(TransparentWrapper, &[]).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}
