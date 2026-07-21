//! `strede-postcard` - Postcard format backend for the `strede` deserialization framework.
//!
//! - [`full`] - in-memory borrow-family deserializer ([`full::PostcardDeserializer`])
//! - [`chunked`] - streaming owned-family deserializer

#![no_std]
#![allow(async_fn_in_trait)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod chunked;
mod error;
pub mod full;
mod impls;
mod varint;
#[cfg(feature = "alloc")]
mod vec;

pub use error::PostcardError;
pub use chunked::access::ChunkedPostcardBytesAccess;
pub use chunked::{ChunkedPostcardClaim, ChunkedPostcardDeserializer};
pub use full::{
    PostcardBytesAccess, PostcardClaim, PostcardDeserializer, PostcardEntry, PostcardEnumAccess,
    PostcardEnumVariantProbe, PostcardMapAccess, PostcardMapKeyProbe, PostcardMapValueProbe,
    PostcardNumberAccess, PostcardSeqAccess, PostcardSeqEntry, PostcardStrAccess,
    PostcardSubDeserializer,
};
