//! `strede-json` - JSON backend for the `strede` deserialization framework.
//!
//! - [`full`] - in-memory borrow-family deserializer ([`full::JsonDeserializer`])
//! - [`chunked`] - streaming owned-family deserializer

#![no_std]
#![allow(async_fn_in_trait)]

pub mod chunked;
mod error;
pub mod full;
pub(crate) mod token;

pub use error::JsonError;
pub use full::{
    JsonBytesAccess, JsonClaim, JsonDeserializer, JsonEntry, JsonMapAccess, JsonMapKeyClaim,
    JsonMapKeyProbe, JsonMapValueClaim, JsonMapValueProbe, JsonSeqAccess, JsonSeqEntry,
    JsonStrAccess,
};
