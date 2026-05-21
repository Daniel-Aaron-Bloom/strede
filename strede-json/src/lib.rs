//! `strede-json` - JSON backend for the `strede` deserialization framework.
//!
//! - [`full`] - in-memory borrow-family deserializer ([`full::JsonDeserializer`])
//! - [`chunked`] - streaming owned-family deserializer

#![no_std]
#![allow(async_fn_in_trait)]

pub mod chunked;
mod error;
pub mod full;
pub mod number;
pub(crate) mod token;
#[cfg(feature = "alloc")]
pub mod value;

pub use error::JsonError;
pub use full::{
    JsonBytesAccess, JsonClaim, JsonDeserializer, JsonEntry, JsonMapAccess, JsonMapKeyProbe,
    JsonMapValueProbe, JsonSeqAccess, JsonSeqEntry, JsonStrAccess, JsonSubDeserializer,
};
pub use number::{NumberBorrowed, NumberOwned};
#[cfg(feature = "alloc")]
pub use value::{RawValueBorrowed, RawValueOwned, ValueBorrowed, ValueOwned};
