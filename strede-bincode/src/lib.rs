//! `strede-bincode` - Bincode format backend for the `strede` deserialization framework.
//!
//! Bincode is schema-driven like postcard (no type tags on the wire), but
//! unlike postcard its wire encoding is itself configurable. This crate
//! supports the real `bincode` crate's wire matrix as a compile-time
//! generic parameter `C: BincodeConfig` (see [`config`]):
//!
//! - **Byte order**: little or big ([`config::ByteOrder`])
//! - **Int encoding**: fixed-width ([`config::Fixint`], matches bincode 1.x
//!   / bincode2's `legacy()`) or bincode's own compact varint
//!   ([`config::Varint`], matches bincode2's `standard()` default)
//!
//! - [`full`] - in-memory borrow-family deserializer ([`full::BincodeDeserializer`])
//! - [`chunked`] - streaming owned-family deserializer
//!
//! # Known limitations
//!
//! Schema-driven like postcard: `skip()` is impossible (field positions are
//! determined by the type, not the wire data), so `allow_unknown_fields`
//! and `#[strede(flatten)]` are unsupported — see [`BincodeError::CannotSkip`].
//!
//! `#[strede(untagged)]` support follows `strede-postcard`'s own precedent
//! of delegating directly to `T::deserialize` and accepting the first
//! declaration-order variant whose shape happens to parse. This carries a
//! known, accepted ambiguity risk: a byte sequence that structurally parses
//! as an earlier untagged variant's shape but was actually written as a
//! later variant will silently produce the wrong value. This is a
//! documented, long-standing limitation of real bincode+serde too — see
//! `servo/bincode#130`, where serde's own author states there is no
//! structural information in a non-self-describing byte stream to
//! distinguish which variant produced a given sequence of bytes.

#![no_std]
#![allow(async_fn_in_trait)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod chunked;
pub mod config;
mod error;
pub mod full;
mod impls;
mod num;
#[cfg(feature = "alloc")]
mod vec;

pub use chunked::access::ChunkedBincodeBytesAccess;
pub use chunked::{ChunkedBincodeClaim, ChunkedBincodeDeserializer};
pub use config::{BigLegacy, BigStandard, BincodeConfig, Fixint, Legacy, Standard, Varint};
pub use error::BincodeError;
pub use full::{
    BincodeBytesAccess, BincodeClaim, BincodeDeserializer, BincodeEntry, BincodeEnumAccess,
    BincodeEnumVariantProbe, BincodeMapAccess, BincodeMapKeyProbe, BincodeMapValueProbe,
    BincodeNumberAccess, BincodeSeqAccess, BincodeSeqEntry, BincodeStrAccess,
    BincodeSubDeserializer,
};
