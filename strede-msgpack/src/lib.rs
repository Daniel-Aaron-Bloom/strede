//! `strede-msgpack` - MessagePack format backend for the `strede` deserialization framework.
//!
//! - [`full`] - in-memory borrow-family deserializer ([`full::MsgpackDeserializer`])
//! - [`chunked`] - streaming owned-family deserializer

#![no_std]
#![allow(async_fn_in_trait)]

pub mod chunked;
mod error;
mod ext;
pub mod full;
mod impls;
mod timestamp;
pub(crate) mod token;

#[cfg(feature = "alloc")]
mod value;
#[cfg(feature = "alloc")]
mod vec;

pub use error::MsgpackError;
pub use timestamp::MsgpackTimestamp;

pub use chunked::access::ChunkedMsgpackBytesAccess;
pub use chunked::{ChunkedMsgpackClaim, ChunkedMsgpackDeserializer};
pub use ext::{
    DeserializeFromExtBytes, DeserializeFromExtBytesOwned, DeserializeFromFixExt, ExtWrapper,
    FixExtWrapper,
};
pub use full::{
    MsgpackBytesAccess, MsgpackClaim, MsgpackDeserializer, MsgpackEntry, MsgpackMapAccess,
    MsgpackMapKeyProbe, MsgpackMapValueProbe, MsgpackSeqAccess, MsgpackSeqEntry, MsgpackStrAccess,
    MsgpackSubDeserializer,
};
#[cfg(feature = "alloc")]
pub use value::MsgpackValue;
