#![no_std]
#![allow(async_fn_in_trait)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod error;
mod varint;
mod impls;
pub mod full;

pub use error::PostcardError;
pub use full::{
    PostcardBytesAccess, PostcardClaim, PostcardDeserializer, PostcardEntry, PostcardEnumAccess,
    PostcardEnumVariantProbe, PostcardMapAccess, PostcardMapKeyProbe, PostcardMapValueProbe,
    PostcardNumberAccess, PostcardSeqAccess, PostcardSeqEntry, PostcardStrAccess,
    PostcardSubDeserializer,
};
