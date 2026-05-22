#![no_std]
#![allow(async_fn_in_trait)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod chunked;
mod error;
mod full;
mod impls;
mod tag;
mod token;

#[cfg(feature = "alloc")]
mod value;

pub use chunked::ChunkedCborDeserializer;
pub use error::CborError;
pub use full::{CborDeserializer, CborSubDeserializer};
pub use tag::{Accepted, Captured, CborTag, Ignored, Required, TagHandler};
pub use token::CborToken;

#[cfg(feature = "alloc")]
pub use value::CborValue;
