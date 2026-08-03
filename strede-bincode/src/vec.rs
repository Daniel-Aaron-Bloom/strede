//! `Vec<T>: Deserialize` / `DeserializeOwned` for bincode.
//!
//! Bincode is schema-driven like postcard: a bare `u8` element is itself
//! encoded (as a raw byte or varint depending on config — see `impls.rs`),
//! so "a sequence of `u8` elements" and "raw bytes" are two genuinely
//! different wire encodings that only coincide for small element values.
//! Mirrors `strede-postcard`'s identical convention: `Vec<u8>` always means
//! raw length-prefixed bytes, never a sequence of individually-encoded
//! elements, so it is never raced against the seq reading the way
//! self-describing formats race their `Vec<u8>`.

use crate::BincodeError;
use crate::chunked::{ChunkedBincodeDeserializer, ChunkedBincodeSubDeserializer};
use crate::config::BincodeConfig;
use crate::full::{BincodeClaim, BincodeDeserializer, BincodeSubDeserializer};
use strede::{
    Buffer, Deserialize, DeserializeOwned, DeserializerOwned, Probe, hit, typeid,
    utils::{
        u8_vec_as_t, vec_u8_bytes_only, vec_u8_bytes_only_owned, vec_via_seq, vec_via_seq_owned,
    },
};

macro_rules! impl_deserialize_vec_borrow {
    ($de:ty) => {
        #[cfg(feature = "alloc")]
        impl<'de, C: BincodeConfig, T> Deserialize<'de, $de> for alloc::vec::Vec<T>
        where
            T: Deserialize<'de, BincodeSubDeserializer<'de, C>, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(BincodeClaim<'de, C>, Self)>, BincodeError> {
                if typeid::of::<T>() == typeid::of::<u8>() {
                    let (claim, v) = hit!(vec_u8_bytes_only(d).await);
                    // Safety: T == u8 confirmed by the TypeId check above.
                    let v = unsafe { u8_vec_as_t(v) };
                    return Ok(Probe::Hit((claim, v)));
                }
                vec_via_seq(d).await
            }
        }
    };
}

impl_deserialize_vec_borrow!(BincodeDeserializer<'de, C>);
impl_deserialize_vec_borrow!(BincodeSubDeserializer<'de, C>);

macro_rules! impl_deserialize_vec_owned {
    ($de:ty) => {
        #[cfg(feature = "alloc")]
        impl<'s, C: BincodeConfig, T, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de>
            for alloc::vec::Vec<T>
        where
            T: DeserializeOwned<ChunkedBincodeSubDeserializer<'s, C, B, F>, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, BincodeError> {
                if typeid::of::<T>() == typeid::of::<u8>() {
                    let (claim, v) = hit!(vec_u8_bytes_only_owned(d).await);
                    // Safety: T == u8 confirmed by the TypeId check above.
                    let v = unsafe { u8_vec_as_t(v) };
                    return Ok(Probe::Hit((claim, v)));
                }
                vec_via_seq_owned(d).await
            }
        }
    };
}

impl_deserialize_vec_owned!(ChunkedBincodeDeserializer<'s, C, B, F>);
impl_deserialize_vec_owned!(ChunkedBincodeSubDeserializer<'s, C, B, F>);
