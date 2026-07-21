//! `Vec<T>: Deserialize` / `DeserializeOwned` for postcard.
//!
//! Postcard is schema-driven: a bare `u8` element is itself varint-encoded
//! (see `impls.rs`), so "a sequence of `u8` elements" and "raw bytes" are two
//! genuinely different wire encodings that only coincide for element values
//! `< 0x80` - there is no wire tag to tell which one a writer meant. This
//! crate's convention (see `tests/collections_borrow.rs` /
//! `tests/collections_owned.rs`) is that `Vec<u8>` always means raw bytes -
//! never a sequence of varint-encoded elements - so it is never raced
//! against the seq reading the way self-describing formats race their
//! `Vec<u8>` (see `strede::utils::vec_u8_bytes_only`). Every other `Vec<T>`
//! goes through the plain seq path (`strede::utils::vec_via_seq`), same as
//! before.

use crate::PostcardError;
use crate::chunked::{ChunkedPostcardDeserializer, ChunkedPostcardSubDeserializer};
use crate::full::{PostcardClaim, PostcardDeserializer, PostcardSubDeserializer};
use strede::{
    Buffer, Deserialize, DeserializeOwned, DeserializerOwned, Probe, hit, typeid,
    utils::{
        u8_vec_as_t, vec_u8_bytes_only, vec_u8_bytes_only_owned, vec_via_seq, vec_via_seq_owned,
    },
};

macro_rules! impl_deserialize_vec_borrow {
    ($de:ty) => {
        #[cfg(feature = "alloc")]
        impl<'de, T> Deserialize<'de, $de> for alloc::vec::Vec<T>
        where
            T: Deserialize<'de, PostcardSubDeserializer<'de>, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(PostcardClaim<'de>, Self)>, PostcardError> {
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

impl_deserialize_vec_borrow!(PostcardDeserializer<'de>);
impl_deserialize_vec_borrow!(PostcardSubDeserializer<'de>);

macro_rules! impl_deserialize_vec_owned {
    ($de:ty) => {
        #[cfg(feature = "alloc")]
        impl<'s, T, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for alloc::vec::Vec<T>
        where
            T: DeserializeOwned<ChunkedPostcardSubDeserializer<'s, B, F>, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, PostcardError> {
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

impl_deserialize_vec_owned!(ChunkedPostcardDeserializer<'s, B, F>);
impl_deserialize_vec_owned!(ChunkedPostcardSubDeserializer<'s, B, F>);
