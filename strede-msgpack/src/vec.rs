//! `Vec<T>: Deserialize` / `DeserializeOwned` for MessagePack.
//!
//! MessagePack is self-describing: `bin8`/`bin16`/`bin32` byte strings and
//! `array16`/`array32`/fixarray tokens are wire-distinguishable, so
//! `Vec<u8>` can safely race "raw bytes" against "a sequence of `u8`
//! elements" via `strede::utils::vec_u8_race` - whichever token is actually
//! on the wire is the only one that can hit. Every other `Vec<T>` goes
//! through the plain seq path (`strede::utils::vec_via_seq`).

extern crate alloc;

use crate::MsgpackError;
use crate::chunked::{ChunkedMsgpackDeserializer, ChunkedMsgpackSubDeserializer};
use crate::full::{MsgpackClaim, MsgpackDeserializer, MsgpackSubDeserializer};
use strede::{
    Buffer, Deserialize, DeserializeOwned, DeserializerOwned, Probe, hit, typeid,
    utils::{u8_vec_as_t, vec_u8_race, vec_u8_race_owned, vec_via_seq, vec_via_seq_owned},
};

macro_rules! impl_deserialize_vec_borrow {
    ($de:ty) => {
        #[cfg(feature = "alloc")]
        impl<'de, T> Deserialize<'de, $de> for alloc::vec::Vec<T>
        where
            T: Deserialize<'de, MsgpackSubDeserializer<'de>, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                if typeid::of::<T>() == typeid::of::<u8>() {
                    let (claim, v) = hit!(vec_u8_race(d).await);
                    // Safety: T == u8 confirmed by the TypeId check above.
                    let v = unsafe { u8_vec_as_t(v) };
                    return Ok(Probe::Hit((claim, v)));
                }
                vec_via_seq(d).await
            }
        }
    };
}

impl_deserialize_vec_borrow!(MsgpackDeserializer<'de>);
impl_deserialize_vec_borrow!(MsgpackSubDeserializer<'de>);

macro_rules! impl_deserialize_vec_owned {
    ($de:ty) => {
        #[cfg(feature = "alloc")]
        impl<'s, T, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for alloc::vec::Vec<T>
        where
            T: DeserializeOwned<ChunkedMsgpackSubDeserializer<'s, B, F>, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                if typeid::of::<T>() == typeid::of::<u8>() {
                    let (claim, v) = hit!(vec_u8_race_owned(d).await);
                    // Safety: T == u8 confirmed by the TypeId check above.
                    let v = unsafe { u8_vec_as_t(v) };
                    return Ok(Probe::Hit((claim, v)));
                }
                vec_via_seq_owned(d).await
            }
        }
    };
}

impl_deserialize_vec_owned!(ChunkedMsgpackDeserializer<'s, B, F>);
impl_deserialize_vec_owned!(ChunkedMsgpackSubDeserializer<'s, B, F>);
