//! Format-specific primitive `Deserialize` and `DeserializeOwned` impls.
//!
//! Msgpack numbers are binary-typed; we dispatch directly on the token value
//! rather than going through a text representation.  The `ParseNum` trait is
//! the integration point for both the borrow and owned families.

use crate::{
    MsgpackError,
    chunked::{ChunkedMsgpackClaim, ChunkedMsgpackDeserializer, ChunkedMsgpackSubDeserializer},
    full::{MsgpackClaim, MsgpackDeserializer, MsgpackSubDeserializer, ParseNum},
    token::MsgpackToken,
};
use strede::{Buffer, Deserialize, DeserializeOwned, Deserializer, DeserializerOwned, Probe};

// ---------------------------------------------------------------------------
// ParseNum implementations
// ---------------------------------------------------------------------------

macro_rules! impl_parse_num_uint {
    ($($t:ty),*) => {
        $(impl ParseNum for $t {
            #[inline(always)]
            fn from_ufixint(v: [u8; 1]) -> Option<Self> { <$t>::try_from(v[0]).ok() }
            #[inline(always)]
            fn from_uint8(v: [u8; 1]) -> Option<Self> { <$t>::try_from(v[0]).ok() }
            #[inline(always)]
            fn from_uint16(v: [u8; 2]) -> Option<Self> { <$t>::try_from(u16::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_uint32(v: [u8; 4]) -> Option<Self> { <$t>::try_from(u32::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_uint64(v: [u8; 8]) -> Option<Self> { <$t>::try_from(u64::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_ifixint(v: [u8; 1]) -> Option<Self> {
                let n = v[0] as i8 as i64;
                if n >= 0 { <$t>::try_from(n as u64).ok() } else { None }
            }
            #[inline(always)]
            fn from_int8(v: [u8; 1]) -> Option<Self> {
                let n = v[0] as i8 as i64;
                if n >= 0 { <$t>::try_from(n as u64).ok() } else { None }
            }
            #[inline(always)]
            fn from_int16(v: [u8; 2]) -> Option<Self> {
                let n = i16::from_be_bytes(v) as i64;
                if n >= 0 { <$t>::try_from(n as u64).ok() } else { None }
            }
            #[inline(always)]
            fn from_int32(v: [u8; 4]) -> Option<Self> {
                let n = i32::from_be_bytes(v) as i64;
                if n >= 0 { <$t>::try_from(n as u64).ok() } else { None }
            }
            #[inline(always)]
            fn from_int64(v: [u8; 8]) -> Option<Self> {
                let n = i64::from_be_bytes(v);
                if n >= 0 { <$t>::try_from(n as u64).ok() } else { None }
            }
            #[inline(always)]
            fn from_f32(v: f32) -> Option<Self> {
                let t = v as $t;
                if t as f32 == v { Some(t) } else { None }
            }
            #[inline(always)]
            fn from_f64(v: f64) -> Option<Self> {
                let t = v as $t;
                if t as f64 == v { Some(t) } else { None }
            }
        })*
    };
}

macro_rules! impl_parse_num_sint {
    ($($t:ty),*) => {
        $(impl ParseNum for $t {
            #[inline(always)]
            fn from_ufixint(v: [u8; 1]) -> Option<Self> { <$t>::try_from(v[0]).ok() }
            #[inline(always)]
            fn from_uint8(v: [u8; 1]) -> Option<Self> { <$t>::try_from(v[0]).ok() }
            #[inline(always)]
            fn from_uint16(v: [u8; 2]) -> Option<Self> { <$t>::try_from(u16::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_uint32(v: [u8; 4]) -> Option<Self> { <$t>::try_from(u32::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_uint64(v: [u8; 8]) -> Option<Self> { <$t>::try_from(u64::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_ifixint(v: [u8; 1]) -> Option<Self> { <$t>::try_from(v[0] as i8).ok() }
            #[inline(always)]
            fn from_int8(v: [u8; 1]) -> Option<Self> { <$t>::try_from(v[0] as i8).ok() }
            #[inline(always)]
            fn from_int16(v: [u8; 2]) -> Option<Self> { <$t>::try_from(i16::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_int32(v: [u8; 4]) -> Option<Self> { <$t>::try_from(i32::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_int64(v: [u8; 8]) -> Option<Self> { <$t>::try_from(i64::from_be_bytes(v)).ok() }
            #[inline(always)]
            fn from_f32(v: f32) -> Option<Self> {
                let t = v as $t;
                if t as f32 == v { Some(t) } else { None }
            }
            #[inline(always)]
            fn from_f64(v: f64) -> Option<Self> {
                let t = v as $t;
                if t as f64 == v { Some(t) } else { None }
            }
        })*
    };
}

impl_parse_num_uint!(u8, u16, u32, u64);
impl_parse_num_sint!(i8, i16, i32, i64);

impl ParseNum for f32 {
    #[inline(always)]
    fn from_ufixint(v: [u8; 1]) -> Option<Self> { Some(v[0] as f32) }
    #[inline(always)]
    fn from_uint8(v: [u8; 1]) -> Option<Self> { Some(v[0] as f32) }
    #[inline(always)]
    fn from_uint16(v: [u8; 2]) -> Option<Self> { Some(u16::from_be_bytes(v) as f32) }
    #[inline(always)]
    fn from_uint32(v: [u8; 4]) -> Option<Self> { Some(u32::from_be_bytes(v) as f32) }
    #[inline(always)]
    fn from_uint64(v: [u8; 8]) -> Option<Self> { Some(u64::from_be_bytes(v) as f32) }
    #[inline(always)]
    fn from_ifixint(v: [u8; 1]) -> Option<Self> { Some(v[0] as i8 as f32) }
    #[inline(always)]
    fn from_int8(v: [u8; 1]) -> Option<Self> { Some(v[0] as i8 as f32) }
    #[inline(always)]
    fn from_int16(v: [u8; 2]) -> Option<Self> { Some(i16::from_be_bytes(v) as f32) }
    #[inline(always)]
    fn from_int32(v: [u8; 4]) -> Option<Self> { Some(i32::from_be_bytes(v) as f32) }
    #[inline(always)]
    fn from_int64(v: [u8; 8]) -> Option<Self> { Some(i64::from_be_bytes(v) as f32) }
    #[inline(always)]
    fn from_f32(v: f32) -> Option<Self> { Some(v) }
    #[inline(always)]
    fn from_f64(v: f64) -> Option<Self> { Some(v as f32) }
}

impl ParseNum for f64 {
    #[inline(always)]
    fn from_ufixint(v: [u8; 1]) -> Option<Self> { Some(v[0] as f64) }
    #[inline(always)]
    fn from_uint8(v: [u8; 1]) -> Option<Self> { Some(v[0] as f64) }
    #[inline(always)]
    fn from_uint16(v: [u8; 2]) -> Option<Self> { Some(u16::from_be_bytes(v) as f64) }
    #[inline(always)]
    fn from_uint32(v: [u8; 4]) -> Option<Self> { Some(u32::from_be_bytes(v) as f64) }
    #[inline(always)]
    fn from_uint64(v: [u8; 8]) -> Option<Self> { Some(u64::from_be_bytes(v) as f64) }
    #[inline(always)]
    fn from_ifixint(v: [u8; 1]) -> Option<Self> { Some(v[0] as i8 as f64) }
    #[inline(always)]
    fn from_int8(v: [u8; 1]) -> Option<Self> { Some(v[0] as i8 as f64) }
    #[inline(always)]
    fn from_int16(v: [u8; 2]) -> Option<Self> { Some(i16::from_be_bytes(v) as f64) }
    #[inline(always)]
    fn from_int32(v: [u8; 4]) -> Option<Self> { Some(i32::from_be_bytes(v) as f64) }
    #[inline(always)]
    fn from_int64(v: [u8; 8]) -> Option<Self> { Some(i64::from_be_bytes(v) as f64) }
    #[inline(always)]
    fn from_f32(v: f32) -> Option<Self> { Some(v as f64) }
    #[inline(always)]
    fn from_f64(v: f64) -> Option<Self> { Some(v) }
}

// ---------------------------------------------------------------------------
// Borrow-family Deserialize impls
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_borrow_one {
    ($de:ty; bool) => {
        impl<'de> Deserialize<'de, $de> for bool {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::Bool(b) => Ok(Probe::Hit((MsgpackClaim { src: e.src, remaining_after: 0 }, b))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
    ($de:ty; ()) => {
        impl<'de> Deserialize<'de, $de> for () {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::Nil => Ok(Probe::Hit((MsgpackClaim { src: e.src, remaining_after: 0 }, ()))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
    ($de:ty; $($t:ty),+) => {
        $(impl<'de> Deserialize<'de, $de> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_borrow_both {
    ($($t:tt),+) => {
        $(
            impl_deserialize_borrow_one!(MsgpackDeserializer<'de>; $t);
            impl_deserialize_borrow_one!(MsgpackSubDeserializer<'de>; $t);
        )+
    };
}

impl_deserialize_borrow_both!(bool, ());
impl_deserialize_borrow_one!(MsgpackDeserializer<'de>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_deserialize_borrow_one!(MsgpackSubDeserializer<'de>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

// ---------------------------------------------------------------------------
// Owned-family DeserializeOwned impls
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_owned_one {
    ($de:ty; bool) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for bool {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::Bool(b) => Ok(Probe::Hit((
                            ChunkedMsgpackClaim { offset: e.offset, handle: e.handle, remaining_after: 0 },
                            b,
                        ))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
    ($de:ty; ()) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for () {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::Nil => Ok(Probe::Hit((
                            ChunkedMsgpackClaim { offset: e.offset, handle: e.handle, remaining_after: 0 },
                            (),
                        ))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
    ($de:ty; $($t:ty),+) => {
        $(impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_owned_both {
    ($($t:tt),+) => {
        $(
            impl_deserialize_owned_one!(ChunkedMsgpackDeserializer<'s, B, F>; $t);
            impl_deserialize_owned_one!(ChunkedMsgpackSubDeserializer<'s, B, F>; $t);
        )+
    };
}

impl_deserialize_owned_both!(bool, ());
impl_deserialize_owned_one!(ChunkedMsgpackDeserializer<'s, B, F>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_deserialize_owned_one!(ChunkedMsgpackSubDeserializer<'s, B, F>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
