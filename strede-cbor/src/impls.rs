//! Format-specific primitive `Deserialize` and `DeserializeOwned` impls.
//!
//! CBOR numbers are binary-typed; we dispatch directly on the token value
//! via `ParseNum`.

use crate::{
    CborError,
    chunked::{ChunkedCborClaim, ChunkedCborDeserializer, ChunkedCborSubDeserializer},
    full::{CborClaim, CborDeserializer, CborSubDeserializer, ParseNum},
    token::CborToken,
};
use strede::{Buffer, Deserialize, DeserializeOwned, Deserializer, DeserializerOwned, Probe};

// ---------------------------------------------------------------------------
// ParseNum implementations
// ---------------------------------------------------------------------------

macro_rules! impl_parse_num_uint {
    ($($t:ty),*) => {
        $(impl ParseNum for $t {
            #[inline(always)]
            fn from_uint(v: u64) -> Option<Self> { <$t>::try_from(v).ok() }
            #[inline(always)]
            fn from_negint(n: u64) -> Option<Self> {
                // actual = -1 - n; unsigned types can never hold negatives
                let _ = n;
                None
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
            fn from_uint(v: u64) -> Option<Self> { <$t>::try_from(v).ok() }
            #[inline(always)]
            fn from_negint(n: u64) -> Option<Self> {
                // actual = -1 - n; use checked arithmetic
                let pos = <$t>::try_from(n).ok()?;
                pos.checked_neg()?.checked_sub(1)
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

impl_parse_num_uint!(u8, u16, u32, u64);
impl_parse_num_sint!(i8, i16, i32, i64);

impl ParseNum for f32 {
    #[inline(always)]
    fn from_uint(v: u64) -> Option<Self> {
        Some(v as f32)
    }
    #[inline(always)]
    fn from_negint(n: u64) -> Option<Self> {
        // -1 - n
        Some(-1.0f32 - n as f32)
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Option<Self> {
        Some(v)
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Option<Self> {
        Some(v as f32)
    }
}

impl ParseNum for f64 {
    #[inline(always)]
    fn from_uint(v: u64) -> Option<Self> {
        Some(v as f64)
    }
    #[inline(always)]
    fn from_negint(n: u64) -> Option<Self> {
        Some(-1.0f64 - n as f64)
    }
    #[inline(always)]
    fn from_f32(v: f32) -> Option<Self> {
        Some(v as f64)
    }
    #[inline(always)]
    fn from_f64(v: f64) -> Option<Self> {
        Some(v)
    }
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
            ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
                d.entry(|[e]| async move {
                    match e.token {
                        CborToken::Bool(b) => Ok(Probe::Hit((CborClaim { src: e.src, remaining_after: 0 }, b))),
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
            ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
                d.entry(|[e]| async move {
                    match e.token {
                        CborToken::Null | CborToken::Undefined => {
                            Ok(Probe::Hit((CborClaim { src: e.src, remaining_after: 0 }, ())))
                        }
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
            ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_borrow_both {
    ($($t:tt),+) => {
        $(
            impl_deserialize_borrow_one!(CborDeserializer<'de>; $t);
            impl_deserialize_borrow_one!(CborSubDeserializer<'de>; $t);
        )+
    };
}

impl_deserialize_borrow_both!(bool, ());
impl_deserialize_borrow_one!(CborDeserializer<'de>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_deserialize_borrow_one!(CborSubDeserializer<'de>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

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
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, CborError> {
                d.entry(|[e]| async move {
                    match e.token {
                        CborToken::Bool(b) => Ok(Probe::Hit((
                            ChunkedCborClaim { offset: e.offset, handle: e.handle, remaining_after: None },
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
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, CborError> {
                d.entry(|[e]| async move {
                    match e.token {
                        CborToken::Null | CborToken::Undefined => Ok(Probe::Hit((
                            ChunkedCborClaim { offset: e.offset, handle: e.handle, remaining_after: None },
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
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, CborError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_owned_both {
    ($($t:tt),+) => {
        $(
            impl_deserialize_owned_one!(ChunkedCborDeserializer<'s, B, F>; $t);
            impl_deserialize_owned_one!(ChunkedCborSubDeserializer<'s, B, F>; $t);
        )+
    };
}

impl_deserialize_owned_both!(bool, ());
impl_deserialize_owned_one!(ChunkedCborDeserializer<'s, B, F>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
impl_deserialize_owned_one!(ChunkedCborSubDeserializer<'s, B, F>; u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
