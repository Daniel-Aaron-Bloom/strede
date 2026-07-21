//! Ext type support traits and wrapper types for strede-msgpack.

use crate::{
    MsgpackError,
    chunked::{
        ChunkedMsgpackDeserializer, ChunkedMsgpackSubDeserializer,
        access::ChunkedMsgpackBytesAccess,
    },
    full::{MsgpackBytesAccess, MsgpackClaim, MsgpackDeserializer, MsgpackSubDeserializer},
    token::MsgpackToken,
};
use strede::{
    Buffer, BytesAccess, BytesAccessOwned, Deserialize, DeserializeOwned, Deserializer,
    DeserializerOwned, Probe,
};

/// Deserialize a fixext value (type_id + fixed-size payload) synchronously.
///
/// Fixext payloads are 1, 2, 4, 8, or 16 bytes. The data slice length always
/// matches one of those sizes. Unified across borrow and owned families.
pub trait DeserializeFromFixExt: Sized {
    type Extra;

    fn deserialize_from_fixext(
        type_id: i8,
        data: &[u8],
        extra: Self::Extra,
    ) -> Result<Probe<Self>, MsgpackError>;
}

/// Deserialize a variable-length ext value (borrow family).
///
/// Called when an ext8/16/32 token is encountered. The payload bytes are
/// accessible via `bytes`; `len` is their total length.
pub trait DeserializeFromExtBytes<B: BytesAccess>: Sized {
    type Extra;

    async fn deserialize_from_ext_bytes(
        type_id: i8,
        len: usize,
        bytes: B,
        extra: Self::Extra,
    ) -> Result<Probe<(B::Claim, Self)>, B::Error>;
}

/// Deserialize a variable-length ext value (owned family).
///
/// Called when an ext8/16/32 token is encountered. The payload bytes are
/// accessible via `bytes`; `len` is their total length.
pub trait DeserializeFromExtBytesOwned<B: BytesAccessOwned>: Sized {
    type Extra;

    async fn deserialize_from_ext_bytes_owned(
        type_id: i8,
        len: usize,
        bytes: B,
        extra: Self::Extra,
    ) -> Result<Probe<(B::Claim, Self)>, B::Error>;
}

/// Wrapper that deserializes `T` from fixext tokens via [`DeserializeFromFixExt`].
pub struct FixExtWrapper<T>(pub T);

/// Wrapper that deserializes `T` from variable-length ext tokens via
/// [`DeserializeFromExtBytes`] or [`DeserializeFromExtBytesOwned`].
pub struct ExtWrapper<T>(pub T);

// ---------------------------------------------------------------------------
// FixExtWrapper — borrow family
// ---------------------------------------------------------------------------

macro_rules! impl_fixext_borrow {
    ($de:ty) => {
        impl<'de, T> Deserialize<'de, $de> for FixExtWrapper<T>
        where
            T: DeserializeFromFixExt,
        {
            type Extra = T::Extra;

            #[inline(always)]
            async fn deserialize(
                d: $de,
                extra: T::Extra,
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                let mut extra = Some(extra);
                d.entry(move |[e]| {
                    let extra = extra.take().unwrap();
                    async move {
                        match e.token {
                            MsgpackToken::FixExt { type_id, data, len } => {
                                match T::deserialize_from_fixext(
                                    type_id,
                                    &data[..len as usize],
                                    extra,
                                )? {
                                    Probe::Hit(val) => Ok(Probe::Hit((
                                        MsgpackClaim {
                                            src: e.src,
                                            remaining_after: 0,
                                        },
                                        FixExtWrapper(val),
                                    ))),
                                    Probe::Miss => Ok(Probe::Miss),
                                }
                            }
                            _ => Ok(Probe::Miss),
                        }
                    }
                })
                .await
            }
        }
    };
}

impl_fixext_borrow!(MsgpackDeserializer<'de>);
impl_fixext_borrow!(MsgpackSubDeserializer<'de>);

// ---------------------------------------------------------------------------
// FixExtWrapper — owned family
// ---------------------------------------------------------------------------

macro_rules! impl_fixext_owned {
    ($de:ty) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B), T> DeserializeOwned<$de> for FixExtWrapper<T>
        where
            T: DeserializeFromFixExt,
        {
            type Extra = T::Extra;

            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                extra: T::Extra,
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                let mut extra = Some(extra);
                d.entry(move |[e]| {
                    let extra = extra.take().unwrap();
                    async move {
                        match e.token {
                            MsgpackToken::FixExt { type_id, data, len } => {
                                match T::deserialize_from_fixext(
                                    type_id,
                                    &data[..len as usize],
                                    extra,
                                )? {
                                    Probe::Hit(val) => {
                                        Ok(Probe::Hit((e.into_claim(), FixExtWrapper(val))))
                                    }
                                    Probe::Miss => Ok(Probe::Miss),
                                }
                            }
                            _ => Ok(Probe::Miss),
                        }
                    }
                })
                .await
            }
        }
    };
}

impl_fixext_owned!(ChunkedMsgpackDeserializer<'s, B, F>);
impl_fixext_owned!(ChunkedMsgpackSubDeserializer<'s, B, F>);

// ---------------------------------------------------------------------------
// ExtWrapper — borrow family
// ---------------------------------------------------------------------------

macro_rules! impl_ext_borrow {
    ($de:ty) => {
        impl<'de, T> Deserialize<'de, $de> for ExtWrapper<T>
        where
            T: DeserializeFromExtBytes<MsgpackBytesAccess<'de>>,
        {
            type Extra = T::Extra;

            #[inline(always)]
            async fn deserialize(
                d: $de,
                extra: T::Extra,
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                let mut extra = Some(extra);
                d.entry(move |[e]| {
                    let extra = extra.take().unwrap();
                    async move {
                        match e.token {
                            MsgpackToken::Ext { type_id, len } => {
                                let access = MsgpackBytesAccess {
                                    src: e.src,
                                    remaining: len,
                                };
                                match T::deserialize_from_ext_bytes(type_id, len, access, extra)
                                    .await?
                                {
                                    Probe::Hit((claim, val)) => {
                                        Ok(Probe::Hit((claim, ExtWrapper(val))))
                                    }
                                    Probe::Miss => Ok(Probe::Miss),
                                }
                            }
                            _ => Ok(Probe::Miss),
                        }
                    }
                })
                .await
            }
        }
    };
}

impl_ext_borrow!(MsgpackDeserializer<'de>);
impl_ext_borrow!(MsgpackSubDeserializer<'de>);

// ---------------------------------------------------------------------------
// ExtWrapper — owned family
// ---------------------------------------------------------------------------

macro_rules! impl_ext_owned {
    ($de:ty) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B), T> DeserializeOwned<$de> for ExtWrapper<T>
        where
            T: DeserializeFromExtBytesOwned<ChunkedMsgpackBytesAccess<'s, B, F>>,
        {
            type Extra = T::Extra;

            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                extra: T::Extra,
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                let mut extra = Some(extra);
                d.entry(move |[e]| {
                    let extra = extra.take().unwrap();
                    async move {
                        match e.token {
                            MsgpackToken::Ext { type_id, len } => {
                                let access =
                                    ChunkedMsgpackBytesAccess::new(e.handle, e.offset, len);
                                match T::deserialize_from_ext_bytes_owned(
                                    type_id, len, access, extra,
                                )
                                .await?
                                {
                                    Probe::Hit((claim, val)) => {
                                        Ok(Probe::Hit((claim, ExtWrapper(val))))
                                    }
                                    Probe::Miss => Ok(Probe::Miss),
                                }
                            }
                            _ => Ok(Probe::Miss),
                        }
                    }
                })
                .await
            }
        }
    };
}

impl_ext_owned!(ChunkedMsgpackDeserializer<'s, B, F>);
impl_ext_owned!(ChunkedMsgpackSubDeserializer<'s, B, F>);
