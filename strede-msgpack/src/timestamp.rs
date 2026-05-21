#[cfg(feature = "alloc")]
extern crate alloc;

use crate::{
    MsgpackError,
    chunked::{ChunkedMsgpackDeserializer, ChunkedMsgpackSubDeserializer},
    full::{MsgpackClaim, MsgpackDeserializer, MsgpackSubDeserializer},
    token::MsgpackToken,
};
use strede::{Buffer, Deserialize, DeserializeOwned, Deserializer, DeserializerOwned, Probe};

/// Decoded msgpack timestamp extension (type_id = -1).
///
/// Three encodings are supported:
/// - **Timestamp 32** (`fixext4`): seconds only, nanoseconds = 0.
/// - **Timestamp 64** (`fixext8`): nanoseconds in bits 63–34, seconds in bits 33–0.
/// - **Timestamp 96** (`ext8`, 12 bytes): u32 nanoseconds + i64 seconds.
///
/// Timestamp 96 in the owned family requires the `alloc` feature. Without it,
/// `ext8` with type_id = -1 and len = 12 returns `Probe::Miss`.
pub struct MsgpackTimestamp {
    pub seconds: i64,
    pub nanoseconds: u32,
}

// ---------------------------------------------------------------------------
// Borrow family
// ---------------------------------------------------------------------------

macro_rules! impl_timestamp_borrow {
    ($de:ty) => {
        impl<'de> Deserialize<'de, $de> for MsgpackTimestamp {
            type Extra = ();

            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::FixExt {
                            type_id: -1,
                            data,
                            len: 4,
                        } => {
                            let sec = u32::from_be_bytes(data[0..4].try_into().unwrap()) as i64;
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src: e.src,
                                    remaining_after: 0,
                                },
                                MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: 0,
                                },
                            )))
                        }
                        MsgpackToken::FixExt {
                            type_id: -1,
                            data,
                            len: 8,
                        } => {
                            let val = u64::from_be_bytes(data[0..8].try_into().unwrap());
                            let nsec = (val >> 34) as u32;
                            let sec = (val & 0x3_FFFF_FFFF) as i64;
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src: e.src,
                                    remaining_after: 0,
                                },
                                MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: nsec,
                                },
                            )))
                        }
                        MsgpackToken::Ext {
                            type_id: -1,
                            len: 12,
                        } => {
                            if e.src.len() < 12 {
                                return Err(MsgpackError::UnexpectedEnd);
                            }
                            let nsec = u32::from_be_bytes(e.src[0..4].try_into().unwrap());
                            let sec = i64::from_be_bytes(e.src[4..12].try_into().unwrap());
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src: &e.src[12..],
                                    remaining_after: 0,
                                },
                                MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: nsec,
                                },
                            )))
                        }
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
}

impl_timestamp_borrow!(MsgpackDeserializer<'de>);
impl_timestamp_borrow!(MsgpackSubDeserializer<'de>);

// ---------------------------------------------------------------------------
// Owned family
// ---------------------------------------------------------------------------

macro_rules! impl_timestamp_owned {
    ($de:ty) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for MsgpackTimestamp {
            type Extra = ();

            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::FixExt {
                            type_id: -1,
                            data,
                            len: 4,
                        } => {
                            let sec = u32::from_be_bytes(data[0..4].try_into().unwrap()) as i64;
                            Ok(Probe::Hit((
                                e.into_claim(),
                                MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: 0,
                                },
                            )))
                        }
                        MsgpackToken::FixExt {
                            type_id: -1,
                            data,
                            len: 8,
                        } => {
                            let val = u64::from_be_bytes(data[0..8].try_into().unwrap());
                            let nsec = (val >> 34) as u32;
                            let sec = (val & 0x3_FFFF_FFFF) as i64;
                            Ok(Probe::Hit((
                                e.into_claim(),
                                MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: nsec,
                                },
                            )))
                        }
                        MsgpackToken::Ext {
                            type_id: -1,
                            len: 12,
                        } => {
                            #[cfg(not(feature = "alloc"))]
                            {
                                return Ok(Probe::Miss);
                            }
                            #[cfg(feature = "alloc")]
                            {
                                use crate::chunked::access::ChunkedMsgpackBytesAccess;
                                use strede::{BytesAccessOwned, Chunk};
                                let mut collected = alloc::vec::Vec::with_capacity(12);
                                let mut acc =
                                    ChunkedMsgpackBytesAccess::new(e.handle, e.offset, 12);
                                loop {
                                    match acc.next_bytes(|b| b.to_vec()).await? {
                                        Chunk::Data((next, chunk)) => {
                                            collected.extend_from_slice(&chunk);
                                            acc = next;
                                        }
                                        Chunk::Done(claim) => {
                                            let nsec = u32::from_be_bytes(
                                                collected[0..4].try_into().unwrap(),
                                            );
                                            let sec = i64::from_be_bytes(
                                                collected[4..12].try_into().unwrap(),
                                            );
                                            return Ok(Probe::Hit((
                                                claim,
                                                MsgpackTimestamp {
                                                    seconds: sec,
                                                    nanoseconds: nsec,
                                                },
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
}

impl_timestamp_owned!(ChunkedMsgpackDeserializer<'s, B, F>);
impl_timestamp_owned!(ChunkedMsgpackSubDeserializer<'s, B, F>);
