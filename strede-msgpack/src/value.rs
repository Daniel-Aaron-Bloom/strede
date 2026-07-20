extern crate alloc;

use crate::{
    MsgpackError,
    chunked::{ChunkedMsgpackClaim, ChunkedMsgpackDeserializer, ChunkedMsgpackSubDeserializer},
    full::{MsgpackClaim, MsgpackDeserializer, MsgpackSubDeserializer},
    timestamp::MsgpackTimestamp,
    token::MsgpackToken,
};
use alloc::string::ToString;
use strede::{Buffer, Deserialize, DeserializeOwned, Deserializer, DeserializerOwned, Probe};

/// A dynamically-typed msgpack value.
///
/// `Map` uses a `Vec` of pairs (not `HashMap`) because msgpack keys can be any
/// type and map order is observable.
///
/// `Timestamp` is a dedicated variant for type_id = -1 ext values decoded via
/// [`MsgpackTimestamp`]. All other ext types appear as [`MsgpackValue::Ext`].
pub enum MsgpackValue {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float32(f32),
    Float64(f64),
    Str(alloc::string::String),
    Bin(alloc::vec::Vec<u8>),
    Array(alloc::vec::Vec<MsgpackValue>),
    Map(alloc::vec::Vec<(MsgpackValue, MsgpackValue)>),
    Ext {
        type_id: i8,
        data: alloc::vec::Vec<u8>,
    },
    Timestamp(MsgpackTimestamp),
}

// ---------------------------------------------------------------------------
// Borrow family
// ---------------------------------------------------------------------------

macro_rules! impl_value_borrow {
    ($de:ty) => {
        impl<'de> Deserialize<'de, $de> for MsgpackValue {
            type Extra = ();

            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::Nil => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Nil,
                        ))),
                        MsgpackToken::Bool(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Bool(b),
                        ))),
                        MsgpackToken::IFixInt(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Int(b[0] as i8 as i64),
                        ))),
                        MsgpackToken::Int8(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Int(b[0] as i8 as i64),
                        ))),
                        MsgpackToken::Int16(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Int(i16::from_be_bytes(b) as i64),
                        ))),
                        MsgpackToken::Int32(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Int(i32::from_be_bytes(b) as i64),
                        ))),
                        MsgpackToken::Int64(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Int(i64::from_be_bytes(b)),
                        ))),
                        MsgpackToken::UFixInt(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::UInt(b[0] as u64),
                        ))),
                        MsgpackToken::UInt8(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::UInt(b[0] as u64),
                        ))),
                        MsgpackToken::UInt16(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::UInt(u16::from_be_bytes(b) as u64),
                        ))),
                        MsgpackToken::UInt32(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::UInt(u32::from_be_bytes(b) as u64),
                        ))),
                        MsgpackToken::UInt64(b) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::UInt(u64::from_be_bytes(b)),
                        ))),
                        MsgpackToken::Float32(f) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Float32(f),
                        ))),
                        MsgpackToken::Float64(f) => Ok(Probe::Hit((
                            MsgpackClaim {
                                src: e.src,
                                remaining_after: 0,
                            },
                            MsgpackValue::Float64(f),
                        ))),
                        // Timestamp 32
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
                                MsgpackValue::Timestamp(MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: 0,
                                }),
                            )))
                        }
                        // Timestamp 64
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
                                MsgpackValue::Timestamp(MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: nsec,
                                }),
                            )))
                        }
                        // Other fixext
                        MsgpackToken::FixExt { type_id, data, len } => {
                            let data_vec = data[..len as usize].to_vec();
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src: e.src,
                                    remaining_after: 0,
                                },
                                MsgpackValue::Ext {
                                    type_id,
                                    data: data_vec,
                                },
                            )))
                        }
                        // Timestamp 96
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
                                MsgpackValue::Timestamp(MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: nsec,
                                }),
                            )))
                        }
                        // Variable-length ext (non-timestamp)
                        MsgpackToken::Ext { type_id, len } => {
                            if e.src.len() < len {
                                return Err(MsgpackError::UnexpectedEnd);
                            }
                            let data = e.src[..len].to_vec();
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src: &e.src[len..],
                                    remaining_after: 0,
                                },
                                MsgpackValue::Ext { type_id, data },
                            )))
                        }
                        MsgpackToken::Str(len) => {
                            if e.src.len() < len {
                                return Err(MsgpackError::UnexpectedEnd);
                            }
                            let s = core::str::from_utf8(&e.src[..len])
                                .map_err(|_| MsgpackError::InvalidUtf8)?
                                .to_string();
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src: &e.src[len..],
                                    remaining_after: 0,
                                },
                                MsgpackValue::Str(s),
                            )))
                        }
                        MsgpackToken::Bin(len) => {
                            if e.src.len() < len {
                                return Err(MsgpackError::UnexpectedEnd);
                            }
                            let data = e.src[..len].to_vec();
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src: &e.src[len..],
                                    remaining_after: 0,
                                },
                                MsgpackValue::Bin(data),
                            )))
                        }
                        MsgpackToken::Array(count) => {
                            let mut src = e.src;
                            let mut items = alloc::vec::Vec::with_capacity(count.min(16));
                            for _ in 0..count {
                                let tok = crate::token::next_token(&mut src)?;
                                let sub = MsgpackSubDeserializer::new(src, tok);
                                // Explicit dyn-Future type breaks the recursive type cycle.
                                let fut: core::pin::Pin<
                                    alloc::boxed::Box<
                                        dyn core::future::Future<
                                                Output = Result<
                                                    Probe<(MsgpackClaim<'de>, MsgpackValue)>,
                                                    MsgpackError,
                                                >,
                                            > + 'de,
                                    >,
                                > = alloc::boxed::Box::pin(MsgpackValue::deserialize(sub, ()));
                                match fut.await? {
                                    Probe::Hit((claim, val)) => {
                                        items.push(val);
                                        src = claim.src;
                                    }
                                    Probe::Miss => return Ok(Probe::Miss),
                                }
                            }
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src,
                                    remaining_after: 0,
                                },
                                MsgpackValue::Array(items),
                            )))
                        }
                        MsgpackToken::Map(count) => {
                            let mut src = e.src;
                            let mut pairs = alloc::vec::Vec::with_capacity(count.min(8));
                            for _ in 0..count {
                                let ktok = crate::token::next_token(&mut src)?;
                                let ksub = MsgpackSubDeserializer::new(src, ktok);
                                let kfut: core::pin::Pin<
                                    alloc::boxed::Box<
                                        dyn core::future::Future<
                                                Output = Result<
                                                    Probe<(MsgpackClaim<'de>, MsgpackValue)>,
                                                    MsgpackError,
                                                >,
                                            > + 'de,
                                    >,
                                > = alloc::boxed::Box::pin(MsgpackValue::deserialize(ksub, ()));
                                let key = match kfut.await? {
                                    Probe::Hit((claim, val)) => {
                                        src = claim.src;
                                        val
                                    }
                                    Probe::Miss => return Ok(Probe::Miss),
                                };
                                let vtok = crate::token::next_token(&mut src)?;
                                let vsub = MsgpackSubDeserializer::new(src, vtok);
                                let vfut: core::pin::Pin<
                                    alloc::boxed::Box<
                                        dyn core::future::Future<
                                                Output = Result<
                                                    Probe<(MsgpackClaim<'de>, MsgpackValue)>,
                                                    MsgpackError,
                                                >,
                                            > + 'de,
                                    >,
                                > = alloc::boxed::Box::pin(MsgpackValue::deserialize(vsub, ()));
                                let val = match vfut.await? {
                                    Probe::Hit((claim, val)) => {
                                        src = claim.src;
                                        val
                                    }
                                    Probe::Miss => return Ok(Probe::Miss),
                                };
                                pairs.push((key, val));
                            }
                            Ok(Probe::Hit((
                                MsgpackClaim {
                                    src,
                                    remaining_after: 0,
                                },
                                MsgpackValue::Map(pairs),
                            )))
                        }
                    }
                })
                .await
            }
        }
    };
}

impl_value_borrow!(MsgpackDeserializer<'de>);
impl_value_borrow!(MsgpackSubDeserializer<'de>);

// ---------------------------------------------------------------------------
// Owned family
// ---------------------------------------------------------------------------

macro_rules! impl_value_owned {
    ($de:ty) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for MsgpackValue {
            type Extra = ();

            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, MsgpackError> {
                d.entry(|[e]| async move {
                    match e.token {
                        MsgpackToken::Nil => Ok(Probe::Hit((e.into_claim(), MsgpackValue::Nil))),
                        MsgpackToken::Bool(b) => {
                            Ok(Probe::Hit((e.into_claim(), MsgpackValue::Bool(b))))
                        }
                        MsgpackToken::IFixInt(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::Int(b[0] as i8 as i64),
                        ))),
                        MsgpackToken::Int8(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::Int(b[0] as i8 as i64),
                        ))),
                        MsgpackToken::Int16(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::Int(i16::from_be_bytes(b) as i64),
                        ))),
                        MsgpackToken::Int32(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::Int(i32::from_be_bytes(b) as i64),
                        ))),
                        MsgpackToken::Int64(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::Int(i64::from_be_bytes(b)),
                        ))),
                        MsgpackToken::UFixInt(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::UInt(b[0] as u64),
                        ))),
                        MsgpackToken::UInt8(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::UInt(b[0] as u64),
                        ))),
                        MsgpackToken::UInt16(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::UInt(u16::from_be_bytes(b) as u64),
                        ))),
                        MsgpackToken::UInt32(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::UInt(u32::from_be_bytes(b) as u64),
                        ))),
                        MsgpackToken::UInt64(b) => Ok(Probe::Hit((
                            e.into_claim(),
                            MsgpackValue::UInt(u64::from_be_bytes(b)),
                        ))),
                        MsgpackToken::Float32(f) => {
                            Ok(Probe::Hit((e.into_claim(), MsgpackValue::Float32(f))))
                        }
                        MsgpackToken::Float64(f) => {
                            Ok(Probe::Hit((e.into_claim(), MsgpackValue::Float64(f))))
                        }
                        // Timestamp 32
                        MsgpackToken::FixExt {
                            type_id: -1,
                            data,
                            len: 4,
                        } => {
                            let sec = u32::from_be_bytes(data[0..4].try_into().unwrap()) as i64;
                            Ok(Probe::Hit((
                                e.into_claim(),
                                MsgpackValue::Timestamp(MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: 0,
                                }),
                            )))
                        }
                        // Timestamp 64
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
                                MsgpackValue::Timestamp(MsgpackTimestamp {
                                    seconds: sec,
                                    nanoseconds: nsec,
                                }),
                            )))
                        }
                        // Other fixext
                        MsgpackToken::FixExt { type_id, data, len } => {
                            let data_vec = data[..len as usize].to_vec();
                            Ok(Probe::Hit((
                                e.into_claim(),
                                MsgpackValue::Ext {
                                    type_id,
                                    data: data_vec,
                                },
                            )))
                        }
                        // Variable-length ext (includes Timestamp 96 by payload check)
                        MsgpackToken::Ext { type_id, len } => {
                            use crate::chunked::access::ChunkedMsgpackBytesAccess;
                            use strede::{BytesAccessOwned, Chunk};
                            let mut data = alloc::vec::Vec::with_capacity(len.min(256));
                            let mut acc = ChunkedMsgpackBytesAccess::new(e.handle, e.offset, len);
                            let claim = loop {
                                match acc.next_bytes(|b| b.to_vec()).await? {
                                    Chunk::Data((next, chunk)) => {
                                        data.extend_from_slice(&chunk);
                                        acc = next;
                                    }
                                    Chunk::Done(claim) => break claim,
                                }
                            };
                            if type_id == -1 && len == 12 {
                                let nsec = u32::from_be_bytes(data[0..4].try_into().unwrap());
                                let sec = i64::from_be_bytes(data[4..12].try_into().unwrap());
                                Ok(Probe::Hit((
                                    claim,
                                    MsgpackValue::Timestamp(MsgpackTimestamp {
                                        seconds: sec,
                                        nanoseconds: nsec,
                                    }),
                                )))
                            } else {
                                Ok(Probe::Hit((claim, MsgpackValue::Ext { type_id, data })))
                            }
                        }
                        MsgpackToken::Str(len) => {
                            use strede::{Chunk, EntryOwned, StrAccessOwned};
                            // MsgpackToken is Copy, so `e` is not consumed by the match.
                            let acc = match EntryOwned::deserialize_str_chunks(e).await? {
                                Probe::Hit(acc) => acc,
                                Probe::Miss => unreachable!(),
                            };
                            let mut s = alloc::string::String::with_capacity(len.min(256));
                            let mut acc = acc;
                            let claim = loop {
                                match acc.next_str(|c| alloc::string::String::from(c)).await? {
                                    Chunk::Data((next, chunk)) => {
                                        s.push_str(&chunk);
                                        acc = next;
                                    }
                                    Chunk::Done(claim) => break claim,
                                }
                            };
                            Ok(Probe::Hit((claim, MsgpackValue::Str(s))))
                        }
                        MsgpackToken::Bin(len) => {
                            use strede::{BytesAccessOwned, Chunk, EntryOwned};
                            let acc = match EntryOwned::deserialize_bytes_chunks(e).await? {
                                Probe::Hit(acc) => acc,
                                Probe::Miss => unreachable!(),
                            };
                            let mut data = alloc::vec::Vec::with_capacity(len.min(256));
                            let mut acc = acc;
                            let claim = loop {
                                match acc.next_bytes(|b| b.to_vec()).await? {
                                    Chunk::Data((next, chunk)) => {
                                        data.extend_from_slice(&chunk);
                                        acc = next;
                                    }
                                    Chunk::Done(claim) => break claim,
                                }
                            };
                            Ok(Probe::Hit((claim, MsgpackValue::Bin(data))))
                        }
                        MsgpackToken::Array(count) => {
                            let mut handle = e.handle;
                            let mut offset = e.offset;
                            let mut items = alloc::vec::Vec::with_capacity(count.min(16));
                            for _ in 0..count {
                                let (h, tok) =
                                    crate::chunked::next_dispatch(handle, &mut offset).await?;
                                let sub = ChunkedMsgpackSubDeserializer::new(h, offset, tok);
                                // Explicit dyn-Future type breaks the recursive type cycle.
                                let fut: core::pin::Pin<
                                    alloc::boxed::Box<
                                        dyn core::future::Future<
                                                Output = Result<
                                                    Probe<(
                                                        ChunkedMsgpackClaim<'s, B, F>,
                                                        MsgpackValue,
                                                    )>,
                                                    MsgpackError,
                                                >,
                                            > + 's,
                                    >,
                                > = alloc::boxed::Box::pin(MsgpackValue::deserialize_owned(
                                    sub,
                                    (),
                                ));
                                match fut.await? {
                                    Probe::Hit((claim, val)) => {
                                        items.push(val);
                                        handle = claim.handle;
                                        offset = claim.offset;
                                    }
                                    Probe::Miss => return Ok(Probe::Miss),
                                }
                            }
                            Ok(Probe::Hit((
                                ChunkedMsgpackClaim {
                                    handle,
                                    offset,
                                    remaining_after: 0,
                                },
                                MsgpackValue::Array(items),
                            )))
                        }
                        MsgpackToken::Map(count) => {
                            let mut handle = e.handle;
                            let mut offset = e.offset;
                            let mut pairs = alloc::vec::Vec::with_capacity(count.min(8));
                            for _ in 0..count {
                                let (kh, ktok) =
                                    crate::chunked::next_dispatch(handle, &mut offset).await?;
                                let ksub = ChunkedMsgpackSubDeserializer::new(kh, offset, ktok);
                                let kfut: core::pin::Pin<
                                    alloc::boxed::Box<
                                        dyn core::future::Future<
                                                Output = Result<
                                                    Probe<(
                                                        ChunkedMsgpackClaim<'s, B, F>,
                                                        MsgpackValue,
                                                    )>,
                                                    MsgpackError,
                                                >,
                                            > + 's,
                                    >,
                                > = alloc::boxed::Box::pin(MsgpackValue::deserialize_owned(
                                    ksub,
                                    (),
                                ));
                                let key = match kfut.await? {
                                    Probe::Hit((claim, val)) => {
                                        handle = claim.handle;
                                        offset = claim.offset;
                                        val
                                    }
                                    Probe::Miss => return Ok(Probe::Miss),
                                };
                                let (vh, vtok) =
                                    crate::chunked::next_dispatch(handle, &mut offset).await?;
                                let vsub = ChunkedMsgpackSubDeserializer::new(vh, offset, vtok);
                                let vfut: core::pin::Pin<
                                    alloc::boxed::Box<
                                        dyn core::future::Future<
                                                Output = Result<
                                                    Probe<(
                                                        ChunkedMsgpackClaim<'s, B, F>,
                                                        MsgpackValue,
                                                    )>,
                                                    MsgpackError,
                                                >,
                                            > + 's,
                                    >,
                                > = alloc::boxed::Box::pin(MsgpackValue::deserialize_owned(
                                    vsub,
                                    (),
                                ));
                                let val = match vfut.await? {
                                    Probe::Hit((claim, val)) => {
                                        handle = claim.handle;
                                        offset = claim.offset;
                                        val
                                    }
                                    Probe::Miss => return Ok(Probe::Miss),
                                };
                                pairs.push((key, val));
                            }
                            Ok(Probe::Hit((
                                ChunkedMsgpackClaim {
                                    handle,
                                    offset,
                                    remaining_after: 0,
                                },
                                MsgpackValue::Map(pairs),
                            )))
                        }
                    }
                })
                .await
            }
        }
    };
}

impl_value_owned!(ChunkedMsgpackDeserializer<'s, B, F>);
impl_value_owned!(ChunkedMsgpackSubDeserializer<'s, B, F>);
