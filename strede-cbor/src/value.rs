extern crate alloc;

use crate::{
    CborError,
    chunked::{ChunkedCborClaim, ChunkedCborDeserializer, ChunkedCborSubDeserializer},
    full::{CborClaim, CborDeserializer, CborSubDeserializer},
    token::CborToken,
};
use alloc::{boxed::Box, string::String, vec::Vec};
use strede::{Buffer, Deserialize, DeserializeOwned, Deserializer, DeserializerOwned, Probe};

/// A dynamically-typed CBOR value.
///
/// `Map` uses `Vec` of pairs (not `HashMap`) because CBOR keys can be any type
/// and map order may be significant.
///
/// `Tag` wraps another `CborValue` with the tag number.
pub enum CborValue {
    Null,
    Undefined,
    Bool(bool),
    UInt(u64),
    /// Signed negative integer: -2^63 .. -1 (i.e. actual = -1 - raw_negint where raw fits i64)
    Int(i64),
    /// Negative integer that doesn't fit i64: actual = -1 - NegIntOverflow
    NegIntOverflow(u64),
    /// All float variants stored as f64 for uniformity
    Float(f64),
    Bstr(Vec<u8>),
    Tstr(String),
    Array(Vec<CborValue>),
    Map(Vec<(CborValue, CborValue)>),
    Tag {
        number: u64,
        value: Box<CborValue>,
    },
}

// ---------------------------------------------------------------------------
// Borrow family
// ---------------------------------------------------------------------------

macro_rules! impl_value_borrow {
    ($de:ty) => {
        impl<'de> Deserialize<'de, $de> for CborValue {
            type Extra = ();

            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
                d.entry(|[e]| async move {
                    let src = e.src;
                    let claim = || CborClaim {
                        src,
                        remaining_after: 0,
                    };
                    match e.token {
                        CborToken::Null => Ok(Probe::Hit((claim(), CborValue::Null))),
                        CborToken::Undefined => Ok(Probe::Hit((claim(), CborValue::Undefined))),
                        CborToken::Bool(b) => Ok(Probe::Hit((claim(), CborValue::Bool(b)))),
                        CborToken::UInt(n) => Ok(Probe::Hit((claim(), CborValue::UInt(n)))),
                        CborToken::NegInt(n) => {
                            let v = if n <= i64::MAX as u64 {
                                CborValue::Int(-1i64 - n as i64)
                            } else {
                                CborValue::NegIntOverflow(n)
                            };
                            Ok(Probe::Hit((claim(), v)))
                        }
                        CborToken::Float16(f) => {
                            Ok(Probe::Hit((claim(), CborValue::Float(f as f64))))
                        }
                        CborToken::Float32(f) => {
                            Ok(Probe::Hit((claim(), CborValue::Float(f as f64))))
                        }
                        CborToken::Float64(f) => Ok(Probe::Hit((claim(), CborValue::Float(f)))),
                        CborToken::Bstr(len) => {
                            if src.len() < len {
                                return Err(CborError::UnexpectedEnd);
                            }
                            let (payload, rest) = src.split_at(len);
                            Ok(Probe::Hit((
                                CborClaim {
                                    src: rest,
                                    remaining_after: 0,
                                },
                                CborValue::Bstr(payload.to_vec()),
                            )))
                        }
                        CborToken::BstrIndef => {
                            // Collect all chunks
                            let mut bytes = Vec::new();
                            let mut s = src;
                            loop {
                                let tok = crate::token::next_token(&mut s)?;
                                match tok {
                                    CborToken::Break => break,
                                    CborToken::Bstr(len) => {
                                        if s.len() < len {
                                            return Err(CborError::UnexpectedEnd);
                                        }
                                        bytes.extend_from_slice(&s[..len]);
                                        s = &s[len..];
                                    }
                                    _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                                }
                            }
                            Ok(Probe::Hit((
                                CborClaim {
                                    src: s,
                                    remaining_after: 0,
                                },
                                CborValue::Bstr(bytes),
                            )))
                        }
                        CborToken::Tstr(len) => {
                            if src.len() < len {
                                return Err(CborError::UnexpectedEnd);
                            }
                            let (payload, rest) = src.split_at(len);
                            let s = core::str::from_utf8(payload)
                                .map_err(|_| CborError::InvalidUtf8)?;
                            Ok(Probe::Hit((
                                CborClaim {
                                    src: rest,
                                    remaining_after: 0,
                                },
                                CborValue::Tstr(s.into()),
                            )))
                        }
                        CborToken::TstrIndef => {
                            let mut out = String::new();
                            let mut s = src;
                            loop {
                                let tok = crate::token::next_token(&mut s)?;
                                match tok {
                                    CborToken::Break => break,
                                    CborToken::Tstr(len) => {
                                        if s.len() < len {
                                            return Err(CborError::UnexpectedEnd);
                                        }
                                        let chunk = core::str::from_utf8(&s[..len])
                                            .map_err(|_| CborError::InvalidUtf8)?;
                                        out.push_str(chunk);
                                        s = &s[len..];
                                    }
                                    _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                                }
                            }
                            Ok(Probe::Hit((
                                CborClaim {
                                    src: s,
                                    remaining_after: 0,
                                },
                                CborValue::Tstr(out),
                            )))
                        }
                        CborToken::Array(count) => {
                            let sub_de = CborSubDeserializer::new(src, CborToken::Array(count));
                            let fut: core::pin::Pin<
                                Box<dyn core::future::Future<Output = _> + 'de>,
                            > = Box::pin(deserialize_array_borrow(sub_de));
                            fut.await
                        }
                        CborToken::Map(count) => {
                            let sub_de = CborSubDeserializer::new(src, CborToken::Map(count));
                            let fut: core::pin::Pin<
                                Box<dyn core::future::Future<Output = _> + 'de>,
                            > = Box::pin(deserialize_map_borrow(sub_de));
                            fut.await
                        }
                        CborToken::Tag(number) => {
                            // Read the tagged value
                            let mut s = src;
                            let raw = crate::token::next_token(&mut s)?;
                            let inner_sub = CborSubDeserializer::new(s, raw);
                            let fut: core::pin::Pin<
                                Box<dyn core::future::Future<Output = _> + 'de>,
                            > = Box::pin(CborValue::deserialize(inner_sub, ()));
                            match fut.await? {
                                Probe::Hit((claim, inner)) => Ok(Probe::Hit((
                                    claim,
                                    CborValue::Tag {
                                        number,
                                        value: Box::new(inner),
                                    },
                                ))),
                                Probe::Miss => Ok(Probe::Miss),
                            }
                        }
                        CborToken::Break => Err(CborError::InvalidBreak),
                    }
                })
                .await
            }
        }
    };
}

async fn deserialize_array_borrow<'de>(
    d: CborSubDeserializer<'de>,
) -> Result<Probe<(CborClaim<'de>, CborValue)>, CborError> {
    use strede::SeqAccess;
    d.entry(|[e]| async move {
        let seq = match e.token {
            CborToken::Array(count) => crate::full::CborSeqAccess {
                src: e.src,
                remaining: count,
            },
            _ => return Ok(Probe::Miss),
        };
        let mut items = alloc::vec::Vec::new();
        let mut seq = seq;
        loop {
            match seq
                .next(|[elem]| async move {
                    let sub = CborSubDeserializer::new(elem.src, elem.elem_tok);
                    let fut: core::pin::Pin<Box<dyn core::future::Future<Output = _> + '_>> =
                        Box::pin(CborValue::deserialize(sub, ()));
                    match fut.await? {
                        Probe::Hit((claim, v)) => Ok(Probe::Hit((claim, v))),
                        Probe::Miss => Ok(Probe::Miss),
                    }
                })
                .await?
            {
                Probe::Hit(strede::Chunk::Done(claim)) => {
                    return Ok(Probe::Hit((claim, CborValue::Array(items))));
                }
                Probe::Hit(strede::Chunk::Data((new_seq, v))) => {
                    items.push(v);
                    seq = new_seq;
                }
                Probe::Miss => return Ok(Probe::Miss),
            }
        }
    })
    .await
}

async fn deserialize_map_borrow<'de>(
    d: CborSubDeserializer<'de>,
) -> Result<Probe<(CborClaim<'de>, CborValue)>, CborError> {
    d.entry(|[e]| async move {
        // Use seq-like iteration manually: for each pair, read key then value
        let (mut src, remaining) = match e.token {
            CborToken::Map(count) => (e.src, count),
            _ => return Ok(Probe::Miss),
        };
        let mut pairs = alloc::vec::Vec::new();

        match remaining {
            Some(0) => {}
            None => {
                // Indefinite
                loop {
                    let raw = crate::token::next_token(&mut src)?;
                    if matches!(raw, CborToken::Break) {
                        break;
                    }
                    let key_sub = CborSubDeserializer::new(src, raw);
                    let (key_claim, key) = match CborValue::deserialize(key_sub, ()).await? {
                        Probe::Hit(x) => x,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    src = key_claim.src;
                    let val_raw = crate::token::next_token(&mut src)?;
                    let val_sub = CborSubDeserializer::new(src, val_raw);
                    let (val_claim, val) = match CborValue::deserialize(val_sub, ()).await? {
                        Probe::Hit(x) => x,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    src = val_claim.src;
                    pairs.push((key, val));
                }
            }
            Some(n) => {
                for _ in 0..n {
                    let raw = crate::token::next_token(&mut src)?;
                    let key_sub = CborSubDeserializer::new(src, raw);
                    let (key_claim, key) = match CborValue::deserialize(key_sub, ()).await? {
                        Probe::Hit(x) => x,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    src = key_claim.src;
                    let val_raw = crate::token::next_token(&mut src)?;
                    let val_sub = CborSubDeserializer::new(src, val_raw);
                    let (val_claim, val) = match CborValue::deserialize(val_sub, ()).await? {
                        Probe::Hit(x) => x,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    src = val_claim.src;
                    pairs.push((key, val));
                }
            }
        }

        Ok(Probe::Hit((
            CborClaim {
                src,
                remaining_after: 0,
            },
            CborValue::Map(pairs),
        )))
    })
    .await
}

impl_value_borrow!(CborDeserializer<'de>);
impl_value_borrow!(CborSubDeserializer<'de>);

// ---------------------------------------------------------------------------
// Owned family
// ---------------------------------------------------------------------------

macro_rules! impl_value_owned {
    ($de:ty) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for CborValue {
            type Extra = ();

            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, CborError> {
                d.entry(|[e]| async move {
                    match e.token {
                        CborToken::Null => Ok(Probe::Hit((e.into_claim(), CborValue::Null))),
                        CborToken::Undefined => {
                            Ok(Probe::Hit((e.into_claim(), CborValue::Undefined)))
                        }
                        CborToken::Bool(b) => Ok(Probe::Hit((e.into_claim(), CborValue::Bool(b)))),
                        CborToken::UInt(n) => Ok(Probe::Hit((e.into_claim(), CborValue::UInt(n)))),
                        CborToken::NegInt(n) => {
                            let v = if n <= i64::MAX as u64 {
                                CborValue::Int(-1i64 - n as i64)
                            } else {
                                CborValue::NegIntOverflow(n)
                            };
                            Ok(Probe::Hit((e.into_claim(), v)))
                        }
                        CborToken::Float16(f) => {
                            Ok(Probe::Hit((e.into_claim(), CborValue::Float(f as f64))))
                        }
                        CborToken::Float32(f) => {
                            Ok(Probe::Hit((e.into_claim(), CborValue::Float(f as f64))))
                        }
                        CborToken::Float64(f) => {
                            Ok(Probe::Hit((e.into_claim(), CborValue::Float(f))))
                        }
                        CborToken::Bstr(len) => {
                            // Collect bytes across chunk boundaries
                            let mut bytes = Vec::with_capacity(len);
                            let access = crate::chunked::access::ChunkedCborBytesAccess {
                                handle: e.handle,
                                offset: e.offset,
                                state: crate::chunked::access::ChunkedBytesState::Definite {
                                    remaining: len,
                                },
                            };
                            let (access, _) = collect_bytes_owned(access, &mut bytes).await?;
                            let c = ChunkedCborClaim {
                                offset: access.offset,
                                handle: access.handle,
                                remaining_after: 0,
                            };
                            Ok(Probe::Hit((c, CborValue::Bstr(bytes))))
                        }
                        CborToken::BstrIndef => {
                            let mut bytes = Vec::new();
                            let access = crate::chunked::access::ChunkedCborBytesAccess {
                                handle: e.handle,
                                offset: e.offset,
                                state: crate::chunked::access::ChunkedBytesState::Indefinite,
                            };
                            let (access, _) = collect_bytes_owned(access, &mut bytes).await?;
                            let c = ChunkedCborClaim {
                                offset: access.offset,
                                handle: access.handle,
                                remaining_after: 0,
                            };
                            Ok(Probe::Hit((c, CborValue::Bstr(bytes))))
                        }
                        CborToken::Tstr(len) => {
                            let mut s = String::new();
                            let access = crate::chunked::access::ChunkedCborStrAccess {
                                handle: e.handle,
                                offset: e.offset,
                                state: crate::chunked::access::ChunkedStrState::Definite {
                                    remaining: len,
                                },
                            };
                            let (access, _) = collect_str_owned(access, &mut s).await?;
                            let c = ChunkedCborClaim {
                                offset: access.offset,
                                handle: access.handle,
                                remaining_after: 0,
                            };
                            Ok(Probe::Hit((c, CborValue::Tstr(s))))
                        }
                        CborToken::TstrIndef => {
                            let mut s = String::new();
                            let access = crate::chunked::access::ChunkedCborStrAccess {
                                handle: e.handle,
                                offset: e.offset,
                                state: crate::chunked::access::ChunkedStrState::Indefinite,
                            };
                            let (access, _) = collect_str_owned(access, &mut s).await?;
                            let c = ChunkedCborClaim {
                                offset: access.offset,
                                handle: access.handle,
                                remaining_after: 0,
                            };
                            Ok(Probe::Hit((c, CborValue::Tstr(s))))
                        }
                        CborToken::Array(count) => {
                            let sub = ChunkedCborSubDeserializer::new(
                                e.handle,
                                e.offset,
                                CborToken::Array(count),
                            );
                            let fut: core::pin::Pin<Box<dyn core::future::Future<Output = _>>> =
                                Box::pin(deserialize_array_owned(sub));
                            fut.await
                        }
                        CborToken::Map(count) => {
                            let sub = ChunkedCborSubDeserializer::new(
                                e.handle,
                                e.offset,
                                CborToken::Map(count),
                            );
                            let fut: core::pin::Pin<Box<dyn core::future::Future<Output = _>>> =
                                Box::pin(deserialize_map_owned(sub));
                            fut.await
                        }
                        CborToken::Tag(number) => {
                            // Read the tagged value
                            let (handle, raw) =
                                crate::chunked::next_dispatch(e.handle, &mut { e.offset }).await?;
                            let inner_sub = ChunkedCborSubDeserializer::new(handle, e.offset, raw);
                            let fut: core::pin::Pin<Box<dyn core::future::Future<Output = _>>> =
                                Box::pin(CborValue::deserialize_owned(inner_sub, ()));
                            match fut.await? {
                                Probe::Hit((claim, inner)) => Ok(Probe::Hit((
                                    claim,
                                    CborValue::Tag {
                                        number,
                                        value: Box::new(inner),
                                    },
                                ))),
                                Probe::Miss => Ok(Probe::Miss),
                            }
                        }
                        CborToken::Break => Err(CborError::InvalidBreak),
                    }
                })
                .await
            }
        }
    };
}

async fn collect_bytes_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut access: crate::chunked::access::ChunkedCborBytesAccess<'s, B, F>,
    out: &mut Vec<u8>,
) -> Result<(crate::chunked::access::ChunkedCborBytesAccess<'s, B, F>, ()), CborError> {
    use strede::BytesAccessOwned;
    loop {
        match access.next_bytes(|b| out.extend_from_slice(b)).await? {
            strede::Chunk::Data((next, _)) => access = next,
            strede::Chunk::Done(claim) => {
                return Ok((
                    crate::chunked::access::ChunkedCborBytesAccess {
                        handle: claim.handle,
                        offset: claim.offset,
                        state: crate::chunked::access::ChunkedBytesState::Definite { remaining: 0 },
                    },
                    (),
                ));
            }
        }
    }
}

async fn collect_str_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut access: crate::chunked::access::ChunkedCborStrAccess<'s, B, F>,
    out: &mut String,
) -> Result<(crate::chunked::access::ChunkedCborStrAccess<'s, B, F>, ()), CborError> {
    use strede::StrAccessOwned;
    loop {
        match access.next_str(|s| out.push_str(s)).await? {
            strede::Chunk::Data((next, _)) => access = next,
            strede::Chunk::Done(claim) => {
                return Ok((
                    crate::chunked::access::ChunkedCborStrAccess {
                        handle: claim.handle,
                        offset: claim.offset,
                        state: crate::chunked::access::ChunkedStrState::Definite { remaining: 0 },
                    },
                    (),
                ));
            }
        }
    }
}

async fn deserialize_array_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    d: ChunkedCborSubDeserializer<'s, B, F>,
) -> Result<Probe<(ChunkedCborClaim<'s, B, F>, CborValue)>, CborError> {
    use strede::SeqAccessOwned;
    d.entry(|[e]| async move {
        let seq = match e.token {
            CborToken::Array(count) => crate::chunked::access::ChunkedCborSeqAccess {
                handle: e.handle,
                offset: e.offset,
                remaining: count,
            },
            _ => return Ok(Probe::Miss),
        };
        let mut items = Vec::new();
        let mut seq = seq;
        loop {
            match seq
                .next(|[elem]| async move {
                    let sub =
                        ChunkedCborSubDeserializer::new(elem.handle, elem.offset, elem.elem_tok);
                    let fut: core::pin::Pin<Box<dyn core::future::Future<Output = _>>> =
                        Box::pin(CborValue::deserialize_owned(sub, ()));
                    match fut.await? {
                        Probe::Hit((claim, v)) => Ok(Probe::Hit((claim, v))),
                        Probe::Miss => Ok(Probe::Miss),
                    }
                })
                .await?
            {
                Probe::Hit(strede::Chunk::Done(claim)) => {
                    return Ok(Probe::Hit((claim, CborValue::Array(items))));
                }
                Probe::Hit(strede::Chunk::Data((new_seq, v))) => {
                    items.push(v);
                    seq = new_seq;
                }
                Probe::Miss => return Ok(Probe::Miss),
            }
        }
    })
    .await
}

async fn deserialize_map_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    d: ChunkedCborSubDeserializer<'s, B, F>,
) -> Result<Probe<(ChunkedCborClaim<'s, B, F>, CborValue)>, CborError> {
    d.entry(|[e]| async move {
        let (mut handle, mut offset, remaining) = match e.token {
            CborToken::Map(count) => (e.handle, e.offset, count),
            _ => return Ok(Probe::Miss),
        };
        let mut pairs = Vec::new();

        match remaining {
            Some(0) => {}
            None => loop {
                let (h, tok) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                handle = h;
                if matches!(tok, CborToken::Break) {
                    break;
                }
                let key_sub = ChunkedCborSubDeserializer::new(handle, offset, tok);
                let (key_claim, key) = match CborValue::deserialize_owned(key_sub, ()).await? {
                    Probe::Hit(x) => x,
                    Probe::Miss => return Ok(Probe::Miss),
                };
                handle = key_claim.handle;
                offset = key_claim.offset;

                let (h, val_tok) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                handle = h;
                let val_sub = ChunkedCborSubDeserializer::new(handle, offset, val_tok);
                let (val_claim, val) = match CborValue::deserialize_owned(val_sub, ()).await? {
                    Probe::Hit(x) => x,
                    Probe::Miss => return Ok(Probe::Miss),
                };
                handle = val_claim.handle;
                offset = val_claim.offset;
                pairs.push((key, val));
            },
            Some(n) => {
                for _ in 0..n {
                    let (h, key_tok) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                    handle = h;
                    let key_sub = ChunkedCborSubDeserializer::new(handle, offset, key_tok);
                    let (key_claim, key) = match CborValue::deserialize_owned(key_sub, ()).await? {
                        Probe::Hit(x) => x,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    handle = key_claim.handle;
                    offset = key_claim.offset;

                    let (h, val_tok) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                    handle = h;
                    let val_sub = ChunkedCborSubDeserializer::new(handle, offset, val_tok);
                    let (val_claim, val) = match CborValue::deserialize_owned(val_sub, ()).await? {
                        Probe::Hit(x) => x,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    handle = val_claim.handle;
                    offset = val_claim.offset;
                    pairs.push((key, val));
                }
            }
        }

        Ok(Probe::Hit((
            ChunkedCborClaim {
                handle,
                offset,
                remaining_after: 0,
            },
            CborValue::Map(pairs),
        )))
    })
    .await
}

impl_value_owned!(ChunkedCborDeserializer<'s, B, F>);
impl_value_owned!(ChunkedCborSubDeserializer<'s, B, F>);
