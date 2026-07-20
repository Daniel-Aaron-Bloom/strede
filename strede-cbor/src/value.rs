extern crate alloc;

use crate::{
    CborError,
    chunked::{ChunkedCborClaim, ChunkedCborDeserializer, ChunkedCborSubDeserializer},
    full::{CborClaim, CborDeserializer, CborSubDeserializer},
    token::CborToken,
};
use alloc::{boxed::Box, string::String, vec::Vec};
use strede::{Buffer, Deserialize, DeserializeOwned, DeserializerOwned, Probe};

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
    /// Simple value other than false/true/null/undefined (see `CborToken::Simple`).
    Simple(u8),
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

type DecodeValueBorrowFuture<'de> =
    core::pin::Pin<Box<dyn core::future::Future<Output = DecodeValueBorrowResult<'de>> + 'de>>;
type DecodeValueBorrowResult<'de> = Result<Probe<(&'de [u8], CborValue)>, CborError>;

/// Recursively decode one CBOR value starting at `raw_start` — a position
/// that has *not* had any leading tags stripped. Reading fresh from here
/// (rather than from an already-stripped `Entry`/`SubDeserializer`) is what
/// lets `CborValue::Tag` actually get constructed: every recursive call
/// (tag payload, array element, map key/value) is handed the true origin of
/// its own value, so tags nested at any depth are preserved rather than
/// silently discarded by the generic entry-construction machinery.
fn decode_value_borrow<'de>(raw_start: &'de [u8]) -> DecodeValueBorrowFuture<'de> {
    Box::pin(async move {
        let mut src = raw_start;
        let tok = crate::token::next_token(&mut src)?;
        match tok {
            CborToken::Null => Ok(Probe::Hit((src, CborValue::Null))),
            CborToken::Undefined => Ok(Probe::Hit((src, CborValue::Undefined))),
            CborToken::Bool(b) => Ok(Probe::Hit((src, CborValue::Bool(b)))),
            CborToken::Simple(v) => Ok(Probe::Hit((src, CborValue::Simple(v)))),
            CborToken::UInt(n) => Ok(Probe::Hit((src, CborValue::UInt(n)))),
            CborToken::NegInt(n) => {
                let v = if n <= i64::MAX as u64 {
                    CborValue::Int(-1i64 - n as i64)
                } else {
                    CborValue::NegIntOverflow(n)
                };
                Ok(Probe::Hit((src, v)))
            }
            CborToken::Float16(f) => Ok(Probe::Hit((src, CborValue::Float(f as f64)))),
            CborToken::Float32(f) => Ok(Probe::Hit((src, CborValue::Float(f as f64)))),
            CborToken::Float64(f) => Ok(Probe::Hit((src, CborValue::Float(f)))),
            CborToken::Bstr(len) => {
                if src.len() < len {
                    return Err(CborError::UnexpectedEnd);
                }
                let (payload, rest) = src.split_at(len);
                Ok(Probe::Hit((rest, CborValue::Bstr(payload.to_vec()))))
            }
            CborToken::BstrIndef => {
                let mut bytes = Vec::new();
                loop {
                    let chunk_tok = crate::token::next_token(&mut src)?;
                    match chunk_tok {
                        CborToken::Break => break,
                        CborToken::Bstr(len) => {
                            if src.len() < len {
                                return Err(CborError::UnexpectedEnd);
                            }
                            bytes.extend_from_slice(&src[..len]);
                            src = &src[len..];
                        }
                        _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                    }
                }
                Ok(Probe::Hit((src, CborValue::Bstr(bytes))))
            }
            CborToken::Tstr(len) => {
                if src.len() < len {
                    return Err(CborError::UnexpectedEnd);
                }
                let (payload, rest) = src.split_at(len);
                let s = core::str::from_utf8(payload).map_err(|_| CborError::InvalidUtf8)?;
                Ok(Probe::Hit((rest, CborValue::Tstr(s.into()))))
            }
            CborToken::TstrIndef => {
                let mut out = String::new();
                loop {
                    let chunk_tok = crate::token::next_token(&mut src)?;
                    match chunk_tok {
                        CborToken::Break => break,
                        CborToken::Tstr(len) => {
                            if src.len() < len {
                                return Err(CborError::UnexpectedEnd);
                            }
                            let chunk =
                                core::str::from_utf8(&src[..len]).map_err(|_| CborError::InvalidUtf8)?;
                            out.push_str(chunk);
                            src = &src[len..];
                        }
                        _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                    }
                }
                Ok(Probe::Hit((src, CborValue::Tstr(out))))
            }
            CborToken::Array(count) => {
                let mut items = Vec::new();
                match count {
                    Some(n) => {
                        for _ in 0..n {
                            match decode_value_borrow(src).await? {
                                Probe::Hit((rest, v)) => {
                                    items.push(v);
                                    src = rest;
                                }
                                Probe::Miss => return Ok(Probe::Miss),
                            }
                        }
                    }
                    None => loop {
                        let mut peek = src;
                        let peeked = crate::token::next_token(&mut peek)?;
                        if matches!(peeked, CborToken::Break) {
                            src = peek;
                            break;
                        }
                        match decode_value_borrow(src).await? {
                            Probe::Hit((rest, v)) => {
                                items.push(v);
                                src = rest;
                            }
                            Probe::Miss => return Ok(Probe::Miss),
                        }
                    },
                }
                Ok(Probe::Hit((src, CborValue::Array(items))))
            }
            CborToken::Map(count) => {
                let mut pairs = Vec::new();
                let read_pair = async |src: &mut &'de [u8]| -> Result<Option<(CborValue, CborValue)>, CborError> {
                    let key = match decode_value_borrow(src).await? {
                        Probe::Hit((rest, k)) => {
                            *src = rest;
                            k
                        }
                        Probe::Miss => return Ok(None),
                    };
                    let val = match decode_value_borrow(src).await? {
                        Probe::Hit((rest, v)) => {
                            *src = rest;
                            v
                        }
                        Probe::Miss => return Ok(None),
                    };
                    Ok(Some((key, val)))
                };
                match count {
                    Some(n) => {
                        for _ in 0..n {
                            match read_pair(&mut src).await? {
                                Some(pair) => pairs.push(pair),
                                None => return Ok(Probe::Miss),
                            }
                        }
                    }
                    None => loop {
                        let mut peek = src;
                        let peeked = crate::token::next_token(&mut peek)?;
                        if matches!(peeked, CborToken::Break) {
                            src = peek;
                            break;
                        }
                        match read_pair(&mut src).await? {
                            Some(pair) => pairs.push(pair),
                            None => return Ok(Probe::Miss),
                        }
                    },
                }
                Ok(Probe::Hit((src, CborValue::Map(pairs))))
            }
            CborToken::Tag(number) => match decode_value_borrow(src).await? {
                Probe::Hit((rest, inner)) => Ok(Probe::Hit((
                    rest,
                    CborValue::Tag {
                        number,
                        value: Box::new(inner),
                    },
                ))),
                Probe::Miss => Ok(Probe::Miss),
            },
            CborToken::Break => Err(CborError::InvalidBreak),
        }
    })
}

impl<'de> Deserialize<'de, CborDeserializer<'de>> for CborValue {
    type Extra = ();

    async fn deserialize(
        d: CborDeserializer<'de>,
        _: (),
    ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
        match decode_value_borrow(d.src).await? {
            Probe::Hit((rest, v)) => {
                if !rest.is_empty() {
                    return Err(CborError::ExpectedEnd);
                }
                Ok(Probe::Hit((
                    CborClaim {
                        src: rest,
                        remaining_after: 0,
                    },
                    v,
                )))
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

impl<'de> Deserialize<'de, CborSubDeserializer<'de>> for CborValue {
    type Extra = ();

    async fn deserialize(
        d: CborSubDeserializer<'de>,
        _: (),
    ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
        match decode_value_borrow(d.raw_start).await? {
            Probe::Hit((rest, v)) => Ok(Probe::Hit((
                CborClaim {
                    src: rest,
                    remaining_after: 0,
                },
                v,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

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
                    // `e.token` is exactly what was read — not resolved past
                    // any leading tags — so the `Tag` arm below is real and
                    // reachable: this is the one place that's allowed to see
                    // tags at all, recursing for as many layers as actually
                    // exist (no fixed capacity, unlike a captured chain).
                    match e.token {
                        CborToken::Null => Ok(Probe::Hit((e.into_claim(), CborValue::Null))),
                        CborToken::Undefined => {
                            Ok(Probe::Hit((e.into_claim(), CborValue::Undefined)))
                        }
                        CborToken::Bool(b) => Ok(Probe::Hit((e.into_claim(), CborValue::Bool(b)))),
                        CborToken::Simple(v) => {
                            Ok(Probe::Hit((e.into_claim(), CborValue::Simple(v))))
                        }
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
                                remaining_after: None,
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
                                remaining_after: None,
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
                                remaining_after: None,
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
                                remaining_after: None,
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
                            let mut offset = e.offset;
                            let (handle, raw) =
                                crate::chunked::next_dispatch(e.handle, &mut offset).await?;
                            let inner_sub = ChunkedCborSubDeserializer::new(handle, offset, raw);
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
                let (h, raw) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                if matches!(raw, CborToken::Break) {
                    handle = h;
                    break;
                }
                let key_sub = ChunkedCborSubDeserializer::new(h, offset, raw);
                let (key_claim, key) = match CborValue::deserialize_owned(key_sub, ()).await? {
                    Probe::Hit(x) => x,
                    Probe::Miss => return Ok(Probe::Miss),
                };
                handle = key_claim.handle;
                offset = key_claim.offset;

                let (h, raw) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                let val_sub = ChunkedCborSubDeserializer::new(h, offset, raw);
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
                    let (h, raw) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                    let key_sub = ChunkedCborSubDeserializer::new(h, offset, raw);
                    let (key_claim, key) = match CborValue::deserialize_owned(key_sub, ()).await? {
                        Probe::Hit(x) => x,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    handle = key_claim.handle;
                    offset = key_claim.offset;

                    let (h, raw) = crate::chunked::next_dispatch(handle, &mut offset).await?;
                    let val_sub = ChunkedCborSubDeserializer::new(h, offset, raw);
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
                remaining_after: None,
            },
            CborValue::Map(pairs),
        )))
    })
    .await
}

impl_value_owned!(ChunkedCborDeserializer<'s, B, F>);
impl_value_owned!(ChunkedCborSubDeserializer<'s, B, F>);
