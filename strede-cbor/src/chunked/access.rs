//! Owned-family accessor types for the chunked CBOR deserializer.

use super::{
    ChunkedCborClaim, ChunkedCborSubDeserializer, next_dispatch, refill, skip_value_chunked,
};
use crate::CborError;
use crate::token::CborToken;
use core::future::Future;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeOwned, Handle, NumberAccessOwned, NumberEncoding,
    Probe, SeqAccessOwned, SeqEntryOwned, StrAccessOwned, hit, utils::repeat,
};

// ---------------------------------------------------------------------------
// StrAccess / BytesAccess
// ---------------------------------------------------------------------------

pub enum ChunkedStrState {
    Definite {
        remaining: usize,
    },
    /// A definite-length sub-chunk inside an indefinite string; when remaining
    /// hits 0, transitions back to `Indefinite` instead of returning `Done`.
    DefiniteInsideIndef {
        remaining: usize,
    },
    Indefinite,
}

pub struct ChunkedCborStrAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) state: ChunkedStrState,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> StrAccessOwned for ChunkedCborStrAccess<'s, B, F> {
    type Claim = ChunkedCborClaim<'s, B, F>;
    type Error = CborError;

    async fn next_str<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        loop {
            match self.state {
                ChunkedStrState::Definite { remaining: 0 } => {
                    return Ok(Chunk::Done(ChunkedCborClaim {
                        offset: self.offset,
                        handle: self.handle,
                        remaining_after: None,
                    }));
                }
                ChunkedStrState::DefiniteInsideIndef { remaining: 0 } => {
                    self.state = ChunkedStrState::Indefinite;
                    // loop: read the next indefinite-string chunk header
                }
                ChunkedStrState::Definite { remaining }
                | ChunkedStrState::DefiniteInsideIndef { remaining } => {
                    let avail = self.handle.buf().len() - self.offset;
                    if avail > 0 {
                        let take = remaining.min(avail);
                        let start = self.offset;
                        let end = start + take;
                        let s = {
                            let buf = self.handle.buf();
                            core::str::from_utf8(&buf[start..end])
                                .map_err(|_| CborError::InvalidUtf8)?
                        };
                        let r = f(s);
                        self.offset += take;
                        self.state = match self.state {
                            ChunkedStrState::Definite { .. } => ChunkedStrState::Definite {
                                remaining: remaining - take,
                            },
                            _ => ChunkedStrState::DefiniteInsideIndef {
                                remaining: remaining - take,
                            },
                        };
                        return Ok(Chunk::Data((self, r)));
                    }
                    self.handle = refill(self.handle, &mut self.offset).await?;
                    // loop to retry with fresh data
                }
                ChunkedStrState::Indefinite => {
                    let (handle, chunk_tok) = next_dispatch(self.handle, &mut self.offset).await?;
                    self.handle = handle;
                    match chunk_tok {
                        CborToken::Break => {
                            return Ok(Chunk::Done(ChunkedCborClaim {
                                offset: self.offset,
                                handle: self.handle,
                                remaining_after: None,
                            }));
                        }
                        CborToken::Tstr(0) => {
                            let r = f("");
                            return Ok(Chunk::Data((self, r)));
                        }
                        CborToken::Tstr(len) => {
                            self.state = ChunkedStrState::DefiniteInsideIndef { remaining: len };
                            // loop: Definite branch will yield the bytes
                        }
                        _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                    }
                }
            }
        }
    }
}

pub enum ChunkedBytesState {
    Definite {
        remaining: usize,
    },
    /// A definite-length sub-chunk inside an indefinite bytes sequence; when
    /// remaining hits 0, transitions back to `Indefinite` instead of `Done`.
    DefiniteInsideIndef {
        remaining: usize,
    },
    Indefinite,
}

pub struct ChunkedCborBytesAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) state: ChunkedBytesState,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> BytesAccessOwned for ChunkedCborBytesAccess<'s, B, F> {
    type Claim = ChunkedCborClaim<'s, B, F>;
    type Error = CborError;

    async fn next_bytes<R>(
        mut self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        loop {
            match self.state {
                ChunkedBytesState::Definite { remaining: 0 } => {
                    return Ok(Chunk::Done(ChunkedCborClaim {
                        offset: self.offset,
                        handle: self.handle,
                        remaining_after: None,
                    }));
                }
                ChunkedBytesState::DefiniteInsideIndef { remaining: 0 } => {
                    self.state = ChunkedBytesState::Indefinite;
                    // loop: read the next indefinite-bytes chunk header
                }
                ChunkedBytesState::Definite { remaining }
                | ChunkedBytesState::DefiniteInsideIndef { remaining } => {
                    let avail = self.handle.buf().len() - self.offset;
                    if avail > 0 {
                        let take = remaining.min(avail);
                        let start = self.offset;
                        let end = start + take;
                        let r = f(&self.handle.buf()[start..end]);
                        self.offset += take;
                        self.state = match self.state {
                            ChunkedBytesState::Definite { .. } => ChunkedBytesState::Definite {
                                remaining: remaining - take,
                            },
                            _ => ChunkedBytesState::DefiniteInsideIndef {
                                remaining: remaining - take,
                            },
                        };
                        return Ok(Chunk::Data((self, r)));
                    }
                    self.handle = refill(self.handle, &mut self.offset).await?;
                    // loop to retry with fresh data
                }
                ChunkedBytesState::Indefinite => {
                    let (handle, chunk_tok) = next_dispatch(self.handle, &mut self.offset).await?;
                    self.handle = handle;
                    match chunk_tok {
                        CborToken::Break => {
                            return Ok(Chunk::Done(ChunkedCborClaim {
                                offset: self.offset,
                                handle: self.handle,
                                remaining_after: None,
                            }));
                        }
                        CborToken::Bstr(len) => {
                            self.state = ChunkedBytesState::DefiniteInsideIndef { remaining: len };
                            // loop: Definite branch will yield the bytes
                        }
                        _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ChunkedCborNumberAccess — owned-family number chunk accessor (BigEndian bignums)
// ---------------------------------------------------------------------------

pub struct ChunkedCborNumberAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) state: ChunkedBytesState,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B), Enc: NumberEncoding> NumberAccessOwned<Enc>
    for ChunkedCborNumberAccess<'s, B, F>
{
    type Claim = ChunkedCborClaim<'s, B, F>;
    type Error = CborError;

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        // Only reachable when Enc == BigEndian; other encodings never produce this accessor.
        loop {
            match self.state {
                ChunkedBytesState::Definite { remaining: 0 } => {
                    return Ok(Chunk::Done(ChunkedCborClaim {
                        offset: self.offset,
                        handle: self.handle,
                        remaining_after: None,
                    }));
                }
                ChunkedBytesState::DefiniteInsideIndef { remaining: 0 } => {
                    self.state = ChunkedBytesState::Indefinite;
                }
                ChunkedBytesState::Definite { remaining }
                | ChunkedBytesState::DefiniteInsideIndef { remaining } => {
                    let avail = self.handle.buf().len() - self.offset;
                    if avail > 0 {
                        let take = remaining.min(avail);
                        let start = self.offset;
                        let end = start + take;
                        let r = f(Enc::from_bytes(&self.handle.buf()[start..end]));
                        self.offset += take;
                        self.state = match self.state {
                            ChunkedBytesState::Definite { .. } => ChunkedBytesState::Definite {
                                remaining: remaining - take,
                            },
                            _ => ChunkedBytesState::DefiniteInsideIndef {
                                remaining: remaining - take,
                            },
                        };
                        return Ok(Chunk::Data((self, r)));
                    }
                    self.handle = refill(self.handle, &mut self.offset).await?;
                }
                ChunkedBytesState::Indefinite => {
                    let (handle, chunk_tok) = next_dispatch(self.handle, &mut self.offset).await?;
                    self.handle = handle;
                    match chunk_tok {
                        CborToken::Break => {
                            return Ok(Chunk::Done(ChunkedCborClaim {
                                offset: self.offset,
                                handle: self.handle,
                                remaining_after: None,
                            }));
                        }
                        CborToken::Bstr(len) => {
                            self.state = ChunkedBytesState::DefiniteInsideIndef { remaining: len };
                        }
                        _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MapAccess — see `strede::PairSeqMapAccess` (generic over `RawSlot`,
// implemented for `ChunkedCborClaim` in `mod.rs`). CBOR maps (definite or
// `Break`-terminated indefinite) are wire-identical to "N pairs in a flat
// stream", so the key/value probe quintet is generic infrastructure rather
// than hand-rolled here.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SeqAccess / SeqEntry
// ---------------------------------------------------------------------------

pub struct ChunkedCborSeqAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    /// `Some(n)` = definite, `None` = indefinite
    pub(crate) remaining: Option<usize>,
}

pub struct ChunkedCborSeqEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    /// The token as read — *not* stripped of leading tags. See the
    /// module-level note in `chunked/mod.rs` on `RawSlot`/
    /// `strip_tags_bignum_dispatch`.
    pub(crate) elem_tok: CborToken,
    pub(crate) offset: usize,
}

async fn seq_next_chunked<'s, B, F, const N: usize, Fn_, Fut, R>(
    mut seq: ChunkedCborSeqAccess<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<Chunk<(ChunkedCborSeqAccess<'s, B, F>, R), ChunkedCborClaim<'s, B, F>>>, CborError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedCborSeqEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedCborClaim<'s, B, F>, R)>, CborError>>,
{
    match seq.remaining {
        Some(0) => Ok(Probe::Hit(Chunk::Done(ChunkedCborClaim {
            offset: seq.offset,
            handle: seq.handle,
            remaining_after: None,
        }))),
        Some(n) => {
            let (handle, elem_tok) = next_dispatch(seq.handle, &mut seq.offset).await?;
            seq.remaining = Some(n - 1);
            let snap_offset = seq.offset;

            let entries: [ChunkedCborSeqEntry<'s, B, F>; N] =
                repeat(handle, Handle::fork).map(|handle| ChunkedCborSeqEntry {
                    handle,
                    elem_tok,
                    offset: snap_offset,
                });

            let (elem_claim, r) = hit!(f(entries).await);
            seq.offset = elem_claim.offset;
            seq.handle = elem_claim.handle;
            Ok(Probe::Hit(Chunk::Data((seq, r))))
        }
        None => {
            // Indefinite: check for break
            let (handle, tok) = next_dispatch(seq.handle, &mut seq.offset).await?;
            seq.handle = handle;
            if matches!(tok, CborToken::Break) {
                return Ok(Probe::Hit(Chunk::Done(ChunkedCborClaim {
                    offset: seq.offset,
                    handle: seq.handle,
                    remaining_after: None,
                })));
            }
            let snap_offset = seq.offset;

            let entries: [ChunkedCborSeqEntry<'s, B, F>; N] =
                repeat(seq.handle, Handle::fork).map(|handle| ChunkedCborSeqEntry {
                    handle,
                    elem_tok: tok,
                    offset: snap_offset,
                });

            let (elem_claim, r) = hit!(f(entries).await);
            seq.offset = elem_claim.offset;
            seq.handle = elem_claim.handle;
            Ok(Probe::Hit(Chunk::Data((seq, r))))
        }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqAccessOwned for ChunkedCborSeqAccess<'s, B, F> {
    type Error = CborError;
    type SeqClaim = ChunkedCborClaim<'s, B, F>;
    type ElemClaim = ChunkedCborClaim<'s, B, F>;
    type Elem = ChunkedCborSeqEntry<'s, B, F>;

    async fn next<const N: usize, Fn_, Fut, R>(
        self,
        f: Fn_,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        Fn_: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        seq_next_chunked(self, f).await
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqEntryOwned for ChunkedCborSeqEntry<'s, B, F> {
    type Error = CborError;
    type Claim = ChunkedCborClaim<'s, B, F>;
    type SubDeserializer = ChunkedCborSubDeserializer<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            elem_tok: self.elem_tok,
            offset: self.offset,
        }
    }

    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let sub = ChunkedCborSubDeserializer::new(self.handle, self.offset, self.elem_tok);
        T::deserialize_owned(sub, extra).await
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut offset = self.offset;
        let handle = skip_value_chunked(self.handle, &mut offset, self.elem_tok).await?;
        Ok(ChunkedCborClaim {
            offset,
            handle,
            remaining_after: None,
        })
    }
}
