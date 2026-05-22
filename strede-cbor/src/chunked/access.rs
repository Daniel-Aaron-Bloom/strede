//! Owned-family accessor types for the chunked CBOR deserializer.

use super::{
    ChunkedCborClaim, ChunkedCborSubDeserializer, next_dispatch, refill, skip_value_chunked,
    strip_tags_dispatch,
};
use crate::CborError;
use crate::token::CborToken;
use core::future::Future;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeOwned, Handle, MapAccessOwned, MapArmStackOwned,
    MapKeyClaimOwned, MapKeyProbeOwned, MapValueClaimOwned, MapValueProbeOwned, NextKey, Probe,
    SeqAccessOwned, SeqEntryOwned, StrAccessOwned, hit, utils::repeat,
};

// ---------------------------------------------------------------------------
// StrAccess / BytesAccess
// ---------------------------------------------------------------------------

pub enum ChunkedStrState {
    Definite { remaining: usize },
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

    fn fork(&mut self) -> Self {
        let state = match &self.state {
            ChunkedStrState::Definite { remaining } => ChunkedStrState::Definite {
                remaining: *remaining,
            },
            ChunkedStrState::Indefinite => ChunkedStrState::Indefinite,
        };
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            state,
        }
    }

    async fn next_str<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.state {
            ChunkedStrState::Definite { remaining: 0 } => Ok(Chunk::Done(ChunkedCborClaim {
                offset: self.offset,
                handle: self.handle,
                remaining_after: 0,
            })),
            ChunkedStrState::Definite { remaining } => loop {
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
                    self.state = ChunkedStrState::Definite {
                        remaining: remaining - take,
                    };
                    return Ok(Chunk::Data((self, r)));
                }
                self.handle = refill(self.handle, &mut self.offset).await?;
            },
            ChunkedStrState::Indefinite => {
                // Read next chunk header
                let (handle, chunk_tok) = next_dispatch(self.handle, &mut self.offset).await?;
                self.handle = handle;
                match chunk_tok {
                    CborToken::Break => Ok(Chunk::Done(ChunkedCborClaim {
                        offset: self.offset,
                        handle: self.handle,
                        remaining_after: 0,
                    })),
                    CborToken::Tstr(0) => {
                        // Zero-length chunk; stay in Indefinite state
                        let r = f("");
                        Ok(Chunk::Data((self, r)))
                    }
                    CborToken::Tstr(len) => {
                        // Delegate to Definite state for this chunk's bytes
                        self.state = ChunkedStrState::Definite { remaining: len };
                        match self.next_str(f).await? {
                            Chunk::Data((mut next, r)) => {
                                // After this definite chunk is fully consumed, go back to Indefinite
                                if let ChunkedStrState::Definite { remaining: 0 } = next.state {
                                    next.state = ChunkedStrState::Indefinite;
                                }
                                Ok(Chunk::Data((next, r)))
                            }
                            Chunk::Done(_claim) => {
                                // Should not happen: Definite{0} would have been caught above
                                unreachable!("zero-length Tstr handled separately")
                            }
                        }
                    }
                    _ => Err(CborError::UnexpectedByte { byte: 0 }),
                }
            }
        }
    }
}

pub enum ChunkedBytesState {
    Definite { remaining: usize },
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

    fn fork(&mut self) -> Self {
        let state = match &self.state {
            ChunkedBytesState::Definite { remaining } => ChunkedBytesState::Definite {
                remaining: *remaining,
            },
            ChunkedBytesState::Indefinite => ChunkedBytesState::Indefinite,
        };
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            state,
        }
    }

    async fn next_bytes<R>(
        mut self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.state {
            ChunkedBytesState::Definite { remaining: 0 } => Ok(Chunk::Done(ChunkedCborClaim {
                offset: self.offset,
                handle: self.handle,
                remaining_after: 0,
            })),
            ChunkedBytesState::Definite { remaining } => loop {
                let avail = self.handle.buf().len() - self.offset;
                if avail > 0 {
                    let take = remaining.min(avail);
                    let start = self.offset;
                    let end = start + take;
                    let r = f(&self.handle.buf()[start..end]);
                    self.offset += take;
                    self.state = ChunkedBytesState::Definite {
                        remaining: remaining - take,
                    };
                    return Ok(Chunk::Data((self, r)));
                }
                self.handle = refill(self.handle, &mut self.offset).await?;
            },
            ChunkedBytesState::Indefinite => {
                let (handle, chunk_tok) = next_dispatch(self.handle, &mut self.offset).await?;
                self.handle = handle;
                match chunk_tok {
                    CborToken::Break => Ok(Chunk::Done(ChunkedCborClaim {
                        offset: self.offset,
                        handle: self.handle,
                        remaining_after: 0,
                    })),
                    CborToken::Bstr(len) => {
                        self.state = ChunkedBytesState::Definite { remaining: len };
                        self.next_bytes(f).await
                    }
                    _ => Err(CborError::UnexpectedByte { byte: 0 }),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MapAccess
// ---------------------------------------------------------------------------

pub struct ChunkedCborMapAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    /// `Some(n)` = definite with n pairs remaining, `None` = indefinite
    pub(super) remaining: Option<usize>,
}

pub struct ChunkedCborKeyProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    key_tok: CborToken,
    offset: usize,
    remaining_after: usize,
}

pub struct ChunkedCborValueProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    value_tok: CborToken,
    offset: usize,
    remaining_after: usize,
}

// MapKeyClaimOwned and MapValueClaimOwned on ChunkedCborClaim

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyClaimOwned for ChunkedCborClaim<'s, B, F> {
    type Error = CborError;
    type MapClaim = ChunkedCborClaim<'s, B, F>;
    type ValueProbe = ChunkedCborValueProbe<'s, B, F>;

    async fn into_value_probe(mut self) -> Result<Self::ValueProbe, Self::Error> {
        let (handle, raw) = next_dispatch(self.handle, &mut self.offset).await?;
        let (handle, value_tok) = strip_tags_dispatch(handle, &mut self.offset, raw).await?;
        Ok(ChunkedCborValueProbe {
            handle,
            value_tok,
            offset: self.offset,
            remaining_after: self.remaining_after,
        })
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueClaimOwned for ChunkedCborClaim<'s, B, F> {
    type Error = CborError;
    type KeyProbe = ChunkedCborKeyProbe<'s, B, F>;
    type MapClaim = ChunkedCborClaim<'s, B, F>;

    async fn next_key(
        mut self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        match self.remaining_after {
            0 => Ok(NextKey::Done(ChunkedCborClaim {
                offset: self.offset,
                handle: self.handle,
                remaining_after: 0,
            })),
            usize::MAX => {
                // Indefinite: check for break
                let (handle, tok) = next_dispatch(self.handle, &mut self.offset).await?;
                self.handle = handle;
                if matches!(tok, CborToken::Break) {
                    return Ok(NextKey::Done(ChunkedCborClaim {
                        offset: self.offset,
                        handle: self.handle,
                        remaining_after: 0,
                    }));
                }
                let (handle, key_tok) =
                    strip_tags_dispatch(self.handle, &mut self.offset, tok).await?;
                Ok(NextKey::Entry(ChunkedCborKeyProbe {
                    handle,
                    key_tok,
                    offset: self.offset,
                    remaining_after: usize::MAX,
                }))
            }
            n => {
                let (handle, raw) = next_dispatch(self.handle, &mut self.offset).await?;
                let (handle, key_tok) = strip_tags_dispatch(handle, &mut self.offset, raw).await?;
                Ok(NextKey::Entry(ChunkedCborKeyProbe {
                    handle,
                    key_tok,
                    offset: self.offset,
                    remaining_after: n - 1,
                }))
            }
        }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyProbeOwned for ChunkedCborKeyProbe<'s, B, F> {
    type Error = CborError;
    type KeyClaim = ChunkedCborClaim<'s, B, F>;
    type KeySubDeserializer = ChunkedCborSubDeserializer<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            key_tok: self.key_tok,
            offset: self.offset,
            remaining_after: self.remaining_after,
        }
    }

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: DeserializeOwned<Self::KeySubDeserializer>,
    {
        let sub = ChunkedCborSubDeserializer::new(self.handle, self.offset, self.key_tok);
        match K::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, k)) => Ok(Probe::Hit((
                ChunkedCborClaim {
                    handle: claim.handle,
                    offset: claim.offset,
                    remaining_after: self.remaining_after,
                },
                k,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueProbeOwned for ChunkedCborValueProbe<'s, B, F> {
    type Error = CborError;
    type MapClaim = ChunkedCborClaim<'s, B, F>;
    type ValueClaim = ChunkedCborClaim<'s, B, F>;
    type ValueSubDeserializer = ChunkedCborSubDeserializer<'s, B, F>;
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            value_tok: self.value_tok,
            offset: self.offset,
            remaining_after: self.remaining_after,
        }
    }

    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeOwned<Self::ValueSubDeserializer>,
    {
        let sub = ChunkedCborSubDeserializer::new(self.handle, self.offset, self.value_tok);
        match V::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((
                ChunkedCborClaim {
                    handle: claim.handle,
                    offset: claim.offset,
                    remaining_after: self.remaining_after,
                },
                v,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let mut offset = self.offset;
        let handle = skip_value_chunked(self.handle, &mut offset, self.value_tok).await?;
        Ok(ChunkedCborClaim {
            handle,
            offset,
            remaining_after: self.remaining_after,
        })
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapAccessOwned for ChunkedCborMapAccess<'s, B, F> {
    type Error = CborError;
    type MapClaim = ChunkedCborClaim<'s, B, F>;
    type KeyProbe = ChunkedCborKeyProbe<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            remaining: self.remaining,
        }
    }

    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        mut self,
        mut arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        // Build first key probe or return early for empty map
        let mut key_probe_opt: Option<ChunkedCborKeyProbe<'s, B, F>> = match self.remaining {
            Some(0) => {
                return Ok(Probe::Hit((
                    ChunkedCborClaim {
                        offset: self.offset,
                        handle: self.handle,
                        remaining_after: 0,
                    },
                    arms.take_outputs(),
                )));
            }
            Some(n) => {
                let (handle, raw) = next_dispatch(self.handle, &mut self.offset).await?;
                let (handle, key_tok) = strip_tags_dispatch(handle, &mut self.offset, raw).await?;
                self.remaining = Some(n - 1);
                Some(ChunkedCborKeyProbe {
                    handle,
                    key_tok,
                    offset: self.offset,
                    remaining_after: n - 1,
                })
            }
            None => {
                // Indefinite: check for immediate break
                let (handle, tok) = next_dispatch(self.handle, &mut self.offset).await?;
                self.handle = handle;
                if matches!(tok, CborToken::Break) {
                    return Ok(Probe::Hit((
                        ChunkedCborClaim {
                            offset: self.offset,
                            handle: self.handle,
                            remaining_after: 0,
                        },
                        arms.take_outputs(),
                    )));
                }
                let (handle, key_tok) =
                    strip_tags_dispatch(self.handle, &mut self.offset, tok).await?;
                Some(ChunkedCborKeyProbe {
                    handle,
                    key_tok,
                    offset: self.offset,
                    remaining_after: usize::MAX,
                })
            }
        };

        loop {
            let key_probe = key_probe_opt.take().unwrap();

            let (arm_index, key_claim) = match arms.race_keys(key_probe).await? {
                Probe::Miss => return Ok(Probe::Miss),
                Probe::Hit(x) => x,
            };

            let value_probe = key_claim.into_value_probe().await?;

            let (value_claim, ()) = match arms.dispatch_value(arm_index, value_probe).await? {
                Probe::Miss => return Ok(Probe::Miss),
                Probe::Hit(x) => x,
            };

            match value_claim
                .next_key(arms.unsatisfied_count(), arms.open_count())
                .await?
            {
                NextKey::Done(map_claim) => {
                    return Ok(Probe::Hit((map_claim, arms.take_outputs())));
                }
                NextKey::Entry(next_kp) => {
                    key_probe_opt = Some(next_kp);
                }
            }
        }
    }
}

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
            remaining_after: 0,
        }))),
        Some(n) => {
            let (handle, raw) = next_dispatch(seq.handle, &mut seq.offset).await?;
            let (handle, elem_tok) = strip_tags_dispatch(handle, &mut seq.offset, raw).await?;
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
                    remaining_after: 0,
                })));
            }
            let (handle, elem_tok) = strip_tags_dispatch(seq.handle, &mut seq.offset, tok).await?;
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
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqAccessOwned for ChunkedCborSeqAccess<'s, B, F> {
    type Error = CborError;
    type SeqClaim = ChunkedCborClaim<'s, B, F>;
    type ElemClaim = ChunkedCborClaim<'s, B, F>;
    type Elem = ChunkedCborSeqEntry<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            remaining: self.remaining,
        }
    }

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
            remaining_after: 0,
        })
    }
}
