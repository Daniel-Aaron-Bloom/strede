//! Owned-family accessor types for the chunked msgpack deserializer.

use super::{
    ChunkedMsgpackClaim, ChunkedMsgpackSubDeserializer, next_dispatch, refill, skip_value_chunked,
};
use crate::MsgpackError;
use crate::token::MsgpackToken;
use core::future::Future;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeFromMapOwned, DeserializeFromSeqOwned,
    DeserializeOwned, Handle, MapAccessOwned, MapArmStackOwned, MapKeyClaimOwned, MapKeyProbeOwned,
    MapValueClaimOwned, MapValueProbeOwned, NextKey, Probe, SeqAccessOwned, SeqEntryOwned,
    StrAccessOwned, hit, utils::repeat,
};

// ---------------------------------------------------------------------------
// StrAccess / BytesAccess
// ---------------------------------------------------------------------------

pub struct ChunkedMsgpackStrAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> StrAccessOwned for ChunkedMsgpackStrAccess<'s, B, F> {
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type Error = MsgpackError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            remaining: self.remaining,
        }
    }

    async fn next_str<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(ChunkedMsgpackClaim {
                offset: self.offset,
                handle: self.handle,
                remaining_after: 0,
            }));
        }
        loop {
            let avail = self.handle.buf().len() - self.offset;
            if avail > 0 {
                let take = self.remaining.min(avail);
                let start = self.offset;
                let end = start + take;
                let s = {
                    let buf = self.handle.buf();
                    core::str::from_utf8(&buf[start..end]).map_err(|_| MsgpackError::InvalidUtf8)?
                };
                let r = f(s);
                self.offset += take;
                self.remaining -= take;
                return Ok(Chunk::Data((self, r)));
            }
            self.handle = refill(self.handle, &mut self.offset).await?;
        }
    }
}

pub struct ChunkedMsgpackBytesAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedMsgpackBytesAccess<'s, B, F> {
    pub(crate) fn new(handle: Handle<'s, B, F>, offset: usize, remaining: usize) -> Self {
        Self {
            handle,
            offset,
            remaining,
        }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> BytesAccessOwned
    for ChunkedMsgpackBytesAccess<'s, B, F>
{
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type Error = MsgpackError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            remaining: self.remaining,
        }
    }

    async fn next_bytes<R>(
        mut self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(ChunkedMsgpackClaim {
                offset: self.offset,
                handle: self.handle,
                remaining_after: 0,
            }));
        }
        loop {
            let avail = self.handle.buf().len() - self.offset;
            if avail > 0 {
                let take = self.remaining.min(avail);
                let start = self.offset;
                let end = start + take;
                let r = f(&self.handle.buf()[start..end]);
                self.offset += take;
                self.remaining -= take;
                return Ok(Chunk::Data((self, r)));
            }
            self.handle = refill(self.handle, &mut self.offset).await?;
        }
    }
}

// ---------------------------------------------------------------------------
// MapAccess
// ---------------------------------------------------------------------------

pub struct ChunkedMsgpackMapAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
}

pub struct ChunkedMsgpackKeyProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    key_tok: MsgpackToken,
    offset: usize,
    remaining_after: usize,
}

pub struct ChunkedMsgpackValueProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    value_tok: MsgpackToken,
    offset: usize,
    remaining_after: usize,
}

// ---------------------------------------------------------------------------
// MapKeyClaimOwned and MapValueClaimOwned on ChunkedMsgpackClaim
//
// ChunkedMsgpackClaim serves as both KeyClaim and ValueClaim.
// remaining_after carries the number of map pairs still to iterate.
// ---------------------------------------------------------------------------

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyClaimOwned for ChunkedMsgpackClaim<'s, B, F> {
    type Error = MsgpackError;
    type MapClaim = ChunkedMsgpackClaim<'s, B, F>;
    type ValueProbe = ChunkedMsgpackValueProbe<'s, B, F>;

    async fn into_value_probe(mut self) -> Result<Self::ValueProbe, Self::Error> {
        let (handle, value_tok) = next_dispatch(self.handle, &mut self.offset).await?;
        Ok(ChunkedMsgpackValueProbe {
            handle,
            value_tok,
            offset: self.offset,
            remaining_after: self.remaining_after,
        })
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueClaimOwned for ChunkedMsgpackClaim<'s, B, F> {
    type Error = MsgpackError;
    type KeyProbe = ChunkedMsgpackKeyProbe<'s, B, F>;
    type MapClaim = ChunkedMsgpackClaim<'s, B, F>;

    async fn next_key(
        mut self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        if self.remaining_after == 0 {
            return Ok(NextKey::Done(ChunkedMsgpackClaim {
                offset: self.offset,
                handle: self.handle,
                remaining_after: 0,
            }));
        }
        let (handle, key_tok) = next_dispatch(self.handle, &mut self.offset).await?;
        Ok(NextKey::Entry(ChunkedMsgpackKeyProbe {
            handle,
            key_tok,
            offset: self.offset,
            remaining_after: self.remaining_after - 1,
        }))
    }
}

// --- MapKeyProbeOwned ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyProbeOwned for ChunkedMsgpackKeyProbe<'s, B, F> {
    type Error = MsgpackError;
    type KeyClaim = ChunkedMsgpackClaim<'s, B, F>;
    type KeySubDeserializer = ChunkedMsgpackSubDeserializer<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            key_tok: self.key_tok,
            offset: self.offset,
            remaining_after: self.remaining_after,
        }
    }

    #[inline(always)]
    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: DeserializeOwned<Self::KeySubDeserializer>,
    {
        let sub = ChunkedMsgpackSubDeserializer::new(self.handle, self.offset, self.key_tok);
        match K::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, k)) => Ok(Probe::Hit((
                ChunkedMsgpackClaim {
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

// --- MapValueProbeOwned ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueProbeOwned
    for ChunkedMsgpackValueProbe<'s, B, F>
{
    type Error = MsgpackError;
    type MapClaim = ChunkedMsgpackClaim<'s, B, F>;
    type ValueClaim = ChunkedMsgpackClaim<'s, B, F>;
    type ValueSubDeserializer = ChunkedMsgpackSubDeserializer<'s, B, F>;
    type ValueMap = ChunkedMsgpackMapAccess<'s, B, F>;
    type ValueSeq = ChunkedMsgpackSeqAccess<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            value_tok: self.value_tok,
            offset: self.offset,
            remaining_after: self.remaining_after,
        }
    }

    #[inline(always)]
    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeOwned<Self::ValueSubDeserializer>,
    {
        let sub = ChunkedMsgpackSubDeserializer::new(self.handle, self.offset, self.value_tok);
        match V::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((
                ChunkedMsgpackClaim {
                    handle: claim.handle,
                    offset: claim.offset,
                    remaining_after: self.remaining_after,
                },
                v,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_map_into<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeFromMapOwned<Self::ValueMap>,
    {
        match self.value_tok {
            MsgpackToken::Map(count) => {
                let map = ChunkedMsgpackMapAccess {
                    handle: self.handle,
                    offset: self.offset,
                    remaining: count,
                };
                match V::deserialize_from_map_owned(map, extra).await? {
                    Probe::Hit((claim, v)) => Ok(Probe::Hit((
                        ChunkedMsgpackClaim {
                            handle: claim.handle,
                            offset: claim.offset,
                            remaining_after: self.remaining_after,
                        },
                        v,
                    ))),
                    Probe::Miss => Ok(Probe::Miss),
                }
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_seq_into<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeFromSeqOwned<Self::ValueSeq>,
    {
        match self.value_tok {
            MsgpackToken::Array(count) => {
                let seq = ChunkedMsgpackSeqAccess {
                    handle: self.handle,
                    offset: self.offset,
                    remaining: count,
                    first: true,
                };
                match V::deserialize_from_seq_owned(seq, extra).await? {
                    Probe::Hit((claim, v)) => Ok(Probe::Hit((
                        ChunkedMsgpackClaim {
                            handle: claim.handle,
                            offset: claim.offset,
                            remaining_after: self.remaining_after,
                        },
                        v,
                    ))),
                    Probe::Miss => Ok(Probe::Miss),
                }
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let mut offset = self.offset;
        let handle = skip_value_chunked(self.handle, &mut offset, self.value_tok).await?;
        Ok(ChunkedMsgpackClaim {
            handle,
            offset,
            remaining_after: self.remaining_after,
        })
    }
}

// --- MapAccessOwned ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapAccessOwned for ChunkedMsgpackMapAccess<'s, B, F> {
    type Error = MsgpackError;
    type MapClaim = ChunkedMsgpackClaim<'s, B, F>;
    type KeyProbe = ChunkedMsgpackKeyProbe<'s, B, F>;

    #[inline(always)]
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
        if self.remaining == 0 {
            return Ok(Probe::Hit((
                ChunkedMsgpackClaim {
                    offset: self.offset,
                    handle: self.handle,
                    remaining_after: 0,
                },
                arms.take_outputs(),
            )));
        }

        let (handle, key_tok) = next_dispatch(self.handle, &mut self.offset).await?;
        self.remaining -= 1;
        let mut key_probe_opt: Option<ChunkedMsgpackKeyProbe<'s, B, F>> =
            Some(ChunkedMsgpackKeyProbe {
                handle,
                key_tok,
                offset: self.offset,
                remaining_after: self.remaining,
            });

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

pub struct ChunkedMsgpackSeqAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
    pub(super) first: bool,
}

pub struct ChunkedMsgpackSeqEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    elem_tok: MsgpackToken,
    offset: usize,
}

async fn seq_next_chunked<'s, B, F, const N: usize, Fn_, Fut, R>(
    mut seq: ChunkedMsgpackSeqAccess<'s, B, F>,
    mut f: Fn_,
) -> Result<
    Probe<Chunk<(ChunkedMsgpackSeqAccess<'s, B, F>, R), ChunkedMsgpackClaim<'s, B, F>>>,
    MsgpackError,
>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedMsgpackSeqEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedMsgpackClaim<'s, B, F>, R)>, MsgpackError>>,
{
    if seq.remaining == 0 {
        return Ok(Probe::Hit(Chunk::Done(ChunkedMsgpackClaim {
            offset: seq.offset,
            handle: seq.handle,
            remaining_after: 0,
        })));
    }

    let (handle, elem_tok) = next_dispatch(seq.handle, &mut seq.offset).await?;
    seq.remaining -= 1;
    let snap_offset = seq.offset;

    let entries: [ChunkedMsgpackSeqEntry<'s, B, F>; N] =
        repeat(handle, Handle::fork).map(|handle| ChunkedMsgpackSeqEntry {
            handle,
            elem_tok,
            offset: snap_offset,
        });

    let (elem_claim, r) = hit!(f(entries).await);
    seq.offset = elem_claim.offset;
    seq.handle = elem_claim.handle;
    seq.first = false;
    Ok(Probe::Hit(Chunk::Data((seq, r))))
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqAccessOwned for ChunkedMsgpackSeqAccess<'s, B, F> {
    type Error = MsgpackError;
    type SeqClaim = ChunkedMsgpackClaim<'s, B, F>;
    type ElemClaim = ChunkedMsgpackClaim<'s, B, F>;
    type Elem = ChunkedMsgpackSeqEntry<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            remaining: self.remaining,
            first: self.first,
        }
    }

    #[inline(always)]
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

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqEntryOwned for ChunkedMsgpackSeqEntry<'s, B, F> {
    type Error = MsgpackError;
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type SubDeserializer = ChunkedMsgpackSubDeserializer<'s, B, F>;
    type Map = ChunkedMsgpackMapAccess<'s, B, F>;
    type Seq = ChunkedMsgpackSeqAccess<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            elem_tok: self.elem_tok,
            offset: self.offset,
        }
    }

    #[inline(always)]
    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let sub = ChunkedMsgpackSubDeserializer::new(self.handle, self.offset, self.elem_tok);
        T::deserialize_owned(sub, extra).await
    }

    #[inline(always)]
    async fn get_map_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMapOwned<Self::Map>,
    {
        match self.elem_tok {
            MsgpackToken::Map(count) => {
                let map = ChunkedMsgpackMapAccess {
                    handle: self.handle,
                    offset: self.offset,
                    remaining: count,
                };
                T::deserialize_from_map_owned(map, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn get_seq_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeqOwned<Self::Seq>,
    {
        match self.elem_tok {
            MsgpackToken::Array(count) => {
                let seq = ChunkedMsgpackSeqAccess {
                    handle: self.handle,
                    offset: self.offset,
                    remaining: count,
                    first: true,
                };
                T::deserialize_from_seq_owned(seq, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut offset = self.offset;
        let handle = skip_value_chunked(self.handle, &mut offset, self.elem_tok).await?;
        Ok(ChunkedMsgpackClaim {
            offset,
            handle,
            remaining_after: 0,
        })
    }
}
