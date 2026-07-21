//! Owned-family accessor types for the chunked msgpack deserializer.

use super::{
    ChunkedMsgpackClaim, ChunkedMsgpackSubDeserializer, next_dispatch, refill, skip_value_chunked,
};
use crate::MsgpackError;
use crate::token::MsgpackToken;
use core::future::Future;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeOwned, Handle, Probe, SeqAccessOwned,
    SeqEntryOwned, StrAccessOwned, hit, utils::repeat,
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
// MapAccess — see `strede::PairSeqMapAccess` (generic over `RawSlot`,
// implemented for `ChunkedMsgpackClaim` in `mod.rs`). Msgpack maps are
// wire-identical to "N pairs in a flat, count-prefixed stream", so the
// key/value probe quintet is generic infrastructure rather than hand-rolled.
// ---------------------------------------------------------------------------

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
