//! Owned-family accessor types for the chunked bincode deserializer.

use core::future::Future;
use core::marker::PhantomData;

use super::{ChunkedBincodeClaim, ChunkedBincodeSubDeserializer, refill};
use crate::BincodeError;
use crate::config::BincodeConfig;
use strede::utils::repeat;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeOwned, Handle, MapKeyProbeOwned,
    MapValueProbeOwned, Probe, SeqAccessOwned, SeqEntryOwned, StrAccessOwned, hit,
};

// ---------------------------------------------------------------------------
// ChunkedBincodeStrAccess / ChunkedBincodeBytesAccess
// ---------------------------------------------------------------------------

pub struct ChunkedBincodeStrAccess<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
    pub(super) _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> StrAccessOwned
    for ChunkedBincodeStrAccess<'s, C, B, F>
{
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type Error = BincodeError;

    /// A chunk yielded here could in principle split mid-UTF-8-codepoint
    /// even though the overall string is valid — the same known,
    /// pre-existing limitation shared by every chunked format in this
    /// workspace (see `strede-postcard`'s identical comment).
    async fn next_str<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(ChunkedBincodeClaim::new(
                self.handle,
                self.offset,
            )));
        }
        loop {
            let avail = self.handle.buf().len() - self.offset;
            if avail > 0 {
                let take = self.remaining.min(avail);
                let start = self.offset;
                let end = start + take;
                let s = {
                    let buf = self.handle.buf();
                    core::str::from_utf8(&buf[start..end]).map_err(|_| BincodeError::InvalidUtf8)?
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

pub struct ChunkedBincodeBytesAccess<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
    pub(super) _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> BytesAccessOwned
    for ChunkedBincodeBytesAccess<'s, C, B, F>
{
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type Error = BincodeError;

    async fn next_bytes<R>(
        mut self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(ChunkedBincodeClaim::new(
                self.handle,
                self.offset,
            )));
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
// Map key/value probe chain
//
// Bincode structs have no wire keys and no wire field count, exactly like
// postcard. Fields are decoded positionally: field 0 first, then field 1,
// etc. `ChunkedBincodeClaim` serves as `KeyClaim`/`ValueClaim` (impls in
// `mod.rs`, alongside `ChunkedBincodeMapAccess`).
// ---------------------------------------------------------------------------

pub struct ChunkedBincodeMapKeyProbe<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) current_idx: usize,
    /// `true` for a dynamic-collection key slot (HashMap/BTreeMap via
    /// `CollectMap`), `false` for a struct field. See
    /// `strede-postcard`'s identical field for the full rationale.
    pub(crate) dynamic: bool,
    pub(crate) _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedBincodeMapKeyProbe<'s, C, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            current_idx: self.current_idx,
            dynamic: self.dynamic,
            _cfg: PhantomData,
        }
    }
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyProbeOwned
    for ChunkedBincodeMapKeyProbe<'s, C, B, F>
{
    type Error = BincodeError;
    type KeyClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type KeySubDeserializer = ChunkedBincodeSubDeserializer<'s, C, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: DeserializeOwned<Self::KeySubDeserializer>,
    {
        if !self.dynamic {
            return Ok(Probe::Miss);
        }
        let sub = ChunkedBincodeSubDeserializer::new(self.handle, self.offset);
        K::deserialize_owned(sub, extra).await
    }

    async fn deserialize_key_by_index(
        self,
        expected: usize,
    ) -> Result<Probe<(Self::KeyClaim, ())>, Self::Error> {
        if self.current_idx == expected {
            Ok(Probe::Hit((
                ChunkedBincodeClaim::new(self.handle, self.offset),
                (),
            )))
        } else {
            Ok(Probe::Miss)
        }
    }
}

pub struct ChunkedBincodeMapValueProbe<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)>
    ChunkedBincodeMapValueProbe<'s, C, B, F>
{
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            _cfg: PhantomData,
        }
    }
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> MapValueProbeOwned
    for ChunkedBincodeMapValueProbe<'s, C, B, F>
{
    type Error = BincodeError;
    type MapClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type ValueClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type ValueSubDeserializer = ChunkedBincodeSubDeserializer<'s, C, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeOwned<Self::ValueSubDeserializer>,
    {
        let sub = ChunkedBincodeSubDeserializer::new(self.handle, self.offset);
        V::deserialize_owned(sub, extra).await
    }

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        Err(BincodeError::CannotSkip)
    }
}

// ---------------------------------------------------------------------------
// SeqAccess / SeqEntry
// ---------------------------------------------------------------------------

pub struct ChunkedBincodeSeqAccess<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) remaining: usize,
    pub(crate) _cfg: PhantomData<C>,
}

pub struct ChunkedBincodeSeqEntry<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedBincodeSeqEntry<'s, C, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            _cfg: PhantomData,
        }
    }
}

#[inline(always)]
async fn bincode_seq_next<'s, C, B, F, const N: usize, Fn_, Fut, R>(
    mut seq: ChunkedBincodeSeqAccess<'s, C, B, F>,
    mut f: Fn_,
) -> Result<
    Probe<Chunk<(ChunkedBincodeSeqAccess<'s, C, B, F>, R), ChunkedBincodeClaim<'s, C, B, F>>>,
    BincodeError,
>
where
    C: BincodeConfig,
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedBincodeSeqEntry<'s, C, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, R)>, BincodeError>>,
{
    if seq.remaining == 0 {
        return Ok(Probe::Hit(Chunk::Done(ChunkedBincodeClaim::new(
            seq.handle,
            seq.offset,
        ))));
    }
    seq.remaining -= 1;
    let entries: [ChunkedBincodeSeqEntry<'s, C, B, F>; N] =
        repeat(seq.handle, Handle::fork).map(|handle| ChunkedBincodeSeqEntry {
            handle,
            offset: seq.offset,
            _cfg: PhantomData,
        });
    let (claim, r) = hit!(f(entries).await);
    seq.handle = claim.handle;
    seq.offset = claim.offset;
    Ok(Probe::Hit(Chunk::Data((seq, r))))
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> SeqAccessOwned
    for ChunkedBincodeSeqAccess<'s, C, B, F>
{
    type Error = BincodeError;
    type SeqClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type ElemClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type Elem = ChunkedBincodeSeqEntry<'s, C, B, F>;

    async fn next<const N: usize, Fn_, Fut, R>(
        self,
        f: Fn_,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        Fn_: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        bincode_seq_next(self, f).await
    }
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> SeqEntryOwned
    for ChunkedBincodeSeqEntry<'s, C, B, F>
{
    type Error = BincodeError;
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type SubDeserializer = ChunkedBincodeSubDeserializer<'s, C, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let sub = ChunkedBincodeSubDeserializer::new(self.handle, self.offset);
        T::deserialize_owned(sub, extra).await
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        Err(BincodeError::CannotSkip)
    }
}
