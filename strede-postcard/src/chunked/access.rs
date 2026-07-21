//! Owned-family accessor types for the chunked postcard deserializer.

use super::{ChunkedPostcardClaim, ChunkedPostcardSubDeserializer, refill};
use crate::PostcardError;
use core::future::Future;
use strede::utils::repeat;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeOwned, Handle, MapKeyProbeOwned,
    MapValueProbeOwned, Probe, SeqAccessOwned, SeqEntryOwned, StrAccessOwned, hit,
};

// ---------------------------------------------------------------------------
// PostcardStrAccess / PostcardBytesAccess
// ---------------------------------------------------------------------------

pub struct ChunkedPostcardStrAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> StrAccessOwned for ChunkedPostcardStrAccess<'s, B, F> {
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type Error = PostcardError;

    /// A chunk yielded here could in principle split mid-UTF-8-codepoint even
    /// though the overall string is valid - the same known, pre-existing
    /// limitation shared by every chunked format in this workspace (e.g.
    /// msgpack's `ChunkedMsgpackStrAccess`), not something new to postcard.
    async fn next_str<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(ChunkedPostcardClaim {
                handle: self.handle,
                offset: self.offset,
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
                    core::str::from_utf8(&buf[start..end]).map_err(|_| PostcardError::InvalidUtf8)?
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

pub struct ChunkedPostcardBytesAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) offset: usize,
    pub(super) remaining: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> BytesAccessOwned
    for ChunkedPostcardBytesAccess<'s, B, F>
{
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type Error = PostcardError;

    async fn next_bytes<R>(
        mut self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(ChunkedPostcardClaim {
                handle: self.handle,
                offset: self.offset,
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
// Map key/value probe chain
//
// Postcard structs have no wire keys and no wire field count. Fields are
// decoded positionally: field 0 first, then field 1, etc. The arm stack's
// key callbacks call `kp.deserialize_key_by_index(arm_idx)` which hits only
// when the probe's `current_idx` counter equals `arm_idx`.
//
// `ChunkedPostcardClaim` serves as `KeyClaim` and `ValueClaim` (impls in
// `mod.rs`, alongside `ChunkedPostcardMapAccess`). It carries the updated
// handle/offset so the next field probe starts from the right position.
// ---------------------------------------------------------------------------

pub struct ChunkedPostcardMapKeyProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) current_idx: usize,
    /// `true` for a dynamic-collection key slot (HashMap/BTreeMap via
    /// `CollectMap`), where the key is real wire bytes to be decoded via
    /// `deserialize_key`. `false` for a struct field, where fields have no
    /// wire key names at all and `deserialize_key` must stay a no-op `Miss`
    /// - dynamic-only, since the derive races `deserialize_key::<Match>`
    /// against `deserialize_key_by_index` for every struct field, and a real
    /// decode attempt there could misparse arbitrary field bytes.
    pub(crate) dynamic: bool,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedPostcardMapKeyProbe<'s, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            current_idx: self.current_idx,
            dynamic: self.dynamic,
        }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyProbeOwned for ChunkedPostcardMapKeyProbe<'s, B, F> {
    type Error = PostcardError;
    type KeyClaim = ChunkedPostcardClaim<'s, B, F>;
    type KeySubDeserializer = ChunkedPostcardSubDeserializer<'s, B, F>;

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
            // Struct fields have no wire key names - name-based matching
            // always misses.
            return Ok(Probe::Miss);
        }
        let sub = ChunkedPostcardSubDeserializer::new(self.handle, self.offset);
        K::deserialize_owned(sub, extra).await
    }

    async fn deserialize_key_by_index(
        self,
        expected: usize,
    ) -> Result<Probe<(Self::KeyClaim, ())>, Self::Error> {
        if self.current_idx == expected {
            Ok(Probe::Hit((
                ChunkedPostcardClaim {
                    handle: self.handle,
                    offset: self.offset,
                },
                (),
            )))
        } else {
            Ok(Probe::Miss)
        }
    }
}

pub struct ChunkedPostcardMapValueProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedPostcardMapValueProbe<'s, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
        }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueProbeOwned
    for ChunkedPostcardMapValueProbe<'s, B, F>
{
    type Error = PostcardError;
    type MapClaim = ChunkedPostcardClaim<'s, B, F>;
    type ValueClaim = ChunkedPostcardClaim<'s, B, F>;
    type ValueSubDeserializer = ChunkedPostcardSubDeserializer<'s, B, F>;

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
        let sub = ChunkedPostcardSubDeserializer::new(self.handle, self.offset);
        V::deserialize_owned(sub, extra).await
    }

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        Err(PostcardError::CannotSkip)
    }
}

// ---------------------------------------------------------------------------
// SeqAccess / SeqEntry
// ---------------------------------------------------------------------------

pub struct ChunkedPostcardSeqAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) remaining: usize,
}

pub struct ChunkedPostcardSeqEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedPostcardSeqEntry<'s, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
        }
    }
}

#[inline(always)]
async fn postcard_seq_next<'s, B, F, const N: usize, Fn_, Fut, R>(
    mut seq: ChunkedPostcardSeqAccess<'s, B, F>,
    mut f: Fn_,
) -> Result<
    Probe<Chunk<(ChunkedPostcardSeqAccess<'s, B, F>, R), ChunkedPostcardClaim<'s, B, F>>>,
    PostcardError,
>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedPostcardSeqEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedPostcardClaim<'s, B, F>, R)>, PostcardError>>,
{
    if seq.remaining == 0 {
        return Ok(Probe::Hit(Chunk::Done(ChunkedPostcardClaim {
            handle: seq.handle,
            offset: seq.offset,
        })));
    }
    seq.remaining -= 1;
    let entries: [ChunkedPostcardSeqEntry<'s, B, F>; N] =
        repeat(seq.handle, Handle::fork).map(|handle| ChunkedPostcardSeqEntry {
            handle,
            offset: seq.offset,
        });
    let (claim, r) = hit!(f(entries).await);
    seq.handle = claim.handle;
    seq.offset = claim.offset;
    Ok(Probe::Hit(Chunk::Data((seq, r))))
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqAccessOwned for ChunkedPostcardSeqAccess<'s, B, F> {
    type Error = PostcardError;
    type SeqClaim = ChunkedPostcardClaim<'s, B, F>;
    type ElemClaim = ChunkedPostcardClaim<'s, B, F>;
    type Elem = ChunkedPostcardSeqEntry<'s, B, F>;

    async fn next<const N: usize, Fn_, Fut, R>(
        self,
        f: Fn_,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        Fn_: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        postcard_seq_next(self, f).await
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqEntryOwned for ChunkedPostcardSeqEntry<'s, B, F> {
    type Error = PostcardError;
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type SubDeserializer = ChunkedPostcardSubDeserializer<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let sub = ChunkedPostcardSubDeserializer::new(self.handle, self.offset);
        T::deserialize_owned(sub, extra).await
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        Err(PostcardError::CannotSkip)
    }
}
