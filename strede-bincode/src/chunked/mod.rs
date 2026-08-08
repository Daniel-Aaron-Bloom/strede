//! Chunked bincode deserializer for async streaming input.
//!
//! Uses [`strede::SharedBuf`]/[`strede::Handle`] to coordinate access to a
//! buffer that is refilled asynchronously by a user-supplied loader closure.
//! Mirrors `strede-postcard`'s chunked deserializer structurally; see that
//! crate's module doc comment for the general shape. The key difference
//! here: the wire encoding itself is a compile-time parameter
//! `C: BincodeConfig`, threaded through every type in this module
//! (including [`ChunkedBincodeClaim`] — see `crate::full`'s module doc
//! comment for why `Claim` can't stay config-independent the way
//! postcard's can).
//!
//! Bincode's varint scheme (unlike postcard's LEB128) needs no
//! byte-at-a-time resumable loop: the prefix byte alone determines the
//! entire remaining tail length, so [`num`] reduces to "read one prefix
//! byte via `read_bytes_exact::<1>`, then at most one more
//! `read_bytes_exact::<N>` for the fixed tail" — the same idiom
//! `strede-msgpack`/`strede-cbor` already use for their own multi-byte
//! headers.

use core::future::Future;
use core::marker::PhantomData;

use crate::BincodeError;
use crate::chunked::access::ChunkedBincodeMapKeyProbe;
use crate::config::BincodeConfig;
use strede::utils::repeat;
use strede::{
    Buffer, Chunk, DeserializeFromEnumOwned, DeserializeFromMapOwned, DeserializeFromSeqOwned,
    DeserializeOwned, DeserializerOwned, EntryOwned, EnumAccessOwned, EnumArmStackOwned,
    EnumVariantProbeOwned, Handle, MapAccessOwned, MapArmStackOwned, MapKeyClaimOwned,
    MapValueClaimOwned, NextKey, NumberAccessOwned, NumberEncoding, Probe, SharedBuf, hit,
};

pub(crate) mod access;
pub(crate) mod num;

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Advance handle to the next chunk, resetting offset. Errors if the new
/// chunk is empty (unexpected EOF mid-value).
#[inline(always)]
pub(super) async fn refill<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<Handle<'s, B, F>, BincodeError> {
    let h = handle.next().await;
    *offset = 0;
    if h.buf().is_empty() {
        return Err(BincodeError::UnexpectedEnd);
    }
    Ok(h)
}

/// Read exactly `N` bytes into a `[u8; N]`, refilling across chunk
/// boundaries.
pub(super) async fn read_bytes_exact<'s, B: Buffer, F: AsyncFnMut(&mut B), const N: usize>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, [u8; N]), BincodeError> {
    let mut out = [0u8; N];
    let mut filled = 0;
    while filled < N {
        let buf = handle.buf();
        let avail = buf.len() - *offset;
        if avail == 0 {
            handle = refill(handle, offset).await?;
            continue;
        }
        let take = (N - filled).min(avail);
        out[filled..filled + take].copy_from_slice(&buf[*offset..*offset + take]);
        *offset += take;
        filled += take;
    }
    Ok((handle, out))
}

// ---------------------------------------------------------------------------
// ChunkedBincodeClaim
// ---------------------------------------------------------------------------

/// Proof of consumption: carries the live handle and offset after consuming
/// a value, plus which `C` is in effect (see the module doc comment).
pub struct ChunkedBincodeClaim<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedBincodeClaim<'s, C, B, F> {
    #[inline(always)]
    pub(crate) fn new(handle: Handle<'s, B, F>, offset: usize) -> Self {
        Self {
            handle,
            offset,
            _cfg: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Deserializer / SubDeserializer
// ---------------------------------------------------------------------------

/// Root deserializer: checks for trailing bytes after the top-level value.
pub struct ChunkedBincodeDeserializer<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: SharedBuf<'s, B, F>,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)>
    ChunkedBincodeDeserializer<'s, C, B, F>
{
    pub fn new(shared: SharedBuf<'s, B, F>) -> Self {
        Self {
            shared,
            _cfg: PhantomData,
        }
    }
}

/// Sub-deserializer for nested values: no trailing-bytes check.
pub struct ChunkedBincodeSubDeserializer<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)>
    ChunkedBincodeSubDeserializer<'s, C, B, F>
{
    #[inline(always)]
    pub(crate) fn new(handle: Handle<'s, B, F>, offset: usize) -> Self {
        Self {
            handle,
            offset,
            _cfg: PhantomData,
        }
    }
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedBincodeDeserializer<'s, C, B, F>
{
    type Error = BincodeError;
    type Claim = ();
    type EntryClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type Entry = ChunkedBincodeEntry<'s, C, B, F>;

    async fn entry<const N: usize, Fn_, Fut, R>(
        self,
        mut f: Fn_,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        Fn_: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let handle = self.shared.fork();
        let entry = ChunkedBincodeEntry {
            handle,
            offset: 0,
            _cfg: PhantomData,
        };
        let (claim, r) = hit!(f(repeat(entry, ChunkedBincodeEntry::clone)).await);

        // Trailing-garbage check: drain remaining buffer, then verify EOF.
        let mut h = claim.handle;
        let mut off = claim.offset;
        loop {
            let buf = h.buf();
            if off < buf.len() {
                return Err(BincodeError::ExpectedEnd);
            }
            let new_h = h.next().await;
            if new_h.buf().is_empty() {
                break;
            }
            h = new_h;
            off = 0;
        }
        Ok(Probe::Hit(((), r)))
    }
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedBincodeSubDeserializer<'s, C, B, F>
{
    type Error = BincodeError;
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type EntryClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type Entry = ChunkedBincodeEntry<'s, C, B, F>;

    async fn entry<const N: usize, Fn_, Fut, R>(
        self,
        mut f: Fn_,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        Fn_: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = ChunkedBincodeEntry {
            handle: self.handle,
            offset: self.offset,
            _cfg: PhantomData,
        };
        f(repeat(entry, ChunkedBincodeEntry::clone)).await
    }
}

// ---------------------------------------------------------------------------
// ChunkedBincodeEntry
// ---------------------------------------------------------------------------

/// One item slot. Holds the live handle/offset. Forkable for `select_probe!`.
pub struct ChunkedBincodeEntry<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedBincodeEntry<'s, C, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            _cfg: PhantomData,
        }
    }

    /// Decode a primitive number from the wire and return it as `T`. Async
    /// counterpart to `full.rs`'s `BincodeEntry::parse_num`.
    pub(crate) async fn parse_num<T: ParseNumOwned<C>>(
        self,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, T)>, BincodeError> {
        T::parse_owned(self.handle, self.offset).await
    }
}

/// Trait for types that can decode themselves from bincode's wire encoding
/// under config `C`, reading asynchronously from a `Handle`. Async
/// counterpart to `full.rs`'s `ParseNum`.
pub(crate) trait ParseNumOwned<C: BincodeConfig>: Sized {
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError>;
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> EntryOwned
    for ChunkedBincodeEntry<'s, C, B, F>
{
    type Error = BincodeError;
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type SubDeserializer = ChunkedBincodeSubDeserializer<'s, C, B, F>;
    type StrChunks = access::ChunkedBincodeStrAccess<'s, C, B, F>;
    type BytesChunks = access::ChunkedBincodeBytesAccess<'s, C, B, F>;
    type NumberChunks<Enc: NumberEncoding> = ChunkedBincodeNumberAccess<'s, C, B, F>;
    type Map = ChunkedBincodeMapAccess<'s, C, B, F>;
    type Seq = access::ChunkedBincodeSeqAccess<'s, C, B, F>;
    type Enum = ChunkedBincodeEnumAccess<'s, C, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    // ---- Strings ------------------------------------------------------------

    async fn deserialize_str_chunks(mut self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        let (handle, len) = num::decode_len::<_, _, C>(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(access::ChunkedBincodeStrAccess {
            handle,
            offset: self.offset,
            remaining: len,
            _cfg: PhantomData,
        }))
    }

    // ---- Bytes --------------------------------------------------------------

    async fn deserialize_bytes_chunks(mut self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        let (handle, len) = num::decode_len::<_, _, C>(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(access::ChunkedBincodeBytesAccess {
            handle,
            offset: self.offset,
            remaining: len,
            _cfg: PhantomData,
        }))
    }

    // ---- Numbers ------------------------------------------------------------

    /// Only meaningful in `Varint` mode — see `full.rs`'s identical
    /// `deserialize_number_chunks` doc comment for why `Fixint` mode always
    /// misses here.
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        mut self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if num::is_fixint::<C>() {
            return Ok(Probe::Miss);
        }
        if Enc::NAME != <C::Order as NumberEncoding>::NAME {
            return Ok(Probe::Miss);
        }
        let (handle, bytes, len) = num::varint_span(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(ChunkedBincodeNumberAccess {
            handle,
            offset: self.offset,
            bytes,
            len,
            done: false,
            _cfg: PhantomData,
        }))
    }

    // ---- Map / Seq / Enum ---------------------------------------------------

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        Ok(Probe::Hit(ChunkedBincodeMapAccess {
            handle: self.handle,
            offset: self.offset,
            current: 0,
            _cfg: PhantomData,
        }))
    }

    /// See `full.rs`'s identical `deserialize_seq` doc comment: `count` is
    /// an element count, not a byte count, so no generic fail-fast bounds
    /// check is sound here either (a corrupt/huge count is instead caught
    /// lazily once iteration runs out of bytes) — mirrors
    /// `strede-postcard`'s own owned-family `deserialize_seq`.
    async fn deserialize_seq(mut self) -> Result<Probe<Self::Seq>, Self::Error> {
        let (handle, count) = num::decode_len::<_, _, C>(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(access::ChunkedBincodeSeqAccess {
            handle,
            offset: self.offset,
            remaining: count,
            _cfg: PhantomData,
        }))
    }

    async fn deserialize_enum(mut self) -> Result<Probe<Self::Enum>, Self::Error> {
        let (handle, discriminant) =
            num::decode_discriminant::<_, _, C>(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(ChunkedBincodeEnumAccess {
            handle,
            offset: self.offset,
            discriminant,
            _cfg: PhantomData,
        }))
    }

    // ---- Option -------------------------------------------------------------

    #[inline(always)]
    async fn deserialize_option<T>(
        mut self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let (handle, [tag]) = read_bytes_exact::<_, _, 1>(self.handle, &mut self.offset).await?;
        match tag {
            0x00 => Ok(Probe::Hit((
                ChunkedBincodeClaim::new(handle, self.offset),
                None,
            ))),
            0x01 => {
                let sub = ChunkedBincodeSubDeserializer::new(handle, self.offset);
                let (claim, v) = hit!(T::deserialize_owned(sub, extra).await);
                Ok(Probe::Hit((claim, Some(v))))
            }
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Value / Map / Seq / Enum forwarding --------------------------------

    #[inline(always)]
    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let sub = ChunkedBincodeSubDeserializer::new(self.handle, self.offset);
        T::deserialize_owned(sub, extra).await
    }

    #[inline(always)]
    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMapOwned<Self::Map>,
    {
        let map = hit!(EntryOwned::deserialize_map(self).await);
        T::deserialize_from_map_owned(map, extra).await
    }

    #[inline(always)]
    async fn deserialize_seq_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeqOwned<Self::Seq>,
    {
        let seq = hit!(EntryOwned::deserialize_seq(self).await);
        T::deserialize_from_seq_owned(seq, extra).await
    }

    #[inline(always)]
    async fn deserialize_enum_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnumOwned<Self::Enum>,
    {
        let e = hit!(EntryOwned::deserialize_enum(self).await);
        T::deserialize_from_enum_owned(e, extra).await
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        Err(BincodeError::CannotSkip)
    }

    /// `#[strede(other)]` only ever targets a unit variant, so once every
    /// named/indexed variant has missed, the unmatched discriminant is
    /// treated as carrying no payload — mirrors `full.rs`'s
    /// `BincodeEntry::skip_other`.
    async fn skip_other(mut self) -> Result<Self::Claim, Self::Error> {
        let (handle, _discriminant) =
            num::decode_discriminant::<_, _, C>(self.handle, &mut self.offset).await?;
        Ok(ChunkedBincodeClaim::new(handle, self.offset))
    }
}

// ---------------------------------------------------------------------------
// ChunkedBincodeNumberAccess — yields owned varint-span bytes as the
// caller's requested `Enc` (only ever constructed when `Enc::NAME` already
// matches `C::Order`'s own encoding — see `deserialize_number_chunks` above).
// ---------------------------------------------------------------------------

pub struct ChunkedBincodeNumberAccess<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    bytes: [u8; 16],
    len: usize,
    done: bool,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B), Enc: NumberEncoding>
    NumberAccessOwned<Enc> for ChunkedBincodeNumberAccess<'s, C, B, F>
{
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type Error = BincodeError;

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.done {
            return Ok(Chunk::Done(ChunkedBincodeClaim::new(
                self.handle,
                self.offset,
            )));
        }
        // `Enc::NAME == <C::Order as NumberEncoding>::NAME` was already
        // checked in `deserialize_number_chunks`.
        let r = f(Enc::from_bytes(&self.bytes[..self.len]));
        self.done = true;
        Ok(Chunk::Data((self, r)))
    }
}

// ---------------------------------------------------------------------------
// Map access — hand-rolled for both static (struct) and dynamic (collection)
// iteration, mirroring `strede-postcard`'s identical situation.
// ---------------------------------------------------------------------------

pub struct ChunkedBincodeMapAccess<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    current: usize,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyClaimOwned
    for ChunkedBincodeClaim<'s, C, B, F>
{
    type Error = BincodeError;
    type MapClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type ValueProbe = access::ChunkedBincodeMapValueProbe<'s, C, B, F>;

    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        Ok(access::ChunkedBincodeMapValueProbe {
            handle: self.handle,
            offset: self.offset,
            _cfg: PhantomData,
        })
    }
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> MapValueClaimOwned
    for ChunkedBincodeClaim<'s, C, B, F>
{
    type Error = BincodeError;
    type KeyProbe = ChunkedBincodeMapKeyProbe<'s, C, B, F>;
    type MapClaim = ChunkedBincodeClaim<'s, C, B, F>;

    /// Unreachable for bincode: both `iterate` and `iterate_dyn` drive
    /// their loops manually rather than going through `next_key` — mirrors
    /// `full.rs`'s identical comment.
    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        Ok(NextKey::Done(self))
    }
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> MapAccessOwned
    for ChunkedBincodeMapAccess<'s, C, B, F>
{
    type Error = BincodeError;
    type MapClaim = ChunkedBincodeClaim<'s, C, B, F>;
    type KeyProbe = ChunkedBincodeMapKeyProbe<'s, C, B, F>;

    /// Struct fields: no wire framing at all; driven by the arm stack
    /// becoming satisfied (`unsatisfied_count() == 0`), matching fields
    /// positionally via `current_idx`. Mirrors `full.rs`'s `iterate_static`.
    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        mut arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        let mut handle = self.handle;
        let mut offset = self.offset;
        let mut current = self.current;
        loop {
            if arms.unsatisfied_count() == 0 {
                return Ok(Probe::Hit((
                    ChunkedBincodeClaim::new(handle, offset),
                    arms.take_outputs(),
                )));
            }

            let kp = ChunkedBincodeMapKeyProbe {
                handle,
                offset,
                current_idx: current,
                dynamic: false,
                _cfg: PhantomData,
            };

            let (arm_index, key_claim) = match arms.race_keys::<true>(kp).await? {
                Probe::Hit(x) => x,
                Probe::Miss => return Ok(Probe::Miss),
            };

            let value_probe = key_claim.into_value_probe().await?;

            let (value_claim, ()) = match arms.dispatch_value(arm_index, value_probe).await? {
                Probe::Hit(x) => x,
                Probe::Miss => return Ok(Probe::Miss),
            };

            handle = value_claim.handle;
            offset = value_claim.offset;
            current += 1;
        }
    }

    /// Unbounded collection (HashMap/BTreeMap via `CollectMap`): bincode
    /// writes an explicit length prefix for these, unlike structs. Loops
    /// exactly `count` times. Mirrors `full.rs`'s `iterate_dynamic`.
    async fn iterate_dyn<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        mut arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        let mut offset = self.offset;
        let (mut handle, count) = num::decode_len::<_, _, C>(self.handle, &mut offset).await?;

        for _ in 0..count {
            let kp = ChunkedBincodeMapKeyProbe {
                handle,
                offset,
                current_idx: 0,
                dynamic: true,
                _cfg: PhantomData,
            };

            let (arm_index, key_claim) = match arms.race_keys::<true>(kp).await? {
                Probe::Hit(x) => x,
                Probe::Miss => return Ok(Probe::Miss),
            };

            let value_probe = key_claim.into_value_probe().await?;

            let (value_claim, ()) = match arms.dispatch_value(arm_index, value_probe).await? {
                Probe::Hit(x) => x,
                Probe::Miss => return Ok(Probe::Miss),
            };

            handle = value_claim.handle;
            offset = value_claim.offset;
        }

        Ok(Probe::Hit((
            ChunkedBincodeClaim::new(handle, offset),
            arms.take_outputs(),
        )))
    }
}

// ---------------------------------------------------------------------------
// EnumAccess / EnumVariantProbe
// ---------------------------------------------------------------------------
//
// Bincode externally-tagged enums: `u32` discriminant (subject to `C`'s int
// encoding), then payload. The discriminant is decoded once, upfront; the
// variant probe itself does zero further I/O to determine Hit/Miss.

pub struct ChunkedBincodeEnumAccess<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    discriminant: usize,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> EnumAccessOwned
    for ChunkedBincodeEnumAccess<'s, C, B, F>
{
    type Error = BincodeError;
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type VariantProbe = ChunkedBincodeEnumVariantProbe<'s, C, B, F>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStackOwned<Self::VariantProbe>,
    {
        let vp = ChunkedBincodeEnumVariantProbe {
            handle: self.handle,
            offset: self.offset,
            discriminant: self.discriminant,
            _cfg: PhantomData,
        };
        let (_idx, claim) = hit!(arms.race::<true>(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

pub struct ChunkedBincodeEnumVariantProbe<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    discriminant: usize,
    _cfg: PhantomData<C>,
}

impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> EnumVariantProbeOwned
    for ChunkedBincodeEnumVariantProbe<'s, C, B, F>
{
    type Error = BincodeError;
    type Claim = ChunkedBincodeClaim<'s, C, B, F>;
    type PayloadDeserializer = ChunkedBincodeSubDeserializer<'s, C, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            discriminant: self.discriminant,
            _cfg: PhantomData,
        }
    }

    // Name-based methods: bincode has no wire names, but the local arm
    // index in each candidate maps directly to the wire discriminant by
    // convention — mirrors `full.rs`.

    async fn deserialize_unit_by_name<W>(
        self,
        candidates: W,
    ) -> Result<Probe<(Self::Claim, usize)>, Self::Error>
    where
        W: strede::ConcatableArray<T = (&'static str, usize)>
            + Copy
            + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        for &(_name, local_idx) in candidates.as_ref() {
            if self.discriminant == local_idx {
                return Ok(Probe::Hit((
                    ChunkedBincodeClaim::new(self.handle, self.offset),
                    local_idx,
                )));
            }
        }
        Ok(Probe::Miss)
    }

    async fn deserialize_payload_by_name<T, W>(
        self,
        candidates: W,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
        W: strede::ConcatableArray<T = (&'static str, usize)>
            + Copy
            + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        for &(_name, local_idx) in candidates.as_ref() {
            if self.discriminant == local_idx {
                let sub = ChunkedBincodeSubDeserializer::new(self.handle, self.offset);
                return match T::deserialize_owned(sub, extra).await? {
                    Probe::Hit((claim, v)) => Ok(Probe::Hit((claim, local_idx, v))),
                    Probe::Miss => Ok(Probe::Miss),
                };
            }
        }
        Ok(Probe::Miss)
    }

    async fn deserialize_unit_by_index(
        self,
        expected_idx: usize,
    ) -> Result<Probe<(Self::Claim, usize)>, Self::Error> {
        if self.discriminant == expected_idx {
            Ok(Probe::Hit((
                ChunkedBincodeClaim::new(self.handle, self.offset),
                expected_idx,
            )))
        } else {
            Ok(Probe::Miss)
        }
    }

    async fn deserialize_payload_by_index<T>(
        self,
        expected_idx: usize,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        if self.discriminant != expected_idx {
            return Ok(Probe::Miss);
        }
        let sub = ChunkedBincodeSubDeserializer::new(self.handle, self.offset);
        match T::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((claim, expected_idx, v))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    /// `#[strede(untagged)]` support — mirrors `full.rs`'s identical
    /// `deserialize_value_by_shape` and its doc comment on the accepted
    /// trade-off (also following `strede-postcard`'s precedent).
    async fn deserialize_value_by_shape<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        let sub = ChunkedBincodeSubDeserializer::new(self.handle, self.offset);
        T::deserialize_owned(sub, extra).await
    }
}
