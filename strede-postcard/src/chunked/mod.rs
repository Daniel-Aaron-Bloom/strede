//! Chunked postcard deserializer for async streaming input.
//!
//! Uses [`strede::SharedBuf`]/[`strede::Handle`] to coordinate access to a
//! buffer that is refilled asynchronously by a user-supplied loader closure.
//!
//! # Key differences from the borrow family
//!
//! - Implements the **owned** trait family only. Zero-copy string/byte
//!   borrowing is unsupported (chunk lifetimes are shorter than the session).
//! - Postcard has no wire type tags, so unlike self-describing formats there
//!   is no shared "read one token" dispatch step: every probe method inlines
//!   its own varint/byte read, exactly like [`crate::full`]'s `PostcardEntry`.
//! - Varint reads (string/bytes lengths, seq/map counts, enum discriminants,
//!   and every integer primitive) may span an arbitrary number of chunk
//!   boundaries; see [`varint`] for the resumable decoder this requires.
//! - Struct fields are matched by position (no wire keys at all), so
//!   [`ChunkedPostcardMapAccess::iterate`] never touches the buffer directly
//!   in its key-matching step - only each field's value deserialization does.
//!   Both the static (struct) and dynamic (HashMap/BTreeMap) map iteration
//!   paths are hand-rolled here rather than built on `strede::PairSeqMapAccess`:
//!   `MapAccessOwned::KeyProbe` is a single associated type shared by both
//!   `iterate` and `iterate_dyn`, and postcard's static case (no wire count,
//!   terminates via `arms.unsatisfied_count() == 0`) has no analogue in
//!   `RawSlot`'s wire-driven termination model, so one key-probe type with a
//!   `dynamic: bool` flag (mirroring `full.rs`'s `PostcardMapKeyProbe`) must
//!   serve both paths instead.

use crate::PostcardError;
use crate::chunked::access::ChunkedPostcardMapKeyProbe;
use core::future::Future;
use strede::utils::repeat;
use strede::{
    Buffer, Chunk, DeserializeFromEnumOwned, DeserializeFromMapOwned, DeserializeFromSeqOwned,
    DeserializeOwned, DeserializerOwned, EntryOwned, EnumAccessOwned, EnumArmStackOwned,
    EnumVariantProbeOwned, Handle, LittleEndian, MapAccessOwned, MapArmStackOwned,
    MapKeyClaimOwned, MapValueClaimOwned, NextKey, NumberAccessOwned, NumberEncoding, Probe,
    SharedBuf, hit,
};

pub(crate) mod access;
pub(crate) mod varint;

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Advance handle to the next chunk, resetting offset. Errors if the new
/// chunk is empty (unexpected EOF mid-value).
#[inline(always)]
pub(super) async fn refill<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<Handle<'s, B, F>, PostcardError> {
    let h = handle.next().await;
    *offset = 0;
    if h.buf().is_empty() {
        return Err(PostcardError::UnexpectedEnd);
    }
    Ok(h)
}

/// Read exactly `N` bytes into a `[u8; N]`, refilling across chunk
/// boundaries. Used for f32/f64 (4/8 bytes) and the bool/Option tag byte (1
/// byte).
pub(super) async fn read_bytes_exact<'s, B: Buffer, F: AsyncFnMut(&mut B), const N: usize>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, [u8; N]), PostcardError> {
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
// ChunkedPostcardClaim
// ---------------------------------------------------------------------------

/// Proof of consumption: carries the live handle and offset after consuming
/// a value. No source lifetime borrow (unlike `PostcardClaim<'de>`) - chunk
/// lifetimes are shorter than the session.
pub struct ChunkedPostcardClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
}

// ---------------------------------------------------------------------------
// Deserializer / SubDeserializer
// ---------------------------------------------------------------------------

/// Root deserializer: checks for trailing bytes after the top-level value.
pub struct ChunkedPostcardDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: SharedBuf<'s, B, F>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedPostcardDeserializer<'s, B, F> {
    pub fn new(shared: SharedBuf<'s, B, F>) -> Self {
        Self { shared }
    }
}

/// Sub-deserializer for nested values: no trailing-bytes check.
pub struct ChunkedPostcardSubDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedPostcardSubDeserializer<'s, B, F> {
    #[inline(always)]
    pub(crate) fn new(handle: Handle<'s, B, F>, offset: usize) -> Self {
        Self { handle, offset }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedPostcardDeserializer<'s, B, F>
{
    type Error = PostcardError;
    type Claim = ();
    type EntryClaim = ChunkedPostcardClaim<'s, B, F>;
    type Entry = ChunkedPostcardEntry<'s, B, F>;

    async fn entry<const N: usize, Fn_, Fut, R>(
        self,
        mut f: Fn_,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        Fn_: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let handle = self.shared.fork();
        let entry = ChunkedPostcardEntry { handle, offset: 0 };
        let (claim, r) = hit!(f(repeat(entry, ChunkedPostcardEntry::clone)).await);

        // Trailing-garbage check: drain remaining buffer, then verify EOF.
        let mut h = claim.handle;
        let mut off = claim.offset;
        loop {
            let buf = h.buf();
            if off < buf.len() {
                return Err(PostcardError::ExpectedEnd);
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

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedPostcardSubDeserializer<'s, B, F>
{
    type Error = PostcardError;
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type EntryClaim = ChunkedPostcardClaim<'s, B, F>;
    type Entry = ChunkedPostcardEntry<'s, B, F>;

    async fn entry<const N: usize, Fn_, Fut, R>(
        self,
        mut f: Fn_,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        Fn_: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = ChunkedPostcardEntry {
            handle: self.handle,
            offset: self.offset,
        };
        f(repeat(entry, ChunkedPostcardEntry::clone)).await
    }
}

// ---------------------------------------------------------------------------
// ChunkedPostcardEntry
// ---------------------------------------------------------------------------

/// One item slot. Holds the live handle/offset. Forkable for `select_probe!`.
pub struct ChunkedPostcardEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedPostcardEntry<'s, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
        }
    }

    /// Decode a primitive number from the wire and return it as `T`. Async
    /// counterpart to `full.rs`'s `PostcardEntry::parse_num`.
    pub(crate) async fn parse_num<T: ParseNumOwned>(
        self,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, T)>, PostcardError> {
        T::parse_owned(self.handle, self.offset).await
    }
}

/// Trait for types that can decode themselves from postcard's wire encoding,
/// reading asynchronously from a `Handle`. Async counterpart to `full.rs`'s
/// `ParseNum`.
pub(crate) trait ParseNumOwned: Sized {
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError>;
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EntryOwned for ChunkedPostcardEntry<'s, B, F> {
    type Error = PostcardError;
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type SubDeserializer = ChunkedPostcardSubDeserializer<'s, B, F>;
    type StrChunks = access::ChunkedPostcardStrAccess<'s, B, F>;
    type BytesChunks = access::ChunkedPostcardBytesAccess<'s, B, F>;
    type NumberChunks<Enc: NumberEncoding> = ChunkedPostcardNumberAccess<'s, B, F>;
    type Map = ChunkedPostcardMapAccess<'s, B, F>;
    type Seq = access::ChunkedPostcardSeqAccess<'s, B, F>;
    type Enum = ChunkedPostcardEnumAccess<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    // ---- Strings ------------------------------------------------------------

    async fn deserialize_str_chunks(mut self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        let (handle, len) = varint::read_varint(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(access::ChunkedPostcardStrAccess {
            handle,
            offset: self.offset,
            remaining: len as usize,
        }))
    }

    // ---- Bytes --------------------------------------------------------------

    async fn deserialize_bytes_chunks(mut self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        let (handle, len) = varint::read_varint(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(access::ChunkedPostcardBytesAccess {
            handle,
            offset: self.offset,
            remaining: len as usize,
        }))
    }

    // ---- Numbers ------------------------------------------------------------

    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        mut self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if Enc::NAME != LittleEndian::NAME {
            return Ok(Probe::Miss);
        }
        let (handle, bytes, len) = varint::read_varint_bytes(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(ChunkedPostcardNumberAccess {
            handle,
            offset: self.offset,
            bytes,
            len,
            done: false,
        }))
    }

    // ---- Map / Seq / Enum ---------------------------------------------------

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        Ok(Probe::Hit(ChunkedPostcardMapAccess {
            handle: self.handle,
            offset: self.offset,
            current: 0,
        }))
    }

    async fn deserialize_seq(mut self) -> Result<Probe<Self::Seq>, Self::Error> {
        let (handle, count) = varint::read_varint(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(access::ChunkedPostcardSeqAccess {
            handle,
            offset: self.offset,
            remaining: count as usize,
        }))
    }

    async fn deserialize_enum(mut self) -> Result<Probe<Self::Enum>, Self::Error> {
        let (handle, discriminant) = varint::read_varint(self.handle, &mut self.offset).await?;
        Ok(Probe::Hit(ChunkedPostcardEnumAccess {
            handle,
            offset: self.offset,
            discriminant: discriminant as usize,
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
                ChunkedPostcardClaim {
                    handle,
                    offset: self.offset,
                },
                None,
            ))),
            0x01 => {
                let sub = ChunkedPostcardSubDeserializer::new(handle, self.offset);
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
        let sub = ChunkedPostcardSubDeserializer::new(self.handle, self.offset);
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
        Err(PostcardError::CannotSkip)
    }

    /// `#[strede(other)]` only ever targets a unit variant (enforced at
    /// derive time), so once every named/indexed variant has missed, the
    /// unmatched discriminant is treated as carrying no payload - mirroring
    /// `full.rs`'s `PostcardEntry::skip_other` and upstream `postcard`+`serde`'s
    /// own `#[serde(other)]` behavior. See that doc comment for the
    /// schema-evolution caveat this carries.
    async fn skip_other(mut self) -> Result<Self::Claim, Self::Error> {
        let (handle, _discriminant) = varint::read_varint(self.handle, &mut self.offset).await?;
        Ok(ChunkedPostcardClaim {
            handle,
            offset: self.offset,
        })
    }
}

// ---------------------------------------------------------------------------
// ChunkedPostcardNumberAccess — yields owned varint bytes as LittleEndian chunks
// ---------------------------------------------------------------------------

pub struct ChunkedPostcardNumberAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    bytes: [u8; 10],
    len: usize,
    done: bool,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B), Enc: NumberEncoding> NumberAccessOwned<Enc>
    for ChunkedPostcardNumberAccess<'s, B, F>
{
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type Error = PostcardError;

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.done {
            return Ok(Chunk::Done(ChunkedPostcardClaim {
                handle: self.handle,
                offset: self.offset,
            }));
        }
        // `Enc::NAME == LittleEndian::NAME` was already checked in
        // `deserialize_number_chunks`, so `Enc::Data == [u8]` here and
        // `Enc::from_bytes` is effectively identity.
        let r = f(Enc::from_bytes(&self.bytes[..self.len]));
        self.done = true;
        Ok(Chunk::Data((self, r)))
    }
}

// ---------------------------------------------------------------------------
// Map access — hand-rolled for both static (struct) and dynamic (collection)
// iteration; see the module doc comment for why `strede::PairSeqMapAccess`
// doesn't fit.
// ---------------------------------------------------------------------------

pub struct ChunkedPostcardMapAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    current: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyClaimOwned for ChunkedPostcardClaim<'s, B, F> {
    type Error = PostcardError;
    type MapClaim = ChunkedPostcardClaim<'s, B, F>;
    type ValueProbe = access::ChunkedPostcardMapValueProbe<'s, B, F>;

    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        Ok(access::ChunkedPostcardMapValueProbe {
            handle: self.handle,
            offset: self.offset,
        })
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueClaimOwned for ChunkedPostcardClaim<'s, B, F> {
    type Error = PostcardError;
    type KeyProbe = ChunkedPostcardMapKeyProbe<'s, B, F>;
    type MapClaim = ChunkedPostcardClaim<'s, B, F>;

    /// Unreachable for postcard: both `iterate` and `iterate_dyn` drive their
    /// loops manually rather than going through `next_key` (see `full.rs`'s
    /// identical comment on `PostcardMapValueClaim::next_key`).
    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        Ok(NextKey::Done(self))
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapAccessOwned for ChunkedPostcardMapAccess<'s, B, F> {
    type Error = PostcardError;
    type MapClaim = ChunkedPostcardClaim<'s, B, F>;
    type KeyProbe = ChunkedPostcardMapKeyProbe<'s, B, F>;

    /// Struct fields: no wire framing at all; driven by the arm stack
    /// becoming satisfied (`unsatisfied_count() == 0`), matching fields
    /// positionally via `current_idx`. Mirrors `full.rs::iterate_static`.
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
                    ChunkedPostcardClaim { handle, offset },
                    arms.take_outputs(),
                )));
            }

            let kp = ChunkedPostcardMapKeyProbe {
                handle,
                offset,
                current_idx: current,
                dynamic: false,
            };

            let (arm_index, key_claim) = match arms.race_keys(kp).await? {
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

    /// Unbounded collection (HashMap/BTreeMap via `CollectMap`): postcard
    /// writes an explicit varint length for these, unlike structs. Loops
    /// exactly `count` times. Mirrors `full.rs::iterate_dynamic`.
    async fn iterate_dyn<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        mut arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        let mut offset = self.offset;
        let (mut handle, count) = varint::read_varint(self.handle, &mut offset).await?;

        for _ in 0..count {
            let kp = ChunkedPostcardMapKeyProbe {
                handle,
                offset,
                current_idx: 0,
                dynamic: true,
            };

            let (arm_index, key_claim) = match arms.race_keys(kp).await? {
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
            ChunkedPostcardClaim { handle, offset },
            arms.take_outputs(),
        )))
    }
}

// ---------------------------------------------------------------------------
// EnumAccess / EnumVariantProbe
// ---------------------------------------------------------------------------
//
// Postcard externally-tagged enums: varint discriminant, then payload.
// The discriminant is decoded once, upfront; the variant probe itself does
// zero further I/O to determine Hit/Miss.

pub struct ChunkedPostcardEnumAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    discriminant: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumAccessOwned for ChunkedPostcardEnumAccess<'s, B, F> {
    type Error = PostcardError;
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type VariantProbe = ChunkedPostcardEnumVariantProbe<'s, B, F>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStackOwned<Self::VariantProbe>,
    {
        let vp = ChunkedPostcardEnumVariantProbe {
            handle: self.handle,
            offset: self.offset,
            discriminant: self.discriminant,
        };
        let (_idx, claim) = hit!(arms.race(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

pub struct ChunkedPostcardEnumVariantProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    discriminant: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumVariantProbeOwned
    for ChunkedPostcardEnumVariantProbe<'s, B, F>
{
    type Error = PostcardError;
    type Claim = ChunkedPostcardClaim<'s, B, F>;
    type PayloadDeserializer = ChunkedPostcardSubDeserializer<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            offset: self.offset,
            discriminant: self.discriminant,
        }
    }

    // Name-based methods: postcard has no wire names, but the local arm index
    // in each candidate maps directly to the wire discriminant by convention
    // (derive assigns arm indices 0, 1, 2, … matching declaration order, which
    // matches postcard's discriminant encoding) - mirrors `full.rs`.

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
                    ChunkedPostcardClaim {
                        handle: self.handle,
                        offset: self.offset,
                    },
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
                let sub = ChunkedPostcardSubDeserializer::new(self.handle, self.offset);
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
                ChunkedPostcardClaim {
                    handle: self.handle,
                    offset: self.offset,
                },
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
        let sub = ChunkedPostcardSubDeserializer::new(self.handle, self.offset);
        match T::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((claim, expected_idx, v))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    async fn deserialize_value_by_shape<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        let sub = ChunkedPostcardSubDeserializer::new(self.handle, self.offset);
        T::deserialize_owned(sub, extra).await
    }
}
