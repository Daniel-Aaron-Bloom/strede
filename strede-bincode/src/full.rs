//! In-memory borrow-family bincode deserializer.
//!
//! Bincode is schema-driven like postcard: there are no type tags on the
//! wire. Type dispatch is handled entirely by which probe method the caller
//! invokes. All probe methods on [`BincodeEntry`] decode inline from the
//! cursor without a pre-loaded token. Unlike postcard, the exact wire
//! encoding (byte order, fixed-width vs varint integers) is a compile-time
//! parameter `C: BincodeConfig` — see [`crate::config`].
//!
//! `C` is threaded through every type in this module, *including*
//! [`BincodeClaim`]. This differs from postcard's single-config
//! `PostcardClaim`: because `MapKeyClaim::into_value_probe` and similar
//! transitions must produce a next-stage type that continues decoding with
//! the *same* `C`, and a single trait impl can't pick a different
//! associated type per call, `Claim` has to remember which `C` is in effect
//! just like `SubDeserializer` does — it can't stay config-independent the
//! way postcard's `Claim` can, since postcard only ever has one config.
//!
//! # Structs and positional fields
//!
//! Named structs deserialize via the map path. [`BincodeMapKeyProbe`]
//! implements [`strede::MapKeyProbe::deserialize_key_by_index`] to match
//! fields by ordinal position — no bytes are consumed for keys.
//!
//! # Enums
//!
//! Externally tagged enums: variant index is encoded as a `u32` (subject to
//! `C`'s int encoding), then the payload follows.
//! [`BincodeEnumVariantProbe`] implements the index-based methods, plus
//! `deserialize_value_by_shape` for `#[strede(untagged)]` support (see its
//! doc comment for the accepted trade-off this carries).

use core::marker::PhantomData;

use crate::config::BincodeConfig;
use crate::num;
use crate::{BincodeError, num::decode_discriminant, num::decode_len};
use strede::{
    BytesAccess, Chunk, Deserialize, DeserializeFromEnum, DeserializeFromMap, DeserializeFromSeq,
    Deserializer, Entry, EnumAccess, EnumArmStack, EnumVariantProbe, MapAccess, MapArmStack,
    MapKeyClaim, MapKeyProbe, MapValueClaim, MapValueProbe, NextKey, NumberAccess, NumberEncoding,
    Probe, SeqAccess, SeqEntry, StrAccess, hit, utils::repeat,
};

// ---------------------------------------------------------------------------
// BincodeClaim
// ---------------------------------------------------------------------------

/// Proof of consumption: carries the cursor position after consuming a
/// value, plus which `C` is in effect (see the module doc comment for why
/// this differs from postcard's config-independent `Claim`).
pub struct BincodeClaim<'de, C: BincodeConfig> {
    pub(crate) src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> core::fmt::Debug for BincodeClaim<'de, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BincodeClaim")
            .field("src", &self.src)
            .finish()
    }
}

impl<'de, C: BincodeConfig> BincodeClaim<'de, C> {
    #[inline(always)]
    pub(crate) fn new(src: &'de [u8]) -> Self {
        Self {
            src,
            _cfg: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// BincodeDeserializer / BincodeSubDeserializer
// ---------------------------------------------------------------------------

/// Root deserializer: checks for trailing bytes after the top-level value.
pub struct BincodeDeserializer<'de, C: BincodeConfig = crate::config::Standard> {
    src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BincodeDeserializer<'de, C> {
    pub fn new(src: &'de [u8]) -> Self {
        Self {
            src,
            _cfg: PhantomData,
        }
    }
}

/// Sub-deserializer for nested values: no trailing-bytes check.
pub struct BincodeSubDeserializer<'de, C: BincodeConfig> {
    pub(crate) src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BincodeSubDeserializer<'de, C> {
    #[inline(always)]
    pub(crate) fn new(src: &'de [u8]) -> Self {
        Self {
            src,
            _cfg: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Deserializer impls
// ---------------------------------------------------------------------------

impl<'de, C: BincodeConfig> Deserializer<'de> for BincodeDeserializer<'de, C> {
    type Error = BincodeError;
    type Claim = BincodeClaim<'de, C>;
    type EntryClaim = BincodeClaim<'de, C>;
    type Entry = BincodeEntry<'de, C>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = BincodeEntry {
            src: self.src,
            _cfg: PhantomData,
        };
        match f(repeat(entry, |e| e.clone())).await? {
            Probe::Hit((claim, r)) => {
                if !claim.src.is_empty() {
                    return Err(BincodeError::ExpectedEnd);
                }
                Ok(Probe::Hit((claim, r)))
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

impl<'de, C: BincodeConfig> Deserializer<'de> for BincodeSubDeserializer<'de, C> {
    type Error = BincodeError;
    type Claim = BincodeClaim<'de, C>;
    type EntryClaim = BincodeClaim<'de, C>;
    type Entry = BincodeEntry<'de, C>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = BincodeEntry {
            src: self.src,
            _cfg: PhantomData,
        };
        f(repeat(entry, |e| e.clone())).await
    }
}

// ---------------------------------------------------------------------------
// BincodeEntry
// ---------------------------------------------------------------------------

/// One item slot. Holds a cursor into the source buffer. Cloneable for `fork`.
pub struct BincodeEntry<'de, C: BincodeConfig> {
    pub(crate) src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BincodeEntry<'de, C> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            _cfg: PhantomData,
        }
    }
}

impl<'de, C: BincodeConfig> Entry<'de> for BincodeEntry<'de, C> {
    type Error = BincodeError;
    type Claim = BincodeClaim<'de, C>;
    type SubDeserializer = BincodeSubDeserializer<'de, C>;
    type StrChunks = BincodeStrAccess<'de, C>;
    type BytesChunks = BincodeBytesAccess<'de, C>;
    type NumberChunks<Enc: NumberEncoding> = BincodeNumberAccess<'de, C>;
    type Map = BincodeMapAccess<'de, C>;
    type Seq = BincodeSeqAccess<'de, C>;
    type Enum = BincodeEnumAccess<'de, C>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    // ---- Strings ------------------------------------------------------------

    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
        let (len, consumed) = decode_len::<C>(self.src)?;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(BincodeError::UnexpectedEnd);
        }
        let (payload, after) = rest.split_at(len);
        let s = core::str::from_utf8(payload).map_err(|_| BincodeError::InvalidUtf8)?;
        Ok(Probe::Hit((BincodeClaim::new(after), s)))
    }

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        let (len, consumed) = decode_len::<C>(self.src)?;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(BincodeError::UnexpectedEnd);
        }
        Ok(Probe::Hit(BincodeStrAccess {
            src: rest,
            len,
            _cfg: PhantomData,
        }))
    }

    // ---- Bytes --------------------------------------------------------------

    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
        let (len, consumed) = decode_len::<C>(self.src)?;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(BincodeError::UnexpectedEnd);
        }
        let (payload, after) = rest.split_at(len);
        Ok(Probe::Hit((BincodeClaim::new(after), payload)))
    }

    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        let (len, consumed) = decode_len::<C>(self.src)?;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(BincodeError::UnexpectedEnd);
        }
        Ok(Probe::Hit(BincodeBytesAccess {
            src: rest,
            len,
            _cfg: PhantomData,
        }))
    }

    // ---- Numbers ------------------------------------------------------------

    /// Only meaningful in `Varint` mode, where the wire byte count is
    /// self-describing (determined by the prefix byte alone). In `Fixint`
    /// mode there is no way to know how many bytes a "number" occupies
    /// without already knowing the target Rust type's width, so this
    /// always misses.
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if num::is_fixint::<C>() {
            return Ok(Probe::Miss);
        }
        if Enc::NAME != <C::Order as NumberEncoding>::NAME {
            return Ok(Probe::Miss);
        }
        let (bytes, consumed) = num::varint_span(self.src)?;
        Ok(Probe::Hit(BincodeNumberAccess {
            bytes,
            after: &self.src[consumed..],
            done: false,
            _cfg: PhantomData,
        }))
    }

    // ---- Map / Seq / Enum ---------------------------------------------------

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        Ok(Probe::Hit(BincodeMapAccess {
            src: self.src,
            current: 0,
            _cfg: PhantomData,
        }))
    }

    /// Unlike `deserialize_str`/`deserialize_bytes`, there is no
    /// `rest.len() < count` fail-fast check here: `count` is an *element*
    /// count, not a byte count, and elements can be zero-width (`()`, unit
    /// structs) — bincode's own `()` decode consumes 0 bytes. A generic
    /// bounds check at this layer (which doesn't know the element type)
    /// would reject a perfectly valid, e.g., `Vec<()>` whenever the buffer
    /// is shorter than the declared count. `strede-postcard`'s own
    /// `deserialize_seq` has the identical shape for the identical reason —
    /// a corrupt/huge count is instead caught lazily once iteration runs out
    /// of bytes, same as postcard.
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        let (count, consumed) = decode_len::<C>(self.src)?;
        Ok(Probe::Hit(BincodeSeqAccess {
            src: &self.src[consumed..],
            remaining: count,
            _cfg: PhantomData,
        }))
    }

    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        let (discriminant, consumed) = decode_discriminant::<C>(self.src)?;
        Ok(Probe::Hit(BincodeEnumAccess {
            discriminant,
            src: &self.src[consumed..],
            _cfg: PhantomData,
        }))
    }

    // ---- Option -------------------------------------------------------------

    #[inline(always)]
    async fn deserialize_option<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        match self.src.first() {
            None => Err(BincodeError::UnexpectedEnd),
            Some(&0x00) => Ok(Probe::Hit((BincodeClaim::new(&self.src[1..]), None))),
            Some(&0x01) => {
                let sub = BincodeSubDeserializer::new(&self.src[1..]);
                let (claim, v) = hit!(T::deserialize(sub, extra).await);
                Ok(Probe::Hit((claim, Some(v))))
            }
            Some(_) => Ok(Probe::Miss),
        }
    }

    // ---- Value / Map / Seq / Enum forwarding --------------------------------

    #[inline(always)]
    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = BincodeSubDeserializer::new(self.src);
        T::deserialize(sub, extra).await
    }

    #[inline(always)]
    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>,
    {
        let map = hit!(Entry::deserialize_map(self).await);
        T::deserialize_from_map(map, extra).await
    }

    #[inline(always)]
    async fn deserialize_seq_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeq<'de, Self::Seq>,
    {
        let seq = hit!(Entry::deserialize_seq(self).await);
        T::deserialize_from_seq(seq, extra).await
    }

    #[inline(always)]
    async fn deserialize_enum_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnum<'de, Self::Enum>,
    {
        let e = hit!(Entry::deserialize_enum(self).await);
        T::deserialize_from_enum(e, extra).await
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        Err(BincodeError::CannotSkip)
    }

    /// `#[strede(other)]` only ever targets a unit variant (enforced at
    /// derive time), so once every named/indexed variant has missed, the
    /// unmatched discriminant is treated as carrying no payload — mirroring
    /// `strede-postcard`'s identical `skip_other` and upstream
    /// `postcard`/`bincode`+`serde`'s own `#[serde(other)]` behavior. If the
    /// real (unrecognized) variant actually carried a payload on the wire,
    /// those bytes are left unconsumed and will surface as a
    /// `BincodeError::ExpectedEnd` (or corrupt a sibling read, if nested) —
    /// the same schema-evolution caveat postcard documents.
    async fn skip_other(self) -> Result<Self::Claim, Self::Error> {
        let (_discriminant, consumed) = decode_discriminant::<C>(self.src)?;
        Ok(BincodeClaim::new(&self.src[consumed..]))
    }
}

// ---------------------------------------------------------------------------
// BincodeEntry::parse_num — format-specific numeric decode helper
// ---------------------------------------------------------------------------

impl<'de, C: BincodeConfig> BincodeEntry<'de, C> {
    /// Decode a primitive number from the wire and return it as `T`.
    /// Returns `Miss` on out-of-range or type mismatch; `Err` on truncation
    /// or corrupt varint framing.
    pub(crate) async fn parse_num<T: ParseNum<C>>(
        self,
    ) -> Result<Probe<(BincodeClaim<'de, C>, T)>, BincodeError> {
        T::parse(self.src)
    }
}

/// Trait for types that can decode themselves from bincode's wire encoding
/// under config `C`.
pub(crate) trait ParseNum<C: BincodeConfig>: Sized {
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError>;
}

// ---------------------------------------------------------------------------
// BincodeStrAccess / BincodeBytesAccess
//
// Neither needs `C` for any actual decoding (string/byte content has no
// config-dependent framing once the length prefix is already consumed by
// the caller), but both carry it as `PhantomData` purely to produce the
// correctly-typed `BincodeClaim<'de, C>` in `Chunk::Done`.
// ---------------------------------------------------------------------------

pub struct BincodeStrAccess<'de, C: BincodeConfig> {
    src: &'de [u8],
    len: usize,
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> StrAccess for BincodeStrAccess<'de, C> {
    type Claim = BincodeClaim<'de, C>;
    type Error = BincodeError;

    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.len == 0 {
            return Ok(Chunk::Done(BincodeClaim::new(self.src)));
        }
        if self.src.len() < self.len {
            return Err(BincodeError::UnexpectedEnd);
        }
        let (payload, after) = self.src.split_at(self.len);
        let s = core::str::from_utf8(payload).map_err(|_| BincodeError::InvalidUtf8)?;
        let r = f(s);
        Ok(Chunk::Data((
            Self {
                src: after,
                len: 0,
                _cfg: PhantomData,
            },
            r,
        )))
    }
}

pub struct BincodeBytesAccess<'de, C: BincodeConfig> {
    src: &'de [u8],
    len: usize,
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BytesAccess for BincodeBytesAccess<'de, C> {
    type Claim = BincodeClaim<'de, C>;
    type Error = BincodeError;

    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.len == 0 {
            return Ok(Chunk::Done(BincodeClaim::new(self.src)));
        }
        if self.src.len() < self.len {
            return Err(BincodeError::UnexpectedEnd);
        }
        let (payload, after) = self.src.split_at(self.len);
        let r = f(payload);
        Ok(Chunk::Data((
            Self {
                src: after,
                len: 0,
                _cfg: PhantomData,
            },
            r,
        )))
    }
}

// ---------------------------------------------------------------------------
// BincodeNumberAccess — yields raw varint-span bytes as the caller's
// requested `Enc` (only ever constructed when `Enc::NAME` already matches
// `C::Order`'s own encoding — see `deserialize_number_chunks` above).
// ---------------------------------------------------------------------------

pub struct BincodeNumberAccess<'de, C: BincodeConfig> {
    bytes: &'de [u8],
    after: &'de [u8],
    done: bool,
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig, Enc: NumberEncoding> NumberAccess<Enc> for BincodeNumberAccess<'de, C> {
    type Claim = BincodeClaim<'de, C>;
    type Error = BincodeError;

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.done {
            return Ok(Chunk::Done(BincodeClaim::new(self.after)));
        }
        let r = f(Enc::from_bytes(self.bytes));
        self.done = true;
        Ok(Chunk::Data((self, r)))
    }
}

// ---------------------------------------------------------------------------
// Map access type chain
//
// Bincode structs have no wire keys and no wire field count, exactly like
// postcard. Fields are decoded positionally: field 0 first, then field 1,
// etc. The arm stack's key callbacks call `kp.deserialize_key_by_index(arm_idx)`
// which hits only when the probe's `current` counter equals `arm_idx`.
// ---------------------------------------------------------------------------

pub struct BincodeMapAccess<'de, C: BincodeConfig> {
    src: &'de [u8],
    current: usize,
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> MapKeyClaim<'de> for BincodeClaim<'de, C> {
    type Error = BincodeError;
    type MapClaim = BincodeClaim<'de, C>;
    type ValueProbe = BincodeMapValueProbe<'de, C>;

    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        Ok(BincodeMapValueProbe {
            src: self.src,
            _cfg: PhantomData,
        })
    }
}

impl<'de, C: BincodeConfig> MapValueClaim<'de> for BincodeClaim<'de, C> {
    type Error = BincodeError;
    type KeyProbe = BincodeMapKeyProbe<'de, C>;
    type MapClaim = BincodeClaim<'de, C>;

    /// Unreachable for bincode: both `iterate` and `iterate_dyn` (below)
    /// drive their loops manually rather than through `next_key` — mirrors
    /// `strede-postcard`'s identical situation and identical comment.
    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        Ok(NextKey::Done(self))
    }
}

// --- Key probe ---

pub struct BincodeMapKeyProbe<'de, C: BincodeConfig> {
    pub(crate) src: &'de [u8],
    pub(crate) current_idx: usize,
    /// `true` for a dynamic-collection key slot (HashMap/BTreeMap via
    /// `CollectMap`), `false` for a struct field. See
    /// `strede-postcard`'s identical field for the full rationale.
    pub(crate) dynamic: bool,
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BincodeMapKeyProbe<'de, C> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            current_idx: self.current_idx,
            dynamic: self.dynamic,
            _cfg: PhantomData,
        }
    }
}

impl<'de, C: BincodeConfig> MapKeyProbe<'de> for BincodeMapKeyProbe<'de, C> {
    type Error = BincodeError;
    type KeyClaim = BincodeClaim<'de, C>;
    type KeySubDeserializer = BincodeSubDeserializer<'de, C>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: Deserialize<'de, Self::KeySubDeserializer>,
    {
        if !self.dynamic {
            return Ok(Probe::Miss);
        }
        let sub = BincodeSubDeserializer::new(self.src);
        K::deserialize(sub, extra).await
    }

    async fn deserialize_key_by_index(
        self,
        expected: usize,
    ) -> Result<Probe<(Self::KeyClaim, ())>, Self::Error> {
        if self.current_idx == expected {
            Ok(Probe::Hit((BincodeClaim::new(self.src), ())))
        } else {
            Ok(Probe::Miss)
        }
    }
}

// --- Value probe ---

pub struct BincodeMapValueProbe<'de, C: BincodeConfig> {
    src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BincodeMapValueProbe<'de, C> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            _cfg: PhantomData,
        }
    }
}

impl<'de, C: BincodeConfig> MapValueProbe<'de> for BincodeMapValueProbe<'de, C> {
    type Error = BincodeError;
    type MapClaim = BincodeClaim<'de, C>;
    type ValueClaim = BincodeClaim<'de, C>;
    type ValueSubDeserializer = BincodeSubDeserializer<'de, C>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: Deserialize<'de, Self::ValueSubDeserializer>,
    {
        let sub = BincodeSubDeserializer::new(self.src);
        V::deserialize(sub, extra).await
    }

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        Err(BincodeError::CannotSkip)
    }
}

// --- MapAccess ---

impl<'de, C: BincodeConfig> MapAccess<'de> for BincodeMapAccess<'de, C> {
    type Error = BincodeError;
    type MapClaim = BincodeClaim<'de, C>;
    type KeyProbe = BincodeMapKeyProbe<'de, C>;

    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        iterate_static::<C, S>(self.src, self.current, arms).await
    }

    async fn iterate_dyn<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        iterate_dynamic::<C, S>(self.src, arms).await
    }
}

/// Unbounded collection (HashMap/BTreeMap via `CollectMap`): bincode writes
/// an explicit length prefix for these, unlike structs. Loops exactly
/// `count` times — see `strede-postcard::full::iterate_dynamic` for why
/// this is a separate fn from [`iterate_static`].
async fn iterate_dynamic<'de, C: BincodeConfig, S: MapArmStack<'de, BincodeMapKeyProbe<'de, C>>>(
    src: &'de [u8],
    mut arms: S,
) -> Result<Probe<(BincodeClaim<'de, C>, S::Outputs)>, BincodeError> {
    let (count, consumed) = decode_len::<C>(src)?;
    let mut src = &src[consumed..];

    for _ in 0..count {
        let kp = BincodeMapKeyProbe {
            src,
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

        src = value_claim.src;
    }

    Ok(Probe::Hit((BincodeClaim::new(src), arms.take_outputs())))
}

/// Struct fields: no wire framing at all; driven by the arm stack becoming
/// satisfied (`unsatisfied_count() == 0`), matching fields positionally via
/// `current_idx`. See [`iterate_dynamic`] for why this is a separate fn.
async fn iterate_static<'de, C: BincodeConfig, S: MapArmStack<'de, BincodeMapKeyProbe<'de, C>>>(
    mut src: &'de [u8],
    mut current: usize,
    mut arms: S,
) -> Result<Probe<(BincodeClaim<'de, C>, S::Outputs)>, BincodeError> {
    loop {
        if arms.unsatisfied_count() == 0 {
            return Ok(Probe::Hit((BincodeClaim::new(src), arms.take_outputs())));
        }

        let kp = BincodeMapKeyProbe {
            src,
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

        src = value_claim.src;
        current += 1;
    }
}

// ---------------------------------------------------------------------------
// SeqAccess / SeqEntry
// ---------------------------------------------------------------------------

pub struct BincodeSeqAccess<'de, C: BincodeConfig> {
    pub(crate) src: &'de [u8],
    pub(crate) remaining: usize,
    _cfg: PhantomData<C>,
}

pub struct BincodeSeqEntry<'de, C: BincodeConfig> {
    src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BincodeSeqEntry<'de, C> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            _cfg: PhantomData,
        }
    }
}

#[inline(always)]
async fn bincode_seq_next<'de, C: BincodeConfig, const N: usize, F, Fut, R>(
    seq: BincodeSeqAccess<'de, C>,
    mut f: F,
) -> Result<Probe<Chunk<(BincodeSeqAccess<'de, C>, R), BincodeClaim<'de, C>>>, BincodeError>
where
    F: FnMut([BincodeSeqEntry<'de, C>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(BincodeClaim<'de, C>, R)>, BincodeError>>,
{
    if seq.remaining == 0 {
        return Ok(Probe::Hit(Chunk::Done(BincodeClaim::new(seq.src))));
    }
    let entry = BincodeSeqEntry {
        src: seq.src,
        _cfg: PhantomData,
    };
    let new_remaining = seq.remaining - 1;
    let (claim, r) = hit!(f(repeat(entry, |e| e.clone())).await);
    let updated_seq = BincodeSeqAccess {
        src: claim.src,
        remaining: new_remaining,
        _cfg: PhantomData,
    };
    Ok(Probe::Hit(Chunk::Data((updated_seq, r))))
}

impl<'de, C: BincodeConfig> SeqAccess<'de> for BincodeSeqAccess<'de, C> {
    type Error = BincodeError;
    type SeqClaim = BincodeClaim<'de, C>;
    type ElemClaim = BincodeClaim<'de, C>;
    type Elem = BincodeSeqEntry<'de, C>;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        bincode_seq_next(self, f).await
    }
}

impl<'de, C: BincodeConfig> SeqEntry<'de> for BincodeSeqEntry<'de, C> {
    type Error = BincodeError;
    type Claim = BincodeClaim<'de, C>;
    type SubDeserializer = BincodeSubDeserializer<'de, C>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = BincodeSubDeserializer::new(self.src);
        T::deserialize(sub, extra).await
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        Err(BincodeError::CannotSkip)
    }
}

// ---------------------------------------------------------------------------
// EnumAccess / EnumVariantProbe
// ---------------------------------------------------------------------------
//
// Bincode externally-tagged enums: `u32` discriminant (subject to `C`'s int
// encoding), then payload. Unit variants: discriminant + zero payload bytes.

pub struct BincodeEnumAccess<'de, C: BincodeConfig> {
    pub(crate) discriminant: usize,
    pub(crate) src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> EnumAccess<'de> for BincodeEnumAccess<'de, C> {
    type Error = BincodeError;
    type Claim = BincodeClaim<'de, C>;
    type VariantProbe = BincodeEnumVariantProbe<'de, C>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStack<'de, Self::VariantProbe>,
    {
        let vp = BincodeEnumVariantProbe {
            discriminant: self.discriminant,
            src: self.src,
            _cfg: PhantomData,
        };
        let (_idx, claim) = hit!(arms.race::<true>(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

pub struct BincodeEnumVariantProbe<'de, C: BincodeConfig> {
    pub(crate) discriminant: usize,
    pub(crate) src: &'de [u8],
    _cfg: PhantomData<C>,
}

impl<'de, C: BincodeConfig> BincodeEnumVariantProbe<'de, C> {
    fn clone(&self) -> Self {
        Self {
            discriminant: self.discriminant,
            src: self.src,
            _cfg: PhantomData,
        }
    }
}

impl<'de, C: BincodeConfig> EnumVariantProbe<'de> for BincodeEnumVariantProbe<'de, C> {
    type Error = BincodeError;
    type Claim = BincodeClaim<'de, C>;
    type PayloadDeserializer = BincodeSubDeserializer<'de, C>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    // Name-based methods: bincode has no wire names, but the local arm
    // index in each candidate maps directly to the wire discriminant by
    // convention (derive assigns arm indices 0, 1, 2, … matching
    // declaration order, which matches bincode's discriminant encoding) —
    // mirrors `strede-postcard`.

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
                return Ok(Probe::Hit((BincodeClaim::new(self.src), local_idx)));
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
        T: Deserialize<'de, Self::PayloadDeserializer>,
        W: strede::ConcatableArray<T = (&'static str, usize)>
            + Copy
            + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        for &(_name, local_idx) in candidates.as_ref() {
            if self.discriminant == local_idx {
                let sub = BincodeSubDeserializer::new(self.src);
                return match T::deserialize(sub, extra).await? {
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
            Ok(Probe::Hit((BincodeClaim::new(self.src), expected_idx)))
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
        T: Deserialize<'de, Self::PayloadDeserializer>,
    {
        if self.discriminant != expected_idx {
            return Ok(Probe::Miss);
        }
        let sub = BincodeSubDeserializer::new(self.src);
        match T::deserialize(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((claim, expected_idx, v))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    /// `#[strede(untagged)]` support. Bincode itself has no shape-based
    /// disambiguation mechanism (the *real* `bincode`+`serde` combination
    /// fundamentally can't support untagged enums — confirmed via
    /// `servo/bincode#130`, where serde's own author states there is no
    /// structural information in the byte stream to distinguish which
    /// variant produced a given sequence of bytes). This impl nonetheless
    /// follows `strede-postcard`'s own precedent (also schema-driven) of
    /// implementing this by simply delegating to `T::deserialize` and
    /// accepting the first declaration-order variant whose shape happens to
    /// parse — the same trade-off postcard already accepts in this
    /// codebase's architecture, where dispatch is driven by the derive's
    /// own arm race rather than a generic `deserialize_any`. A byte sequence
    /// that structurally parses as an earlier untagged variant's shape but
    /// was actually written as a later variant will silently produce the
    /// wrong value — a known, accepted limitation shared with postcard, not
    /// unique to this format.
    async fn deserialize_value_by_shape<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
    {
        let sub = BincodeSubDeserializer::new(self.src);
        T::deserialize(sub, extra).await
    }
}
