//! In-memory borrow-family postcard deserializer.
//!
//! Postcard is schema-driven: there are no type tags on the wire. Type dispatch
//! is handled entirely by which probe method the caller invokes. All probe
//! methods on [`PostcardEntry`] decode inline from the cursor without a
//! pre-loaded token.
//!
//! # Structs and positional fields
//!
//! Named structs deserialize via the map path. [`PostcardMapKeyProbe`] implements
//! [`strede::MapKeyProbe::deserialize_key_by_index`] to match fields by ordinal
//! position — no bytes are consumed for keys. The derive macro's arm closures
//! receive `(kp, arm_index)` and call `kp.deserialize_key_by_index(arm_index)`.
//!
//! # Enums
//!
//! Externally tagged enums: variant index is encoded as a varint, then the
//! payload follows. [`PostcardEnumVariantProbe`] implements the index-based
//! methods only; name-based methods return `Miss`.

use crate::{
    PostcardError,
    varint::{decode_varint, varint_bytes},
};
use strede::{
    BytesAccess, Chunk, Deserialize, DeserializeFromEnum, DeserializeFromMap, DeserializeFromSeq,
    Deserializer, Entry, EnumAccess, EnumArmStack, EnumVariantProbe, LittleEndian, MapAccess,
    MapArmStack, MapKeyClaim, MapKeyProbe, MapValueClaim, MapValueProbe, NextKey, NumberAccess,
    NumberEncoding, Probe, SeqAccess, SeqEntry, StrAccess, hit, utils::repeat,
};

// ---------------------------------------------------------------------------
// PostcardClaim
// ---------------------------------------------------------------------------

/// Proof of consumption: carries the cursor position after consuming a value.
#[derive(Debug)]
pub struct PostcardClaim<'de> {
    pub(crate) src: &'de [u8],
}

// ---------------------------------------------------------------------------
// PostcardDeserializer / PostcardSubDeserializer
// ---------------------------------------------------------------------------

/// Root deserializer: checks for trailing bytes after the top-level value.
pub struct PostcardDeserializer<'de> {
    src: &'de [u8],
}

impl<'de> PostcardDeserializer<'de> {
    pub fn new(src: &'de [u8]) -> Self {
        Self { src }
    }
}

/// Sub-deserializer for nested values: no trailing-bytes check.
pub struct PostcardSubDeserializer<'de> {
    pub(crate) src: &'de [u8],
}

impl<'de> PostcardSubDeserializer<'de> {
    #[inline(always)]
    pub(crate) fn new(src: &'de [u8]) -> Self {
        Self { src }
    }
}

// ---------------------------------------------------------------------------
// Deserializer impls
// ---------------------------------------------------------------------------

impl<'de> Deserializer<'de> for PostcardDeserializer<'de> {
    type Error = PostcardError;
    type Claim = PostcardClaim<'de>;
    type EntryClaim = PostcardClaim<'de>;
    type Entry = PostcardEntry<'de>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = PostcardEntry { src: self.src };
        match f(repeat(entry, |e| e.clone())).await? {
            Probe::Hit((claim, r)) => {
                if !claim.src.is_empty() {
                    return Err(PostcardError::ExpectedEnd);
                }
                Ok(Probe::Hit((claim, r)))
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

impl<'de> Deserializer<'de> for PostcardSubDeserializer<'de> {
    type Error = PostcardError;
    type Claim = PostcardClaim<'de>;
    type EntryClaim = PostcardClaim<'de>;
    type Entry = PostcardEntry<'de>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = PostcardEntry { src: self.src };
        f(repeat(entry, |e| e.clone())).await
    }
}

// ---------------------------------------------------------------------------
// PostcardEntry
// ---------------------------------------------------------------------------

/// One item slot. Holds a cursor into the source buffer. Cloneable for `fork`.
pub struct PostcardEntry<'de> {
    pub(crate) src: &'de [u8],
}

impl<'de> PostcardEntry<'de> {
    fn clone(&self) -> Self {
        Self { src: self.src }
    }
}

impl<'de> Entry<'de> for PostcardEntry<'de> {
    type Error = PostcardError;
    type Claim = PostcardClaim<'de>;
    type SubDeserializer = PostcardSubDeserializer<'de>;
    type StrChunks = PostcardStrAccess<'de>;
    type BytesChunks = PostcardBytesAccess<'de>;
    type NumberChunks<Enc: NumberEncoding> = PostcardNumberAccess<'de>;
    type Map = PostcardMapAccess<'de>;
    type Seq = PostcardSeqAccess<'de>;
    type Enum = PostcardEnumAccess<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    // ---- Strings ------------------------------------------------------------

    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
        let (len, consumed) = decode_varint(self.src)?;
        let len = len as usize;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(PostcardError::UnexpectedEnd);
        }
        let (payload, after) = rest.split_at(len);
        let s = core::str::from_utf8(payload).map_err(|_| PostcardError::InvalidUtf8)?;
        Ok(Probe::Hit((PostcardClaim { src: after }, s)))
    }

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        let (len, consumed) = decode_varint(self.src)?;
        let len = len as usize;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(PostcardError::UnexpectedEnd);
        }
        Ok(Probe::Hit(PostcardStrAccess { src: rest, len }))
    }

    // ---- Bytes --------------------------------------------------------------

    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
        let (len, consumed) = decode_varint(self.src)?;
        let len = len as usize;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(PostcardError::UnexpectedEnd);
        }
        let (payload, after) = rest.split_at(len);
        Ok(Probe::Hit((PostcardClaim { src: after }, payload)))
    }

    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        let (len, consumed) = decode_varint(self.src)?;
        let len = len as usize;
        let rest = &self.src[consumed..];
        if rest.len() < len {
            return Err(PostcardError::UnexpectedEnd);
        }
        Ok(Probe::Hit(PostcardBytesAccess { src: rest, len }))
    }

    // ---- Numbers ------------------------------------------------------------

    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if Enc::NAME != LittleEndian::NAME {
            return Ok(Probe::Miss);
        }
        let (bytes, consumed) = varint_bytes(self.src)?;
        Ok(Probe::Hit(PostcardNumberAccess {
            bytes,
            after: &self.src[consumed..],
            done: false,
        }))
    }

    // ---- Map / Seq / Enum ---------------------------------------------------

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        Ok(Probe::Hit(PostcardMapAccess {
            src: self.src,
            current: 0,
        }))
    }

    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        let (count, consumed) = decode_varint(self.src)?;
        Ok(Probe::Hit(PostcardSeqAccess {
            src: &self.src[consumed..],
            remaining: count as usize,
        }))
    }

    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        let (discriminant, consumed) = decode_varint(self.src)?;
        Ok(Probe::Hit(PostcardEnumAccess {
            discriminant: discriminant as usize,
            src: &self.src[consumed..],
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
            None => Err(PostcardError::UnexpectedEnd),
            Some(&0x00) => Ok(Probe::Hit((
                PostcardClaim {
                    src: &self.src[1..],
                },
                None,
            ))),
            Some(&0x01) => {
                let sub = PostcardSubDeserializer::new(&self.src[1..]);
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
        let sub = PostcardSubDeserializer::new(self.src);
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
        Err(PostcardError::CannotSkip)
    }
}

// ---------------------------------------------------------------------------
// PostcardEntry::parse_num — format-specific numeric decode helper
// ---------------------------------------------------------------------------

impl<'de> PostcardEntry<'de> {
    /// Decode a primitive number from the wire and return it as `T`.
    ///
    /// Unsigned integers and `bool` use varint encoding.
    /// Signed integers use zigzag + varint.
    /// `f32`/`f64` are 4/8 little-endian bytes.
    /// Returns `Miss` on out-of-range or type mismatch; `Err` on truncation.
    pub(crate) async fn parse_num<T: ParseNum>(
        self,
    ) -> Result<Probe<(PostcardClaim<'de>, T)>, PostcardError> {
        T::parse(self.src)
    }
}

/// Trait for types that can decode themselves from postcard's wire encoding.
pub(crate) trait ParseNum: Sized {
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError>;
}

// ---------------------------------------------------------------------------
// PostcardStrAccess / PostcardBytesAccess
// ---------------------------------------------------------------------------

pub struct PostcardStrAccess<'de> {
    src: &'de [u8],
    len: usize,
}

impl<'de> StrAccess for PostcardStrAccess<'de> {
    type Claim = PostcardClaim<'de>;
    type Error = PostcardError;

    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.len == 0 {
            return Ok(Chunk::Done(PostcardClaim { src: self.src }));
        }
        if self.src.len() < self.len {
            return Err(PostcardError::UnexpectedEnd);
        }
        let (payload, after) = self.src.split_at(self.len);
        let s = core::str::from_utf8(payload).map_err(|_| PostcardError::InvalidUtf8)?;
        let r = f(s);
        Ok(Chunk::Data((Self { src: after, len: 0 }, r)))
    }
}

pub struct PostcardBytesAccess<'de> {
    src: &'de [u8],
    len: usize,
}

impl<'de> BytesAccess for PostcardBytesAccess<'de> {
    type Claim = PostcardClaim<'de>;
    type Error = PostcardError;

    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.len == 0 {
            return Ok(Chunk::Done(PostcardClaim { src: self.src }));
        }
        if self.src.len() < self.len {
            return Err(PostcardError::UnexpectedEnd);
        }
        let (payload, after) = self.src.split_at(self.len);
        let r = f(payload);
        Ok(Chunk::Data((Self { src: after, len: 0 }, r)))
    }
}

// ---------------------------------------------------------------------------
// PostcardNumberAccess — yields raw varint bytes as LittleEndian chunks
// ---------------------------------------------------------------------------

pub struct PostcardNumberAccess<'de> {
    bytes: &'de [u8],
    after: &'de [u8],
    done: bool,
}

impl<'de, Enc: NumberEncoding> NumberAccess<Enc> for PostcardNumberAccess<'de> {
    type Claim = PostcardClaim<'de>;
    type Error = PostcardError;

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.done {
            return Ok(Chunk::Done(PostcardClaim { src: self.after }));
        }
        let r = f(Enc::from_bytes(self.bytes));
        self.done = true;
        Ok(Chunk::Data((self, r)))
    }
}

// ---------------------------------------------------------------------------
// Map access type chain
//
// Postcard structs have no wire keys and no wire field count. Fields are
// decoded positionally: field 0 first, then field 1, etc. The arm stack's
// key callbacks call `kp.deserialize_key_by_index(arm_idx)` which hits only
// when the probe's `current` counter equals `arm_idx`.
//
// PostcardClaim serves as KeyClaim and ValueClaim. It carries the updated
// cursor so the next field probe starts from the right position.
// ---------------------------------------------------------------------------

pub struct PostcardMapAccess<'de> {
    src: &'de [u8],
    current: usize,
}

impl<'de> MapKeyClaim<'de> for PostcardClaim<'de> {
    type Error = PostcardError;
    type MapClaim = PostcardClaim<'de>;
    type ValueProbe = PostcardMapValueProbe<'de>;

    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        Ok(PostcardMapValueProbe { src: self.src })
    }
}

impl<'de> MapValueClaim<'de> for PostcardClaim<'de> {
    type Error = PostcardError;
    type KeyProbe = PostcardMapKeyProbe<'de>;
    type MapClaim = PostcardClaim<'de>;

    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        // The map iteration is driven by MapAccess::iterate, which creates key
        // probes with incrementing indices. We signal "more" by returning Entry
        // with a sentinel probe; MapAccess::iterate decides when to stop.
        // We encode the next field index in the probe via a "next_idx" field
        // that MapAccess::iterate had set. But MapValueClaim doesn't carry that.
        //
        // Solution: always return Entry — MapAccess::iterate stops when the arm
        // stack signals unsatisfied_count() == 0 in its own loop. We embed the
        // next index in the probe as `next_idx` which MapAccess stores in MapValueProbe.
        // We need to carry the next index through the claim.
        //
        // Instead: since PostcardClaim doesn't carry `next_idx`, MapAccess::iterate
        // tracks the index itself and does NOT use next_key at all — it drives the
        // loop manually.
        //
        // This default impl is unreachable for postcard struct deserialization.
        // For dynamic map use (e.g. HashMap via map_arms!), return Done immediately
        // since postcard's map path is only used for structs via positional keys.
        Ok(NextKey::Done(self))
    }
}

// --- Key probe ---

pub struct PostcardMapKeyProbe<'de> {
    pub(crate) src: &'de [u8],
    pub(crate) current_idx: usize,
    /// `true` for a dynamic-collection key slot (HashMap/BTreeMap via
    /// `CollectMap`), where the key is real wire bytes to be decoded via
    /// `deserialize_key`. `false` for a struct field, where fields have no
    /// wire key names at all and `deserialize_key` must stay a no-op `Miss`
    /// — dynamic-only, since the derive races `deserialize_key::<Match>`
    /// against `deserialize_key_by_index` for every struct field, and a real
    /// decode attempt there could misparse arbitrary field bytes.
    pub(crate) dynamic: bool,
}

impl<'de> PostcardMapKeyProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            current_idx: self.current_idx,
            dynamic: self.dynamic,
        }
    }
}

impl<'de> MapKeyProbe<'de> for PostcardMapKeyProbe<'de> {
    type Error = PostcardError;
    type KeyClaim = PostcardClaim<'de>;
    type KeySubDeserializer = PostcardSubDeserializer<'de>;

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
            // Struct fields have no wire key names — name-based matching always misses.
            return Ok(Probe::Miss);
        }
        let sub = PostcardSubDeserializer::new(self.src);
        K::deserialize(sub, extra).await
    }

    async fn deserialize_key_by_index(
        self,
        expected: usize,
    ) -> Result<Probe<(Self::KeyClaim, ())>, Self::Error> {
        if self.current_idx == expected {
            Ok(Probe::Hit((PostcardClaim { src: self.src }, ())))
        } else {
            Ok(Probe::Miss)
        }
    }
}

// --- Value probe ---

pub struct PostcardMapValueProbe<'de> {
    src: &'de [u8],
}

impl<'de> PostcardMapValueProbe<'de> {
    fn clone(&self) -> Self {
        Self { src: self.src }
    }
}

impl<'de> MapValueProbe<'de> for PostcardMapValueProbe<'de> {
    type Error = PostcardError;
    type MapClaim = PostcardClaim<'de>;
    type ValueClaim = PostcardClaim<'de>;
    type ValueSubDeserializer = PostcardSubDeserializer<'de>;

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
        let sub = PostcardSubDeserializer::new(self.src);
        V::deserialize(sub, extra).await
    }

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        Err(PostcardError::CannotSkip)
    }
}

// --- MapAccess ---

impl<'de> MapAccess<'de> for PostcardMapAccess<'de> {
    type Error = PostcardError;
    type MapClaim = PostcardClaim<'de>;
    type KeyProbe = PostcardMapKeyProbe<'de>;

    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        iterate_static(self.src, self.current, arms).await
    }

    async fn iterate_dyn<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        iterate_dynamic(self.src, arms).await
    }
}

/// Unbounded collection (HashMap/BTreeMap via `CollectMap`): postcard writes an
/// explicit varint length for these, unlike structs. Loops exactly `count`
/// times — NOT via `unsatisfied_count() == 0` like [`iterate_static`], since
/// `CollectMap`'s `unsatisfied_count()` is hardcoded to 0 and would terminate
/// before reading anything.
///
/// Split out from [`MapAccess::iterate`] (rather than an `if`/`else` inline)
/// so each mode gets its own async state machine instead of one fn's layout
/// covering the union of both — halves the await-point count the compiler has
/// to lay out per call site, which matters because `iterate` is monomorphized
/// deeply (every struct field type nests another `iterate` call).
async fn iterate_dynamic<'de, S: MapArmStack<'de, PostcardMapKeyProbe<'de>>>(
    src: &'de [u8],
    mut arms: S,
) -> Result<Probe<(PostcardClaim<'de>, S::Outputs)>, PostcardError> {
    let (count, consumed) = decode_varint(src)?;
    let mut src = &src[consumed..];

    for _ in 0..count {
        let kp = PostcardMapKeyProbe {
            src,
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

        src = value_claim.src;
    }

    Ok(Probe::Hit((PostcardClaim { src }, arms.take_outputs())))
}

/// Struct fields: no wire framing at all; driven by the arm stack becoming
/// satisfied (`unsatisfied_count() == 0`), matching fields positionally via
/// `current_idx`. See [`iterate_dynamic`] for why this is a separate fn.
async fn iterate_static<'de, S: MapArmStack<'de, PostcardMapKeyProbe<'de>>>(
    mut src: &'de [u8],
    mut current: usize,
    mut arms: S,
) -> Result<Probe<(PostcardClaim<'de>, S::Outputs)>, PostcardError> {
    loop {
        if arms.unsatisfied_count() == 0 {
            return Ok(Probe::Hit((PostcardClaim { src }, arms.take_outputs())));
        }

        let kp = PostcardMapKeyProbe {
            src,
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

        src = value_claim.src;
        current += 1;
    }
}

// ---------------------------------------------------------------------------
// SeqAccess / SeqEntry
// ---------------------------------------------------------------------------

pub struct PostcardSeqAccess<'de> {
    pub(crate) src: &'de [u8],
    pub(crate) remaining: usize,
}

pub struct PostcardSeqEntry<'de> {
    src: &'de [u8],
}

impl<'de> PostcardSeqEntry<'de> {
    fn clone(&self) -> Self {
        Self { src: self.src }
    }
}

#[inline(always)]
async fn postcard_seq_next<'de, const N: usize, F, Fut, R>(
    seq: PostcardSeqAccess<'de>,
    mut f: F,
) -> Result<Probe<Chunk<(PostcardSeqAccess<'de>, R), PostcardClaim<'de>>>, PostcardError>
where
    F: FnMut([PostcardSeqEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(PostcardClaim<'de>, R)>, PostcardError>>,
{
    if seq.remaining == 0 {
        return Ok(Probe::Hit(Chunk::Done(PostcardClaim { src: seq.src })));
    }
    let entry = PostcardSeqEntry { src: seq.src };
    let new_seq = PostcardSeqAccess {
        src: seq.src,
        remaining: seq.remaining - 1,
    };
    let (claim, r) = hit!(f(repeat(entry, |e| e.clone())).await);
    let updated_seq = PostcardSeqAccess {
        src: claim.src,
        remaining: new_seq.remaining,
    };
    Ok(Probe::Hit(Chunk::Data((updated_seq, r))))
}

impl<'de> SeqAccess<'de> for PostcardSeqAccess<'de> {
    type Error = PostcardError;
    type SeqClaim = PostcardClaim<'de>;
    type ElemClaim = PostcardClaim<'de>;
    type Elem = PostcardSeqEntry<'de>;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        postcard_seq_next(self, f).await
    }
}

impl<'de> SeqEntry<'de> for PostcardSeqEntry<'de> {
    type Error = PostcardError;
    type Claim = PostcardClaim<'de>;
    type SubDeserializer = PostcardSubDeserializer<'de>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = PostcardSubDeserializer::new(self.src);
        T::deserialize(sub, extra).await
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        Err(PostcardError::CannotSkip)
    }
}

// ---------------------------------------------------------------------------
// EnumAccess / EnumVariantProbe
// ---------------------------------------------------------------------------
//
// Postcard externally-tagged enums: varint discriminant, then payload.
// Unit variants: discriminant + zero payload bytes.
// Non-unit variants: discriminant + payload (struct as seq, tuple as seq, etc.)

pub struct PostcardEnumAccess<'de> {
    pub(crate) discriminant: usize,
    pub(crate) src: &'de [u8],
}

impl<'de> EnumAccess<'de> for PostcardEnumAccess<'de> {
    type Error = PostcardError;
    type Claim = PostcardClaim<'de>;
    type VariantProbe = PostcardEnumVariantProbe<'de>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStack<'de, Self::VariantProbe>,
    {
        let vp = PostcardEnumVariantProbe {
            discriminant: self.discriminant,
            src: self.src,
        };
        let (_idx, claim) = hit!(arms.race(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

pub struct PostcardEnumVariantProbe<'de> {
    pub(crate) discriminant: usize,
    pub(crate) src: &'de [u8],
}

impl<'de> PostcardEnumVariantProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            discriminant: self.discriminant,
            src: self.src,
        }
    }
}

impl<'de> EnumVariantProbe<'de> for PostcardEnumVariantProbe<'de> {
    type Error = PostcardError;
    type Claim = PostcardClaim<'de>;
    type PayloadDeserializer = PostcardSubDeserializer<'de>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    // Name-based methods: postcard has no wire names, but the local arm index
    // in each candidate maps directly to the wire discriminant by convention
    // (derive assigns arm indices 0, 1, 2, … matching declaration order, which
    // matches postcard's discriminant encoding).

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
                return Ok(Probe::Hit((PostcardClaim { src: self.src }, local_idx)));
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
                let sub = PostcardSubDeserializer::new(self.src);
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
            Ok(Probe::Hit((PostcardClaim { src: self.src }, expected_idx)))
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
        let sub = PostcardSubDeserializer::new(self.src);
        match T::deserialize(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((claim, expected_idx, v))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    async fn deserialize_value_by_shape<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
    {
        let sub = PostcardSubDeserializer::new(self.src);
        T::deserialize(sub, extra).await
    }
}
