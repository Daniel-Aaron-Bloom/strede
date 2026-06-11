use const_array_concat::ConcatableArray;
use core::future::Future;

use crate::map_arm::NextKey;
use crate::{Chunk, DeserializeError, Probe};

// ---------------------------------------------------------------------------
// Deserialize - the "what"  (mirrors serde::Deserialize)
// ---------------------------------------------------------------------------

/// Types that can deserialize themselves from a specific [`Deserializer`] `D`.
///
/// `D` is a trait-level generic so derive can express bounds like
/// `T: Deserialize<'de, <D::Entry as Entry<'de>>::SubDeserializer>` on the
/// impl block. `Extra` is an associated type so caller-supplied context flows
/// transparently through chained dispatches (`Option<T>::Extra = T::Extra`).
pub trait Deserialize<'de, D: Deserializer<'de>>: Sized {
    /// Caller-supplied side-channel context.  Most types use `Extra = ()`.
    type Extra;

    /// Deserialize `Self` from `d`, with caller-supplied side-channel `extra`.
    ///
    /// - `Ok(Probe::Hit((claim, value)))` - succeeded; thread `claim` back to the caller.
    /// - `Ok(Probe::Miss)` - token type didn't match; stream not advanced.
    /// - `Err(e)` - fatal format error (malformed data, I/O failure).
    async fn deserialize(d: D, extra: Self::Extra) -> Result<Probe<(D::Claim, Self)>, D::Error>;
}

/// Types whose deserialization starts from an already-opened [`MapAccess`].
///
/// Used for nested map-shaped fields (struct values inside a struct map),
/// flatten payloads, and tagged-enum variant payloads. The access type `M` is
/// a trait parameter — bounds on `M`'s projections live on the impl block,
/// not the method, so derive's generated `where` clauses compose cleanly.
pub trait DeserializeFromMap<'de, M: MapAccess<'de>>: Sized {
    type Extra;
    async fn deserialize_from_map(
        map: M,
        extra: Self::Extra,
    ) -> Result<Probe<(M::MapClaim, Self)>, M::Error>;
}

/// Types whose deserialization starts from an already-opened [`SeqAccess`].
///
/// Parallel to [`DeserializeFromMap`] for sequence-shaped values (tuples,
/// `Vec<T>`, tuple-variant payloads).
pub trait DeserializeFromSeq<'de, S: SeqAccess<'de>>: Sized {
    type Extra;
    async fn deserialize_from_seq(
        seq: S,
        extra: Self::Extra,
    ) -> Result<Probe<(S::SeqClaim, Self)>, S::Error>;
}

// ---------------------------------------------------------------------------
// Deserializer - stream handle
// ---------------------------------------------------------------------------

/// A stream of tokens that can be decoded into Rust values.
///
/// The deserializer owns the stream and is the sole means of advancing it -
/// all advancement goes through [`Deserializer::entry`].  Type probing is done
/// through [`Entry`] handles handed to the closure.
pub trait Deserializer<'de>: Sized {
    /// Fatal error type produced by this format.
    type Error: DeserializeError;

    /// Proof-of-consumption token returned from [`Deserializer::entry`].
    type Claim: 'de;

    /// The claim type produced by entry handles ([`Entry::Claim`]).
    /// Distinct from `Claim` to allow implementations to use different claim
    /// types at the deserializer level vs the entry level (e.g. flatten facades).
    type EntryClaim: 'de;

    /// Owned handle for one item slot.  See [`Entry`].
    type Entry: Entry<'de, Claim = Self::EntryClaim, Error = Self::Error>;

    /// Advance to the next item in the stream.
    ///
    /// Passes `N` owned [`Entry`] handles to `f`.  When a probe inside `f`
    /// resolves `Ok(Probe::Hit((claim, r)))`, the winning arm returns it;
    /// `entry` verifies the claim and returns `Ok(Probe::Hit((claim, r)))`.
    /// Returns `Err(e)` if a fatal error occurs before or during `f`.
    /// `Ok(Probe::Miss)` propagates upward if no probe matched.
    /// `Pending` until a token is available.
    ///
    /// Use `N > 1` to race multiple probe arms via [`select_probe!`](crate::select_probe) without
    /// borrow conflicts.  Handles dropped without resolving do not advance
    /// the stream.
    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>;
}

// ---------------------------------------------------------------------------
// Entry - type-probing handle for one item slot
// ---------------------------------------------------------------------------

/// Owned handle for one item slot, passed into the closure of [`Deserializer::entry`].
///
/// Each probe consumes `self` and resolves to `Ok(Probe::Hit((Claim, T)))` when
/// the current token is of type `T`, `Ok(Probe::Miss)` if the type doesn't
/// match or the token is a different kind, or `Err(e)` on a fatal format error.
/// The `Claim` must be returned from the closure so `next` can advance the stream.
///
/// # For implementors
///
/// Type mismatches **must** return `Ok(Probe::Miss)` - never `Err`.  `Err` is
/// reserved for fatal format errors (malformed data, I/O failure).
///
/// Use `N > 1` in `entry` to race arms via [`select_probe!`](crate::select_probe):
///
/// ```rust,ignore
/// let value = d.entry(|[e1, e2, e3, e4]| async {
///     select_probe! {
///         e1.deserialize_value::<bool, ()>(()),
///         e2.deserialize_value::<i64, ()>(()),
///         async move {
///             let (c, v) = hit!(e3.deserialize_str().await);
///             Ok(Probe::Hit((c, Value::Str(v))))
///         },
///         async move {
///             let mut m = hit!(e4.deserialize_map().await);
///             let (c, v) = collect_map(m).await?;
///             Ok(Probe::Hit((c, Value::Map(v))))
///         },
///     }
/// }).await?;
/// ```
pub trait Entry<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Proof-of-consumption token.  Must be returned from the `next` closure
    /// alongside the caller's value so the deserializer can advance the stream.
    type Claim: 'de;

    /// Concrete sub-deserializer this Entry spawns to delegate
    /// [`Deserialize::deserialize`] for arbitrary `T`. Format-specific.
    type SubDeserializer: Deserializer<'de, Claim = Self::Claim, Error = Self::Error>;

    type StrChunks: StrAccess<Claim = Self::Claim, Error = Self::Error>;
    type BytesChunks: BytesAccess<Claim = Self::Claim, Error = Self::Error>;
    type NumberChunks<Enc: NumberEncoding>: NumberAccess<Enc, Claim = Self::Claim, Error = Self::Error>;
    type Map: MapAccess<'de, MapClaim = Self::Claim, Error = Self::Error>;
    type Seq: SeqAccess<'de, SeqClaim = Self::Claim, Error = Self::Error>;
    type Enum: EnumAccess<'de, Claim = Self::Claim, Error = Self::Error>;

    /// Attempt a **zero-copy** string borrow.
    ///
    /// Hits only when the format can return the entire string as a single
    /// contiguous `&'de str` slice - i.e. with no escape sequences or
    /// transcoding.  Returns `Ok(Probe::Miss)` when the token is a string
    /// but cannot be represented that way (e.g. JSON `"\n"`), so that a
    /// concurrent [`Entry::deserialize_str_chunks`] arm can take over via
    /// `select_probe!`.  This makes `Cow<str>` callers easy to write: race
    /// the two methods and borrow when free, allocate only when necessary.
    ///
    /// `Ok(Probe::Miss)` on type mismatch *or* when zero-copy is unavailable;
    /// `Err` on fatal format error only.
    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error>;
    /// Begin streaming a string chunk-by-chunk.  The [`Entry::Claim`] is
    /// returned by [`StrAccess::next_str`] when the string is exhausted.
    /// Handles all strings including those with escape sequences.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error>;
    /// Attempt a **zero-copy** bytes borrow.  Same Miss semantics as
    /// [`Entry::deserialize_str`]: hits only when the entire value is
    /// available as a contiguous `&'de [u8]` slice; returns `Miss` otherwise
    /// so a [`Entry::deserialize_bytes_chunks`] arm can handle the rest.
    /// `Ok(Probe::Miss)` on type mismatch *or* when zero-copy is unavailable;
    /// `Err` on fatal format error only.
    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error>;
    /// Begin streaming a byte string chunk-by-chunk.  The [`Entry::Claim`] is
    /// returned by [`BytesAccess::next_bytes`] when the byte string is exhausted.
    /// Handles all byte strings including those requiring transcoding.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error>;
    /// Begin streaming a variable-length number chunk-by-chunk.
    /// The [`Entry::Claim`] is returned by [`NumberAccess::next_number_chunk`] when the
    /// number is exhausted. `Enc` selects the wire encoding; formats hit only on
    /// encodings they natively support and return `Ok(Probe::Miss)` otherwise.
    /// `Ok(Probe::Miss)` on type or encoding mismatch; `Err` on fatal format error only.
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error>;

    /// Begin reading a map.  The [`Entry::Claim`] is returned by
    /// [`MapAccess::iterate`] when the map is exhausted.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error>;
    /// Begin reading a sequence.  The [`Entry::Claim`] is returned by
    /// [`SeqAccess::next`] when the sequence is exhausted.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error>;

    /// Deserialize a value of type `T` by spawning a [`Self::SubDeserializer`]
    /// and delegating to `T::deserialize`.
    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>;

    /// Deserialize an optional value.
    ///
    /// - Null token → `Ok(Probe::Hit((claim, None)))`
    /// - Token matching `T` → `Ok(Probe::Hit((claim, Some(v))))`
    /// - Token matching neither → `Ok(Probe::Miss)`
    /// - Fatal format error → `Err(e)`
    async fn deserialize_option<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>;

    /// Open a map and forward into `T`'s [`DeserializeFromMap`] impl.
    ///
    /// Saves the sub-deserializer trampoline for nested struct fields, and
    /// is the dispatch point for flatten/tagged-enum payloads.
    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>;

    /// Open a seq and forward into `T`'s [`DeserializeFromSeq`] impl.
    async fn deserialize_seq_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeq<'de, Self::Seq>;

    /// Begin externally-tagged enum dispatch.
    ///
    /// The returned [`EnumAccess`] handle drives variant identification via
    /// [`EnumAccess::iterate`].  `Ok(Probe::Miss)` on type mismatch; `Err`
    /// on fatal format error only.
    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error>;

    /// Open an enum and forward into `T`'s [`DeserializeFromEnum`] impl.
    ///
    /// Convenience wrapper over [`Entry::deserialize_enum`] + [`EnumAccess::iterate`].
    async fn deserialize_enum_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnum<'de, Self::Enum>;

    /// Fork a sibling entry handle for the same item slot.
    ///
    /// Both `self` and the returned handle refer to the same slot.
    /// Whichever resolves a probe first claims the slot; the other
    /// becomes inert and may be dropped without advancing the stream.
    fn fork(&mut self) -> Self;

    /// Consume and discard the current token regardless of its type
    /// (scalar, string, map, or sequence).
    ///
    /// Always succeeds on well-formed input.  Returns the [`Claim`](Entry::Claim)
    /// so the stream can advance.  `Err` only on malformed data.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ---------------------------------------------------------------------------
// StrAccess
// ---------------------------------------------------------------------------

/// Streams a string in zero-copy chunks.  Obtained from [`Entry::deserialize_str_chunks`].
///
/// Strings are primitives - the type is already known, so no probing or
/// racing is needed.  This adapter exists solely for formats that cannot
/// deliver the value as a single contiguous slice (e.g. escape-sequence
/// synthesis in JSON).
///
/// ```rust,ignore
/// let mut chunks = hit!(e.deserialize_str_chunks().await);
/// let mut out = String::new();
/// let claim = loop {
///     match chunks.next_str(|s| out.push_str(s)).await? {
///         Chunk::Data((new, ())) => chunks = new,
///         Chunk::Done(claim) => break claim,
///     }
/// };
/// ```
pub trait StrAccess: Sized {
    /// Proof-of-consumption token, returned when the string is exhausted.
    /// Must match the enclosing [`Entry::Claim`].
    type Claim;
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Advance to the next chunk, passing it to `f`.
    ///
    /// - `Ok(Chunk::Data((self, r)))` - next chunk processed; accessor returned for the next call.
    /// - `Ok(Chunk::Done(claim))` - string exhausted; claim is now valid.
    /// - `Err(e)` - fatal format error.
    /// - `Pending` - no data yet.
    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// BytesAccess
// ---------------------------------------------------------------------------

/// Streams a byte string in zero-copy chunks.  Obtained from [`Entry::deserialize_bytes_chunks`].
///
/// Byte strings are primitives - the type is already known, so no probing or
/// racing is needed.  This adapter exists solely for formats that cannot
/// deliver the value as a single contiguous slice.
///
/// ```rust,ignore
/// let mut chunks = hit!(e.deserialize_bytes_chunks().await);
/// let mut out = Vec::new();
/// let claim = loop {
///     match chunks.next_bytes(|b| out.extend_from_slice(b)).await? {
///         Chunk::Data((new, ())) => chunks = new,
///         Chunk::Done(claim) => break claim,
///     }
/// };
/// ```
pub trait BytesAccess: Sized {
    /// Proof-of-consumption token, returned when the byte string is exhausted.
    /// Must match the enclosing [`Entry::Claim`].
    type Claim;
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Advance to the next chunk, passing it to `f`.
    ///
    /// - `Ok(Chunk::Data((self, r)))` - next chunk processed; accessor returned for the next call.
    /// - `Ok(Chunk::Done(claim))` - byte string exhausted; claim is now valid.
    /// - `Err(e)` - fatal format error.
    /// - `Pending` - no data yet.
    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// NumberEncoding
// ---------------------------------------------------------------------------

/// Describes the wire encoding of a variable-length number token.
///
/// Each implementor declares `Data` (either `str` for text or `[u8]` for
/// binary blobs) and provides conversions from the two raw representations
/// formats hold internally.  The `NAME` const lets format impls branch at
/// compile time (e.g. `if Enc::NAME == Ascii::NAME`) without specialization.
///
/// Three built-in encodings are provided: [`Ascii`], [`BigEndian`], and
/// [`LittleEndian`].  Third-party formats can add their own.
pub trait NumberEncoding {
    /// The slice type yielded to callers: `str` for text, `[u8]` for binary.
    type Data: ?Sized;

    /// Stable string identifier for this encoding.
    /// Format impls use `Enc::NAME == SomeEncoding::NAME` to decide Hit/Miss.
    const NAME: &'static str;

    /// Convert a raw byte slice to `&Self::Data`.
    /// Panics if the bytes are not valid for this encoding (e.g. non-UTF-8 for `Ascii`).
    fn from_bytes(bytes: &[u8]) -> &Self::Data;

    /// Convert a `&str` to `&Self::Data`.
    fn from_str(s: &str) -> &Self::Data;
}

/// ASCII text encoding — formats that emit numbers as decimal text (e.g. JSON).
pub struct Ascii;
/// Big-endian binary encoding — formats that emit numbers as big-endian byte blobs.
pub struct BigEndian;
/// Little-endian binary encoding — formats that emit numbers as little-endian byte blobs.
pub struct LittleEndian;

impl NumberEncoding for Ascii {
    type Data = str;
    const NAME: &'static str = "ascii";
    fn from_bytes(bytes: &[u8]) -> &str {
        core::str::from_utf8(bytes).unwrap()
    }
    fn from_str(s: &str) -> &str {
        s
    }
}

impl NumberEncoding for BigEndian {
    type Data = [u8];
    const NAME: &'static str = "big-endian";
    fn from_bytes(b: &[u8]) -> &[u8] {
        b
    }
    fn from_str(s: &str) -> &[u8] {
        s.as_bytes()
    }
}

impl NumberEncoding for LittleEndian {
    type Data = [u8];
    const NAME: &'static str = "little-endian";
    fn from_bytes(b: &[u8]) -> &[u8] {
        b
    }
    fn from_str(s: &str) -> &[u8] {
        s.as_bytes()
    }
}

// ---------------------------------------------------------------------------
// NumberAccess
// ---------------------------------------------------------------------------

/// Streams a variable-length number in zero-copy chunks.
/// Obtained from [`Entry::deserialize_number_chunks`].
///
/// `Enc` controls the wire encoding: [`Ascii`] for text formats (JSON),
/// [`BigEndian`] / [`LittleEndian`] for binary formats (msgpack bigints, etc.).
/// Each chunk is passed as `&Enc::Data` — `&str` for `Ascii`, `&[u8]` for binary.
///
/// Useful for arbitrary-precision number types that cannot be represented by
/// fixed primitives like `i64` or `f64`. Fixed-width natives use their own
/// `deserialize_number_*` methods instead.
///
/// ```rust,ignore
/// let mut chunks = hit!(e.deserialize_number_chunks::<Ascii>().await);
/// let mut out = String::new();
/// let claim = loop {
///     match chunks.next_number_chunk(|s| out.push_str(s)).await? {
///         Chunk::Data((new, ())) => chunks = new,
///         Chunk::Done(claim) => break claim,
///     }
/// };
/// ```
pub trait NumberAccess<Enc: NumberEncoding = Ascii>: Sized {
    /// Proof-of-consumption token, returned when the number is exhausted.
    /// Must match the enclosing [`Entry::Claim`].
    type Claim;
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Advance to the next chunk, passing it to `f`.
    ///
    /// - `Ok(Chunk::Data((self, r)))` — next chunk processed; accessor returned for the next call.
    /// - `Ok(Chunk::Done(claim))` — number exhausted; claim is now valid.
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next_number_chunk<R>(
        self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Map access
// ---------------------------------------------------------------------------

/// A key probe for a single map key in the borrow family. Forkable for racing multiple arms.
pub trait MapKeyProbe<'de>: Sized {
    type Error: DeserializeError;
    type KeyClaim: MapKeyClaim<'de, Error = Self::Error>;

    /// Concrete sub-deserializer this key probe spawns for [`Deserialize`] dispatch.
    type KeySubDeserializer: Deserializer<'de, Claim = Self::KeyClaim, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: Deserialize<'de, Self::KeySubDeserializer>;

    /// Match the key by positional index rather than by name.
    ///
    /// Returns `Ok(Probe::Hit((claim, ())))` if this key corresponds to the
    /// field at position `expected` (0-based). Returns `Ok(Probe::Miss)` by
    /// default — formats that support positional access override this.
    async fn deserialize_key_by_index(
        self,
        _expected: usize,
    ) -> Result<Probe<(Self::KeyClaim, ())>, Self::Error> {
        Ok(Probe::Miss)
    }
}

/// Proof that a key was consumed. Converts into a value probe.
pub trait MapKeyClaim<'de>: Sized {
    type Error: DeserializeError;
    type MapClaim: 'de;
    type ValueProbe: MapValueProbe<'de, MapClaim = Self::MapClaim, Error = Self::Error>;

    /// Consume this key claim and produce a value probe for the corresponding
    /// map value. Format-specific (e.g. JSON reads `:` and the value start token).
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error>;
}

/// A value probe that can deserialize a value or skip it (borrow family).
pub trait MapValueProbe<'de>: Sized {
    type Error: DeserializeError;
    type MapClaim: 'de;
    type ValueClaim: MapValueClaim<'de, MapClaim = Self::MapClaim, Error = Self::Error>;

    /// Concrete sub-deserializer this value probe spawns for [`Deserialize`] dispatch.
    type ValueSubDeserializer: Deserializer<'de, Claim = Self::ValueClaim, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: Deserialize<'de, Self::ValueSubDeserializer>;

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error>;
}

/// Proof that a value was consumed. Advances to the next key or ends the map (borrow family).
pub trait MapValueClaim<'de>: Sized {
    type Error: DeserializeError;
    type KeyProbe: MapKeyProbe<'de, Error = Self::Error>;
    type MapClaim: 'de;

    async fn next_key(
        self,
        unsatisfied: usize,
        open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// MapAccess - iterates key-value pairs via arm stacks
// ---------------------------------------------------------------------------

/// Iterates the key-value pairs of a map.  Obtained from [`Entry::deserialize_map`].
pub trait MapAccess<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Proof-of-consumption token returned on map exhaustion; must match
    /// the enclosing [`Entry::Claim`].
    type MapClaim: 'de;

    type KeyProbe: MapKeyProbe<'de, Error = Self::Error>;

    /// Drive the map iteration with the given arm stack.
    ///
    /// Returns `Hit((MapClaim, Outputs))` on success, `Miss` if a value
    /// type mismatched or a required field was missing, `Err` on format errors.
    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Type aliases for borrow map access
// ---------------------------------------------------------------------------

/// Shorthand for the key probe type reachable from a `Deserializer`.
pub type KP<'de, D> =
    <<<D as Deserializer<'de>>::Entry as Entry<'de>>::Map as MapAccess<'de>>::KeyProbe;

/// Shorthand for the value claim type reachable from a borrow key probe type.
pub type VC<'de, KP> =
    <<<KP as MapKeyProbe<'de>>::KeyClaim as MapKeyClaim<'de>>::ValueProbe as MapValueProbe<'de>>::ValueClaim;

/// Shorthand for the value probe type reachable from a borrow key probe type.
pub type VP<'de, KP> = <<KP as MapKeyProbe<'de>>::KeyClaim as MapKeyClaim<'de>>::ValueProbe;

/// Shorthand for the value probe type reachable directly from a `Deserializer`.
pub type VP2<'de, D> = <<KP<'de, D> as MapKeyProbe<'de>>::KeyClaim as MapKeyClaim<'de>>::ValueProbe;

/// Shorthand for the sequence element entry type reachable directly from a `Deserializer`.
pub type SE<'de, D> =
    <<<D as Deserializer<'de>>::Entry as Entry<'de>>::Seq as SeqAccess<'de>>::Elem;

/// Shorthand for the enum variant probe type reachable from a `Deserializer`.
pub type EVP<'de, D> =
    <<<D as Deserializer<'de>>::Entry as Entry<'de>>::Enum as EnumAccess<'de>>::VariantProbe;

pub use crate::enum_arm::EnumArmStack;
pub use crate::map_arm::MapArmStack;

// ---------------------------------------------------------------------------
// SeqAccess
// ---------------------------------------------------------------------------

/// Iterates the elements of a sequence.  Obtained from [`Entry::deserialize_seq`].
///
/// ```rust,ignore
/// let mut out = Vec::new();
/// loop {
///     match hit!(seq.next(|[e]| async {
///         let (claim, v) = hit!(e.get::<u32, ()>(()).await);
///         Ok(Probe::Hit((claim, v)))
///     }).await) {
///         Chunk::Data((n_seq, v)) => {
///             out.push(v)
///             seq = n_seq;
///         },
///         Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
///     }
/// }
/// ```
pub trait SeqAccess<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Proof-of-consumption token returned on sequence exhaustion; must match
    /// the enclosing [`Entry::Claim`].
    type SeqClaim: 'de;

    /// Proof-of-consumption token produced by [`SeqEntry`] probes; threaded
    /// back through the closure return value to advance the sequence.
    type ElemClaim: 'de;

    /// Owned handle for one sequence element.  See [`SeqEntry`].
    type Elem: SeqEntry<'de, Claim = Self::ElemClaim, Error = Self::Error>;

    /// Advance to the next element, passing `N` owned [`SeqEntry`] handles to `f`.
    ///
    /// Returns:
    /// - `Ok(Probe::Hit(Chunk::Data(r)))` - an element was consumed.
    /// - `Ok(Probe::Hit(Chunk::Done(claim)))` - sequence exhausted.
    /// - `Ok(Probe::Miss)` - closure returned Miss.
    /// - `Err(e)` - fatal format error.
    /// - `Pending` - no data yet.
    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>;
}

/// Owned handle for one element in a sequence.
pub trait SeqEntry<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;
    /// Proof-of-consumption token; must match the enclosing [`SeqAccess::ElemClaim`].
    type Claim: 'de;

    /// Concrete sub-deserializer this seq element spawns for [`Deserialize`] dispatch.
    type SubDeserializer: Deserializer<'de, Claim = Self::Claim, Error = Self::Error>;

    /// Deserialize the element as `T` by spawning a [`Self::SubDeserializer`]
    /// and delegating to `T::deserialize`.
    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>;

    /// Fork a sibling element handle for the same sequence slot.
    fn fork(&mut self) -> Self;

    /// Consume and discard the element without deserializing it.
    /// Returns the [`Claim`](SeqEntry::Claim) so the stream can advance.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ---------------------------------------------------------------------------
// EnumAccess - variant identification + payload dispatch
// ---------------------------------------------------------------------------

/// The format's variant-identification surface, analogous to [`MapKeyProbe`].
///
/// Each arm in an [`EnumArmStack`] calls one of the probe methods on a forked
/// `EnumVariantProbe` to simultaneously identify its variant *and* deserialize
/// its payload.  Because identification and payload are read in a single call,
/// formats where the discriminant and payload are interleaved (e.g. a msgpack
/// array whose first element is the variant index and second is the payload) are
/// handled naturally.
///
/// Formats implement the methods they actually support and return
/// `Ok(Probe::Miss)` from the rest.  A default `Miss` impl is provided for every
/// method so formats only need to fill in what applies to them.
pub trait EnumVariantProbe<'de>: Sized {
    type Error: DeserializeError;

    /// Proof-of-consumption produced by a successful probe.  Must match the
    /// enclosing [`Entry::Claim`].
    type Claim: 'de;

    /// Sub-deserializer used to deserialize non-unit variant payloads.
    type PayloadDeserializer: Deserializer<'de, Claim = Self::Claim, Error = Self::Error>;

    /// Fork a sibling probe at the same read position.
    ///
    /// All forked probes must be driven concurrently (e.g. via [`select_probe!`](crate::select_probe)).
    /// See [`crate::owned::DeserializerOwned`] for the deadlock hazard.
    fn fork(&mut self) -> Self;

    // --- string-name based (JSON, CBOR text, msgpack str-tagged) ---

    /// Match a **unit** variant against `candidates` by comparing the current
    /// token to each `(wire_name, local_idx)` pair.
    ///
    /// Returns `Ok(Probe::Hit((Claim, matched_local_idx)))` on a match,
    /// `Ok(Probe::Miss)` if the current token is not a string or does not match
    /// any candidate.
    async fn deserialize_unit_by_name<W>(
        self,
        _candidates: W,
    ) -> Result<Probe<(Self::Claim, usize)>, Self::Error>
    where
        W: ConcatableArray<T = (&'static str, usize)> + Copy + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        Ok(Probe::Miss)
    }

    /// Match a **non-unit** variant by name and deserialize its payload as `T`.
    ///
    /// Returns `Ok(Probe::Hit((Claim, matched_local_idx, value)))` on a match,
    /// `Ok(Probe::Miss)` if the current token is not a single-key map or the key
    /// does not match any candidate.
    async fn deserialize_payload_by_name<T, W>(
        self,
        _candidates: W,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
        W: ConcatableArray<T = (&'static str, usize)> + Copy + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        Ok(Probe::Miss)
    }

    // --- index/positional based (msgpack int-tagged, positional formats) ---

    /// Match a **unit** variant by integer or positional index.
    ///
    /// Returns `Ok(Probe::Hit((Claim, expected_idx)))` on a match,
    /// `Ok(Probe::Miss)` otherwise.
    async fn deserialize_unit_by_index(
        self,
        _expected_idx: usize,
    ) -> Result<Probe<(Self::Claim, usize)>, Self::Error> {
        Ok(Probe::Miss)
    }

    /// Match a **non-unit** variant by index and deserialize its payload as `T`.
    ///
    /// Returns `Ok(Probe::Hit((Claim, expected_idx, value)))` on a match,
    /// `Ok(Probe::Miss)` otherwise.
    async fn deserialize_payload_by_index<T>(
        self,
        _expected_idx: usize,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
    {
        Ok(Probe::Miss)
    }

    // --- untagged (shape-based) ---

    /// Try to deserialize `T` directly from the current token with no discriminant.
    ///
    /// Used for untagged variants: each arm races concurrently with a forked probe
    /// via `EnumArmStack::race`. Returns `Hit((Claim, value))` if the token matches
    /// `T`'s shape, `Miss` otherwise.  Default impl: `Ok(Probe::Miss)`.
    async fn deserialize_value_by_shape<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
    {
        Ok(Probe::Miss)
    }
}

/// An in-progress enum access handle, analogous to [`MapAccess`].
///
/// The format drives variant identification by calling [`EnumAccess::iterate`]
/// with an [`EnumArmStack`].  Each arm's closure receives a forked
/// [`EnumVariantProbe`] and is fully responsible for identifying its variant
/// and deserializing its payload in a single call.
///
/// Obtained from [`Entry::deserialize_enum`].
pub trait EnumAccess<'de>: Sized {
    type Error: DeserializeError;

    /// Proof-of-consumption returned on successful variant dispatch.  Must
    /// match the enclosing [`Entry::Claim`].
    type Claim: 'de;

    type VariantProbe: EnumVariantProbe<'de, Claim = Self::Claim, Error = Self::Error>;

    /// Drive variant dispatch with the given arm stack.
    ///
    /// Returns `Hit((Claim, Outputs))` when an arm matched,
    /// `Miss` when no arm matched, `Err` on a fatal format error.
    async fn iterate<S>(self, arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: crate::enum_arm::EnumArmStack<'de, Self::VariantProbe>;
}

/// Types whose deserialization starts from an already-opened [`EnumAccess`].
///
/// Analogous to [`DeserializeFromMap`].  Used for externally-tagged enum dispatch:
/// the derive macro emits a `DeserializeFromEnum` impl for each enum type and
/// the `Deserialize` impl delegates to [`Entry::deserialize_enum_into`].
pub trait DeserializeFromEnum<'de, E: EnumAccess<'de>>: Sized {
    type Extra;
    async fn deserialize_from_enum(
        e: E,
        extra: Self::Extra,
    ) -> Result<Probe<(E::Claim, Self)>, E::Error>;
}

// ---------------------------------------------------------------------------
// Universal Deserialize impls (primitives ship per-format)
// ---------------------------------------------------------------------------

impl<'de: 'a, 'a, D: Deserializer<'de>> Deserialize<'de, D> for &'a str {
    type Extra = ();
    async fn deserialize(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_str()).await
    }
}

impl<'de: 'a, 'a, D: Deserializer<'de>> Deserialize<'de, D> for &'a [u8] {
    type Extra = ();
    async fn deserialize(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_bytes()).await
    }
}

impl<'de, D, T> Deserialize<'de, D> for Option<T>
where
    D: Deserializer<'de>,
    T: Deserialize<'de, <D::Entry as Entry<'de>>::SubDeserializer>,
{
    type Extra = T::Extra;
    async fn deserialize(d: D, extra: Self::Extra) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut extra = Some(extra);
        d.entry(move |[e]| {
            let extra = extra.take().expect("entry closure called more than once");
            async move { e.deserialize_option::<T>(extra).await }
        })
        .await
    }
}
