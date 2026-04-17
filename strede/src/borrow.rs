use core::future::Future;

use crate::{Chunk, DeserializeError, Probe};

// ---------------------------------------------------------------------------
// Deserialize — the "what"  (mirrors serde::Deserialize)
// ---------------------------------------------------------------------------

/// Types that can deserialize themselves from a [`Deserializer`] stream.
///
/// Implement this to make a type readable by any format that implements
/// [`Deserializer`].  The method drives the deserializer forward and returns
/// the fully constructed value, or a fatal format error.
///
/// The `Extra` type parameter is side-channel context passed into
/// [`Entry::deserialize_value`], [`SeqEntry::get`], [`MapKeyEntry::key`], and
/// [`MapValueEntry::value`] at the call site.  Defaults to `()` for types that
/// need no extra context.
pub trait Deserialize<'de, Extra = ()>: Sized {
    /// Deserialize `Self` from `d`, with caller-supplied side-channel `extra`.
    ///
    /// - `Ok(Probe::Hit((claim, value)))` — succeeded; thread `claim` back to the caller.
    /// - `Ok(Probe::Miss)` — token type didn't match; stream not advanced.
    /// - `Err(e)` — fatal format error (malformed data, I/O failure).
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>;
}

// ---------------------------------------------------------------------------
// Deserializer — stream handle
// ---------------------------------------------------------------------------

/// A stream of tokens that can be decoded into Rust values.
///
/// The deserializer owns the stream and is the sole means of advancing it —
/// all advancement goes through [`Deserializer::entry`].  Type probing is done
/// through [`Entry`] handles handed to the closure.
pub trait Deserializer<'de>: Sized {
    /// Fatal error type produced by this format.
    type Error: DeserializeError;

    /// Proof-of-consumption token returned from [`Deserializer::entry`].
    type Claim: 'de;

    /// Owned handle for one item slot.  See [`Entry`].
    type Entry: Entry<'de, Error = Self::Error>;

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
        Fut: Future<Output = Result<Probe<(<Self::Entry as Entry<'de>>::Claim, R)>, Self::Error>>;
}

// ---------------------------------------------------------------------------
// Entry — type-probing handle for one item slot
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
/// Type mismatches **must** return `Ok(Probe::Miss)` — never `Err`.  `Err` is
/// reserved for fatal format errors (malformed data, I/O failure).
///
/// Use `N > 1` in `entry` to race arms via [`select_probe!`](crate::select_probe):
///
/// ```rust,ignore
/// let value = d.entry(|[e1, e2, e3, e4]| async {
///     select_probe! {
///         e1.deserialize_bool(),
///         e2.deserialize_i64(),
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

    type StrChunks: StrAccess<Claim = Self::Claim, Error = Self::Error>;
    type BytesChunks: BytesAccess<Claim = Self::Claim, Error = Self::Error>;
    type Map: MapAccess<'de, Claim = Self::Claim, Error = Self::Error>;
    type Seq: SeqAccess<'de, Claim = Self::Claim, Error = Self::Error>;

    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error>;
    /// Attempt a **zero-copy** string borrow.
    ///
    /// Hits only when the format can return the entire string as a single
    /// contiguous `&'de str` slice — i.e. with no escape sequences or
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
    /// Begin reading a map.  The [`Entry::Claim`] is returned by
    /// [`MapAccess::next_kv`] when the map is exhausted.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error>;
    /// Begin reading a sequence.  The [`Entry::Claim`] is returned by
    /// [`SeqAccess::next`] when the sequence is exhausted.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error>;

    /// Deserialize an optional value.
    ///
    /// - Null token → `Ok(Probe::Hit((claim, None)))`
    /// - Token matching `T::deserialize` → `Ok(Probe::Hit((claim, Some(v))))`
    /// - Token matching neither → `Ok(Probe::Miss)`
    /// - Fatal format error → `Err(e)`
    async fn deserialize_option<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>;

    /// Probe for a null token.
    ///
    /// - `Ok(Probe::Hit(claim))` — null token consumed.
    /// - `Ok(Probe::Miss)` — token is not null.
    /// - `Err(e)` — fatal format error.
    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error>;

    /// Delegate to `T::deserialize` from this entry handle, forwarding `extra`.
    ///
    /// Creates a sub-deserializer with the current token pre-loaded and
    /// calls `T::deserialize`.  Returns `Hit` if `T` matched, `Miss` if
    /// `T::deserialize` returned `Miss`.
    async fn deserialize_value<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

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
/// Strings are primitives — the type is already known, so no probing or
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

    /// Fork a sibling accessor at the same read position.
    ///
    /// Both `self` and the returned accessor are independent: advancing one
    /// does not affect the other.  Both start from the current chunk
    /// position and must each be driven to `Done` independently.
    fn fork(&mut self) -> Self;

    /// Advance to the next chunk, passing it to `f`.
    ///
    /// - `Ok(Chunk::Data((self, r)))` — next chunk processed; accessor returned for the next call.
    /// - `Ok(Chunk::Done(claim))` — string exhausted; claim is now valid.
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
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
/// Byte strings are primitives — the type is already known, so no probing or
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

    /// Fork a sibling accessor at the same read position.
    ///
    /// See [`StrAccess::fork`] for full semantics.
    fn fork(&mut self) -> Self;

    /// Advance to the next chunk, passing it to `f`.
    ///
    /// - `Ok(Chunk::Data((self, r)))` — next chunk processed; accessor returned for the next call.
    /// - `Ok(Chunk::Done(claim))` — byte string exhausted; claim is now valid.
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// MapAccess
// ---------------------------------------------------------------------------

/// Iterates the key-value pairs of a map.  Obtained from [`Entry::deserialize_map`].
pub trait MapAccess<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Proof-of-consumption token returned on map exhaustion; must match
    /// the enclosing [`Entry::Claim`].
    type Claim: 'de;

    /// Owned handle for reading one key.  See [`MapKeyEntry`].
    type KeyEntry: MapKeyEntry<'de, Claim = Self::Claim, Error = Self::Error>;

    /// Fork a sibling accessor at the same map position.
    ///
    /// Both `self` and the returned accessor are independent iterators
    /// starting from the current pair.  Each must consume all remaining
    /// pairs (or be dropped) independently.
    fn fork(&mut self) -> Self;

    /// Advance to the next key-value pair, passing `N` owned [`MapKeyEntry`]
    /// handles to `f`.  The closure must return `Ok(Probe::Hit((claim, r)))` where
    /// `claim` is the [`MapValueEntry::Claim`] threaded out through
    /// [`MapKeyEntry::key`], or `Ok(Probe::Miss)` if no key matched.
    ///
    /// Returns:
    /// - `Ok(Probe::Hit(Chunk::Data((self, r))))` — a pair was consumed.
    /// - `Ok(Probe::Hit(Chunk::Done(claim)))` — map exhausted.
    /// - `Ok(Probe::Miss)` — closure returned Miss (no key probe matched).
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next_kv<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
    where
        F: FnMut([Self::KeyEntry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned handle for reading the key of one map pair.
///
/// `key` follows the same lambda pattern as [`Deserializer::entry`]: the closure
/// receives `(&K, [ValueEntry; N])` — a reference to the decoded key and `N`
/// owned [`MapValueEntry`] handles — so it can inspect the key and read the
/// value, returning `Ok(Probe::Hit((ValueClaim, R)))` or `Ok(Probe::Miss)` for
/// unknown keys.  `key` returns `Ok(Probe::Hit((Claim, K, R)))` on success.
///
/// ```rust,ignore
/// let mut out = HashMap::new();
/// loop {
///     match map.next_kv(|[ke]| async {
///         match ke.key(|_k: &String, [ve]| async {
///             let (claim, v) = hit!(ve.value::<u32, ()>(()).await);
///             Ok(Probe::Hit((claim, v)))
///         }).await? {
///             Probe::Hit((claim, k, v)) => Ok(Probe::Hit((claim, (k, v)))),
///             Probe::Miss => Ok(Probe::Miss),
///         }
///     }).await? {
///         Probe::Hit(Chunk::Data((k, v))) => { out.insert(k, v); }
///         Probe::Hit(Chunk::Done(claim)) => return Ok(Probe::Hit((claim, out))),
///         Probe::Miss => { /* unknown key */ }
///     }
/// }
/// ```
pub trait MapKeyEntry<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;
    /// Proof-of-consumption token; must match the enclosing [`MapAccess::Claim`].
    type Claim: 'de;
    type ValueEntry: MapValueEntry<'de, Claim = Self::Claim, Error = Self::Error>;

    /// Fork a sibling key-entry handle for the same map pair.
    ///
    /// Both handles refer to the same key slot; whichever resolves `key`
    /// first claims the pair.
    fn fork(&mut self) -> Self
    where
        Self: Sized;

    async fn key<K: Deserialize<'de, KExtra>, KExtra, const N: usize, F, Fut, R>(
        self,
        extra: KExtra,
        f: F,
    ) -> Result<Probe<(Self::Claim, K, R)>, Self::Error>
    where
        F: FnMut(&K, [Self::ValueEntry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned handle for reading the value of one map pair.
///
/// Constructed by the [`MapKeyEntry::key`] implementation and passed as the
/// `[ValueEntry; N]` argument to its closure.
pub trait MapValueEntry<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;
    /// Proof-of-consumption token; must match the enclosing [`MapAccess::Claim`].
    type Claim: 'de;

    /// Deserialize the value as `V`, forwarding `extra` into `V::deserialize`.
    /// Returns `Ok(Probe::Hit((claim, value)))`; the claim must be threaded out
    /// through [`MapKeyEntry::key`] and back to [`MapAccess::next_kv`] to advance
    /// the stream.  `Ok(Probe::Miss)` if `V::deserialize` misses; `Err` on
    /// fatal format error only.
    async fn value<V: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, V)>, Self::Error>;

    /// Fork a sibling value-entry handle for the same map value slot.
    ///
    /// Both handles refer to the same value; whichever resolves `value`
    /// first claims it.
    fn fork(&mut self) -> Self;

    /// Consume and discard the value without deserializing it.
    /// Returns the [`Claim`](MapValueEntry::Claim) so the stream can advance.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

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
    type Claim: 'de;

    /// Owned handle for one sequence element.  See [`SeqEntry`].
    type Elem: SeqEntry<'de, Claim = Self::Claim, Error = Self::Error>;

    /// Fork a sibling accessor at the same sequence position.
    ///
    /// See [`MapAccess::fork`] for full semantics.
    fn fork(&mut self) -> Self;

    /// Advance to the next element, passing `N` owned [`SeqEntry`] handles to `f`.
    ///
    /// Returns:
    /// - `Ok(Probe::Hit(Chunk::Data(r)))` — an element was consumed.
    /// - `Ok(Probe::Hit(Chunk::Done(claim)))` — sequence exhausted.
    /// - `Ok(Probe::Miss)` — closure returned Miss.
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned handle for one element in a sequence.
pub trait SeqEntry<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;
    /// Proof-of-consumption token; must match the enclosing [`SeqAccess::Claim`].
    type Claim: 'de;

    /// Deserialize the element as `T`, forwarding `extra` into `T::deserialize`.
    /// Returns `Ok(Probe::Hit((claim, value)))`; the claim must be returned from
    /// the closure passed to [`SeqAccess::next`] to advance the stream.
    /// `Ok(Probe::Miss)` if `T::deserialize` misses; `Err` on fatal format error only.
    async fn get<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

    /// Fork a sibling element handle for the same sequence slot.
    ///
    /// Both handles refer to the same element; whichever resolves `get`
    /// first claims it.
    fn fork(&mut self) -> Self
    where
        Self: Sized;

    /// Consume and discard the element without deserializing it.
    /// Returns the [`Claim`](SeqEntry::Claim) so the stream can advance.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ---------------------------------------------------------------------------
// Never impls — borrow family
// ---------------------------------------------------------------------------

impl<'n, C, E: DeserializeError> StrAccess for crate::Never<'n, C, E> {
    type Claim = C;
    type Error = E;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn next_str<R>(
        self,
        _f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> BytesAccess for crate::Never<'n, C, E> {
    type Claim = C;
    type Error = E;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn next_bytes<R>(
        self,
        _f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> SeqAccess<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type Elem = crate::Never<'n, C, E>;

    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn next<const N: usize, F, Fut, R>(
        self,
        _f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
    {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> SeqEntry<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn get<T: Deserialize<'de, Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        match self.0 {}
    }
}

// ---------------------------------------------------------------------------
// Deserialize impls for primitives
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_primitive {
    ($ty:ty, $method:ident) => {
        impl<'de> Deserialize<'de> for $ty {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async { e.$method().await }).await
            }
        }
    };
}

impl_deserialize_primitive!(bool, deserialize_bool);
impl_deserialize_primitive!(u8, deserialize_u8);
impl_deserialize_primitive!(u16, deserialize_u16);
impl_deserialize_primitive!(u32, deserialize_u32);
impl_deserialize_primitive!(u64, deserialize_u64);
impl_deserialize_primitive!(u128, deserialize_u128);
impl_deserialize_primitive!(i8, deserialize_i8);
impl_deserialize_primitive!(i16, deserialize_i16);
impl_deserialize_primitive!(i32, deserialize_i32);
impl_deserialize_primitive!(i64, deserialize_i64);
impl_deserialize_primitive!(i128, deserialize_i128);
impl_deserialize_primitive!(f32, deserialize_f32);
impl_deserialize_primitive!(f64, deserialize_f64);
impl_deserialize_primitive!(char, deserialize_char);

impl<'de: 'a, 'a> Deserialize<'de> for &'a str {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_str()).await
    }
}

impl<'de: 'a, 'a> Deserialize<'de> for &'a [u8] {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_bytes()).await
    }
}

impl<'de, Extra: Copy, T: Deserialize<'de, Extra>> Deserialize<'de, Extra> for Option<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_option::<T, Extra>(extra)).await
    }
}
