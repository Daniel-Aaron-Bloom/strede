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
pub trait Deserialize<'de>: Sized {
    /// Deserialize `Self` from `d`.
    ///
    /// - `Ok(Probe::Hit(value))` — succeeded.
    /// - `Ok(Probe::Miss)` — token type didn't match; stream not advanced.
    /// - `Err(e)` — fatal format error (malformed data, I/O failure).
    async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error>;
}

// ---------------------------------------------------------------------------
// Deserializer — stream handle
// ---------------------------------------------------------------------------

/// A stream of tokens that can be decoded into Rust values.
///
/// The deserializer owns the stream and is the sole means of advancing it —
/// all advancement goes through [`Deserializer::next`].  Type probing is done
/// through [`Entry`] handles handed to the closure.
pub trait Deserializer<'de> {
    /// Fatal error type produced by this format.
    type Error: DeserializeError;

    /// Owned handle for one item slot.  See [`Entry`].
    type Entry<'a>: Entry<'de, Error = Self::Error>
    where
        Self: 'a,
        'de: 'a;

    /// Advance to the next item in the stream.
    ///
    /// Passes `N` owned [`Entry`] handles to `f`.  When a probe inside `f`
    /// resolves `Ok(Probe::Hit((claim, r)))`, the winning arm returns it;
    /// `next` verifies the claim and returns `Ok(Probe::Hit(r))`.  Returns
    /// `Err(e)` if a fatal error occurs before or during `f`.
    /// `Ok(Probe::Miss)` propagates upward if no probe matched.
    /// `Pending` until a token is available.
    ///
    /// Use `N > 1` to race multiple probe arms via [`select_probe!`](crate::select_probe) without
    /// borrow conflicts.  Handles dropped without resolving do not advance
    /// the stream.
    async fn next<'s, const N: usize, F, Fut, R>(
        &'s mut self,
        f: F,
    ) -> Result<Probe<R>, Self::Error>
    where
        'de: 's,
        F: FnOnce([Self::Entry<'s>; N]) -> Fut,
        Fut: Future<
            Output = Result<Probe<(<Self::Entry<'s> as Entry<'de>>::Claim, R)>, Self::Error>,
        >;
}

// ---------------------------------------------------------------------------
// Entry — type-probing handle for one item slot
// ---------------------------------------------------------------------------

/// Owned handle for one item slot, passed into the closure of [`Deserializer::next`].
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
/// Use `N > 1` in `next` to race arms via [`select_probe!`](crate::select_probe):
///
/// ```rust,ignore
/// let value = d.next(|[e1, e2, e3, e4]| async move {
///     select_probe! {
///         (c, v) = e1.deserialize_bool()  => Ok(Probe::Hit((c, Value::Bool(v)))),
///         (c, v) = e2.deserialize_i64()   => Ok(Probe::Hit((c, Value::Int(v)))),
///         (c, v) = e3.deserialize_str()   => Ok(Probe::Hit((c, Value::Str(v)))),
///         m      = e4.deserialize_map()   => {
///             let (c, v) = collect_map(m).await?;
///             Ok(Probe::Hit((c, Value::Map(v))))
///         },
///     }
/// }).await?;
/// ```
pub trait Entry<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error;

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
    /// returned by [`StrAccess::next`] when the string is exhausted.
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
    /// returned by [`BytesAccess::next`] when the byte string is exhausted.
    /// Handles all byte strings including those requiring transcoding.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error>;
    /// Begin reading a map.  The [`Entry::Claim`] is returned by
    /// [`MapAccess::next`] when the map is exhausted.
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
    async fn deserialize_option<T: Deserialize<'de>>(
        self,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>;

    /// Probe for a null token.
    ///
    /// - `Ok(Probe::Hit(claim))` — null token consumed.
    /// - `Ok(Probe::Miss)` — token is not null.
    /// - `Err(e)` — fatal format error.
    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error>;

    /// Delegate to `T::deserialize` from this entry handle.
    ///
    /// Creates a sub-deserializer with the current token pre-loaded and
    /// calls `T::deserialize`.  Returns `Hit` if `T` matched, `Miss` if
    /// `T::deserialize` returned `Miss`.
    async fn deserialize_value<T: Deserialize<'de>>(
        self,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;
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
///     match chunks.next().await? {
///         Chunk::Data(chunk) => out.push_str(chunk),
///         Chunk::Done(claim) => break claim,
///     }
/// };
/// ```
pub trait StrAccess {
    /// Proof-of-consumption token, returned when the string is exhausted.
    /// Must match the enclosing [`Entry::Claim`].
    type Claim;
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error;

    /// - `Ok(Chunk::Data(chunk))` — next chunk; valid until the next call.
    /// - `Ok(Chunk::Done(claim))` — string exhausted; claim is now valid.
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next(&mut self) -> Result<Chunk<&str, Self::Claim>, Self::Error>;
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
///     match chunks.next().await? {
///         Chunk::Data(chunk) => out.extend_from_slice(chunk),
///         Chunk::Done(claim) => break claim,
///     }
/// };
/// ```
pub trait BytesAccess {
    /// Proof-of-consumption token, returned when the byte string is exhausted.
    /// Must match the enclosing [`Entry::Claim`].
    type Claim;
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error;

    /// - `Ok(Chunk::Data(chunk))` — next chunk; valid until the next call.
    /// - `Ok(Chunk::Done(claim))` — byte string exhausted; claim is now valid.
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next(&mut self) -> Result<Chunk<&[u8], Self::Claim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// MapAccess
// ---------------------------------------------------------------------------

/// Iterates the key-value pairs of a map.  Obtained from [`Entry::deserialize_map`].
pub trait MapAccess<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error;

    /// Proof-of-consumption token returned on map exhaustion; must match
    /// the enclosing [`Entry::Claim`].
    type Claim: 'de;

    /// Owned handle for reading one key.  See [`MapKeyEntry`].
    type KeyEntry<'a>: MapKeyEntry<'de, Claim = Self::Claim, Error = Self::Error>
    where
        Self: 'a,
        'de: 'a;

    /// Advance to the next key-value pair, passing `N` owned [`MapKeyEntry`]
    /// handles to `f`.  The closure must return `Ok(Probe::Hit((claim, r)))` where
    /// `claim` is the [`MapValueEntry::Claim`] threaded out through
    /// [`MapKeyEntry::key`], or `Ok(Probe::Miss)` if no key matched.
    ///
    /// Returns:
    /// - `Ok(Probe::Hit(Chunk::Data(r)))` — a pair was consumed.
    /// - `Ok(Probe::Hit(Chunk::Done(claim)))` — map exhausted.
    /// - `Ok(Probe::Miss)` — closure returned Miss (no key probe matched).
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next<'s, const N: usize, F, Fut, R>(
        &'s mut self,
        f: F,
    ) -> Result<Probe<Chunk<R, Self::Claim>>, Self::Error>
    where
        'de: 's,
        F: FnOnce([Self::KeyEntry<'s>; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned handle for reading the key of one map pair.
///
/// `key` follows the same lambda pattern as [`Deserializer::next`]: the closure
/// receives `(&K, [ValueEntry; N])` — a reference to the decoded key and `N`
/// owned [`MapValueEntry`] handles — so it can inspect the key and read the
/// value, returning `Ok(Probe::Hit((ValueClaim, R)))` or `Ok(Probe::Miss)` for
/// unknown keys.  `key` returns `Ok(Probe::Hit((Claim, K, R)))` on success.
///
/// ```rust,ignore
/// let mut out = HashMap::new();
/// loop {
///     match map.next(|[ke]| async move {
///         match ke.key::<String, 1, _, _, _>(|_k, [ve]| async move {
///             let (claim, v) = hit!(ve.value::<u32>().await);
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
    type Error;
    /// Proof-of-consumption token; must match the enclosing [`MapAccess::Claim`].
    type Claim: 'de;
    type ValueEntry: MapValueEntry<'de, Claim = Self::Claim, Error = Self::Error>;

    async fn key<K: Deserialize<'de>, const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, K, R)>, Self::Error>
    where
        F: FnOnce(&K, [Self::ValueEntry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned handle for reading the value of one map pair.
///
/// Constructed by the [`MapKeyEntry::key`] implementation and passed as the
/// `[ValueEntry; N]` argument to its closure.
pub trait MapValueEntry<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error;
    /// Proof-of-consumption token; must match the enclosing [`MapAccess::Claim`].
    type Claim: 'de;

    /// Deserialize the value as `V`.  Returns `Ok(Probe::Hit((claim, value)))`;
    /// the claim must be threaded out through [`MapKeyEntry::key`] and back to
    /// [`MapAccess::next`] to advance the stream.  `Ok(Probe::Miss)` if
    /// `V::deserialize` misses; `Err` on fatal format error only.
    async fn value<V: Deserialize<'de>>(self) -> Result<Probe<(Self::Claim, V)>, Self::Error>;
}

// ---------------------------------------------------------------------------
// SeqAccess
// ---------------------------------------------------------------------------

/// Iterates the elements of a sequence.  Obtained from [`Entry::deserialize_seq`].
///
/// ```rust,ignore
/// let mut out = Vec::new();
/// loop {
///     match seq.next(|[e]| async move {
///         let (claim, v) = hit!(e.get::<u32>().await);
///         Ok(Probe::Hit((claim, v)))
///     }).await? {
///         Probe::Hit(Chunk::Data(v)) => out.push(v),
///         Probe::Hit(Chunk::Done(claim)) => return Ok(Probe::Hit((claim, out))),
///         Probe::Miss => { /* unexpected element type */ }
///     }
/// }
/// ```
pub trait SeqAccess<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error;

    /// Proof-of-consumption token returned on sequence exhaustion; must match
    /// the enclosing [`Entry::Claim`].
    type Claim: 'de;

    /// Owned handle for one sequence element.  See [`SeqEntry`].
    type Elem<'a>: SeqEntry<'de, Claim = Self::Claim, Error = Self::Error>
    where
        Self: 'a,
        'de: 'a;

    /// Advance to the next element, passing `N` owned [`SeqEntry`] handles to `f`.
    ///
    /// Returns:
    /// - `Ok(Probe::Hit(Chunk::Data(r)))` — an element was consumed.
    /// - `Ok(Probe::Hit(Chunk::Done(claim)))` — sequence exhausted.
    /// - `Ok(Probe::Miss)` — closure returned Miss.
    /// - `Err(e)` — fatal format error.
    /// - `Pending` — no data yet.
    async fn next<'s, const N: usize, F, Fut, R>(
        &'s mut self,
        f: F,
    ) -> Result<Probe<Chunk<R, Self::Claim>>, Self::Error>
    where
        'de: 's,
        F: FnOnce([Self::Elem<'s>; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned handle for one element in a sequence.
pub trait SeqEntry<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error;
    /// Proof-of-consumption token; must match the enclosing [`SeqAccess::Claim`].
    type Claim: 'de;

    /// Deserialize the element as `T`.  Returns `Ok(Probe::Hit((claim, value)))`;
    /// the claim must be returned from the closure passed to [`SeqAccess::next`]
    /// to advance the stream.  `Ok(Probe::Miss)` if `T::deserialize` misses;
    /// `Err` on fatal format error only.
    async fn get<T: Deserialize<'de>>(self) -> Result<Probe<(Self::Claim, T)>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Deserialize impls for primitives
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_primitive {
    ($ty:ty, $method:ident) => {
        impl<'de> Deserialize<'de> for $ty {
            async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error> {
                d.next(|[e]| async move { e.$method().await }).await
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

impl<'de> Deserialize<'de> for &'de str {
    async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error> {
        d.next(|[e]| async move { e.deserialize_str().await }).await
    }
}

impl<'de> Deserialize<'de> for &'de [u8] {
    async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error> {
        d.next(|[e]| async move { e.deserialize_bytes().await })
            .await
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Option<T> {
    async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error> {
        d.next(|[e]| async move { e.deserialize_option::<T>().await })
            .await
    }
}
