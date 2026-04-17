use core::future::Future;

use crate::{Chunk, DeserializeError, Probe};

// ===========================================================================
// Owned family — parallel trait hierarchy for formats that cannot deliver
// borrowed-for-`'de` slices (e.g. chunked/streaming input).
//
// Mirrors the borrow family, minus `deserialize_str` / `deserialize_bytes`.
// The `'s` lifetime is the deserializer's *session* — the region over which
// the deserializer and its `Claim`s remain valid. Analogous to the borrow
// family's `'de`, but with no zero-copy borrow methods.
//
// `Probe`, `Chunk`, `StrAccess`, `BytesAccess`, `hit!`, `DeserializeError`,
// and `select_probe!` are shared between families.
//
// The two families are independent: no supertrait relationship, no blanket
// impls. A format implements whichever family (or both) it can support.
//
// # Parallel scanning — deadlock hazard
//
// The owned family reads from a streaming source where data arrives
// incrementally.  Every reader (entry handle, map/seq accessor, or
// str/bytes chunk accessor) shares the same underlying buffer and
// advances through it cooperatively via `fork`.
//
// **For callers:**  You must not read one forked handle to completion
// and then decide what to do with another — that will deadlock.  The
// first handle may block waiting for more data to arrive, but the
// buffer cannot advance until *all* sibling handles have consumed the
// current chunk.  Instead, race all handles concurrently (e.g. via
// `select_probe!`).  This is safe: forked handles never interfere with
// each other, and every reader is polled and paused as new data becomes
// available, as long as all of them are making forward progress together.
//
// **For implementers:**  Your `fork` implementation must ensure that
// the underlying buffer does not advance past data that any live handle
// still needs to read.  Every forked reader must be independently
// resumable: when new data arrives, all suspended readers must be woken
// and given the opportunity to process it.  You must never require one
// reader to finish before another can make progress — doing so creates
// a circular dependency that deadlocks the single-threaded executor.
// The `shared_buf` module provides a reference implementation of this
// contract.
// ===========================================================================

/// Owned counterpart to [`Deserialize`](crate::Deserialize).
///
/// The `Extra` type parameter is side-channel context passed into
/// [`EntryOwned::deserialize_value`], [`SeqEntryOwned::get`],
/// [`MapKeyEntryOwned::key`], and [`MapValueEntryOwned::value`] at the call
/// site.  Defaults to `()` for types that need no extra context.
pub trait DeserializeOwned<'s, Extra = ()>: Sized {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>;
}

/// Owned counterpart to [`Deserializer`](crate::Deserializer).
///
/// # Streaming and parallel scanning
///
/// Unlike the borrow family, the owned family reads from a streaming source
/// where data arrives incrementally.  When `entry` passes `N` handles to
/// the closure (or when you `fork` an accessor), the resulting readers share
/// the same underlying buffer and advance through it cooperatively.
///
/// **You must drive all forked readers concurrently** — typically via
/// [`select_probe!`](crate::select_probe).  Sequentially awaiting one reader
/// to completion before polling another will deadlock: the first reader may
/// block waiting for buffer data that cannot arrive until all sibling readers
/// have consumed the current chunk.
///
/// This is safe to do: forked readers are guaranteed not to interfere with
/// each other, and every reader is automatically suspended and resumed as
/// new data becomes available, provided all readers are being polled.
pub trait DeserializerOwned<'s>: Sized {
    type Error: DeserializeError;

    type Claim: 's;
    type Entry: EntryOwned<'s, Error = Self::Error>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<
            Output = Result<Probe<(<Self::Entry as EntryOwned<'s>>::Claim, R)>, Self::Error>,
        >;
}

/// Owned counterpart to [`Entry`](crate::Entry). Drops `deserialize_str` / `deserialize_bytes`
/// (the borrow-only methods); strings and bytes must be read via
/// [`StrAccessOwned`] / [`BytesAccessOwned`].
///
/// # For implementors — `fork` contract
///
/// [`EntryOwned::fork`] creates a sibling handle for the same token slot.
/// Your implementation must guarantee that:
///
/// 1. The underlying buffer does not advance past data any live handle still
///    needs.  All forked handles read the same data independently.
/// 2. When new data arrives, **every** suspended reader is woken and given the
///    chance to process it.
/// 3. No reader is required to finish before another can make progress.
///    Violating this creates a circular wait that deadlocks the
///    single-threaded executor.
pub trait EntryOwned<'s>: Sized {
    type Error: DeserializeError;
    type Claim: 's;

    type StrChunks: StrAccessOwned<'s, Claim = Self::Claim, Error = Self::Error>;
    type BytesChunks: BytesAccessOwned<'s, Claim = Self::Claim, Error = Self::Error>;
    type Map: MapAccessOwned<'s, Claim = Self::Claim, Error = Self::Error>;
    type Seq: SeqAccessOwned<'s, Claim = Self::Claim, Error = Self::Error>;

    async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error>;
    async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error>;
    async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error>;
    async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error>;
    async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error>;
    async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error>;
    async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error>;
    async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error>;
    async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error>;
    async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error>;
    async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error>;
    async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error>;
    async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error>;
    async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error>;

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error>;
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error>;
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error>;
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error>;

    async fn deserialize_option<T: DeserializeOwned<'s, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>;

    /// Probe for a null token.
    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error>;

    /// Delegate to `T::deserialize` from this entry handle, forwarding `extra`.
    async fn deserialize_value<T: DeserializeOwned<'s, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

    /// Fork a sibling entry handle for the same item slot.
    ///
    /// Both `self` and the returned handle refer to the same slot.
    /// Whichever resolves a probe first claims the slot; the other
    /// becomes inert and may be dropped without advancing the stream.
    fn fork(&mut self) -> Self;

    /// Consume and discard the current token regardless of its type.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

/// Owned counterpart to [`StrAccess`]. Takes `self` by value and a sync
/// closure that borrows the chunk `&str`. The closure maps the short-lived
/// borrow to an owned `R`; the accessor is handed back alongside `R` on
/// `Data` and the claim emerges on `Done`.
///
/// **Callers:** forked accessors must be driven concurrently (e.g. via
/// `select_probe!`).  Awaiting one fork to `Done` before polling another
/// will deadlock — the buffer cannot advance until all forks have consumed
/// the current chunk.
pub trait StrAccessOwned<'s>: Sized {
    type Claim: 's;
    type Error: DeserializeError;

    /// Fork a sibling accessor at the same read position.
    ///
    /// Both `self` and the returned accessor are independent: each must be
    /// driven to `Done` (or dropped) before the underlying buffer can
    /// advance.  Neither reader replays data — both continue from the
    /// current position, consuming the same remaining chunks.
    ///
    /// # For implementors
    ///
    /// The forked accessor must hold the buffer open at its current position.
    /// All forks are woken when new data arrives; none may require another
    /// to complete first.
    fn fork(&mut self) -> Self;

    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`BytesAccess`]. Same closure pattern as
/// [`StrAccessOwned`].
///
/// **Callers:** forked accessors must be driven concurrently.
/// See [`StrAccessOwned`] for the deadlock hazard.
pub trait BytesAccessOwned<'s>: Sized {
    type Claim: 's;
    type Error: DeserializeError;

    /// Fork a sibling accessor at the same read position.
    ///
    /// Both `self` and the returned accessor are independent: each must be
    /// driven to `Done` (or dropped) before the underlying buffer can
    /// advance.  Neither reader replays data — both continue from the
    /// current position, consuming the same remaining chunks.
    ///
    /// # For implementors
    ///
    /// Same contract as [`StrAccessOwned::fork`].
    fn fork(&mut self) -> Self;

    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`MapAccess`](crate::MapAccess).
///
/// **Callers:** forked map accessors must be driven concurrently.
/// See [`DeserializerOwned`] for the deadlock hazard.
pub trait MapAccessOwned<'s>: Sized {
    type Error: DeserializeError;
    type Claim: 's;
    type KeyEntry: MapKeyEntryOwned<'s, Claim = Self::Claim, Error = Self::Error>;

    /// Fork a sibling accessor at the same map position.
    ///
    /// Both forks must consume all remaining pairs (or be dropped) before
    /// the underlying buffer can advance.
    ///
    /// # For implementors
    ///
    /// Same contract as [`StrAccessOwned::fork`]: the buffer stays pinned
    /// at the current position, all forks are woken on new data, and no
    /// fork may require another to complete first.
    fn fork(&mut self) -> Self;

    async fn next_kv<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
    where
        F: FnMut([Self::KeyEntry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned counterpart to [`MapKeyEntry`](crate::MapKeyEntry).
pub trait MapKeyEntryOwned<'s> {
    type Error: DeserializeError;
    type Claim: 's;
    type ValueEntry: MapValueEntryOwned<'s, Claim = Self::Claim, Error = Self::Error>;

    /// Fork a sibling key-entry handle for the same map pair.
    fn fork(&mut self) -> Self
    where
        Self: Sized;

    async fn key<K: DeserializeOwned<'s, KExtra>, KExtra, const N: usize, F, Fut, R>(
        self,
        extra: KExtra,
        f: F,
    ) -> Result<Probe<(Self::Claim, K, R)>, Self::Error>
    where
        F: FnMut(&K, [Self::ValueEntry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned counterpart to [`MapValueEntry`](crate::MapValueEntry).
pub trait MapValueEntryOwned<'s> {
    type Error: DeserializeError;
    type Claim: 's;

    /// Fork a sibling value-entry handle for the same map value slot.
    fn fork(&mut self) -> Self
    where
        Self: Sized;

    async fn value<V: DeserializeOwned<'s, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, V)>, Self::Error>;

    /// Consume and discard the value without deserializing it.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

/// Owned counterpart to [`SeqAccess`](crate::SeqAccess).
///
/// **Callers:** forked sequence accessors must be driven concurrently.
/// See [`DeserializerOwned`] for the deadlock hazard.
pub trait SeqAccessOwned<'s>: Sized {
    type Error: DeserializeError;
    type Claim: 's;
    type Elem: SeqEntryOwned<'s, Claim = Self::Claim, Error = Self::Error>;

    /// Fork a sibling accessor at the same sequence position.
    ///
    /// # For implementors
    ///
    /// Same contract as [`StrAccessOwned::fork`].
    fn fork(&mut self) -> Self;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned counterpart to [`SeqEntry`](crate::SeqEntry).
pub trait SeqEntryOwned<'s> {
    type Error: DeserializeError;
    type Claim: 's;

    /// Fork a sibling element handle for the same sequence slot.
    fn fork(&mut self) -> Self
    where
        Self: Sized;

    async fn get<T: DeserializeOwned<'s, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

    /// Consume and discard the element without deserializing it.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ---------------------------------------------------------------------------
// Never impls — owned family
// ---------------------------------------------------------------------------

impl<'n, 's, C: 's, E: DeserializeError> StrAccessOwned<'s> for crate::Never<'n, C, E> {
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

impl<'n, 's, C: 's, E: DeserializeError> BytesAccessOwned<'s> for crate::Never<'n, C, E> {
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

impl<'n, 's, C: 's, E: DeserializeError> SeqAccessOwned<'s> for crate::Never<'n, C, E> {
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

impl<'n, 's, C: 's, E: DeserializeError> SeqEntryOwned<'s> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn get<T: DeserializeOwned<'s, Extra>, Extra>(
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
// DeserializeOwned impls for primitives
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_owned_primitive {
    ($ty:ty, $method:ident) => {
        impl<'s> DeserializeOwned<'s> for $ty {
            async fn deserialize_owned<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async { e.$method().await }).await
            }
        }
    };
}

impl_deserialize_owned_primitive!(bool, deserialize_bool);
impl_deserialize_owned_primitive!(u8, deserialize_u8);
impl_deserialize_owned_primitive!(u16, deserialize_u16);
impl_deserialize_owned_primitive!(u32, deserialize_u32);
impl_deserialize_owned_primitive!(u64, deserialize_u64);
impl_deserialize_owned_primitive!(u128, deserialize_u128);
impl_deserialize_owned_primitive!(i8, deserialize_i8);
impl_deserialize_owned_primitive!(i16, deserialize_i16);
impl_deserialize_owned_primitive!(i32, deserialize_i32);
impl_deserialize_owned_primitive!(i64, deserialize_i64);
impl_deserialize_owned_primitive!(i128, deserialize_i128);
impl_deserialize_owned_primitive!(f32, deserialize_f32);
impl_deserialize_owned_primitive!(f64, deserialize_f64);
impl_deserialize_owned_primitive!(char, deserialize_char);

impl<'s, Extra: Copy, T: DeserializeOwned<'s, Extra>> DeserializeOwned<'s, Extra> for Option<T> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async { e.deserialize_option::<T, Extra>(extra).await })
            .await
    }
}
