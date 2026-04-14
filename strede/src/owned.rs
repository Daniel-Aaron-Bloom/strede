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
// ===========================================================================

/// Owned counterpart to [`Deserialize`](crate::Deserialize).
pub trait DeserializeOwned<'s>: Sized {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>;
}

/// Owned counterpart to [`Deserializer`](crate::Deserializer).
pub trait DeserializerOwned<'s>: Sized {
    type Error: DeserializeError;

    type Claim: 's;
    type Entry: EntryOwned<'s, Error = Self::Error>;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnOnce([Self::Entry; N]) -> Fut,
        Fut: Future<
            Output = Result<Probe<(<Self::Entry as EntryOwned<'s>>::Claim, R)>, Self::Error>,
        >;
}

/// Owned counterpart to [`Entry`](crate::Entry). Drops `deserialize_str` / `deserialize_bytes`
/// (the borrow-only methods); strings and bytes must be read via
/// [`StrAccessOwned`] / [`BytesAccessOwned`].
pub trait EntryOwned<'s>: Sized {
    type Error;
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

    async fn deserialize_option<T: DeserializeOwned<'s>>(
        self,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>;

    /// Probe for a null token.
    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error>;

    /// Delegate to `T::deserialize` from this entry handle.
    async fn deserialize_value<T: DeserializeOwned<'s>>(
        self,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;
}

/// Owned counterpart to [`StrAccess`]. Takes `self` by value and a sync
/// closure that borrows the chunk `&str`. The closure maps the short-lived
/// borrow to an owned `R`; the accessor is handed back alongside `R` on
/// `Data` and the claim emerges on `Done`.
pub trait StrAccessOwned<'s>: Sized {
    type Claim: 's;
    type Error;

    async fn next<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`BytesAccess`]. Same closure pattern as
/// [`StrAccessOwned`].
pub trait BytesAccessOwned<'s>: Sized {
    type Claim: 's;
    type Error;

    async fn next<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`MapAccess`](crate::MapAccess).
pub trait MapAccessOwned<'s>: Sized {
    type Error;
    type Claim: 's;
    type KeyEntry: MapKeyEntryOwned<'s, Claim = Self::Claim, Error = Self::Error>;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
    where
        F: FnOnce([Self::KeyEntry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned counterpart to [`MapKeyEntry`](crate::MapKeyEntry).
pub trait MapKeyEntryOwned<'s> {
    type Error;
    type Claim: 's;
    type ValueEntry: MapValueEntryOwned<'s, Claim = Self::Claim, Error = Self::Error>;

    async fn key<K: DeserializeOwned<'s>, const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, K, R)>, Self::Error>
    where
        F: FnOnce(&K, [Self::ValueEntry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned counterpart to [`MapValueEntry`](crate::MapValueEntry).
pub trait MapValueEntryOwned<'s> {
    type Error;
    type Claim: 's;

    async fn value<V: DeserializeOwned<'s>>(self) -> Result<Probe<(Self::Claim, V)>, Self::Error>;
}

/// Owned counterpart to [`SeqAccess`](crate::SeqAccess).
pub trait SeqAccessOwned<'s>: Sized {
    type Error;
    type Claim: 's;
    type Elem: SeqEntryOwned<'s, Claim = Self::Claim, Error = Self::Error>;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
    where
        F: FnOnce([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>;
}

/// Owned counterpart to [`SeqEntry`](crate::SeqEntry).
pub trait SeqEntryOwned<'s> {
    type Error;
    type Claim: 's;

    async fn get<T: DeserializeOwned<'s>>(self) -> Result<Probe<(Self::Claim, T)>, Self::Error>;
}

// ---------------------------------------------------------------------------
// DeserializeOwned impls for primitives
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_owned_primitive {
    ($ty:ty, $method:ident) => {
        impl<'s> DeserializeOwned<'s> for $ty {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.next(|[e]| async move { e.$method().await }).await
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

impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for Option<T> {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.next(|[e]| async move { e.deserialize_option::<T>().await })
            .await
    }
}
