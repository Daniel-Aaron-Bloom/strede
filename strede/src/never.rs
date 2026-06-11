use core::convert::Infallible;
use core::future::Future;
use core::marker::PhantomData;

/// The uninhabited bottom type - equivalent to `!` but stable on all editions.
///
/// Used as the associated `StrChunks`, `BytesChunks`, and `Seq` types on
/// [`crate::Entry`] / [`crate::EntryOwned`] implementations that never produce those
/// accessor kinds.  Because `Never` has no values, all trait method bodies
/// on `Never` are written as `match self {}` and the compiler accepts them
/// without any reachable code.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub struct Never<'a, Claim, Error>(
    Infallible,
    PhantomData<(fn(*const Claim), fn(*const Error), fn(&'a ()))>,
);

use crate::borrow::{
    BytesAccess, Deserialize, DeserializeFromEnum, DeserializeFromMap, DeserializeFromSeq,
    Deserializer, Entry, EnumAccess, EnumArmStack, EnumVariantProbe, MapAccess, MapArmStack,
    MapKeyClaim, MapKeyProbe, MapValueClaim, MapValueProbe, NumberAccess, NumberEncoding, SeqAccess,
    SeqEntry, StrAccess,
};
use crate::owned::{
    BytesAccessOwned, DeserializeFromEnumOwned, DeserializeFromMapOwned, DeserializeFromSeqOwned,
    DeserializeOwned, DeserializerOwned, EntryOwned, EnumAccessOwned, EnumArmStackOwned,
    EnumVariantProbeOwned, MapAccessOwned, MapArmStackOwned, MapKeyClaimOwned, MapKeyProbeOwned,
    MapValueClaimOwned, MapValueProbeOwned, NextKey, NumberAccessOwned, SeqAccessOwned,
    SeqEntryOwned, StrAccessOwned,
};
use crate::{Chunk, DeserializeError, Probe};

// ---------------------------------------------------------------------------
// Never impls - borrow family
// ---------------------------------------------------------------------------

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> Deserializer<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type EntryClaim = C;
    type Entry = crate::Never<'n, C, E>;
    async fn entry<const N: usize, F, Fut, R>(
        self,
        _f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> Entry<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type SubDeserializer = crate::Never<'n, C, E>;
    type StrChunks = crate::Never<'n, C, E>;
    type BytesChunks = crate::Never<'n, C, E>;
    type NumberChunks<Enc: NumberEncoding> = crate::Never<'n, C, E>;
    type Map = crate::Never<'n, C, E>;
    type Seq = crate::Never<'n, C, E>;
    type Enum = crate::Never<'n, C, E>;
    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_value<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        match self.0 {}
    }
    async fn deserialize_option<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        match self.0 {}
    }
    async fn deserialize_map_into<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>,
    {
        match self.0 {}
    }
    async fn deserialize_seq_into<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeq<'de, Self::Seq>,
    {
        match self.0 {}
    }
    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_enum_into<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnum<'de, Self::Enum>,
    {
        match self.0 {}
    }
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> EnumAccess<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type VariantProbe = crate::Never<'n, C, E>;
    async fn iterate<S>(self, _arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStack<'de, Self::VariantProbe>,
    {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> EnumVariantProbe<'de>
    for crate::Never<'n, C, E>
{
    type Error = E;
    type Claim = C;
    type PayloadDeserializer = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_value_by_shape<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
    {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> StrAccess for crate::Never<'n, C, E> {
    type Claim = C;
    type Error = E;
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
    async fn next_bytes<R>(
        self,
        _f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError, Enc: NumberEncoding> NumberAccess<Enc> for crate::Never<'n, C, E> {
    type Claim = C;
    type Error = E;
    async fn next_number_chunk<R>(
        self,
        _f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> SeqAccess<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type SeqClaim = C;
    type ElemClaim = C;
    type Elem = crate::Never<'n, C, E>;
    async fn next<const N: usize, F, Fut, R>(
        self,
        _f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> SeqEntry<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type SubDeserializer = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn get<T>(self, _extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> MapAccess<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type KeyProbe = crate::Never<'n, C, E>;
    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        _arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> MapKeyProbe<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type KeyClaim = crate::Never<'n, C, E>;
    type KeySubDeserializer = crate::Never<'n, crate::Never<'n, C, E>, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_key<K>(
        self,
        _extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: Deserialize<'de, Self::KeySubDeserializer>,
    {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> MapKeyClaim<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type ValueProbe = crate::Never<'n, C, E>;
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> MapValueProbe<'de>
    for crate::Never<'n, C, E>
{
    type Error = E;
    type MapClaim = C;
    type ValueClaim = crate::Never<'n, C, E>;
    type ValueSubDeserializer = crate::Never<'n, crate::Never<'n, C, E>, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_value<V>(
        self,
        _extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: Deserialize<'de, Self::ValueSubDeserializer>,
    {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        match self.0 {}
    }
}

impl<'n: 'de, 'de, C: 'de, E: DeserializeError + 'de> MapValueClaim<'de>
    for crate::Never<'n, C, E>
{
    type Error = E;
    type KeyProbe = crate::Never<'n, C, E>;
    type MapClaim = C;
    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        match self.0 {}
    }
}

// ---------------------------------------------------------------------------
// Never impls - owned family
// ---------------------------------------------------------------------------

impl<'n, C, E: DeserializeError> DeserializerOwned for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type EntryClaim = C;
    type Entry = crate::Never<'n, C, E>;
    async fn entry<const N: usize, F, Fut, R>(
        self,
        _f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> EntryOwned for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type SubDeserializer = crate::Never<'n, C, E>;
    type StrChunks = crate::Never<'n, C, E>;
    type BytesChunks = crate::Never<'n, C, E>;
    type NumberChunks<Enc: NumberEncoding> = crate::Never<'n, C, E>;
    type Map = crate::Never<'n, C, E>;
    type Seq = crate::Never<'n, C, E>;
    type Enum = crate::Never<'n, C, E>;
    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_value<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        match self.0 {}
    }
    async fn deserialize_option<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        match self.0 {}
    }
    async fn deserialize_map_into<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMapOwned<Self::Map>,
    {
        match self.0 {}
    }
    async fn deserialize_seq_into<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeqOwned<Self::Seq>,
    {
        match self.0 {}
    }
    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        match self.0 {}
    }
    async fn deserialize_enum_into<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnumOwned<Self::Enum>,
    {
        match self.0 {}
    }
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> EnumAccessOwned for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type VariantProbe = crate::Never<'n, C, E>;
    async fn iterate<S>(self, _arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStackOwned<Self::VariantProbe>,
    {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> EnumVariantProbeOwned for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type PayloadDeserializer = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_value_by_shape<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        match self.0 {}
    }
}

impl<C, E: DeserializeError> StrAccessOwned for crate::Never<'_, C, E> {
    type Claim = C;
    type Error = E;
    async fn next_str<R>(
        self,
        _f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<C, E: DeserializeError> BytesAccessOwned for crate::Never<'_, C, E> {
    type Claim = C;
    type Error = E;
    async fn next_bytes<R>(
        self,
        _f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<C, E: DeserializeError, Enc: NumberEncoding> NumberAccessOwned<Enc> for crate::Never<'_, C, E> {
    type Claim = C;
    type Error = E;
    async fn next_number_chunk<R>(
        self,
        _f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> SeqAccessOwned for crate::Never<'n, C, E> {
    type Error = E;
    type SeqClaim = C;
    type ElemClaim = C;
    type Elem = crate::Never<'n, C, E>;
    async fn next<const N: usize, F, Fut, R>(
        self,
        _f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> SeqEntryOwned for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    type SubDeserializer = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn get<T>(self, _extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapAccessOwned for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type KeyProbe = crate::Never<'n, C, E>;
    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        _arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapKeyProbeOwned for crate::Never<'n, C, E> {
    type Error = E;
    type KeyClaim = crate::Never<'n, C, E>;
    type KeySubDeserializer = crate::Never<'n, crate::Never<'n, C, E>, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_key<K>(
        self,
        _extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: DeserializeOwned<Self::KeySubDeserializer>,
    {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapKeyClaimOwned for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type ValueProbe = crate::Never<'n, C, E>;
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapValueProbeOwned for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type ValueClaim = crate::Never<'n, C, E>;
    type ValueSubDeserializer = crate::Never<'n, crate::Never<'n, C, E>, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_value<V>(
        self,
        _extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeOwned<Self::ValueSubDeserializer>,
    {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapValueClaimOwned for crate::Never<'n, C, E> {
    type Error = E;
    type KeyProbe = crate::Never<'n, C, E>;
    type MapClaim = C;
    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        match self.0 {}
    }
}
