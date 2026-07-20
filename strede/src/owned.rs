use const_array_concat::ConcatableArray;
use core::future::Future;

pub use crate::map_arm::{
    ArmState, DetectDuplicates, False, MapArm, MapArmBase, MapArmSlot, NextKey, StackConcat,
    TagInjectingStack, True, VirtualArmSlot,
};
use crate::borrow::{Ascii, NumberEncoding};
use crate::{Chunk, DeserializeError, Probe};

// ===========================================================================
// Owned family - parallel trait hierarchy for formats that cannot deliver
// borrowed-for-`'de` slices (e.g. chunked/streaming input).
//
// Mirrors the borrow family, minus `deserialize_str` / `deserialize_bytes`.
// The `'s` lifetime is the deserializer's *session* - the region over which
// the deserializer and its `Claim`s remain valid. Analogous to the borrow
// family's `'de`, but with no zero-copy borrow methods.
//
// `Probe`, `Chunk`, `StrAccess`, `BytesAccess`, `hit!`, `DeserializeError`,
// and `select_probe!` are shared between families.
//
// The two families are independent: no supertrait relationship, no blanket
// impls. A format implements whichever family (or both) it can support.
//
// # Parallel scanning - deadlock hazard
//
// The owned family reads from a streaming source where data arrives
// incrementally.  Every reader (entry handle, map/seq accessor, or
// str/bytes chunk accessor) shares the same underlying buffer and
// advances through it cooperatively via `fork`.
//
// **For callers:**  You must not read one forked handle to completion
// and then decide what to do with another - that will deadlock.  The
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
// reader to finish before another can make progress - doing so creates
// a circular dependency that deadlocks the single-threaded executor.
// The `shared_buf` module provides a reference implementation of this
// contract.
// ===========================================================================

/// Owned counterpart to [`Deserialize`](crate::Deserialize).
///
/// `D` is a trait-level generic so derive-generated `impl` blocks can express
/// bounds on `D::Entry`, `D::Error`, and related associated types.  `Extra` is
/// an associated type so format crates can ship blanket and explicit impls in
/// the same crate without coherence conflicts (see [`crate::Deserialize`] for
/// rationale).
pub trait DeserializeOwned<D: DeserializerOwned>: Sized {
    type Extra;
    async fn deserialize_owned(
        d: D,
        extra: Self::Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>;
}

/// Owned counterpart to [`DeserializeFromMap`](crate::DeserializeFromMap).
pub trait DeserializeFromMapOwned<M: MapAccessOwned>: Sized {
    type Extra;
    async fn deserialize_from_map_owned(
        map: M,
        extra: Self::Extra,
    ) -> Result<Probe<(M::MapClaim, Self)>, M::Error>;
}

/// Owned counterpart to [`DeserializeFromSeq`](crate::DeserializeFromSeq).
pub trait DeserializeFromSeqOwned<S: SeqAccessOwned>: Sized {
    type Extra;
    async fn deserialize_from_seq_owned(
        seq: S,
        extra: Self::Extra,
    ) -> Result<Probe<(S::SeqClaim, Self)>, S::Error>;
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
/// **You must drive all forked readers concurrently** - typically via
/// [`select_probe!`](crate::select_probe).  Sequentially awaiting one reader
/// to completion before polling another will deadlock: the first reader may
/// block waiting for buffer data that cannot arrive until all sibling readers
/// have consumed the current chunk.
///
/// This is safe to do: forked readers are guaranteed not to interfere with
/// each other, and every reader is automatically suspended and resumed as
/// new data becomes available, provided all readers are being polled.
pub trait DeserializerOwned: Sized {
    type Error: DeserializeError;

    type Claim;
    type EntryClaim;
    type Entry: EntryOwned<Claim = Self::EntryClaim, Error = Self::Error>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>;
}

/// Owned counterpart to [`Entry`](crate::Entry). Drops `deserialize_str` / `deserialize_bytes`
/// (the borrow-only methods); strings and bytes must be read via
/// [`StrAccessOwned`] / [`BytesAccessOwned`].
///
/// # For implementors - `fork` contract
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
pub trait EntryOwned: Sized {
    type Error: DeserializeError;
    type Claim;

    /// Concrete sub-deserializer this Entry spawns for [`DeserializeOwned`] dispatch.
    type SubDeserializer: DeserializerOwned<Claim = Self::Claim, Error = Self::Error>;

    type StrChunks: StrAccessOwned<Claim = Self::Claim, Error = Self::Error>;
    type BytesChunks: BytesAccessOwned<Claim = Self::Claim, Error = Self::Error>;
    type NumberChunks<Enc: NumberEncoding>: NumberAccessOwned<Enc, Claim = Self::Claim, Error = Self::Error>;
    type Map: MapAccessOwned<MapClaim = Self::Claim, Error = Self::Error>;
    type Seq: SeqAccessOwned<SeqClaim = Self::Claim, Error = Self::Error>;
    type Enum: EnumAccessOwned<Claim = Self::Claim, Error = Self::Error>;

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error>;
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error>;
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error>;
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error>;
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error>;

    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>;

    async fn deserialize_option<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>;

    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMapOwned<Self::Map>;

    async fn deserialize_seq_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeqOwned<Self::Seq>;

    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error>;

    async fn deserialize_enum_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnumOwned<Self::Enum>;

    fn fork(&mut self) -> Self;
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

/// Owned counterpart to [`crate::StrAccess`]. Takes `self` by value and a sync
/// closure that borrows the chunk `&str`. The closure maps the short-lived
/// borrow to an owned `R`; the accessor is handed back alongside `R` on
/// `Data` and the claim emerges on `Done`.
///
/// **Callers:** forked accessors must be driven concurrently (e.g. via
/// `select_probe!`).  Awaiting one fork to `Done` before polling another
/// will deadlock - the buffer cannot advance until all forks have consumed
/// the current chunk.
pub trait StrAccessOwned: Sized {
    type Claim;
    type Error: DeserializeError;

    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`crate::BytesAccess`]. Same closure pattern as
/// [`StrAccessOwned`].
///
/// **Callers:** forked accessors must be driven concurrently.
/// See [`StrAccessOwned`] for the deadlock hazard.
pub trait BytesAccessOwned: Sized {
    type Claim;
    type Error: DeserializeError;

    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`crate::NumberAccess`]. Same closure pattern as
/// [`StrAccessOwned`].
///
/// **Callers:** forked accessors must be driven concurrently.
/// See [`StrAccessOwned`] for the deadlock hazard.
pub trait NumberAccessOwned<Enc: NumberEncoding = Ascii>: Sized {
    type Claim;
    type Error: DeserializeError;

    async fn next_number_chunk<R>(
        self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`SeqAccess`](crate::SeqAccess).
///
/// **Callers:** forked sequence accessors must be driven concurrently.
/// See [`DeserializerOwned`] for the deadlock hazard.
pub trait SeqAccessOwned: Sized {
    type Error: DeserializeError;
    type SeqClaim;
    type ElemClaim;
    type Elem: SeqEntryOwned<Claim = Self::ElemClaim, Error = Self::Error>;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>;
}

/// Owned counterpart to [`SeqEntry`](crate::SeqEntry).
pub trait SeqEntryOwned: Sized {
    type Error: DeserializeError;
    type Claim;

    type SubDeserializer: DeserializerOwned<Claim = Self::Claim, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>;

    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ===========================================================================
// Map Access
// ===========================================================================

// ---------------------------------------------------------------------------
// Claim chain: MapKeyProbeOwned → MapKeyClaimOwned → MapValueProbeOwned → MapValueClaimOwned → next MapKeyProbeOwned
// ---------------------------------------------------------------------------

/// A key probe for a single map key. Forkable for racing multiple arms.
pub trait MapKeyProbeOwned: Sized {
    type Error: DeserializeError;
    type KeyClaim: MapKeyClaimOwned<Error = Self::Error>;

    type KeySubDeserializer: DeserializerOwned<Claim = Self::KeyClaim, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: DeserializeOwned<Self::KeySubDeserializer>;

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
///
/// Separate type from `ValueClaim` so you can't use a key claim to
/// advance past a value without consuming it.
pub trait MapKeyClaimOwned: Sized {
    type Error: DeserializeError;
    type MapClaim;
    type ValueProbe: MapValueProbeOwned<MapClaim = Self::MapClaim, Error = Self::Error>;

    /// Consume this key claim and produce a value probe for the corresponding
    /// map value. Format-specific (e.g. JSON reads `:` and the value start token).
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error>;
}

/// A value probe that can deserialize a value or skip it.
pub trait MapValueProbeOwned: Sized {
    type Error: DeserializeError;
    type MapClaim;
    type ValueClaim: MapValueClaimOwned<MapClaim = Self::MapClaim, Error = Self::Error>;

    type ValueSubDeserializer: DeserializerOwned<Claim = Self::ValueClaim, Error = Self::Error>;
    fn fork(&mut self) -> Self;

    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeOwned<Self::ValueSubDeserializer>;

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error>;
}

/// Proof that a value was consumed. Advances to the next key or ends the map.
pub trait MapValueClaimOwned: Sized {
    type Error: DeserializeError;
    type KeyProbe: MapKeyProbeOwned<Error = Self::Error>;
    type MapClaim;

    /// Consume this value claim and advance the map.
    ///
    /// `unsatisfied` is the number of arms that still require a value (required
    /// fields not yet matched). `open` is the number of arms still willing to
    /// run, including both unsatisfied required-field arms and virtual arms
    /// (skip-unknown, dup-detect, tag-inject) that are always active. When
    /// `open == 0`, the format may skip remaining entries and close the map
    /// early. When `unsatisfied == 0` but `open > 0`, the format should
    /// continue iterating to let virtual arms consume or inspect remaining keys.
    async fn next_key(
        self,
        unsatisfied: usize,
        open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Arm stack - tuple-stack of closure-based map field arms
// ---------------------------------------------------------------------------

/// Shorthand for the key probe type reachable from a DeserializerOwned.
pub type KP<D> = <<<D as DeserializerOwned>::Entry as EntryOwned>::Map as MapAccessOwned>::KeyProbe;
/// Shorthand for the value claim type reachable from a key probe type.
pub type VC<KP> = <<<KP as MapKeyProbeOwned>::KeyClaim as MapKeyClaimOwned>::ValueProbe as MapValueProbeOwned>::ValueClaim;
/// Shorthand for the value probe type reachable from a key probe type.
pub type VP<KP> = <<KP as MapKeyProbeOwned>::KeyClaim as MapKeyClaimOwned>::ValueProbe;
/// Shorthand for the value probe type reachable from a DeserializerOwned.
pub type VP2<D> = <<KP<D> as MapKeyProbeOwned>::KeyClaim as MapKeyClaimOwned>::ValueProbe;

/// Shorthand for the sequence element entry type reachable from a DeserializerOwned.
pub type SE<D> = <<<D as DeserializerOwned>::Entry as EntryOwned>::Seq as SeqAccessOwned>::Elem;

pub use crate::enum_arm::EnumArmStackOwned;
pub use crate::map_arm::MapArmStackOwned;

// ===========================================================================
// Owned-family enum access
// ===========================================================================

/// Owned counterpart to [`crate::EnumVariantProbe`].
pub trait EnumVariantProbeOwned: Sized {
    type Error: DeserializeError;
    type Claim;
    type PayloadDeserializer: DeserializerOwned<Claim = Self::Claim, Error = Self::Error>;

    fn fork(&mut self) -> Self;

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

    async fn deserialize_payload_by_name<T, W>(
        self,
        _candidates: W,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
        W: ConcatableArray<T = (&'static str, usize)> + Copy + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        Ok(Probe::Miss)
    }

    async fn deserialize_unit_by_index(
        self,
        _expected_idx: usize,
    ) -> Result<Probe<(Self::Claim, usize)>, Self::Error> {
        Ok(Probe::Miss)
    }

    async fn deserialize_payload_by_index<T>(
        self,
        _expected_idx: usize,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        Ok(Probe::Miss)
    }

    /// Try to deserialize `T` directly from the current token with no discriminant.
    /// Used for untagged variants. Default impl: `Ok(Probe::Miss)`.
    async fn deserialize_value_by_shape<T>(
        self,
        _extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        Ok(Probe::Miss)
    }
}

/// Owned counterpart to [`crate::EnumAccess`].
pub trait EnumAccessOwned: Sized {
    type Error: DeserializeError;
    type Claim;
    type VariantProbe: EnumVariantProbeOwned<Claim = Self::Claim, Error = Self::Error>;

    async fn iterate<S>(self, arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStackOwned<Self::VariantProbe>;
}

/// Owned counterpart to [`crate::DeserializeFromEnum`].
pub trait DeserializeFromEnumOwned<E: EnumAccessOwned>: Sized {
    type Extra;
    async fn deserialize_from_enum_owned(
        e: E,
        extra: Self::Extra,
    ) -> Result<Probe<(E::Claim, Self)>, E::Error>;
}

/// An in-progress map for the owned family.
///
/// Takes a [`MapArmStackOwned`] - a left-nested tuple of [`MapArmSlot`]s.
/// The map impl owns the iteration loop:
///
/// 1. Produce the first key probe (or detect empty map).
/// 2. Each round: call `arms.race_keys(kp)` to race key callbacks,
///    then on hit: `key_claim.into_value_probe()` → `arms.dispatch_value(idx, vp)`.
/// 3. `value_claim.next_key(unsatisfied, open)` → next round or done.
/// 4. Extract `arms.take_outputs()`.
pub trait MapAccessOwned: Sized {
    type Error: DeserializeError;
    type MapClaim;
    type KeyProbe: MapKeyProbeOwned<Error = Self::Error>;

    /// Drive the map iteration with the given arm stack, for a fixed
    /// compile-time field set (structs, enums) whose end is signaled by the
    /// arm stack becoming satisfied. See [`crate::MapArmStackOwned::Dynamic`].
    ///
    /// Returns `Hit((MapClaim, Outputs))` on success, `Miss` if a value
    /// type mismatched or a required field was missing, `Err` on format errors.
    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error>;

    /// Drive the map iteration for an unbounded/runtime-sized collection
    /// (e.g. HashMap's `CollectMap`) rather than a fixed schema — see
    /// [`crate::MapArmStackOwned::Dynamic`]. No default: most formats' maps
    /// are wire-uniform regardless of consumer shape, in which case both this
    /// and [`Self::iterate`] should delegate to one shared helper rather than
    /// one calling the other. Formats with a genuinely different wire shape
    /// for schema-less collections give this a real, different implementation
    /// instead.
    async fn iterate_dyn<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Universal DeserializeOwned impls
// ---------------------------------------------------------------------------

impl<D, T> DeserializeOwned<D> for Option<T>
where
    D: DeserializerOwned,
    T: DeserializeOwned<<D::Entry as EntryOwned>::SubDeserializer>,
{
    type Extra = T::Extra;
    async fn deserialize_owned(
        d: D,
        extra: Self::Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut extra = Some(extra);
        d.entry(move |[e]| {
            let extra = extra.take().expect("entry closure called more than once");
            async move { e.deserialize_option::<T>(extra).await }
        })
        .await
    }
}
