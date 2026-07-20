use const_array_concat::ConcatableArray;

use crate::borrow::{MapArmStack, MapKeyProbe};
use crate::owned::{MapArmStackOwned, MapKeyProbeOwned};

/// Types that expose their arm stacks for map-based deserialization.
///
/// Unlike [`crate::borrow::DeserializeFromMap`] (which owns the `iterate`
/// call), this trait hands the arm stack to the caller. Parent structs can
/// `StackConcat` arm stacks from all fields—regular and flatten alike—and
/// call `iterate` exactly once, so flatten composition needs no runtime
/// adapter or continuation chain.
///
/// Non-flatten fields contribute a single-slot arm stack; flatten fields
/// contribute their full arm stack recursively. The outer struct's impl
/// combines them with `StackConcat`.
pub trait MapFieldProvider<'de, KP: MapKeyProbe<'de>>: Sized {
    /// Combined output type from `iterate` on `make_arms()`. Typically a
    /// left-nested tuple of `Option<(K, V)>` per field.
    type Outputs;

    /// Total number of arm slots in the stack produced by `make_arms()`.
    /// Regular fields contribute 1; flatten fields contribute their own ARMS.
    const ARMS: usize;

    /// Array type holding `(wire_name, arm_index)` pairs for all fields.
    /// Arm indices are relative (0-based within this struct's arm stack).
    /// Used by `DetectDuplicates` to detect duplicate keys across all fields.
    type WireNames: ConcatableArray<T = (&'static str, usize)> + Copy;

    /// Build a fresh arm stack for this type's fields.
    ///
    /// `Dynamic = crate::False`: every `MapFieldProvider` is a derived
    /// struct's fixed field set, never an unbounded collection (only
    /// `CollectMap` is `crate::True`, and it never implements this trait) —
    /// constraining it here lets `StackConcat`'s `Dynamic` equality bound
    /// type-check across flatten composition without the derive needing to
    /// know the flattened type's concrete arm-stack type.
    fn make_arms() -> impl MapArmStack<'de, KP, Outputs = Self::Outputs, Dynamic = crate::False>;

    /// Reconstruct `Self` from the outputs after `iterate` has completed.
    /// Returns `None` if a required field was absent; the caller should
    /// return `Ok(Probe::Miss)`.
    fn from_outputs(outputs: Self::Outputs) -> Option<Self>;

    /// Return the wire names with their relative arm indices.
    /// Flatten fields shift their sub-type's indices by the cumulative arm
    /// offset so that `DetectDuplicates` sees correct absolute indices.
    fn wire_names() -> Self::WireNames;
}

/// Owned-family counterpart to [`MapFieldProvider`].
pub trait MapFieldProviderOwned<KP: MapKeyProbeOwned>: Sized {
    type Outputs;
    const ARMS: usize;
    type WireNames: ConcatableArray<T = (&'static str, usize)> + Copy;
    /// See [`MapFieldProvider::make_arms`] for why `Dynamic = crate::False`.
    fn make_arms() -> impl MapArmStackOwned<KP, Outputs = Self::Outputs, Dynamic = crate::False>;
    fn from_outputs(outputs: Self::Outputs) -> Option<Self>;
    fn wire_names() -> Self::WireNames;
}
