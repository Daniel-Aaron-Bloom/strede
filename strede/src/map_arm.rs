use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use pin_project::pin_project;

use crate::Probe;

// ===========================================================================
// Shared map arm infrastructure - used by both the borrow and owned families.
//
// The borrow family's `MapArmStack<'de, KP>` and the owned family's
// `MapArmStackOwned<KP>` are separate traits (different `KP` constraints,
// different value-probe type aliases), but the data structures, pin-projection
// helpers, and the `poll_key_slot` helper function are identical. They live
// here so neither family module needs to import them from the other.
// ===========================================================================

// ---------------------------------------------------------------------------
// NextKey - shared by both families' value-claim traits
// ---------------------------------------------------------------------------

/// Returned by the value-claim's `next_key` method to either continue or end
/// map iteration.  Used by both [`crate::ValueClaim`] and [`crate::MapValueClaim`].
pub enum NextKey<KeyProbe, MapClaim> {
    /// Another KV pair is available; here is the key probe.
    Entry(KeyProbe),
    /// Map is exhausted; here is the proof-of-consumption.
    Done(MapClaim),
}

// ---------------------------------------------------------------------------
// Arm-stack data structures
// ---------------------------------------------------------------------------

/// State of a single arm in the map iteration.
///
/// - `Empty` - no key matched yet for this field.
/// - `Key(K)` - key matched this round, waiting for value dispatch.
/// - `Done(K, V)` - both key and value resolved; arm is satisfied.
pub enum ArmState<K, V> {
    Empty,
    Key(K),
    Done(K, V),
}

impl<K, V> ArmState<K, V> {
    pub fn is_done(&self) -> bool {
        matches!(self, ArmState::Done(..))
    }
}

/// One slot in the arm stack. Holds the key callback, value callback, and
/// accumulated state for a single struct field.
///
/// - `KeyFn: FnMut(KP) -> KeyFut` where `KeyFut: Future<Output = Result<Probe<(KeyClaim, K)>, Error>>`
/// - `ValFn: FnMut(ValueProbe, K) -> ValFut` where `ValFut: Future<Output = Result<Probe<(ValueClaim, V)>, Error>>`
pub struct MapArmSlot<K, V, KeyFn, ValFn> {
    pub key_fn: KeyFn,
    pub val_fn: ValFn,
    pub state: ArmState<K, V>,
}

impl<K, V, KeyFn, ValFn> MapArmSlot<K, V, KeyFn, ValFn> {
    pub fn new(key_fn: KeyFn, val_fn: ValFn) -> Self {
        Self {
            key_fn,
            val_fn,
            state: ArmState::Empty,
        }
    }
}

/// Base of the arm tuple stack. Analogous to `SelectProbeBase`.
pub struct MapArmBase;

/// Wrapper that marks a [`MapArmSlot`] as one arm in a [`map_arms!`] call.
///
/// Used with `+` on [`MapArmBase`] to build the arm stack without recursive macros:
/// `MapArmBase + MapArm(slot0) + MapArm(slot1) + ...`
pub struct MapArm<S>(pub S);

impl<S> core::ops::Add<MapArm<S>> for MapArmBase {
    type Output = (MapArmBase, S);
    fn add(self, rhs: MapArm<S>) -> (MapArmBase, S) {
        (self, rhs.0)
    }
}

impl<Rest, S, T> core::ops::Add<MapArm<T>> for (Rest, S) {
    type Output = ((Rest, S), T);
    fn add(self, rhs: MapArm<T>) -> ((Rest, S), T) {
        (self, rhs.0)
    }
}

/// A virtual arm slot for wrapper-style arms (skip, dup-detect, tag-inject).
///
/// Unlike [`MapArmSlot`], a virtual arm:
/// - Is **never satisfied** - excluded from `unsatisfied_count`; contributes 1 to `open_count`.
/// - Produces **no output** in `take_outputs`.
/// - Stores `K` from the key race in `pending_key` so `init_dispatch` can
///   pass it to `val_fn`.
///
/// `KeyFn: FnMut(KP) -> KeyFut` - creates the key-matching future.
/// `ValFn: FnMut(VP, K) -> ValFut` - creates the value-dispatch future.
pub struct VirtualArmSlot<K, KeyFn, ValFn> {
    pub key_fn: KeyFn,
    pub val_fn: ValFn,
    pub pending_key: Option<K>,
}

impl<K, KeyFn, ValFn> VirtualArmSlot<K, KeyFn, ValFn> {
    pub fn new(key_fn: KeyFn, val_fn: ValFn) -> Self {
        Self {
            key_fn,
            val_fn,
            pending_key: None,
        }
    }
}

/// Wraps a [`crate::MapArmStackOwned`] / [`crate::MapArmStack`] to return a
/// duplicate-field error when a wire key that already matched an arm appears a
/// second time.
///
/// `KeyFn` produces the dup arm's key-race future (calls `deserialize_key`).
/// `SkipFn` produces the dup arm's value-skip future (calls `vp.skip()`).
/// Both are closures whose types are inferred at construction.
pub struct DetectDuplicatesOwned<S, const M: usize, KeyFn, SkipFn> {
    pub inner: S,
    pub key_fn: KeyFn,
    pub skip_fn: SkipFn,
    pub wire_names: [(&'static str, usize); M],
    pub dup: &'static str,
}

impl<S, const M: usize, KeyFn, SkipFn> DetectDuplicatesOwned<S, M, KeyFn, SkipFn> {
    pub fn new(
        inner: S,
        wire_names: [(&'static str, usize); M],
        key_fn: KeyFn,
        skip_fn: SkipFn,
    ) -> Self {
        Self {
            inner,
            key_fn,
            skip_fn,
            wire_names,
            dup: "unknown",
        }
    }
}

/// Wraps a [`crate::MapArmStackOwned`] / [`crate::MapArmStack`] to intercept a
/// tag field and capture the matched variant index into a `Cell<Option<usize>>`.
///
/// Tag arm is at index 0 (highest priority). Inner arms at indices 1..SIZE.
pub struct TagInjectingStackOwned<'v, S, const N: usize, TagKeyFn, TagValFn> {
    pub inner: S,
    pub tag_key_fn: TagKeyFn,
    pub tag_val_fn: TagValFn,
    pub tag_field: &'static str,
    pub tag_candidates: [(&'static str, usize); N],
    pub tag_value: &'v core::cell::Cell<Option<usize>>,
}

impl<'v, S, const N: usize, TagKeyFn, TagValFn>
    TagInjectingStackOwned<'v, S, N, TagKeyFn, TagValFn>
{
    pub fn new(
        inner: S,
        tag_field: &'static str,
        tag_candidates: [(&'static str, usize); N],
        tag_value: &'v core::cell::Cell<Option<usize>>,
        tag_key_fn: TagKeyFn,
        tag_val_fn: TagValFn,
    ) -> Self {
        Self {
            inner,
            tag_key_fn,
            tag_val_fn,
            tag_field,
            tag_candidates,
            tag_value,
        }
    }
}

/// Concatenates two arm stacks into one, running both concurrently.
///
/// Arm indices from `A` are `0..A::SIZE`; arm indices from `B` are offset
/// by `A::SIZE`. Outputs are `(A::Outputs, B::Outputs)`.
pub struct StackConcat<A, B>(pub A, pub B);

// ---------------------------------------------------------------------------
// Pin-projection helpers - shared between both arm-stack impls
// ---------------------------------------------------------------------------

/// Pinned race state for `(Rest, MapArmSlot)` and `(Rest, VirtualArmSlot)`.
#[pin_project]
pub struct SlotRaceState<RestState, KeyFut> {
    #[pin]
    pub rest: RestState,
    #[pin]
    pub this: Option<KeyFut>,
}

/// Pinned dispatch state for `(Rest, MapArmSlot)` and `(Rest, VirtualArmSlot)`.
#[pin_project(project = SlotDispatchProj)]
pub enum SlotDispatchState<RestState, ValFut> {
    ThisArm(#[pin] ValFut),
    Delegated(#[pin] RestState),
}

/// Pinned race state for [`StackConcat`].
#[pin_project]
pub struct ConcatRaceState<AState, BState> {
    #[pin]
    pub a: AState,
    #[pin]
    pub b: BState,
}

/// Pinned dispatch state for [`StackConcat`].
#[pin_project(project = ConcatDispatchProj)]
pub enum ConcatDispatchState<AState, BState> {
    InA(#[pin] AState),
    InB(#[pin] BState),
}

/// Pinned race state for wrappers that add a virtual arm (`SkipUnknownOwned`, `DetectDuplicatesOwned`).
#[pin_project]
pub struct WrapperRaceState<InnerState, VirtualFut> {
    #[pin]
    pub inner: InnerState,
    #[pin]
    pub virtual_arm: Option<VirtualFut>,
}

/// Pinned dispatch state for wrappers that add a virtual arm.
#[pin_project(project = WrapperDispatchProj)]
pub enum WrapperDispatchState<InnerState, VirtualFut> {
    Virtual(#[pin] VirtualFut),
    Inner(#[pin] InnerState),
}

/// Race state for [`TagInjectingStackOwned`]: tag future + inner state.
/// Tag arm is at index 0, inner arms at 1..SIZE.
#[pin_project]
pub struct TagRaceState<TagFut, InnerState> {
    #[pin]
    pub tag_fut: Option<TagFut>,
    #[pin]
    pub inner: InnerState,
}

/// Dispatch state for [`TagInjectingStackOwned`].
#[pin_project(project = TagDispatchProj)]
pub enum TagDispatchState<TagFut, InnerState> {
    Tag(#[pin] TagFut),
    Inner(#[pin] InnerState),
}

// ---------------------------------------------------------------------------
// poll_key_slot - shared helper
// ---------------------------------------------------------------------------

/// Poll an `Option<Future>` slot, returning `Miss` if `None` or already done.
///
/// Used by both families' `poll_race_one` implementations.
pub(crate) fn poll_key_slot<F, KC, K, E>(
    slot: Pin<&mut Option<F>>,
    done: &mut bool,
    cx: &mut Context<'_>,
) -> Poll<Result<Probe<(KC, K)>, E>>
where
    F: Future<Output = Result<Probe<(KC, K)>, E>>,
{
    if *done {
        return Poll::Ready(Ok(Probe::Miss));
    }
    match slot.as_pin_mut() {
        None => {
            *done = true;
            Poll::Ready(Ok(Probe::Miss))
        }
        Some(fut) => match fut.poll(cx) {
            Poll::Ready(Ok(Probe::Hit(v))) => {
                *done = true;
                Poll::Ready(Ok(Probe::Hit(v)))
            }
            Poll::Ready(Ok(Probe::Miss)) => {
                *done = true;
                Poll::Ready(Ok(Probe::Miss))
            }
            Poll::Ready(Err(e)) => {
                *done = true;
                Poll::Ready(Err(e))
            }
            Poll::Pending => Poll::Pending,
        },
    }
}

// ---------------------------------------------------------------------------
// map_arms! and map_outputs! macros
// ---------------------------------------------------------------------------

/// Build a left-nested arm tuple from a flat list of arm definitions.
///
/// Each arm is `key_closure => value_closure`, which expands to a
/// [`MapArmSlot`] wrapping the two closures.
///
/// ```rust,ignore
/// let arms = map_arms! {
///     |kp| kp.deserialize_key::<Match, _>("secs") => |vp, k| { ... },
///     |kp| kp.deserialize_key::<Match, _>("nanos") => |vp, k| { ... },
/// };
/// ```
///
/// Expands to `((MapArmBase, MapArmSlot::new(key0, val0)), MapArmSlot::new(key1, val1))`.
#[macro_export]
macro_rules! map_arms {
    ($key_fn:expr => $val_fn:expr $(, $rest_key:expr => $rest_val:expr)* $(,)?) => {
        $crate::MapArmBase
            + $crate::MapArm($crate::MapArmSlot::new($key_fn, $val_fn))
            $(+ $crate::MapArm($crate::MapArmSlot::new($rest_key, $rest_val)))*
    };
}

/// Destructure a left-nested output tuple from [`crate::MapArmStackOwned::take_outputs`]
/// or [`crate::MapArmStack::take_outputs`].
///
/// ```rust,ignore
/// let (claim, map_outputs!(opt_secs, opt_nanos)) = hit!(map.iterate(arms).await);
/// ```
///
/// Expands to the nested pattern `(((), opt_secs), opt_nanos)`.
#[macro_export]
macro_rules! map_outputs {
    ($first:pat $(, $rest:pat)* $(,)?) => {
        $crate::__left_nest_pat!((), $first $(, $rest)*)
    };
}
