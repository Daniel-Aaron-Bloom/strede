use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use pin_project::pin_project;

use crate::Probe;

pub mod borrow;
pub mod owned;
pub use borrow::MapArmStack;
pub use owned::MapArmStackOwned;

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
// True / False - type-level booleans for MapArmStack::Dynamic / MapArmStackOwned::Dynamic
// ---------------------------------------------------------------------------

/// Type-level "true", paired with [`False`]. Used as
/// [`crate::MapArmStack::Dynamic`] / [`crate::MapArmStackOwned::Dynamic`] for
/// arm stacks representing an unbounded/runtime-sized collection (e.g.
/// HashMap's `CollectMap`) that requires the format to read an explicit
/// wire-level length before iterating.
///
/// Encoding this as a type (selected via associated-type dispatch) rather
/// than a `bool` const lets a format provide two genuinely separate
/// implementations for the two iteration strategies — one per marker type —
/// instead of one shared function with a runtime `if`. The latter forces the
/// compiler to lay out the union of both branches' state for every
/// monomorphization, even though only one branch is ever reachable for a
/// given concrete arm stack.
pub struct True;

/// Type-level "false", paired with [`True`]. The default/expected value for
/// arm stacks with a fixed compile-time field set (structs, enums) whose end
/// is signaled by the arm stack becoming satisfied rather than by a wire
/// length.
pub struct False;

// ---------------------------------------------------------------------------
// NextKey - shared by both families' value-claim traits
// ---------------------------------------------------------------------------

/// Returned by [`crate::MapValueClaim::next_key`] / [`crate::MapValueClaimOwned::next_key`]
/// to either yield the next key probe or signal map exhaustion.
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
/// - `KeyFn: FnMut(KP, usize) -> KeyFut` — races against an incoming key probe from the
///   format. The `usize` is this arm's global positional index (0-based), computed at
///   `init_race` time. Named-only arms ignore it; arms that also support positional access
///   can call `kp.deserialize_key_by_index(i)` and race it via `select_probe!`.
/// - `ValFn: FnMut(ValueProbe, K) -> ValFut` — dispatches the value once a key is resolved.
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

/// Wrapper that marks a [`MapArmSlot`] as one arm in a [`crate::map_arms!`] call.
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
/// `KeyFn: FnMut(KP, usize) -> KeyFut` - creates the key-matching future.
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
pub struct DetectDuplicates<S, W, KeyFn, SkipFn> {
    pub inner: S,
    pub key_fn: KeyFn,
    pub skip_fn: SkipFn,
    pub wire_names: W,
    pub dup: &'static str,
}

impl<S, W, KeyFn, SkipFn> DetectDuplicates<S, W, KeyFn, SkipFn> {
    pub fn new(inner: S, wire_names: W, key_fn: KeyFn, skip_fn: SkipFn) -> Self {
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
pub struct TagInjectingStack<'v, S, W, TagKeyFn, TagValFn> {
    pub inner: S,
    pub tag_key_fn: TagKeyFn,
    pub tag_val_fn: TagValFn,
    pub tag_field: &'static str,
    pub tag_candidates: W,
    pub tag_value: &'v core::cell::Cell<Option<usize>>,
}

impl<'v, S, W, TagKeyFn, TagValFn> TagInjectingStack<'v, S, W, TagKeyFn, TagValFn> {
    pub fn new(
        inner: S,
        tag_field: &'static str,
        tag_candidates: W,
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

/// Pinned race state for wrappers that add a virtual arm (`SkipUnknownOwned`, `DetectDuplicates`).
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

/// Race state for [`TagInjectingStack`]: tag future + inner state.
/// Tag arm is at index 0, inner arms at 1..SIZE.
#[pin_project]
pub struct TagRaceState<TagFut, InnerState> {
    #[pin]
    pub tag_fut: Option<TagFut>,
    #[pin]
    pub inner: InnerState,
}

/// Dispatch state for [`TagInjectingStack`].
#[pin_project(project = TagDispatchProj)]
pub enum TagDispatchState<TagFut, InnerState> {
    Tag(#[pin] TagFut),
    Inner(#[pin] InnerState),
}

// ---------------------------------------------------------------------------
// poll_key_slot - shared helper
// ---------------------------------------------------------------------------

/// Poll an `Option<Future>` slot, returning `Miss` if the slot is `None`.
///
/// Used by both families' `poll_race_one` implementations.
#[inline(always)]
pub(crate) fn poll_key_slot<F, KC, K, E>(
    mut slot: Pin<&mut Option<F>>,
    cx: &mut Context<'_>,
) -> Poll<Result<Probe<(KC, K)>, E>>
where
    F: Future<Output = Result<Probe<(KC, K)>, E>>,
{
    match slot.as_mut().as_pin_mut() {
        None => Poll::Ready(Ok(Probe::Miss)),
        Some(fut) => match fut.poll(cx) {
            Poll::Ready(Ok(Probe::Hit(v))) => {
                slot.set(None);
                Poll::Ready(Ok(Probe::Hit(v)))
            }
            Poll::Ready(Ok(Probe::Miss)) => {
                slot.set(None);
                Poll::Ready(Ok(Probe::Miss))
            }
            Poll::Ready(Err(e)) => {
                slot.set(None);
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
/// Each arm is `key_closure => value_closure`. Key closures receive `(KP, usize)` where
/// the `usize` is the arm's global positional index — ignore it (`_i`) for named-only
/// matching, or pass it to `kp.deserialize_key_by_index(i)` for positional support.
///
/// ```rust,ignore
/// let arms = map_arms! {
///     |kp, _i| kp.deserialize_key::<Match, _>("secs") => |vp, k| { ... },
///     |kp, _i| kp.deserialize_key::<Match, _>("nanos") => |vp, k| { ... },
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

// ---------------------------------------------------------------------------
// Helper macros - borrow family
// ---------------------------------------------------------------------------

/// Wraps a [`MapArmStack`] so that unknown map keys are silently consumed (borrow family).
#[macro_export]
macro_rules! SkipUnknown {
    ($inner:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbe as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp: $kp, _i: usize| kp.deserialize_key::<$crate::Skip>(()),
                |vp: $vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbe as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
    ($inner:expr) => {{
        use $crate::MapKeyProbe as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp, _i: usize| kp.deserialize_key::<$crate::Skip>(()),
                |vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbe as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
}

/// Wraps a [`MapArmStack`] to return a duplicate-field error (borrow family).
#[macro_export]
macro_rules! DetectDuplicates {
    ($inner:expr, $wire_names:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbe as _;
        let __wn = $wire_names;
        $crate::DetectDuplicates::new(
            $inner,
            __wn,
            move |kp: $kp, _i: usize| kp.deserialize_key::<$crate::MatchVals<usize, _>>(__wn),
            |vp: $vp| vp.skip(),
        )
    }};
}

/// Wraps a [`MapArmStack`] to intercept a tag field (borrow family).
#[macro_export]
macro_rules! TagInjectingStack {
    ($inner:expr, $tag_field:expr, $tag_candidates:expr, $tag_value:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbe as _;
        use $crate::MapValueProbe as _;
        let __tf = $tag_field;
        let __tc = $tag_candidates;
        $crate::TagInjectingStack::new(
            $inner,
            __tf,
            __tc,
            $tag_value,
            move |kp: $kp, _i: usize| kp.deserialize_key::<$crate::Match>(__tf),
            move |vp: $vp| vp.deserialize_value::<$crate::MatchVals<usize, _>>(__tc),
        )
    }};
}

// ---------------------------------------------------------------------------
// Helper macros - owned family
// ---------------------------------------------------------------------------

/// Wraps a [`MapArmStackOwned`] so that unknown map keys are silently consumed.
///
/// Expands to `(arms, VirtualArmSlot::new(...))` with a skip key/value arm.
#[macro_export]
macro_rules! SkipUnknownOwned {
    // 3-arg form: explicit KP/VP types for closure annotations (used by derive).
    ($inner:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbeOwned as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp: $kp, _i: usize| kp.deserialize_key::<$crate::Skip>(()),
                |vp: $vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbeOwned as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
    // 1-arg form: types inferred from context (for hand-written code).
    ($inner:expr) => {{
        use $crate::MapKeyProbeOwned as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp, _i: usize| kp.deserialize_key::<$crate::Skip>(()),
                |vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbeOwned as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
}

/// Wraps a [`MapArmStackOwned`] to return a duplicate-field error (owned family).
///
/// `DetectDuplicatesOwned!(inner, wire_names, KP, VP)` expands to
/// `DetectDuplicates::new(inner, wire_names, key_fn, skip_fn)` with typed closures.
#[macro_export]
macro_rules! DetectDuplicatesOwned {
    ($inner:expr, $wire_names:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbeOwned as _;
        use $crate::MapValueProbeOwned as _;
        let __wn = $wire_names;
        $crate::DetectDuplicates::new(
            $inner,
            __wn,
            move |kp: $kp, _i: usize| kp.deserialize_key::<$crate::MatchVals<usize, _>>(__wn),
            |vp: $vp| vp.skip(),
        )
    }};
}

/// Wraps a [`MapArmStackOwned`] to intercept a tag field (owned family).
///
/// `TagInjectingStack!(inner, tag_field, tag_candidates, tag_value, KP, VP)`
#[macro_export]
macro_rules! TagInjectingStackOwned {
    ($inner:expr, $tag_field:expr, $tag_candidates:expr, $tag_value:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbeOwned as _;
        use $crate::MapValueProbeOwned as _;
        let __tf = $tag_field;
        let __tc = $tag_candidates;
        $crate::TagInjectingStack::new(
            $inner,
            __tf,
            __tc,
            $tag_value,
            move |kp: $kp, _i: usize| kp.deserialize_key::<$crate::Match>(__tf),
            move |vp: $vp| vp.deserialize_value::<$crate::MatchVals<usize, _>>(__tc),
        )
    }};
}
