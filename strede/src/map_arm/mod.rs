use core::future::Future;
use core::marker::PhantomData;
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
// CandidateArmStack infrastructure - internally-tagged enum flatten
// ---------------------------------------------------------------------------
//
// Supports `#[strede(flatten)]` on a field whose type is an internally-tagged
// enum (`#[strede(tag = "t")]`). Unlike a struct's ordinary flatten
// composition (one shared arm stack via `StackConcat`), an internally-tagged
// enum's variants are mutually exclusive: only one variant's fields are
// actually present on the wire, but which one it is may not be known until
// the tag key arrives (map key order is not guaranteed). `CandidateArmStack`
// races every variant's own arm stack concurrently against the parent's
// shared key stream, and the moment the tag key resolves, permanently
// excludes every other variant's arms from further racing (and, in the same
// pass, prevents them from stealing a wire key that a later-declared correct
// variant also happens to share a field name with - see `CandidateList::poll_race_one`).
//
// `NoTag` (untagged enums, soft cross-candidate elimination) and adjacently-
// tagged support are deliberately out of scope for this primitive - see
// TESTING_GAPS.md item #3(B-2).

/// One candidate variant's own arm stack, participating in a
/// [`crate::map_arm::borrow::CandidateArmStack`] /
/// [`crate::map_arm::owned::CandidateArmStackOwned`].
///
/// `index` is this candidate's 0-based position among all candidates, in
/// declaration order - it must match the position used to build the tag's
/// `tag_candidates` array, since the tag arm identifies a winning candidate
/// by this same index.
///
/// `C`: the candidate's own arm stack (e.g. `<VariantHelper as
/// MapFieldProvider<'de, KP>>::make_arms()` for a struct/newtype variant, or
/// `MapArmBase` for a unit variant - a unit variant has no fields of its own,
/// it exists purely to be selected once the tag matches).
///
/// `BuildFn: FnMut(C::Outputs) -> Option<EnumOut>`: reconstructs the
/// containing enum's variant from this candidate's outputs once the tag has
/// selected it. Returns `None` if a required field was absent (mirrors
/// [`crate::MapFieldProvider::from_outputs`]). Called at most once per
/// `iterate()` call, only for the tag-selected candidate.
pub struct Candidate<C, BuildFn> {
    pub index: usize,
    pub arms: C,
    pub build: BuildFn,
    /// Only meaningful to [`crate::map_arm::borrow::NoTagCandidateList`] /
    /// [`crate::map_arm::owned::NoTagCandidateListOwned`] (untagged-enum
    /// flatten's soft-elimination race). The tag-based `CandidateList` /
    /// `CandidateListOwned` impls never read or write this field - elimination
    /// there is driven entirely by `tag_matched`, not per-candidate liveness.
    pub live: bool,
    /// Transient, only meaningful to `NoTagCandidateList`/`NoTagCandidateListOwned`:
    /// did this candidate's own arms hit *this round*? Recomputed at the start
    /// of every round (`NoTagCandidateList::init_round`) before being read by
    /// the elimination pass at the end of that same round - never carries
    /// meaning across rounds.
    pub round_hit: bool,
}

impl<C, BuildFn> Candidate<C, BuildFn> {
    pub fn new(index: usize, arms: C, build: BuildFn) -> Self {
        Self {
            index,
            arms,
            build,
            live: true,
            round_hit: false,
        }
    }
}

/// Base of the candidate list. Left-nested with `+` via
/// `Add<CandidateArm<Candidate<C, BuildFn>>>`, mirroring [`crate::EnumArmBase`].
pub struct CandidateBase;

/// Wrapper that marks a [`Candidate`] as one entry in a candidate list.
///
/// Used with `+` on [`CandidateBase`]:
/// `CandidateBase + CandidateArm(cand0) + CandidateArm(cand1) + ...`
pub struct CandidateArm<S>(pub S);

impl<C, BuildFn> core::ops::Add<CandidateArm<Candidate<C, BuildFn>>> for CandidateBase {
    type Output = (CandidateBase, Candidate<C, BuildFn>);
    fn add(self, rhs: CandidateArm<Candidate<C, BuildFn>>) -> Self::Output {
        (self, rhs.0)
    }
}

impl<Rest, S, C, BuildFn> core::ops::Add<CandidateArm<Candidate<C, BuildFn>>> for (Rest, S) {
    type Output = ((Rest, S), Candidate<C, BuildFn>);
    fn add(self, rhs: CandidateArm<Candidate<C, BuildFn>>) -> Self::Output {
        (self, rhs.0)
    }
}

/// Wraps a candidate list to add the shared discriminant (tag) arm.
///
/// Tag arm is always global index 0; candidate arms occupy `1..SIZE`. Once
/// the tag matches candidate `idx`, every other candidate's arms permanently
/// stop racing (see `CandidateList::poll_race_one`/`init_race`'s
/// `tag_matched` gating) and `take_outputs` builds `EnumOut` from candidate
/// `idx` alone.
pub struct CandidateArmStack<Candidates, EnumOut, W, TagKeyFn, TagValFn> {
    pub candidates: Candidates,
    pub tag_field: &'static str,
    pub tag_candidates: W,
    pub tag_key_fn: TagKeyFn,
    pub tag_val_fn: TagValFn,
    pub tag_matched: Option<usize>,
    // Ties `EnumOut` to the concrete type so `MapArmStack`/`MapArmStackOwned`'s
    // `Outputs = Option<EnumOut>` impl is a well-formed, coherence-checkable
    // type parameter (it otherwise appears only in the `Candidates:
    // CandidateList<'de, KP, EnumOut>` where-bound, which alone doesn't
    // constrain an impl's generic parameters).
    _enum_out: PhantomData<fn() -> EnumOut>,
}

impl<Candidates, EnumOut, W, TagKeyFn, TagValFn>
    CandidateArmStack<Candidates, EnumOut, W, TagKeyFn, TagValFn>
{
    pub fn new(
        tag_field: &'static str,
        tag_candidates: W,
        tag_key_fn: TagKeyFn,
        tag_val_fn: TagValFn,
        candidates: Candidates,
    ) -> Self {
        Self {
            candidates,
            tag_field,
            tag_candidates,
            tag_key_fn,
            tag_val_fn,
            tag_matched: None,
            _enum_out: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// NoTagCandidateArmStack - untagged enum flatten
// ---------------------------------------------------------------------------
//
// Supports `#[strede(flatten)]` on a field whose type is a purely untagged
// enum (`#[strede(untagged)]`, no `tag`). There is no discriminant key at
// all, so every live candidate's own arm stack races directly against the
// parent's shared key stream from round one. A candidate is permanently
// eliminated (`Candidate::live = false`) the first round some *other* live
// candidate hits a key that this candidate's own arms do not recognize -
// proof this candidate can't be the real variant, mirroring exactly what its
// own standalone `deserialize_map` would do encountering the same
// unrecognized key. See `crate::map_arm::borrow::NoTagCandidateList` /
// `crate::map_arm::owned::NoTagCandidateListOwned` for the round-settling
// algorithm, and CLAUDE.md's "Untagged flatten" section for the full design
// write-up (why this is the one case that genuinely needs a `race_keys`
// override, unlike `CandidateArmStack` above).

/// Untagged counterpart to [`CandidateArmStack`] - no tag field, no tag
/// candidates, no tag key/value closures. `Candidates` must implement both
/// [`crate::map_arm::borrow::CandidateList`] / [`crate::map_arm::owned::CandidateListOwned`]
/// (dispatch, reused unchanged from the tag-based primitive) and
/// [`crate::map_arm::borrow::NoTagCandidateList`] /
/// [`crate::map_arm::owned::NoTagCandidateListOwned`] (the new round-settling
/// race, used in place of the default `race_keys`).
pub struct NoTagCandidateArmStack<Candidates, EnumOut> {
    pub candidates: Candidates,
    // See `CandidateArmStack::_enum_out` - same rationale.
    _enum_out: PhantomData<fn() -> EnumOut>,
}

impl<Candidates, EnumOut> NoTagCandidateArmStack<Candidates, EnumOut> {
    pub fn new(candidates: Candidates) -> Self {
        Self {
            candidates,
            _enum_out: PhantomData,
        }
    }
}

/// Pinned per-round state for `(Rest, Candidate<C, BuildFn>)`'s
/// `NoTagCandidateList` / `NoTagCandidateListOwned` impl.
///
/// `this` holds the live candidate's own in-progress race state, cleared to
/// `None` the instant it resolves (mirroring [`poll_key_slot`]'s own
/// one-shot-then-clear discipline - a resolved arm must never be polled
/// again, since re-polling an already-consumed slot silently reads back
/// `Miss`, which would corrupt the round's soft-elimination decision).
/// `resolved` records this round's outcome once settled: `None` while still
/// racing, `Some(None)` once every one of the candidate's own local arms has
/// missed, `Some(Some((global_arm_index, claim)))` once a local arm has hit.
/// A candidate that isn't live at `init_round` time starts pre-resolved to
/// `Some(None)` (contributes nothing, races nothing) and is never touched
/// again until the next round.
#[pin_project]
pub struct NoTagRoundState<RestState, CRaceState, KeyClaim> {
    #[pin]
    pub rest: RestState,
    #[pin]
    pub this: Option<CRaceState>,
    pub resolved: Option<Option<(usize, KeyClaim)>>,
}

/// Top-level `RaceState` for [`NoTagCandidateArmStack`]'s `MapArmStack` /
/// `MapArmStackOwned` impl.
///
/// Composition primitives elsewhere in this module (`StackConcat`,
/// `TagInjectingStack`, `DetectDuplicates`, and `CandidateArmStack` itself)
/// all reach a nested arm stack's arms via `init_race`/`poll_race_one`
/// directly - none of them call a nested stack's `race_keys()`. Since
/// `#[strede(flatten)]` splices a field's `make_arms()` into the parent via
/// exactly this `StackConcat` path, `NoTagCandidateArmStack` must expose its
/// soft-elimination race through `init_race`/`poll_race_one` too, not a
/// `race_keys` override (an earlier draft of this primitive made that
/// mistake - it worked in isolation, calling `iterate()` directly, but was
/// silently never invoked once nested inside a flatten `StackConcat`).
///
/// `poll_race_one` may be called many times per round for different
/// requested arm indices - once per index the outer `race_keys` loop visits
/// before finding a `Hit`, and again for the `BIASED` re-check of every
/// lower index once one is found (all indices *other* than the winner's own,
/// which `race_keys` never re-queries once found). `winner_index` is
/// deliberately *sticky*: once the round settles it is never reset back to
/// `None`, even after `winner_claim` has been taken - otherwise a later call
/// for a *different, non-winning* index would see "not yet settled" again
/// and re-drive `poll_sweep`/`take_winner` on per-candidate state that's
/// already been consumed (an earlier draft of this stack conflated "settled"
/// with "claim not yet taken" into one `Option`, taking it on the winning
/// call - the very next `BIASED`-recheck call for an earlier index then saw
/// "unsettled" again and panicked trying to re-sweep an already-resolved,
/// already-cleared candidate). `eliminated` similarly guards the one-time
/// elimination pass.
///
/// Indices stored here are purely *local* to this stack's own `0..SIZE`
/// range, matching `CandidateList`/`CandidateListOwned`'s existing
/// convention (see e.g. `StackConcat::poll_race_one`, which adds its own
/// `A::SIZE` offset on the way back up) - `arm_index` as received by
/// `poll_race_one` is likewise already local, since the caller subtracts its
/// own offset before calling in. Neither this stack nor its `arm_base`
/// parameter (required by the `MapArmStack`/`MapArmStackOwned` trait
/// signature, but otherwise unused here - untagged candidates never support
/// positional dispatch) ever needs to add or subtract that offset itself.
#[pin_project]
pub struct NoTagArmRaceState<RoundState, KeyClaim> {
    #[pin]
    pub round: RoundState,
    /// `None` until the round has fully settled; `Some(winning_index)`
    /// afterward, where `winning_index` is itself `None` when nobody hit
    /// this round. Sticky - never reset once `Some`.
    pub winner_index: Option<Option<usize>>,
    /// The winner's claim, present until the winning index is actually
    /// queried (taken exactly once at that point). `None` before the round
    /// settles, if there was no winner, or after the claim's been taken.
    pub winner_claim: Option<KeyClaim>,
    pub eliminated: bool,
}

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

// ---------------------------------------------------------------------------
// candidate_arms! and CandidateArmStack! / CandidateArmStackOwned! macros
// ---------------------------------------------------------------------------

/// Build a left-nested candidate list from a flat list of `index => arms => build` triples.
///
/// `index` must match this candidate's position in the `tag_candidates` array
/// passed to [`CandidateArmStack!`]/[`CandidateArmStackOwned!`].
///
/// ```rust,ignore
/// let candidates = candidate_arms! {
///     0 => <CircleHelper as MapFieldProvider<'de, _>>::make_arms() => |o| CircleHelper::from_outputs(o).map(Shape::Circle),
///     1 => MapArmBase => |()| Some(Shape::Ping),
/// };
/// ```
#[macro_export]
macro_rules! candidate_arms {
    ($($index:expr => $arms:expr => $build:expr),+ $(,)?) => {
        $crate::CandidateBase
            $(+ $crate::CandidateArm($crate::Candidate::new($index, $arms, $build)))+
    };
}

/// Constructs a [`CandidateArmStack`] with typed key/value tag closures (borrow family).
///
/// `CandidateArmStack!(candidates, tag_field, tag_candidates, KP, VP)`
#[macro_export]
macro_rules! CandidateArmStack {
    ($candidates:expr, $tag_field:expr, $tag_candidates:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbe as _;
        use $crate::MapValueProbe as _;
        let __tf = $tag_field;
        let __tc = $tag_candidates;
        $crate::CandidateArmStack::new(
            __tf,
            __tc,
            move |kp: $kp, _i: usize| kp.deserialize_key::<$crate::Match>(__tf),
            move |vp: $vp| vp.deserialize_value::<$crate::MatchVals<usize, _>>(__tc),
            $candidates,
        )
    }};
}

/// Constructs a [`CandidateArmStack`] with typed key/value tag closures (owned family).
///
/// `CandidateArmStackOwned!(candidates, tag_field, tag_candidates, KP, VP)`
#[macro_export]
macro_rules! CandidateArmStackOwned {
    ($candidates:expr, $tag_field:expr, $tag_candidates:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbeOwned as _;
        use $crate::MapValueProbeOwned as _;
        let __tf = $tag_field;
        let __tc = $tag_candidates;
        $crate::CandidateArmStack::new(
            __tf,
            __tc,
            move |kp: $kp, _i: usize| kp.deserialize_key::<$crate::Match>(__tf),
            move |vp: $vp| vp.deserialize_value::<$crate::MatchVals<usize, _>>(__tc),
            $candidates,
        )
    }};
}
