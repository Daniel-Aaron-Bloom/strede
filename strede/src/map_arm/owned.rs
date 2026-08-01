use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};

use super::{
    ArmState, Candidate, CandidateArmStack, CandidateBase, ConcatDispatchProj, ConcatDispatchState,
    ConcatRaceState, DetectDuplicates, False, MapArmBase, MapArmSlot, NoTagArmRaceState,
    NoTagCandidateArmStack, NoTagRoundState, SlotDispatchProj, SlotDispatchState, SlotRaceState,
    StackConcat, TagDispatchProj, TagDispatchState, TagInjectingStack, TagRaceState, VirtualArmSlot,
    WrapperDispatchProj, WrapperDispatchState, WrapperRaceState, poll_key_slot,
};
use crate::Probe;
use crate::owned::{MapKeyProbeOwned, VC as OVC, VP as OVP};

// ---------------------------------------------------------------------------
// MapArmStackOwned<KP> - owned-family arm stack
// ---------------------------------------------------------------------------

/// A left-nested tuple stack of [`MapArmSlot`]s: `((MapArmBase, Slot0), Slot1)`.
///
/// The map impl drives the iteration loop. Each round it calls the
/// [`race_keys`](Self::race_keys) free function which forks the key probe, creates per-arm
/// key futures via [`init_race`](Self::init_race), and polls them
/// flat via [`poll_race_one`](Self::poll_race_one). On a hit,
/// [`dispatch_value`](Self::dispatch_value) converts the key claim to a value probe and polls
/// the winning arm's value callback via [`poll_dispatch`](Self::poll_dispatch).
///
/// All poll methods are sync - recursion through `(Rest, Slot)` tuples is
/// ordinary call-stack recursion, not nested async state machines. This
/// avoids the compiler recursion depth limits that `async fn` nesting causes.
pub trait MapArmStackOwned<KP: MapKeyProbeOwned>: Sized {
    const SIZE: usize;

    /// Number of real (non-virtual) arms. Virtual arms (skip-unknown, dup-detect,
    /// tag-inject) do not contribute. Used to compute positional field indices
    /// for formats like postcard that identify fields by position rather than name.
    const FIELD_COUNT: usize;

    /// [`crate::True`] for arm stacks representing an unbounded/runtime-sized
    /// collection (e.g. HashMap's `CollectMap`) that requires the format to
    /// read an explicit wire-level length before iterating; [`crate::False`]
    /// for a fixed compile-time field set (structs, enums) whose end is
    /// signaled by the arm stack becoming satisfied. No default: every impl
    /// must pick one explicitly. See [`crate::MapArmStack::Dynamic`].
    type Dynamic;

    /// Left-nested tuple of `Option<(K, V)>` for each arm.
    type Outputs;

    /// Number of arms that still require a value (required fields not yet matched).
    /// Virtual arms (skip-unknown, dup-detect, tag-inject) are excluded.
    fn unsatisfied_count(&self) -> usize;

    /// Number of arms still willing to run, including both unsatisfied
    /// required-field arms and always-active virtual arms.
    fn open_count(&self) -> usize;

    // --- race_keys (init/poll) ---
    //
    // The init/poll API is the primary interface. `(Rest, Slot)` and
    // `StackConcat` implement these directly, producing flat poll loops
    // instead of nested async state machines.
    //
    // The `async fn race_keys` / `dispatch_value` default methods wrap
    // init/poll for convenience. Wrappers (SkipUnknownOwned, DetectDuplicates,
    // TagInjectingStack) override the async methods to add their virtual
    // arms while delegating inner arms via the init/poll path.

    /// Pinned state holding per-arm key futures for one round of racing.
    type RaceState;

    /// Fork `kp` for each unsatisfied arm, call each arm's key callback to
    /// create its future, and return the combined pinned state.
    ///
    /// - `arm_base`: global arm index of the first arm in this sub-stack, used
    ///   for arm routing (poll_race_one, dispatch_value) and virtual-arm identity.
    /// - `field_base`: positional field index of the first *real* arm in this
    ///   sub-stack, passed to key_fn closures for `deserialize_key_by_index`.
    ///   Virtual arms (dup-detect, skip-unknown, tag-inject) do not increment
    ///   this counter.
    fn init_race(&mut self, kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState;

    /// Poll a single arm's key future. **Sync** - recursion through
    /// `(Rest, Slot)` is ordinary function calls, not async nesting.
    #[allow(clippy::type_complexity)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>>;

    // --- dispatch_value (init/poll) ---

    /// Pinned state for dispatching the winning arm's value callback.
    /// Only one arm is dispatched per call (the race winner).
    type DispatchState;

    /// Create the dispatch state for the winning arm at `arm_index`.
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState;

    /// Poll the dispatch future.
    #[allow(clippy::type_complexity)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>>;

    // --- Provided async methods (convenience / wrapper override points) ---

    /// Race all unsatisfied arms' key callbacks against the given key probe.
    ///
    /// Default implementation wraps `init_race` + `poll_race_one` in a flat
    /// poll loop. Wrappers that add virtual arms override this.
    ///
    /// `BIASED` mirrors [`select_probe!`](crate::select_probe)'s `biased`
    /// mode: when arm `i` resolves `Hit`, arms `0..i` are re-polled once
    /// before committing to `i`, so a lower-index arm that's also ready
    /// wins. This matters because an arm only reaches `Hit` by fully
    /// draining its forked reader, and whichever arm happens to be the one
    /// that drives the underlying buffer's refill runs synchronously to
    /// completion in the same poll - it can otherwise "finish" before an
    /// earlier arm that was mid-`next()` gets a chance to observe the same
    /// newly-loaded data. Without the re-check, a later-declared arm (e.g.
    /// `DetectDuplicates`'s virtual dup-check arm, which races the same key
    /// content as the real field arm) can win a tie it should always lose.
    /// Callers for whom arm priority never matters (no two arms can validly
    /// hit on the same data) can pass `BIASED = false` to skip the re-poll.
    async fn race_keys<const BIASED: bool>(
        &mut self,
        kp: KP,
    ) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
        if Self::SIZE == 0 {
            return Ok(Probe::Miss);
        }
        let mut race_state = core::pin::pin!(self.init_race(kp, 0, 0));
        core::future::poll_fn(|cx| {
            let mut all_miss = true;
            for i in 0..Self::SIZE {
                match self.poll_race_one(race_state.as_mut(), i, cx) {
                    Poll::Ready(Ok(Probe::Hit(v))) => {
                        if !BIASED {
                            return Poll::Ready(Ok(Probe::Hit(v)));
                        }
                        let mut winner = v;
                        for j in 0..i {
                            match self.poll_race_one(race_state.as_mut(), j, cx) {
                                Poll::Ready(Ok(Probe::Hit(earlier))) => {
                                    winner = earlier;
                                    break;
                                }
                                Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                                Poll::Ready(Ok(Probe::Miss)) | Poll::Pending => {}
                            }
                        }
                        return Poll::Ready(Ok(Probe::Hit(winner)));
                    }
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Ready(Ok(Probe::Miss)) => {}
                    Poll::Pending => {
                        all_miss = false;
                    }
                }
            }
            if all_miss {
                return Poll::Ready(Ok(Probe::Miss));
            }
            Poll::Pending
        })
        .await
    }

    /// Dispatch the value probe to the arm at `arm_index`.
    ///
    /// Default implementation wraps `init_dispatch` + `poll_dispatch`.
    /// Wrappers that add virtual arms override this.
    async fn dispatch_value(
        &mut self,
        arm_index: usize,
        vp: OVP<KP>,
    ) -> Result<Probe<(OVC<KP>, ())>, KP::Error> {
        let dispatch_state = self.init_dispatch(arm_index, vp);
        let mut dispatch_state = core::pin::pin!(dispatch_state);
        core::future::poll_fn(|cx| self.poll_dispatch(dispatch_state.as_mut(), cx)).await
    }

    /// Extract all outputs.
    fn take_outputs(&mut self) -> Self::Outputs;
}

// ---------------------------------------------------------------------------
// MapArmStackOwned impls
// ---------------------------------------------------------------------------

// --- MapArmBase impl ---

impl<KP: MapKeyProbeOwned> MapArmStackOwned<KP> for MapArmBase {
    const SIZE: usize = 0;
    const FIELD_COUNT: usize = 0;
    type Dynamic = False;
    type Outputs = ();

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        0
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        0
    }

    type RaceState = ();

    #[inline(always)]
    fn init_race(&mut self, _kp: KP, _arm_base: usize, _field_base: usize) {}
    #[inline(always)]
    fn poll_race_one(
        &mut self,
        _state: Pin<&mut ()>,
        _arm_index: usize,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        unreachable!("poll_race_one called on MapArmBase (SIZE=0)")
    }

    type DispatchState = core::convert::Infallible;

    #[inline(always)]
    fn init_dispatch(&mut self, _arm_index: usize, _vp: OVP<KP>) -> Self::DispatchState {
        unreachable!("init_dispatch called on MapArmBase")
    }
    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        _state: Pin<&mut Self::DispatchState>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        unreachable!("poll_dispatch called on MapArmBase")
    }

    #[inline(always)]
    fn take_outputs(&mut self) {}
}

// --- Recursive (Rest, Slot) impl ---
//
// `init_race` forks the key probe and creates per-arm key futures, storing
// them in `SlotRaceState`. `poll_race_one` polls a single arm by index -
// sync recursion through `self.0.poll_race_one()` for rest arms.
//
// `init_dispatch` creates a `SlotDispatchState` enum selecting this arm's
// val future or delegating to rest. `poll_dispatch` polls the active variant.

impl<KP, Rest, K, V, KeyFn, KeyFut, ValFn, ValFut> MapArmStackOwned<KP>
    for (Rest, MapArmSlot<K, V, KeyFn, ValFn>)
where
    KP: MapKeyProbeOwned,
    Rest: MapArmStackOwned<KP>,
    KeyFn: FnMut(KP, usize) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(OVP<KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(OVC<KP>, (K, V))>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    const FIELD_COUNT: usize = Rest::FIELD_COUNT + 1;
    type Dynamic = Rest::Dynamic;
    type Outputs = (Rest::Outputs, Option<(K, V)>);

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count() + if self.1.state.is_done() { 0 } else { 1 }
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.0.open_count() + if self.1.state.is_done() { 0 } else { 1 }
    }

    type RaceState = SlotRaceState<Rest::RaceState, KeyFut>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let rest_kp = kp.fork();
        let this_fut = if self.1.state.is_done() {
            None
        } else {
            Some((self.1.key_fn)(kp, field_base + Self::FIELD_COUNT - 1))
        };
        SlotRaceState {
            rest: self.0.init_race(rest_kp, arm_base, field_base),
            this: this_fut,
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == Self::SIZE - 1 {
            match poll_key_slot(projected.this, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, k)))) => {
                    self.1.state = ArmState::Key(k);
                    Poll::Ready(Ok(Probe::Hit((Self::SIZE - 1, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            self.0.poll_race_one(projected.rest, arm_index, cx)
        }
    }

    type DispatchState = SlotDispatchState<Rest::DispatchState, ValFut>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            let k = match mem::replace(&mut self.1.state, ArmState::Empty) {
                ArmState::Key(k) => k,
                _ => unreachable!("init_dispatch called but arm not in Key state"),
            };
            SlotDispatchState::ThisArm((self.1.val_fn)(vp, k))
        } else {
            SlotDispatchState::Delegated(self.0.init_dispatch(arm_index, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        match state.project() {
            SlotDispatchProj::ThisArm(fut) => fut.poll(cx).map(|r| {
                r.map(|probe| match probe {
                    Probe::Hit((vc, (k, v))) => {
                        self.1.state = ArmState::Done(k, v);
                        Probe::Hit((vc, ()))
                    }
                    Probe::Miss => Probe::Miss,
                })
            }),
            SlotDispatchProj::Delegated(rest_state) => self.0.poll_dispatch(rest_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        let out = match mem::replace(&mut self.1.state, ArmState::Empty) {
            ArmState::Done(k, v) => Some((k, v)),
            _ => None,
        };
        (self.0.take_outputs(), out)
    }
}

// --- (Rest, VirtualArmSlot) impl ---

/// `(Rest, VirtualArmSlot)` impl - virtual arm at index `Rest::SIZE`.
/// The virtual arm is always active (never satisfied) and produces no output.
impl<KP, Rest, K, KeyFn, KeyFut, ValFn, ValFut> MapArmStackOwned<KP>
    for (Rest, VirtualArmSlot<K, KeyFn, ValFn>)
where
    KP: MapKeyProbeOwned,
    Rest: MapArmStackOwned<KP>,
    KeyFn: FnMut(KP, usize) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(OVP<KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(OVC<KP>, ())>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    const FIELD_COUNT: usize = Rest::FIELD_COUNT;
    type Dynamic = Rest::Dynamic;
    type Outputs = Rest::Outputs;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.0.open_count() + 1
    }

    type RaceState = SlotRaceState<Rest::RaceState, KeyFut>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let rest_kp = kp.fork();
        let this_fut = (self.1.key_fn)(kp, arm_base + Self::SIZE - 1);
        SlotRaceState {
            rest: self.0.init_race(rest_kp, arm_base, field_base),
            this: Some(this_fut),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == Self::SIZE - 1 {
            match poll_key_slot(projected.this, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, k)))) => {
                    self.1.pending_key = Some(k);
                    Poll::Ready(Ok(Probe::Hit((Self::SIZE - 1, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            self.0.poll_race_one(projected.rest, arm_index, cx)
        }
    }

    type DispatchState = SlotDispatchState<Rest::DispatchState, ValFut>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            let k = self
                .1
                .pending_key
                .take()
                .expect("init_dispatch on virtual arm without pending key");
            SlotDispatchState::ThisArm((self.1.val_fn)(vp, k))
        } else {
            SlotDispatchState::Delegated(self.0.init_dispatch(arm_index, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        match state.project() {
            SlotDispatchProj::ThisArm(fut) => fut.poll(cx),
            SlotDispatchProj::Delegated(rest_state) => self.0.poll_dispatch(rest_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.0.take_outputs()
    }
}

// ---------------------------------------------------------------------------
// Wrapper impls for MapArmStackOwned (owned family)
// ---------------------------------------------------------------------------

// --- DetectDuplicates impl ---

impl<KP, S, W, KeyFn, KeyFut, SkipFn, SkipFut> MapArmStackOwned<KP>
    for DetectDuplicates<S, W, KeyFn, SkipFn>
where
    KP: MapKeyProbeOwned,
    S: MapArmStackOwned<KP>,
    W: AsRef<[(&'static str, usize)]>,
    KeyFn: FnMut(KP, usize) -> KeyFut,
    KeyFut: Future<
        Output = Result<Probe<(KP::KeyClaim, crate::impls::MatchVals<usize, W>)>, KP::Error>,
    >,
    SkipFn: FnMut(OVP<KP>) -> SkipFut,
    SkipFut: Future<Output = Result<OVC<KP>, KP::Error>>,
{
    const SIZE: usize = S::SIZE + 1;
    const FIELD_COUNT: usize = S::FIELD_COUNT;
    type Dynamic = S::Dynamic;
    type Outputs = S::Outputs;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.inner.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.inner.open_count() + 1
    }

    type RaceState = WrapperRaceState<S::RaceState, KeyFut>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let dup_kp = kp.fork();
        let dup_fut = (self.key_fn)(dup_kp, arm_base + Self::SIZE - 1);
        WrapperRaceState {
            inner: self.inner.init_race(kp, arm_base, field_base),
            virtual_arm: Some(dup_fut),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == Self::SIZE - 1 {
            match poll_key_slot(projected.virtual_arm, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, matched)))) => {
                    self.dup = self.wire_names.as_ref()[matched.0].0;
                    Poll::Ready(Ok(Probe::Hit((Self::SIZE - 1, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            self.inner.poll_race_one(projected.inner, arm_index, cx)
        }
    }

    type DispatchState = WrapperDispatchState<S::DispatchState, SkipFut>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            WrapperDispatchState::Virtual((self.skip_fn)(vp))
        } else {
            WrapperDispatchState::Inner(self.inner.init_dispatch(arm_index, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        match state.project() {
            WrapperDispatchProj::Virtual(fut) => fut.poll(cx).map(|r| match r {
                Ok(_vc) => Err(<KP::Error as crate::DeserializeError>::duplicate_field(
                    self.dup,
                )),
                Err(e) => Err(e),
            }),
            WrapperDispatchProj::Inner(inner_state) => self.inner.poll_dispatch(inner_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.inner.take_outputs()
    }
}

// --- TagInjectingStack impl ---

impl<'v, KP, S, W, TagKeyFn, TagKeyFut, TagValFn, TagValFut> MapArmStackOwned<KP>
    for TagInjectingStack<'v, S, W, TagKeyFn, TagValFn>
where
    KP: MapKeyProbeOwned,
    S: MapArmStackOwned<KP>,
    TagKeyFn: FnMut(KP, usize) -> TagKeyFut,
    TagKeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::impls::Match)>, KP::Error>>,
    TagValFn: FnMut(OVP<KP>) -> TagValFut,
    TagValFut:
        Future<Output = Result<Probe<(OVC<KP>, crate::impls::MatchVals<usize, W>)>, KP::Error>>,
{
    const SIZE: usize = S::SIZE + 1;
    const FIELD_COUNT: usize = S::FIELD_COUNT;
    type Dynamic = S::Dynamic;
    type Outputs = S::Outputs;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.inner.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.inner.open_count() + 1
    }

    type RaceState = TagRaceState<TagKeyFut, S::RaceState>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let inner_kp = kp.fork();
        // Tag arm is always at global index arm_base (index 0 within this wrapper).
        let tag_fut = (self.tag_key_fn)(kp, arm_base);
        TagRaceState {
            tag_fut: Some(tag_fut),
            inner: self.inner.init_race(inner_kp, arm_base + 1, field_base),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == 0 {
            match poll_key_slot(projected.tag_fut, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, _match)))) => Poll::Ready(Ok(Probe::Hit((0, kc)))),
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            match self.inner.poll_race_one(projected.inner, arm_index - 1, cx) {
                Poll::Ready(Ok(Probe::Hit((idx, kc)))) => {
                    Poll::Ready(Ok(Probe::Hit((idx + 1, kc))))
                }
                other => other,
            }
        }
    }

    type DispatchState = TagDispatchState<TagValFut, S::DispatchState>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        if arm_index == 0 {
            TagDispatchState::Tag((self.tag_val_fn)(vp))
        } else {
            TagDispatchState::Inner(self.inner.init_dispatch(arm_index - 1, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        match state.project() {
            TagDispatchProj::Tag(fut) => fut.poll(cx).map(|r| {
                r.map(|probe| match probe {
                    Probe::Hit((vc, crate::impls::MatchVals(idx, _))) => {
                        self.tag_value.set(Some(idx));
                        Probe::Hit((vc, ()))
                    }
                    Probe::Miss => Probe::Miss,
                })
            }),
            TagDispatchProj::Inner(inner_state) => self.inner.poll_dispatch(inner_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.inner.take_outputs()
    }
}

// --- StackConcat impl ---

impl<KP, A, B> MapArmStackOwned<KP> for StackConcat<A, B>
where
    KP: MapKeyProbeOwned,
    A: MapArmStackOwned<KP>,
    // Enforced at the type level, not via a runtime/const-eval assertion:
    // a StackConcat mixing a DYNAMIC (unbounded collection) side with a
    // non-DYNAMIC side simply fails to type-check.
    B: MapArmStackOwned<KP, Dynamic = A::Dynamic>,
{
    const SIZE: usize = A::SIZE + B::SIZE;
    const FIELD_COUNT: usize = A::FIELD_COUNT + B::FIELD_COUNT;
    type Dynamic = A::Dynamic;
    type Outputs = (A::Outputs, B::Outputs);

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count() + self.1.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.0.open_count() + self.1.open_count()
    }

    type RaceState = ConcatRaceState<A::RaceState, B::RaceState>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let b_kp = kp.fork();
        ConcatRaceState {
            a: self.0.init_race(kp, arm_base, field_base),
            b: self
                .1
                .init_race(b_kp, arm_base + A::SIZE, field_base + A::FIELD_COUNT),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index < A::SIZE {
            self.0.poll_race_one(projected.a, arm_index, cx)
        } else {
            match self.1.poll_race_one(projected.b, arm_index - A::SIZE, cx) {
                Poll::Ready(Ok(Probe::Hit((idx, kc)))) => {
                    Poll::Ready(Ok(Probe::Hit((A::SIZE + idx, kc))))
                }
                other => other,
            }
        }
    }

    type DispatchState = ConcatDispatchState<A::DispatchState, B::DispatchState>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        if arm_index < A::SIZE {
            ConcatDispatchState::InA(self.0.init_dispatch(arm_index, vp))
        } else {
            ConcatDispatchState::InB(self.1.init_dispatch(arm_index - A::SIZE, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        match state.project() {
            ConcatDispatchProj::InA(a_state) => self.0.poll_dispatch(a_state, cx),
            ConcatDispatchProj::InB(b_state) => self.1.poll_dispatch(b_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        (self.0.take_outputs(), self.1.take_outputs())
    }
}

// ---------------------------------------------------------------------------
// CandidateListOwned<KP, EnumOut> - internal recursion for CandidateArmStack
// ---------------------------------------------------------------------------

/// Owned-family counterpart to [`crate::map_arm::borrow::CandidateList`]. See
/// there for the `tag_matched` gating rationale.
pub trait CandidateListOwned<KP: MapKeyProbeOwned, EnumOut>: Sized {
    const SIZE: usize;

    fn unsatisfied_count(&self, target_index: usize) -> usize;
    fn open_count(&self, tag_matched: Option<usize>) -> usize;

    type RaceState;
    fn init_race(&mut self, kp: KP, arm_base: usize, tag_matched: Option<usize>)
    -> Self::RaceState;
    #[allow(clippy::type_complexity)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
        tag_matched: Option<usize>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>>;

    type DispatchState;
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState;
    #[allow(clippy::type_complexity)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>>;

    fn build_winner(&mut self, target_index: usize) -> Option<EnumOut>;
}

impl<KP: MapKeyProbeOwned, EnumOut> CandidateListOwned<KP, EnumOut> for CandidateBase {
    const SIZE: usize = 0;

    #[inline(always)]
    fn unsatisfied_count(&self, _target_index: usize) -> usize {
        unreachable!("target candidate index not found (CandidateBase)")
    }
    #[inline(always)]
    fn open_count(&self, _tag_matched: Option<usize>) -> usize {
        0
    }

    type RaceState = ();
    #[inline(always)]
    fn init_race(&mut self, _kp: KP, _arm_base: usize, _tag_matched: Option<usize>) {}
    #[inline(always)]
    fn poll_race_one(
        &mut self,
        _state: Pin<&mut ()>,
        _arm_index: usize,
        _cx: &mut Context<'_>,
        _tag_matched: Option<usize>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        unreachable!("poll_race_one called on CandidateBase (SIZE=0)")
    }

    type DispatchState = core::convert::Infallible;
    #[inline(always)]
    fn init_dispatch(&mut self, _arm_index: usize, _vp: OVP<KP>) -> Self::DispatchState {
        unreachable!("init_dispatch called on CandidateBase")
    }
    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        _state: Pin<&mut Self::DispatchState>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        unreachable!("poll_dispatch called on CandidateBase")
    }

    #[inline(always)]
    fn build_winner(&mut self, _target_index: usize) -> Option<EnumOut> {
        unreachable!("build_winner: target index not found among candidates")
    }
}

impl<KP, Rest, C, BuildFn, EnumOut> CandidateListOwned<KP, EnumOut>
    for (Rest, Candidate<C, BuildFn>)
where
    KP: MapKeyProbeOwned,
    Rest: CandidateListOwned<KP, EnumOut>,
    C: MapArmStackOwned<KP>,
    BuildFn: FnMut(C::Outputs) -> Option<EnumOut>,
{
    const SIZE: usize = Rest::SIZE + C::SIZE;

    #[inline(always)]
    fn unsatisfied_count(&self, target_index: usize) -> usize {
        if self.1.index == target_index {
            self.1.arms.unsatisfied_count()
        } else {
            self.0.unsatisfied_count(target_index)
        }
    }
    #[inline(always)]
    fn open_count(&self, tag_matched: Option<usize>) -> usize {
        let this_active = tag_matched.is_none_or(|idx| idx == self.1.index);
        let this = if this_active {
            self.1.arms.open_count()
        } else {
            0
        };
        self.0.open_count(tag_matched) + this
    }

    type RaceState = SlotRaceState<Rest::RaceState, C::RaceState>;

    #[inline(always)]
    fn init_race(
        &mut self,
        mut kp: KP,
        arm_base: usize,
        tag_matched: Option<usize>,
    ) -> Self::RaceState {
        let rest_kp = kp.fork();
        let this_active = tag_matched.is_none_or(|idx| idx == self.1.index);
        let this = if this_active {
            Some(self.1.arms.init_race(kp, arm_base + Rest::SIZE, 0))
        } else {
            None
        };
        SlotRaceState {
            rest: self.0.init_race(rest_kp, arm_base, tag_matched),
            this,
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
        tag_matched: Option<usize>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let mut projected = state.project();
        if arm_index < Rest::SIZE {
            return self
                .0
                .poll_race_one(projected.rest, arm_index, cx, tag_matched);
        }
        let local_index = arm_index - Rest::SIZE;
        match projected.this.as_mut().as_pin_mut() {
            None => Poll::Ready(Ok(Probe::Miss)),
            Some(this_state) => match self.1.arms.poll_race_one(this_state, local_index, cx) {
                Poll::Ready(Ok(Probe::Hit((idx, kc)))) => {
                    Poll::Ready(Ok(Probe::Hit((Rest::SIZE + idx, kc))))
                }
                other => other,
            },
        }
    }

    type DispatchState = ConcatDispatchState<Rest::DispatchState, C::DispatchState>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        if arm_index < Rest::SIZE {
            ConcatDispatchState::InA(self.0.init_dispatch(arm_index, vp))
        } else {
            ConcatDispatchState::InB(self.1.arms.init_dispatch(arm_index - Rest::SIZE, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        match state.project() {
            ConcatDispatchProj::InA(s) => self.0.poll_dispatch(s, cx),
            ConcatDispatchProj::InB(s) => self.1.arms.poll_dispatch(s, cx),
        }
    }

    #[inline(always)]
    fn build_winner(&mut self, target_index: usize) -> Option<EnumOut> {
        if self.1.index == target_index {
            let outputs = self.1.arms.take_outputs();
            (self.1.build)(outputs)
        } else {
            self.0.build_winner(target_index)
        }
    }
}

// ---------------------------------------------------------------------------
// CandidateArmStack impl (owned family)
// ---------------------------------------------------------------------------

impl<KP, Candidates, EnumOut, W, TagKeyFn, TagKeyFut, TagValFn, TagValFut> MapArmStackOwned<KP>
    for CandidateArmStack<Candidates, EnumOut, W, TagKeyFn, TagValFn>
where
    KP: MapKeyProbeOwned,
    Candidates: CandidateListOwned<KP, EnumOut>,
    W: Copy,
    TagKeyFn: FnMut(KP, usize) -> TagKeyFut,
    TagKeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::impls::Match)>, KP::Error>>,
    TagValFn: FnMut(OVP<KP>) -> TagValFut,
    TagValFut:
        Future<Output = Result<Probe<(OVC<KP>, crate::impls::MatchVals<usize, W>)>, KP::Error>>,
{
    const SIZE: usize = Candidates::SIZE + 1;
    const FIELD_COUNT: usize = 1;
    type Dynamic = False;
    type Outputs = Option<EnumOut>;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        match self.tag_matched {
            None => 1,
            Some(idx) => self.candidates.unsatisfied_count(idx),
        }
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        match self.tag_matched {
            None => 1 + self.candidates.open_count(None),
            Some(idx) => self.candidates.open_count(Some(idx)),
        }
    }

    type RaceState = TagRaceState<TagKeyFut, Candidates::RaceState>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, _field_base: usize) -> Self::RaceState {
        let inner_kp = kp.fork();
        let tag_fut = if self.tag_matched.is_some() {
            None
        } else {
            Some((self.tag_key_fn)(kp, arm_base))
        };
        TagRaceState {
            tag_fut,
            inner: self
                .candidates
                .init_race(inner_kp, arm_base + 1, self.tag_matched),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == 0 {
            match poll_key_slot(projected.tag_fut, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, _match)))) => Poll::Ready(Ok(Probe::Hit((0, kc)))),
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            match self.candidates.poll_race_one(
                projected.inner,
                arm_index - 1,
                cx,
                self.tag_matched,
            ) {
                Poll::Ready(Ok(Probe::Hit((idx, kc)))) => {
                    Poll::Ready(Ok(Probe::Hit((idx + 1, kc))))
                }
                other => other,
            }
        }
    }

    type DispatchState = TagDispatchState<TagValFut, Candidates::DispatchState>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        if arm_index == 0 {
            TagDispatchState::Tag((self.tag_val_fn)(vp))
        } else {
            TagDispatchState::Inner(self.candidates.init_dispatch(arm_index - 1, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        match state.project() {
            TagDispatchProj::Tag(fut) => fut.poll(cx).map(|r| {
                r.map(|probe| match probe {
                    Probe::Hit((vc, crate::impls::MatchVals(idx, _))) => {
                        self.tag_matched = Some(idx);
                        Probe::Hit((vc, ()))
                    }
                    Probe::Miss => Probe::Miss,
                })
            }),
            TagDispatchProj::Inner(inner_state) => self.candidates.poll_dispatch(inner_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        match self.tag_matched {
            Some(idx) => self.candidates.build_winner(idx),
            None => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NoTagCandidateListOwned<KP, EnumOut> - untagged enum flatten
// ---------------------------------------------------------------------------

/// Owned-family counterpart to [`crate::map_arm::borrow::NoTagCandidateList`].
/// See there for the full algorithm write-up.
pub trait NoTagCandidateListOwned<KP: MapKeyProbeOwned, EnumOut>: Sized {
    type RoundState;

    fn init_round(&mut self, kp: KP) -> Self::RoundState;

    /// Indices recorded internally (and later returned by `take_winner`) are
    /// purely *local* to this whole `NoTagCandidateListOwned`'s own
    /// `0..SIZE` range - see the borrow-family counterpart's doc comment.
    fn poll_sweep(
        &mut self,
        state: Pin<&mut Self::RoundState>,
        cx: &mut Context<'_>,
    ) -> Result<bool, KP::Error>;

    fn take_winner(&mut self, state: Pin<&mut Self::RoundState>) -> Option<(usize, KP::KeyClaim)>;

    fn eliminate(&mut self, any_hit: bool);

    fn first_satisfied_live(&self) -> Option<usize>;
}

impl<KP: MapKeyProbeOwned, EnumOut> NoTagCandidateListOwned<KP, EnumOut> for CandidateBase {
    type RoundState = ();

    #[inline(always)]
    fn init_round(&mut self, _kp: KP) {}

    #[inline(always)]
    fn poll_sweep(&mut self, _state: Pin<&mut ()>, _cx: &mut Context<'_>) -> Result<bool, KP::Error> {
        Ok(true)
    }

    #[inline(always)]
    fn take_winner(&mut self, _state: Pin<&mut ()>) -> Option<(usize, KP::KeyClaim)> {
        None
    }

    #[inline(always)]
    fn eliminate(&mut self, _any_hit: bool) {}

    #[inline(always)]
    fn first_satisfied_live(&self) -> Option<usize> {
        None
    }
}

impl<KP, Rest, C, BuildFn, EnumOut> NoTagCandidateListOwned<KP, EnumOut>
    for (Rest, Candidate<C, BuildFn>)
where
    KP: MapKeyProbeOwned,
    Rest: NoTagCandidateListOwned<KP, EnumOut> + CandidateListOwned<KP, EnumOut>,
    C: MapArmStackOwned<KP>,
    BuildFn: FnMut(C::Outputs) -> Option<EnumOut>,
{
    type RoundState = NoTagRoundState<Rest::RoundState, C::RaceState, KP::KeyClaim>;

    #[inline(always)]
    fn init_round(&mut self, mut kp: KP) -> Self::RoundState {
        let rest_kp = kp.fork();
        self.1.round_hit = false;
        let (this, resolved) = if self.1.live {
            (Some(self.1.arms.init_race(kp, 0, 0)), None)
        } else {
            (None, Some(None))
        };
        NoTagRoundState {
            rest: self.0.init_round(rest_kp),
            this,
            resolved,
        }
    }

    #[inline(always)]
    fn poll_sweep(
        &mut self,
        state: Pin<&mut Self::RoundState>,
        cx: &mut Context<'_>,
    ) -> Result<bool, KP::Error> {
        let mut projected = state.project();
        let rest_settled = self.0.poll_sweep(projected.rest, cx)?;

        let this_settled = if projected.resolved.is_none() {
            let this_base = <Rest as CandidateListOwned<KP, EnumOut>>::SIZE;
            let mut this_pin = projected
                .this
                .as_mut()
                .as_pin_mut()
                .expect("live, unresolved candidate must have a RaceState");
            let mut hit: Option<(usize, KP::KeyClaim)> = None;
            let mut any_pending = false;
            for i in 0..C::SIZE {
                match self.1.arms.poll_race_one(this_pin.as_mut(), i, cx) {
                    Poll::Ready(Ok(Probe::Hit((_local, kc)))) => {
                        hit = Some((this_base + i, kc));
                        break;
                    }
                    Poll::Ready(Ok(Probe::Miss)) => {}
                    Poll::Ready(Err(e)) => return Err(e),
                    Poll::Pending => any_pending = true,
                }
            }
            match hit {
                Some((idx, kc)) => {
                    self.1.round_hit = true;
                    *projected.resolved = Some(Some((idx, kc)));
                    projected.this.set(None);
                    true
                }
                None if !any_pending => {
                    *projected.resolved = Some(None);
                    projected.this.set(None);
                    true
                }
                None => false,
            }
        } else {
            true
        };

        Ok(rest_settled && this_settled)
    }

    #[inline(always)]
    fn take_winner(&mut self, state: Pin<&mut Self::RoundState>) -> Option<(usize, KP::KeyClaim)> {
        let projected = state.project();
        let rest_winner = self.0.take_winner(projected.rest);
        if rest_winner.is_some() {
            return rest_winner;
        }
        match projected.resolved.take() {
            Some(Some((idx, kc))) => Some((idx, kc)),
            Some(None) => None,
            None => unreachable!("take_winner called before the round settled"),
        }
    }

    #[inline(always)]
    fn eliminate(&mut self, any_hit: bool) {
        self.0.eliminate(any_hit);
        if any_hit && self.1.live && !self.1.round_hit {
            self.1.live = false;
        }
    }

    #[inline(always)]
    fn first_satisfied_live(&self) -> Option<usize> {
        if let Some(idx) = self.0.first_satisfied_live() {
            return Some(idx);
        }
        if self.1.live && self.1.arms.unsatisfied_count() == 0 {
            Some(self.1.index)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// NoTagCandidateArmStack impl (owned family)
// ---------------------------------------------------------------------------

impl<KP, Candidates, EnumOut> MapArmStackOwned<KP> for NoTagCandidateArmStack<Candidates, EnumOut>
where
    KP: MapKeyProbeOwned,
    Candidates: CandidateListOwned<KP, EnumOut> + NoTagCandidateListOwned<KP, EnumOut>,
{
    const SIZE: usize = <Candidates as CandidateListOwned<KP, EnumOut>>::SIZE;
    const FIELD_COUNT: usize = 0;
    type Dynamic = False;
    type Outputs = Option<EnumOut>;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        if self.candidates.first_satisfied_live().is_some() {
            0
        } else {
            1
        }
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        1
    }

    // See `NoTagArmRaceState`'s doc comment: this soft-elimination race must
    // be driven through `init_race`/`poll_race_one` (the interface every
    // composition primitive in this module actually calls), not a
    // `race_keys` override - `StackConcat` (how `#[strede(flatten)]` splices
    // this in) reaches nested arms this way and never calls a nested
    // stack's `race_keys()`.
    type RaceState = NoTagArmRaceState<Candidates::RoundState, KP::KeyClaim>;

    #[inline(always)]
    fn init_race(&mut self, kp: KP, _arm_base: usize, _field_base: usize) -> Self::RaceState {
        NoTagArmRaceState {
            round: self.candidates.init_round(kp),
            winner_index: None,
            winner_claim: None,
            eliminated: false,
        }
    }

    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let mut projected = state.project();

        if projected.winner_index.is_none() {
            match self.candidates.poll_sweep(projected.round.as_mut(), cx) {
                Ok(true) => match self.candidates.take_winner(projected.round.as_mut()) {
                    Some((idx, kc)) => {
                        *projected.winner_index = Some(Some(idx));
                        *projected.winner_claim = Some(kc);
                    }
                    None => *projected.winner_index = Some(None),
                },
                Ok(false) => return Poll::Pending,
                Err(e) => return Poll::Ready(Err(e)),
            }
        }

        if !*projected.eliminated {
            let any_hit = matches!(projected.winner_index, Some(Some(_)));
            self.candidates.eliminate(any_hit);
            *projected.eliminated = true;
        }

        let is_winner = matches!(projected.winner_index, Some(Some(idx)) if *idx == arm_index);
        if is_winner {
            let kc = projected
                .winner_claim
                .take()
                .expect("winning arm_index queried more than once in a single round");
            Poll::Ready(Ok(Probe::Hit((arm_index, kc))))
        } else {
            Poll::Ready(Ok(Probe::Miss))
        }
    }

    type DispatchState = <Candidates as CandidateListOwned<KP, EnumOut>>::DispatchState;
    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: OVP<KP>) -> Self::DispatchState {
        self.candidates.init_dispatch(arm_index, vp)
    }
    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(OVC<KP>, ())>, KP::Error>> {
        self.candidates.poll_dispatch(state, cx)
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        match self.candidates.first_satisfied_live() {
            Some(idx) => self.candidates.build_winner(idx),
            None => None,
        }
    }
}
