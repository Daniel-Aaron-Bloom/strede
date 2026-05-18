use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};

use super::{
    ArmState, ConcatDispatchProj, ConcatDispatchState, ConcatRaceState, DetectDuplicatesOwned,
    MapArmBase, MapArmSlot, SlotDispatchProj, SlotDispatchState, SlotRaceState, StackConcat,
    TagDispatchProj, TagDispatchState, TagInjectingStackOwned, TagRaceState, VirtualArmSlot,
    WrapperDispatchProj, WrapperDispatchState, WrapperRaceState, poll_key_slot,
};
use crate::Probe;
use crate::owned::{MapAccessOwned, MapKeyProbeOwned, VC as OVC, VP as OVP};

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
    // init/poll for convenience. Wrappers (SkipUnknownOwned, DetectDuplicatesOwned,
    // TagInjectingStackOwned) override the async methods to add their virtual
    // arms while delegating inner arms via the init/poll path.

    /// Pinned state holding per-arm key futures for one round of racing.
    type RaceState;

    /// Fork `kp` for each unsatisfied arm, call each arm's key callback to
    /// create its future, and return the combined pinned state.
    fn init_race(&mut self, kp: KP) -> Self::RaceState;

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
    async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
        if Self::SIZE == 0 {
            return Ok(Probe::Miss);
        }
        let mut race_state = core::pin::pin!(self.init_race(kp));
        core::future::poll_fn(|cx| {
            let mut all_miss = true;
            for i in 0..Self::SIZE {
                match self.poll_race_one(race_state.as_mut(), i, cx) {
                    Poll::Ready(Ok(Probe::Hit(v))) => return Poll::Ready(Ok(Probe::Hit(v))),
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
    fn init_race(&mut self, _kp: KP) {}
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
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(OVP<KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(OVC<KP>, (K, V))>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
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
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let rest_kp = kp.fork();
        let this_fut = if self.1.state.is_done() {
            None
        } else {
            Some((self.1.key_fn)(kp))
        };
        SlotRaceState {
            rest: self.0.init_race(rest_kp),
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
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(OVP<KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(OVC<KP>, ())>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
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
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let rest_kp = kp.fork();
        let this_fut = (self.1.key_fn)(kp);
        SlotRaceState {
            rest: self.0.init_race(rest_kp),
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
// FlattenContOwned - continuation trait for multi-flatten chains (owned family)
// ---------------------------------------------------------------------------

/// Owned-family counterpart to [`crate::FlattenCont`]. See its documentation.
pub trait FlattenContOwned<M: MapAccessOwned>: Sized {
    async fn finish<Arms: MapArmStackOwned<M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs)>, M::Error>;
}

impl<M: MapAccessOwned> FlattenContOwned<M> for crate::FlattenTerminal {
    async fn finish<Arms: MapArmStackOwned<M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs)>, M::Error> {
        let (claim, out) = crate::hit!(map.iterate(arms).await);
        Ok(Probe::Hit((claim, out)))
    }
}

#[cfg(feature = "alloc")]
impl<M: MapAccessOwned> FlattenContOwned<M> for crate::FlattenTerminalBoxed {
    async fn finish<Arms: MapArmStackOwned<M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs)>, M::Error> {
        #[allow(clippy::type_complexity)]
        let r: Result<Probe<(M::MapClaim, Arms::Outputs)>, M::Error> =
            alloc::boxed::Box::pin(map.iterate(arms)).await;
        let (claim, out) = crate::hit!(r);
        Ok(Probe::Hit((claim, out)))
    }
}

// ---------------------------------------------------------------------------
// Wrapper impls for MapArmStackOwned (owned family)
// ---------------------------------------------------------------------------

// --- DetectDuplicatesOwned impl ---

impl<KP, S, const M: usize, KeyFn, KeyFut, SkipFn, SkipFut> MapArmStackOwned<KP>
    for DetectDuplicatesOwned<S, M, KeyFn, SkipFn>
where
    KP: MapKeyProbeOwned,
    S: MapArmStackOwned<KP>,
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<
        Output = Result<Probe<(KP::KeyClaim, crate::impls::MatchVals<usize, { M }>)>, KP::Error>,
    >,
    SkipFn: FnMut(OVP<KP>) -> SkipFut,
    SkipFut: Future<Output = Result<OVC<KP>, KP::Error>>,
{
    const SIZE: usize = S::SIZE + 1;
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
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let dup_kp = kp.fork();
        let dup_fut = (self.key_fn)(dup_kp);
        WrapperRaceState {
            inner: self.inner.init_race(kp),
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
                    self.dup = self.wire_names[matched.0].0;
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

// --- TagInjectingStackOwned impl ---

impl<'v, KP, S, const N: usize, TagKeyFn, TagKeyFut, TagValFn, TagValFut> MapArmStackOwned<KP>
    for TagInjectingStackOwned<'v, S, N, TagKeyFn, TagValFn>
where
    KP: MapKeyProbeOwned,
    S: MapArmStackOwned<KP>,
    TagKeyFn: FnMut(KP) -> TagKeyFut,
    TagKeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::impls::Match)>, KP::Error>>,
    TagValFn: FnMut(OVP<KP>) -> TagValFut,
    TagValFut:
        Future<Output = Result<Probe<(OVC<KP>, crate::impls::MatchVals<usize, { N }>)>, KP::Error>>,
{
    const SIZE: usize = S::SIZE + 1;
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
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let inner_kp = kp.fork();
        let tag_fut = (self.tag_key_fn)(kp);
        TagRaceState {
            tag_fut: Some(tag_fut),
            inner: self.inner.init_race(inner_kp),
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
    B: MapArmStackOwned<KP>,
{
    const SIZE: usize = A::SIZE + B::SIZE;
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
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let b_kp = kp.fork();
        ConcatRaceState {
            a: self.0.init_race(kp),
            b: self.1.init_race(b_kp),
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
