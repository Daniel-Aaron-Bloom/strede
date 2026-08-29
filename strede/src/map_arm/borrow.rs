use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};

use super::{
    ArmState, Candidate, CandidateArmStack, CandidateBase, ConcatDispatchProj, ConcatDispatchState,
    ConcatRaceState, DetectDuplicates, False, MapArmBase, MapArmSlot, NoTagArmRaceState,
    NoTagCandidateArmStack, NoTagRoundState, SlotDispatchProj, SlotDispatchState, SlotRaceState,
    StackConcat, TagDispatchProj, TagDispatchState, TagInjectingStack, TagRaceState,
    VirtualArmSlot, WrapperDispatchProj, WrapperDispatchState, WrapperRaceState, poll_key_slot,
};
use crate::Probe;
use crate::borrow::{MapKeyProbe, VC as BVC, VP as BVP};

// ---------------------------------------------------------------------------
// MapArmStack<'de, KP> - borrow-family arm stack
// ---------------------------------------------------------------------------

/// Borrow-family counterpart to [`crate::MapArmStackOwned`].
///
/// A left-nested tuple stack of [`MapArmSlot`]s parameterized by `'de`.
/// The map impl drives the iteration loop using this trait's `init_race` /
/// `poll_race_one` / `init_dispatch` / `poll_dispatch` methods.
pub trait MapArmStack<'de, KP: MapKeyProbe<'de>>: Sized {
    const SIZE: usize;

    /// Number of real (non-virtual) arms. See [`crate::MapArmStackOwned::FIELD_COUNT`].
    const FIELD_COUNT: usize;

    /// [`crate::True`] for arm stacks representing an unbounded/runtime-sized
    /// collection (e.g. HashMap's `CollectMap`) that requires the format to
    /// read an explicit wire-level length before iterating; [`crate::False`]
    /// for a fixed compile-time field set (structs, enums) whose end is
    /// signaled by the arm stack becoming satisfied. No default: every impl
    /// must pick one explicitly. See [`crate::MapArmStackOwned::Dynamic`].
    type Dynamic;

    /// Left-nested tuple of `Option<(K, V)>` for each arm.
    type Outputs;

    /// Number of arms that still require a value (required fields not yet matched).
    /// Virtual arms are excluded.
    fn unsatisfied_count(&self) -> usize;

    /// Number of arms still willing to run, including virtual arms.
    fn open_count(&self) -> usize;

    type RaceState;

    fn init_race(&mut self, kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState;
    #[allow(clippy::type_complexity)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>>;

    type DispatchState;

    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState;
    #[allow(clippy::type_complexity)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>>;

    /// Race all unsatisfied arms' key callbacks against the given key probe.
    ///
    /// See [`MapArmStackOwned::race_keys`](crate::map_arm::owned::MapArmStackOwned::race_keys)
    /// for why `BIASED` exists and what it does.
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
    async fn dispatch_value(
        &mut self,
        arm_index: usize,
        vp: BVP<'de, KP>,
    ) -> Result<Probe<(BVC<'de, KP>, ())>, KP::Error> {
        let dispatch_state = self.init_dispatch(arm_index, vp);
        let mut dispatch_state = core::pin::pin!(dispatch_state);
        core::future::poll_fn(|cx| self.poll_dispatch(dispatch_state.as_mut(), cx)).await
    }

    fn take_outputs(&mut self) -> Self::Outputs;
}

// ---------------------------------------------------------------------------
// MapArmStack impls
// ---------------------------------------------------------------------------

// --- MapArmBase impl ---

impl<'de, KP: MapKeyProbe<'de>> MapArmStack<'de, KP> for MapArmBase {
    const SIZE: usize = 0;
    const FIELD_COUNT: usize = 0;
    type Dynamic = False;
    type Outputs = ();

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        0
    }
    #[inline]
    fn open_count(&self) -> usize {
        0
    }

    type RaceState = ();

    #[inline]
    fn init_race(&mut self, _kp: KP, _arm_base: usize, _field_base: usize) {}
    #[inline]
    fn poll_race_one(
        &mut self,
        _state: Pin<&mut ()>,
        _arm_index: usize,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        unreachable!("poll_race_one called on MapArmBase (SIZE=0)")
    }

    type DispatchState = core::convert::Infallible;

    #[inline]
    fn init_dispatch(&mut self, _arm_index: usize, _vp: BVP<'de, KP>) -> Self::DispatchState {
        unreachable!("init_dispatch called on MapArmBase")
    }
    #[inline]
    fn poll_dispatch(
        &mut self,
        _state: Pin<&mut Self::DispatchState>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
        unreachable!("poll_dispatch called on MapArmBase")
    }

    #[inline]
    fn take_outputs(&mut self) {}
}

// --- Recursive (Rest, Slot) impl ---

impl<'de, KP, Rest, K, V, KeyFn, KeyFut, ValFn, ValFut> MapArmStack<'de, KP>
    for (Rest, MapArmSlot<K, V, KeyFn, ValFn>)
where
    KP: MapKeyProbe<'de>,
    Rest: MapArmStack<'de, KP>,
    KeyFn: FnMut(KP, usize) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(BVP<'de, KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(BVC<'de, KP>, (K, V))>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    const FIELD_COUNT: usize = Rest::FIELD_COUNT + 1;
    type Dynamic = Rest::Dynamic;
    type Outputs = (Rest::Outputs, Option<(K, V)>);

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count() + if self.1.state.is_done() { 0 } else { 1 }
    }
    #[inline]
    fn open_count(&self) -> usize {
        self.0.open_count() + if self.1.state.is_done() { 0 } else { 1 }
    }

    type RaceState = SlotRaceState<Rest::RaceState, KeyFut>;

    #[inline]
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

    #[inline]
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

    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            let k = match core::mem::replace(&mut self.1.state, ArmState::Empty) {
                ArmState::Key(k) => k,
                _ => unreachable!("init_dispatch called but arm not in Key state"),
            };
            SlotDispatchState::ThisArm((self.1.val_fn)(vp, k))
        } else {
            SlotDispatchState::Delegated(self.0.init_dispatch(arm_index, vp))
        }
    }

    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
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

    #[inline]
    fn take_outputs(&mut self) -> Self::Outputs {
        let out = match mem::replace(&mut self.1.state, ArmState::Empty) {
            ArmState::Done(k, v) => Some((k, v)),
            _ => None,
        };
        (self.0.take_outputs(), out)
    }
}

// --- (Rest, VirtualArmSlot) impl ---

impl<'de, KP, Rest, K, KeyFn, KeyFut, ValFn, ValFut> MapArmStack<'de, KP>
    for (Rest, VirtualArmSlot<K, KeyFn, ValFn>)
where
    KP: MapKeyProbe<'de>,
    Rest: MapArmStack<'de, KP>,
    KeyFn: FnMut(KP, usize) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(BVP<'de, KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(BVC<'de, KP>, ())>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    const FIELD_COUNT: usize = Rest::FIELD_COUNT;
    type Dynamic = Rest::Dynamic;
    type Outputs = Rest::Outputs;

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count()
    }
    #[inline]
    fn open_count(&self) -> usize {
        self.0.open_count() + 1
    }

    type RaceState = SlotRaceState<Rest::RaceState, KeyFut>;

    #[inline]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let rest_kp = kp.fork();
        let this_fut = (self.1.key_fn)(kp, arm_base + Self::SIZE - 1);
        SlotRaceState {
            rest: self.0.init_race(rest_kp, arm_base, field_base),
            this: Some(this_fut),
        }
    }

    #[inline]
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

    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
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

    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
        match state.project() {
            SlotDispatchProj::ThisArm(fut) => fut.poll(cx),
            SlotDispatchProj::Delegated(rest_state) => self.0.poll_dispatch(rest_state, cx),
        }
    }

    #[inline]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.0.take_outputs()
    }
}

// ---------------------------------------------------------------------------
// Wrapper impls for MapArmStack (borrow family)
// ---------------------------------------------------------------------------

// --- DetectDuplicates impl ---

impl<'de, KP, S, W, KeyFn, KeyFut, SkipFn, SkipFut> MapArmStack<'de, KP>
    for DetectDuplicates<S, W, KeyFn, SkipFn>
where
    KP: MapKeyProbe<'de>,
    S: MapArmStack<'de, KP>,
    W: AsRef<[(&'static str, usize)]>,
    KeyFn: FnMut(KP, usize) -> KeyFut,
    KeyFut: Future<
        Output = Result<Probe<(KP::KeyClaim, crate::impls::MatchVals<usize, W>)>, KP::Error>,
    >,
    SkipFn: FnMut(BVP<'de, KP>) -> SkipFut,
    SkipFut: Future<Output = Result<BVC<'de, KP>, KP::Error>>,
{
    const SIZE: usize = S::SIZE + 1;
    const FIELD_COUNT: usize = S::FIELD_COUNT;
    type Dynamic = S::Dynamic;
    type Outputs = S::Outputs;

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        self.inner.unsatisfied_count()
    }
    #[inline]
    fn open_count(&self) -> usize {
        self.inner.open_count() + 1
    }

    type RaceState = WrapperRaceState<S::RaceState, KeyFut>;

    #[inline]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let dup_kp = kp.fork();
        let dup_fut = (self.key_fn)(dup_kp, arm_base + Self::SIZE - 1);
        WrapperRaceState {
            inner: self.inner.init_race(kp, arm_base, field_base),
            virtual_arm: Some(dup_fut),
        }
    }

    #[inline]
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

    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            WrapperDispatchState::Virtual((self.skip_fn)(vp))
        } else {
            WrapperDispatchState::Inner(self.inner.init_dispatch(arm_index, vp))
        }
    }

    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
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

    #[inline]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.inner.take_outputs()
    }
}

// --- TagInjectingStack impl ---

impl<'de, 'v, KP, S, W, TagKeyFn, TagKeyFut, TagValFn, TagValFut> MapArmStack<'de, KP>
    for TagInjectingStack<'v, S, W, TagKeyFn, TagValFn>
where
    KP: MapKeyProbe<'de>,
    S: MapArmStack<'de, KP>,
    TagKeyFn: FnMut(KP, usize) -> TagKeyFut,
    TagKeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::impls::Match)>, KP::Error>>,
    TagValFn: FnMut(BVP<'de, KP>) -> TagValFut,
    TagValFut: Future<
        Output = Result<Probe<(BVC<'de, KP>, crate::impls::MatchVals<usize, W>)>, KP::Error>,
    >,
{
    const SIZE: usize = S::SIZE + 1;
    const FIELD_COUNT: usize = S::FIELD_COUNT;
    type Dynamic = S::Dynamic;
    type Outputs = S::Outputs;

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        self.inner.unsatisfied_count()
    }
    #[inline]
    fn open_count(&self) -> usize {
        self.inner.open_count() + 1
    }

    type RaceState = TagRaceState<TagKeyFut, S::RaceState>;

    #[inline]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let inner_kp = kp.fork();
        // Tag arm is always at global index 0; arm_base is irrelevant for it
        // but we pass arm_base to the closure for consistency.
        let tag_fut = (self.tag_key_fn)(kp, arm_base);
        TagRaceState {
            tag_fut: Some(tag_fut),
            inner: self.inner.init_race(inner_kp, arm_base + 1, field_base),
        }
    }

    #[inline]
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

    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
        if arm_index == 0 {
            TagDispatchState::Tag((self.tag_val_fn)(vp))
        } else {
            TagDispatchState::Inner(self.inner.init_dispatch(arm_index - 1, vp))
        }
    }

    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
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

    #[inline]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.inner.take_outputs()
    }
}

// --- StackConcat impl ---

impl<'de, KP, A, B> MapArmStack<'de, KP> for StackConcat<A, B>
where
    KP: MapKeyProbe<'de>,
    A: MapArmStack<'de, KP>,
    // Enforced at the type level, not via a runtime/const-eval assertion:
    // a StackConcat mixing a DYNAMIC (unbounded collection) side with a
    // non-DYNAMIC side simply fails to type-check.
    B: MapArmStack<'de, KP, Dynamic = A::Dynamic>,
{
    const SIZE: usize = A::SIZE + B::SIZE;
    const FIELD_COUNT: usize = A::FIELD_COUNT + B::FIELD_COUNT;
    type Dynamic = A::Dynamic;
    type Outputs = (A::Outputs, B::Outputs);

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count() + self.1.unsatisfied_count()
    }
    #[inline]
    fn open_count(&self) -> usize {
        self.0.open_count() + self.1.open_count()
    }

    type RaceState = ConcatRaceState<A::RaceState, B::RaceState>;

    #[inline]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, field_base: usize) -> Self::RaceState {
        let b_kp = kp.fork();
        ConcatRaceState {
            a: self.0.init_race(kp, arm_base, field_base),
            b: self
                .1
                .init_race(b_kp, arm_base + A::SIZE, field_base + A::FIELD_COUNT),
        }
    }

    #[inline]
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

    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
        if arm_index < A::SIZE {
            ConcatDispatchState::InA(self.0.init_dispatch(arm_index, vp))
        } else {
            ConcatDispatchState::InB(self.1.init_dispatch(arm_index - A::SIZE, vp))
        }
    }

    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
        match state.project() {
            ConcatDispatchProj::InA(a_state) => self.0.poll_dispatch(a_state, cx),
            ConcatDispatchProj::InB(b_state) => self.1.poll_dispatch(b_state, cx),
        }
    }

    #[inline]
    fn take_outputs(&mut self) -> Self::Outputs {
        (self.0.take_outputs(), self.1.take_outputs())
    }
}

// ---------------------------------------------------------------------------
// CandidateList<'de, KP, EnumOut> - internal recursion for CandidateArmStack
// ---------------------------------------------------------------------------

/// A left-nested list of [`Candidate`]s, each contributing its own arm stack.
///
/// `tag_matched` (threaded through every method rather than stored per-node)
/// gates racing/dispatch: once `Some(idx)`, every candidate other than `idx`
/// permanently stops racing - its arm futures are never created (`init_race`)
/// nor polled (`poll_race_one` returns `Miss` without touching them), so a
/// wrong candidate can never "steal" a wire key that the tag-selected
/// candidate also happens to declare (see `CandidateArmStack`'s module docs).
pub trait CandidateList<'de, KP: MapKeyProbe<'de>, EnumOut>: Sized {
    /// Pessimistic sum of every candidate's own arm-stack `SIZE`.
    const SIZE: usize;

    /// Number of unsatisfied fields on the candidate at `target_index`. Only
    /// meaningful once the tag has selected that candidate.
    fn unsatisfied_count(&self, target_index: usize) -> usize;
    /// Sum of `open_count()` over every candidate still eligible given
    /// `tag_matched` (all of them if `None`, only the matched one otherwise).
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
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState;
    #[allow(clippy::type_complexity)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>>;

    /// Build `EnumOut` from the candidate at `target_index`'s accumulated
    /// outputs. Called at most once, from `CandidateArmStack::take_outputs`.
    fn build_winner(&mut self, target_index: usize) -> Option<EnumOut>;
}

impl<'de, KP: MapKeyProbe<'de>, EnumOut> CandidateList<'de, KP, EnumOut> for CandidateBase {
    const SIZE: usize = 0;

    #[inline]
    fn unsatisfied_count(&self, _target_index: usize) -> usize {
        unreachable!("target candidate index not found (CandidateBase)")
    }
    #[inline]
    fn open_count(&self, _tag_matched: Option<usize>) -> usize {
        0
    }

    type RaceState = ();
    #[inline]
    fn init_race(&mut self, _kp: KP, _arm_base: usize, _tag_matched: Option<usize>) {}
    #[inline]
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
    #[inline]
    fn init_dispatch(&mut self, _arm_index: usize, _vp: BVP<'de, KP>) -> Self::DispatchState {
        unreachable!("init_dispatch called on CandidateBase")
    }
    #[inline]
    fn poll_dispatch(
        &mut self,
        _state: Pin<&mut Self::DispatchState>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
        unreachable!("poll_dispatch called on CandidateBase")
    }

    #[inline]
    fn build_winner(&mut self, _target_index: usize) -> Option<EnumOut> {
        unreachable!("build_winner: target index not found among candidates")
    }
}

impl<'de, KP, Rest, C, BuildFn, EnumOut> CandidateList<'de, KP, EnumOut>
    for (Rest, Candidate<C, BuildFn>)
where
    KP: MapKeyProbe<'de>,
    Rest: CandidateList<'de, KP, EnumOut>,
    C: MapArmStack<'de, KP>,
    BuildFn: FnMut(C::Outputs) -> Option<EnumOut>,
{
    const SIZE: usize = Rest::SIZE + C::SIZE;

    #[inline]
    fn unsatisfied_count(&self, target_index: usize) -> usize {
        if self.1.index == target_index {
            self.1.arms.unsatisfied_count()
        } else {
            self.0.unsatisfied_count(target_index)
        }
    }
    #[inline]
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

    #[inline]
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

    #[inline]
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

    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
        if arm_index < Rest::SIZE {
            ConcatDispatchState::InA(self.0.init_dispatch(arm_index, vp))
        } else {
            ConcatDispatchState::InB(self.1.arms.init_dispatch(arm_index - Rest::SIZE, vp))
        }
    }

    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
        match state.project() {
            ConcatDispatchProj::InA(s) => self.0.poll_dispatch(s, cx),
            ConcatDispatchProj::InB(s) => self.1.arms.poll_dispatch(s, cx),
        }
    }

    #[inline]
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
// CandidateArmStack impl (borrow family)
// ---------------------------------------------------------------------------

impl<'de, KP, Candidates, EnumOut, W, TagKeyFn, TagKeyFut, TagValFn, TagValFut> MapArmStack<'de, KP>
    for CandidateArmStack<Candidates, EnumOut, W, TagKeyFn, TagValFn>
where
    KP: MapKeyProbe<'de>,
    Candidates: CandidateList<'de, KP, EnumOut>,
    W: Copy,
    TagKeyFn: FnMut(KP, usize) -> TagKeyFut,
    TagKeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::impls::Match)>, KP::Error>>,
    TagValFn: FnMut(BVP<'de, KP>) -> TagValFut,
    TagValFut: Future<
        Output = Result<Probe<(BVC<'de, KP>, crate::impls::MatchVals<usize, W>)>, KP::Error>,
    >,
{
    const SIZE: usize = Candidates::SIZE + 1;
    const FIELD_COUNT: usize = 1;
    type Dynamic = False;
    type Outputs = Option<EnumOut>;

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        match self.tag_matched {
            None => 1,
            Some(idx) => self.candidates.unsatisfied_count(idx),
        }
    }
    #[inline]
    fn open_count(&self) -> usize {
        match self.tag_matched {
            None => 1 + self.candidates.open_count(None),
            Some(idx) => self.candidates.open_count(Some(idx)),
        }
    }

    type RaceState = TagRaceState<TagKeyFut, Candidates::RaceState>;

    #[inline]
    fn init_race(&mut self, mut kp: KP, arm_base: usize, _field_base: usize) -> Self::RaceState {
        let tag_fut = if self.tag_matched.is_some() {
            None
        } else {
            Some((self.tag_key_fn)(kp.fork(), arm_base))
        };
        TagRaceState {
            tag_fut,
            inner: self
                .candidates
                .init_race(kp, arm_base + 1, self.tag_matched),
        }
    }

    #[inline]
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

    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
        if arm_index == 0 {
            TagDispatchState::Tag((self.tag_val_fn)(vp))
        } else {
            TagDispatchState::Inner(self.candidates.init_dispatch(arm_index - 1, vp))
        }
    }

    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
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

    #[inline]
    fn take_outputs(&mut self) -> Self::Outputs {
        match self.tag_matched {
            Some(idx) => self.candidates.build_winner(idx),
            None => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NoTagCandidateList<'de, KP, EnumOut> - untagged enum flatten
// ---------------------------------------------------------------------------

/// Round-settling counterpart to [`CandidateList`] for untagged-enum flatten.
/// Implemented for the *same* concrete recursive types (`CandidateBase`,
/// `(Rest, Candidate<C, BuildFn>)`) that `CandidateList` already covers -
/// dispatch (`init_dispatch`/`poll_dispatch`/`build_winner`) is reused
/// unchanged from that trait; this one only adds the soft-elimination race.
///
/// See [`NoTagCandidateArmStack`]'s module doc and CLAUDE.md's "Untagged
/// flatten" section for the algorithm.
pub trait NoTagCandidateList<'de, KP: MapKeyProbe<'de>, EnumOut>: Sized {
    type RoundState;

    /// Fork `kp` for every currently-live candidate and initialize its own
    /// arm race (`arm_base`/`field_base` = 0 - untagged candidates never
    /// support positional dispatch, there's no tag to race against).
    /// Resets `Candidate::round_hit` on every node for this new round.
    fn init_round(&mut self, kp: KP) -> Self::RoundState;

    /// Sweep every not-yet-resolved live candidate's own local arm range
    /// once. Returns `Ok(true)` once every candidate in this subtree has
    /// resolved (Hit or Miss) this round - never stops early just because
    /// *one* candidate resolved, since soft elimination needs the *whole*
    /// round settled before it can safely decide who to eliminate.
    ///
    /// Indices recorded internally (and later returned by `take_winner`) are
    /// purely *local* to this whole `NoTagCandidateList`'s own `0..SIZE`
    /// range - matching `CandidateList`/`CandidateListOwned`'s convention,
    /// the caller (e.g. `StackConcat`) adds its own outer offset on the way
    /// back up, so no global `arm_base` is threaded through here.
    fn poll_sweep(
        &mut self,
        state: Pin<&mut Self::RoundState>,
        cx: &mut Context<'_>,
    ) -> Result<bool, KP::Error>;

    /// Once `poll_sweep` has returned `Ok(true)`, extract the lowest-index
    /// (declaration-order) live candidate that hit this round, if any.
    fn take_winner(&mut self, state: Pin<&mut Self::RoundState>) -> Option<(usize, KP::KeyClaim)>;

    /// Eliminate (permanently, `Candidate::live = false`) every live
    /// candidate whose `Candidate::round_hit` is still `false`. No-op when
    /// `any_hit` is `false` - "nobody recognized this key" says nothing
    /// about which live candidate is right, so nobody is eliminated.
    fn eliminate(&mut self, any_hit: bool);

    /// First-declared, still-live candidate whose own arm stack reports
    /// `unsatisfied_count() == 0` (fully satisfied).
    fn first_satisfied_live(&self) -> Option<usize>;
}

impl<'de, KP: MapKeyProbe<'de>, EnumOut> NoTagCandidateList<'de, KP, EnumOut> for CandidateBase {
    type RoundState = ();

    #[inline]
    fn init_round(&mut self, _kp: KP) {}

    #[inline]
    fn poll_sweep(
        &mut self,
        _state: Pin<&mut ()>,
        _cx: &mut Context<'_>,
    ) -> Result<bool, KP::Error> {
        Ok(true)
    }

    #[inline]
    fn take_winner(&mut self, _state: Pin<&mut ()>) -> Option<(usize, KP::KeyClaim)> {
        None
    }

    #[inline]
    fn eliminate(&mut self, _any_hit: bool) {}

    #[inline]
    fn first_satisfied_live(&self) -> Option<usize> {
        None
    }
}

impl<'de, KP, Rest, C, BuildFn, EnumOut> NoTagCandidateList<'de, KP, EnumOut>
    for (Rest, Candidate<C, BuildFn>)
where
    KP: MapKeyProbe<'de>,
    Rest: NoTagCandidateList<'de, KP, EnumOut> + CandidateList<'de, KP, EnumOut>,
    C: MapArmStack<'de, KP>,
    BuildFn: FnMut(C::Outputs) -> Option<EnumOut>,
{
    type RoundState = NoTagRoundState<Rest::RoundState, C::RaceState, KP::KeyClaim>;

    #[inline]
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

    #[inline]
    fn poll_sweep(
        &mut self,
        state: Pin<&mut Self::RoundState>,
        cx: &mut Context<'_>,
    ) -> Result<bool, KP::Error> {
        let mut projected = state.project();
        let rest_settled = self.0.poll_sweep(projected.rest, cx)?;

        let this_settled = if projected.resolved.is_none() {
            let this_base = <Rest as CandidateList<'de, KP, EnumOut>>::SIZE;
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

    #[inline]
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

    #[inline]
    fn eliminate(&mut self, any_hit: bool) {
        self.0.eliminate(any_hit);
        if any_hit && self.1.live && !self.1.round_hit {
            self.1.live = false;
        }
    }

    #[inline]
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
// NoTagCandidateArmStack impl (borrow family)
// ---------------------------------------------------------------------------

impl<'de, KP, Candidates, EnumOut> MapArmStack<'de, KP>
    for NoTagCandidateArmStack<Candidates, EnumOut>
where
    KP: MapKeyProbe<'de>,
    Candidates: CandidateList<'de, KP, EnumOut> + NoTagCandidateList<'de, KP, EnumOut>,
{
    const SIZE: usize = <Candidates as CandidateList<'de, KP, EnumOut>>::SIZE;
    const FIELD_COUNT: usize = 0;
    type Dynamic = False;
    type Outputs = Option<EnumOut>;

    #[inline]
    fn unsatisfied_count(&self) -> usize {
        if self.candidates.first_satisfied_live().is_some() {
            0
        } else {
            1
        }
    }
    #[inline]
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

    #[inline]
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

    type DispatchState = <Candidates as CandidateList<'de, KP, EnumOut>>::DispatchState;
    #[inline]
    fn init_dispatch(&mut self, arm_index: usize, vp: BVP<'de, KP>) -> Self::DispatchState {
        self.candidates.init_dispatch(arm_index, vp)
    }
    #[inline]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(BVC<'de, KP>, ())>, KP::Error>> {
        self.candidates.poll_dispatch(state, cx)
    }

    #[inline]
    fn take_outputs(&mut self) -> Self::Outputs {
        match self.candidates.first_satisfied_live() {
            Some(idx) => self.candidates.build_winner(idx),
            None => None,
        }
    }
}
