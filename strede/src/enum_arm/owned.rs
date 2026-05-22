use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use super::{EnumArmBase, EnumArmSlot, EnumArmState, SlotRaceState, poll_arm_slot};
use crate::Probe;
use crate::owned::EnumVariantProbeOwned;

// ---------------------------------------------------------------------------
// EnumArmStackOwned<VP> - owned-family enum arm stack
// ---------------------------------------------------------------------------

/// Owned-family enum arm stack. Parallel to [`super::borrow::EnumArmStack`].
pub trait EnumArmStackOwned<VP: EnumVariantProbeOwned>: Sized {
    const SIZE: usize;

    type Outputs;
    type RaceState;

    fn init_race(&mut self, vp: VP) -> Self::RaceState;

    #[allow(clippy::type_complexity)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, VP::Claim)>, VP::Error>>;

    async fn race(&mut self, vp: VP) -> Result<Probe<(usize, VP::Claim)>, VP::Error> {
        if Self::SIZE == 0 {
            return Ok(Probe::Miss);
        }
        let mut race_state = core::pin::pin!(self.init_race(vp));
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

    fn take_outputs(&mut self) -> Self::Outputs;
}

// ---------------------------------------------------------------------------
// Impl for EnumArmBase
// ---------------------------------------------------------------------------

impl<VP: EnumVariantProbeOwned> EnumArmStackOwned<VP> for EnumArmBase {
    const SIZE: usize = 0;
    type Outputs = ();
    type RaceState = ();

    #[inline(always)]
    fn init_race(&mut self, _vp: VP) {}

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        _state: Pin<&mut ()>,
        _arm_index: usize,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, VP::Claim)>, VP::Error>> {
        unreachable!("poll_race_one on EnumArmBase (SIZE=0)")
    }

    #[inline(always)]
    fn take_outputs(&mut self) {}
}

// ---------------------------------------------------------------------------
// Impl for (Rest, EnumArmSlot<Out, ArmFn>)
// ---------------------------------------------------------------------------

impl<VP, Rest, Out, ArmFn, ArmFut> EnumArmStackOwned<VP> for (Rest, EnumArmSlot<Out, ArmFn>)
where
    VP: EnumVariantProbeOwned,
    Rest: EnumArmStackOwned<VP>,
    ArmFn: FnMut(VP) -> ArmFut,
    ArmFut: Future<Output = Result<Probe<(VP::Claim, Out)>, VP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    type Outputs = (Rest::Outputs, Option<Out>);

    type RaceState = SlotRaceState<Rest::RaceState, ArmFut>;

    fn init_race(&mut self, mut vp: VP) -> Self::RaceState {
        let this_vp = vp.fork();
        let rest_state = self.0.init_race(vp);
        let this_fut = if self.1.state.is_done() {
            None
        } else {
            Some((self.1.arm_fn)(this_vp))
        };
        SlotRaceState {
            rest: rest_state,
            this: this_fut,
        }
    }

    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, VP::Claim)>, VP::Error>> {
        let state = state.project();
        let my_index = Self::SIZE - 1;
        if arm_index < my_index {
            return self.0.poll_race_one(state.rest, arm_index, cx);
        }
        match poll_arm_slot(state.this, cx) {
            Poll::Ready(Ok(Probe::Hit((claim, out)))) => {
                self.1.state = EnumArmState::Done(out);
                Poll::Ready(Ok(Probe::Hit((my_index, claim))))
            }
            Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }

    fn take_outputs(&mut self) -> Self::Outputs {
        let rest = self.0.take_outputs();
        let this = match core::mem::replace(&mut self.1.state, EnumArmState::Empty) {
            EnumArmState::Done(v) => Some(v),
            EnumArmState::Empty => None,
        };
        (rest, this)
    }
}
