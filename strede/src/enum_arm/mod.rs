use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use pin_project::pin_project;

use crate::Probe;

pub mod borrow;
pub mod owned;
pub use borrow::EnumArmStack;
pub use owned::EnumArmStackOwned;

// ===========================================================================
// Shared enum arm infrastructure - used by both families.
//
// Mirrors map_arm/mod.rs but for enum variant dispatch instead of map field
// iteration. Key differences:
//
// - Each EnumArmSlot holds a single closure that both identifies the variant
//   AND deserializes the payload in one call (discriminant + payload are not
//   separated, because interleaved formats like msgpack arrays can't split them).
//
// - There is no "dispatch" phase separate from the "race" phase. The arm's
//   ArmFn receives an EnumVariantProbe, calls identify_* on it, and returns
//   Probe<(Claim, Out)> directly.
//
// - There is no MapKeyClaim / MapValueClaim chain; the probe method returns
//   the claim directly alongside the output value.
// ===========================================================================

// ---------------------------------------------------------------------------
// Arm state
// ---------------------------------------------------------------------------

/// State of a single arm slot in enum dispatch.
///
/// - `Empty` — no variant matched yet.
/// - `Done(Out)` — variant matched and payload deserialized.
pub enum EnumArmState<Out> {
    Empty,
    Done(Out),
}

impl<Out> EnumArmState<Out> {
    pub fn is_done(&self) -> bool {
        matches!(self, EnumArmState::Done(_))
    }
}

// ---------------------------------------------------------------------------
// EnumArmSlot
// ---------------------------------------------------------------------------

/// One variant arm slot.  `ArmFn` receives a forked [`EnumVariantProbe`] and
/// returns `Probe<(Claim, Out)>` — the arm is fully responsible for both
/// recognizing the variant and deserializing the payload.
///
/// - `ArmFn: FnMut(VP) -> ArmFut` where `ArmFut: Future<Output = Result<Probe<(VP::Claim, Out)>, VP::Error>>`
pub struct EnumArmSlot<Out, ArmFn> {
    pub arm_fn: ArmFn,
    pub state: EnumArmState<Out>,
}

impl<Out, ArmFn> EnumArmSlot<Out, ArmFn> {
    pub fn new(arm_fn: ArmFn) -> Self {
        Self {
            arm_fn,
            state: EnumArmState::Empty,
        }
    }
}

/// Newtype wrapper used with `+` on [`EnumArmBase`] to build an arm stack.
pub struct EnumArm<S>(pub S);

/// Base of the enum arm tuple stack.
pub struct EnumArmBase;

impl<S> core::ops::Add<EnumArm<S>> for EnumArmBase {
    type Output = (EnumArmBase, S);
    fn add(self, rhs: EnumArm<S>) -> (EnumArmBase, S) {
        (self, rhs.0)
    }
}

impl<Rest, S, T> core::ops::Add<EnumArm<T>> for (Rest, S) {
    type Output = ((Rest, S), T);
    fn add(self, rhs: EnumArm<T>) -> ((Rest, S), T) {
        (self, rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Pin-projection helpers
// ---------------------------------------------------------------------------

/// Race state for `(Rest, EnumArmSlot)`.
#[pin_project]
pub struct SlotRaceState<RestState, ArmFut> {
    #[pin]
    pub rest: RestState,
    #[pin]
    pub this: Option<ArmFut>,
}

// ---------------------------------------------------------------------------
// poll_arm_slot - shared helper
// ---------------------------------------------------------------------------

/// Poll an `Option<Future>` slot for an enum arm, returning `Miss` if `None`.
#[inline(always)]
pub(crate) fn poll_arm_slot<F, Claim, Out, E>(
    mut slot: Pin<&mut Option<F>>,
    cx: &mut Context<'_>,
) -> Poll<Result<Probe<(Claim, Out)>, E>>
where
    F: Future<Output = Result<Probe<(Claim, Out)>, E>>,
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
// enum_arms! macro
// ---------------------------------------------------------------------------

/// Build a left-nested enum arm tuple from a flat list of arm closures.
///
/// Each arm is a `FnMut(VP) -> Future<Output = Result<Probe<(Claim, Out)>, Error>>`.
///
/// ```rust,ignore
/// let arms = enum_arms! {
///     |vp| vp.deserialize_unit_by_name([("Foo", 0)]),
///     |vp| vp.deserialize_payload_by_name::<Bar>([("Bar", 1)], ()),
/// };
/// ```
#[macro_export]
macro_rules! enum_arms {
    ($arm_fn:expr $(, $rest:expr)* $(,)?) => {
        $crate::EnumArmBase
            + $crate::EnumArm($crate::EnumArmSlot::new($arm_fn))
            $(+ $crate::EnumArm($crate::EnumArmSlot::new($rest)))*
    };
}
