//! `strede` - async, zero-alloc, pull-based deserialization.
//!
//! # Core idea
//!
//! Stream advancement and type probing are separated:
//!
//! - [`Deserializer::entry`] (`self`) - the *only* way to advance the
//!   stream; passes `N` owned entry handles to an async closure and returns
//!   whatever the closure produces.
//! - [`Entry`] probe methods (`self`) - consume the entry; resolve to
//!   `Ok(Probe::Hit((Claim, T)))` when the token matches type `T`,
//!   `Ok(Probe::Miss)` if the type doesn't match, or `Err(e)` on a fatal
//!   format error.
//!
//! Use `N > 1` to obtain multiple handles for the same slot and race them
//! via [`select_probe!`].  The winning arm returns `Ok(Probe::Hit((claim, value)))`;
//! the claim is forwarded to `entry` as proof of consumption, and `entry` returns
//! the value:
//!
//! ```rust,ignore
//! let v = d.entry(|[e1, e2, e3]| async {
//!     select_probe! {
//!         async move {
//!             let (claim, f) = hit!(e1.deserialize_f32().await);
//!             Ok(Probe::Hit((claim, Value::Float(f))))
//!         },
//!         async move {
//!             let (claim, s) = hit!(e2.deserialize_str().await);
//!             Ok(Probe::Hit((claim, Value::Str(s))))
//!         },
//!         async move {
//!             let (claim, m) = hit!(e3.deserialize_map().await);
//!             let (claim, v) = collect_map(m).await?;
//!             Ok(Probe::Hit((claim, Value::Map(v))))
//!         },
//!         @miss => Ok(Probe::Miss),
//!     }
//! }).await?;
//! ```
//!
//! Schema mismatches (wrong token type for a probe) resolve `Ok(Probe::Miss)` -
//! they are not errors and do not advance the stream.  `Pending` on a probe
//! future means *only* "no data available yet."  Fatal format errors
//! (malformed data, I/O failures) resolve `Err(e)` and propagate through `?`.

// `async fn` in public traits can't express `Send` bounds on the returned
// futures - intentional here so implementations aren't forced to be Send.
#![no_std]
#![allow(async_fn_in_trait)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod borrow;
mod error;
mod impls;
pub mod map_arm;
pub mod owned;
pub mod probe;
pub mod shared_buf;

use core::{convert::Infallible, marker::PhantomData};

// -- proc-macros --
pub use strede_derive::{Deserialize, DeserializeOwned};

// -- shared types --
pub use error::DeserializeError;
pub use probe::{Chunk, Probe};
pub use shared_buf::{Buffer, Handle, SharedBuf};

/// The uninhabited bottom type - equivalent to `!` but stable on all editions.
///
/// Used as the associated `StrChunks`, `BytesChunks`, and `Seq` types on
/// [`Entry`] / [`EntryOwned`] implementations that never produce those
/// accessor kinds.  Because `Never` has no values, all trait method bodies
/// on `Never` are written as `match self {}` and the compiler accepts them
/// without any reachable code.
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub struct Never<'a, Claim, Error>(
    Infallible,
    PhantomData<(fn(*const Claim), fn(*const Error), fn(&'a ()))>,
);

/// Terminal flatten continuation: calls `map.iterate(SkipUnknownOwned(arms))` directly.
/// Used for the last (or only) flatten field in a struct.
/// Zero-alloc - may stack-overflow with deeply nested `StackConcat` types
/// (typically 3+ flatten fields). Use [`FlattenTerminalBoxed`] to avoid this.
pub struct FlattenTerminal;

/// Terminal flatten continuation: calls `map.iterate(SkipUnknownOwned(arms))` via `Box::pin`.
/// Heap-allocates the future to break the deeply-nested async state-machine chain
/// produced by `StackConcat`, preventing stack overflow with 3+ flatten fields.
/// Generated when `#[strede(flatten(boxed))]` is used.
#[cfg(feature = "alloc")]
pub struct FlattenTerminalBoxed;

// -- borrow family --
pub use borrow::{
    BytesAccess, Deserialize, Deserializer, Entry, FlattenCont, FlattenDeserializer, FlattenEntry,
    FlattenMapAccess, MapAccess, MapArmStack, MapKeyClaim, MapKeyProbe, MapValueClaim,
    MapValueProbe, SeqAccess, SeqEntry, StrAccess,
};

// -- default expression helper --
// Used by the derive macro to support both `default = "path"` (called as a
// function) and `default = "expr"` (used as-is).  Inherent method resolution
// beats trait methods, so `DefaultWrapper(fn_path).value()` calls the function
// while `DefaultWrapper(expr).value()` returns the value directly.
#[doc(hidden)]
pub trait DefaultValue {
    type Value;
    fn value(self) -> Self::Value;
}

#[doc(hidden)]
pub struct DefaultWrapper<T>(pub T);

impl<T> DefaultValue for DefaultWrapper<T> {
    type Value = T;
    #[inline]
    fn value(self) -> T {
        self.0
    }
}

impl<T, F: FnOnce() -> T> DefaultWrapper<F> {
    #[inline]
    pub fn value(self) -> T {
        self.0()
    }
}

/// Left-nest a comma-separated list of patterns under a given base pattern.
///
/// `__left_nest_pat!(base, a, b, c)` expands to `((base, a), b), c)`.
#[doc(hidden)]
#[macro_export]
macro_rules! __left_nest_pat {
    (@built $built:pat ,) => { $built };
    (@built $built:pat , $a:pat, $b:pat, $c:pat, $d:pat $(, $rest:pat)*) => {
        $crate::__left_nest_pat!(@built (((($built, $a), $b), $c), $d) , $($rest),*)
    };
    (@built $built:pat , $a:pat, $b:pat $(, $rest:pat)*) => {
        $crate::__left_nest_pat!(@built (($built, $a), $b) , $($rest),*)
    };
    (@built $built:pat , $next:pat $(, $rest:pat)*) => {
        $crate::__left_nest_pat!(@built ($built, $next) , $($rest),*)
    };
    ($base:pat, $first:pat $(, $rest:pat)*) => {
        $crate::__left_nest_pat!(@built ($base, $first) , $($rest),*)
    };
}

// -- utility types --
pub use impls::{Match, MatchVals, Skip, UnwrapOrElse, map_facade, tag_facade};

// -- owned family --
pub use owned::{
    ArmState, BytesAccessOwned, DeserializeOwned, DeserializerOwned, DetectDuplicatesOwned,
    EntryOwned, FlattenContOwned, FlattenDeserializerOwned, FlattenEntryOwned,
    FlattenMapAccessOwned, MapAccessOwned, MapArm, MapArmBase, MapArmSlot, MapArmStackOwned,
    MapKeyClaimOwned, MapKeyProbeOwned, MapValueClaimOwned, MapValueProbeOwned, NextKey,
    SeqAccessOwned, SeqEntryOwned, StackConcat, StrAccessOwned, TagInjectingStackOwned,
    VirtualArmSlot,
};

#[cfg(feature = "alloc")]
pub use alloc::boxed::Box;

pub mod utils {
    /// Like [`core::array::repeat`], but for `clone(&mut self) -> Self`
    #[inline(always)]
    pub fn repeat<T, const N: usize>(f: T, mut clone: impl FnMut(&mut T) -> T) -> [T; N] {
        let mut f = Some(f);
        core::array::from_fn(|i| {
            if i == N - 1 {
                f.take().unwrap()
            } else {
                clone(f.as_mut().unwrap())
            }
        })
    }
}
