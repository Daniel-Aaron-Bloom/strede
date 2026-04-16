//! `strede` — async, zero-alloc, pull-based deserialization.
//!
//! # Core idea
//!
//! Stream advancement and type probing are separated:
//!
//! - [`Deserializer::next`] (`&mut self`) — the *only* way to advance the
//!   stream; passes `N` owned entry handles to an async closure and returns
//!   whatever the closure produces.
//! - [`Entry`] probe methods (`self`) — consume the entry; resolve to
//!   `Ok(Probe::Hit((Claim, T)))` when the token matches type `T`,
//!   `Ok(Probe::Miss)` if the type doesn't match, or `Err(e)` on a fatal
//!   format error.
//!
//! Use `N > 1` to obtain multiple handles for the same slot and race them
//! via [`select_probe!`].  The winning arm returns `Ok(Probe::Hit((claim, value)))`;
//! the claim is forwarded to `next` as proof of consumption, and `next` returns
//! the value:
//!
//! ```rust,ignore
//! let v = d.next(|[e1, e2, e3]| async {
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
//!         miss => Ok(Probe::Miss),
//!     }
//! }).await?;
//! ```
//!
//! Schema mismatches (wrong token type for a probe) resolve `Ok(Probe::Miss)` —
//! they are not errors and do not advance the stream.  `Pending` on a probe
//! future means *only* "no data available yet."  Fatal format errors
//! (malformed data, I/O failures) resolve `Err(e)` and propagate through `?`.

// `async fn` in public traits can't express `Send` bounds on the returned
// futures — intentional here so implementations aren't forced to be Send.
// #![no_std]
#![allow(async_fn_in_trait)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod borrow;
mod error;
mod impls;
mod owned;
mod probe;
pub mod shared_buf;

use core::{convert::Infallible, marker::PhantomData};

// -- proc-macros --
#[doc(hidden)]
pub use strede_derive::select_probe_inner;
pub use strede_derive::{Deserialize, DeserializeOwned};

// -- shared types --
pub use error::DeserializeError;
pub use probe::{Chunk, Probe};
pub use shared_buf::{Buffer, Handle, SharedBuf};

/// The uninhabited bottom type — equivalent to `!` but stable on all editions.
///
/// Used as the associated `StrChunks`, `BytesChunks`, and `Seq` types on
/// [`Entry`] / [`EntryOwned`] implementations that never produce those
/// accessor kinds.  Because `Never` has no values, all trait method bodies
/// on `Never` are written as `match self {}` and the compiler accepts them
/// without any reachable code.
#[doc(hidden)]
pub struct Never<'a, Claim, Error>(
    Infallible,
    PhantomData<(fn(*const Claim), fn(*const Error), fn(&'a ()))>,
);

// -- borrow family --
pub use borrow::{
    BytesAccess, Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry,
    SeqAccess, SeqEntry, StrAccess,
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

// -- utility types --
pub use impls::{Match, MatchVals, Skip, UnwrapOrElse, tag_facade};

// -- owned family --
pub use owned::{
    BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned,
    MapKeyEntryOwned, MapValueEntryOwned, SeqAccessOwned, SeqEntryOwned, StrAccessOwned,
};
