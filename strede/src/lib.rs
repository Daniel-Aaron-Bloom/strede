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
//! let v = d.next(|[e1, e2, e3]| async move {
//!     select_probe! {
//!         (claim, f) = e1.deserialize_f32()  => Ok(Probe::Hit((claim, Value::Float(f)))),
//!         (claim, s) = e2.deserialize_str()  => Ok(Probe::Hit((claim, Value::Str(s)))),
//!         m          = e3.deserialize_map()  => {
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
#![no_std]
#![allow(async_fn_in_trait)]

mod borrow;
mod error;
mod owned;
mod probe;
pub mod shared_buf;

// -- proc-macros --
pub use strede_derive::DeserializeOwned;
#[doc(hidden)]
pub use strede_derive::select_probe_inner;

#[macro_export]
macro_rules! select_probe {
    ($($tt:tt)*) => {
        $crate::select_probe_inner!(@$crate $($tt)*)
    };
}

// -- shared types --
pub use error::DeserializeError;
pub use probe::{Chunk, Probe};
pub use shared_buf::{Buffer, Handle, SharedBuf};

// -- borrow family --
pub use borrow::{
    BytesAccess, Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry,
    SeqAccess, SeqEntry, StrAccess,
};

// -- owned family --
pub use owned::{
    BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned,
    MapKeyEntryOwned, MapValueEntryOwned, SeqAccessOwned, SeqEntryOwned, StrAccessOwned,
};
