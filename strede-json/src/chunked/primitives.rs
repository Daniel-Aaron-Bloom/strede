//! Chunked JSON deserializer for async streaming input.
//!
//! Uses [`strede::SharedBuf`]/[`strede::Handle`] to coordinate access to a
//! buffer that is refilled asynchronously by a user-supplied loader closure.
//! The deserializer holds either a `&mut SharedBuf` (top-level) or an
//! `Option<Handle>` (sub-deserializer for option/map values/seq elements).
//!
//! # Capabilities vs. [`crate::JsonDeserializer`]
//!
//! - Implements the **owned** trait family only ([`DeserializerOwned`],
//!   [`EntryOwned`], etc.). Zero-copy `&'de str` / `&'de [u8]` borrowing is
//!   unsupported by design - buffer-chunk lifetimes are shorter than the
//!   caller-facing deserialization session, so there is no `'de` to borrow
//!   for. Use [`EntryOwned::deserialize_str_chunks`] /
//!   [`EntryOwned::deserialize_bytes_chunks`] for string and byte data.
//! - `#[derive(Deserialize)]` today emits a borrow-family impl; types meant
//!   for use with this deserializer must hand-roll a [`DeserializeOwned`]
//!   impl (typically using `deserialize_str_chunks` for keys).
//! - Sub-deserializer Miss recovery is best-effort: if a sub-probe partially
//!   consumed the buffer (advanced past chunk boundaries) and then missed,
//!   the parent's replay state may be inconsistent. Common cases (sync probes
//!   that don't advance) work correctly.
//! - Loader errors must be communicated via the `Buffer` value (e.g. an empty
//!   slice signals EOF). The `F: AsyncFnMut(&mut B)` signature does not
//!   support `Result`.

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::JsonError;
use crate::token::{SimpleToken, Token};
use strede::{
    Buffer, Chunk, DeserializeOwned, DeserializerOwned, EntryOwned, Probe, StrAccessOwned,
};

use super::{ChunkedJsonClaim, ChunkedJsonDeserializer, ChunkedJsonSubDeserializer};

// ---------------------------------------------------------------------------
// Per-format primitive DeserializeOwned impls
//
// Each type gets impls for both `ChunkedJsonDeserializer` and
// `ChunkedJsonSubDeserializer`. The orphan rule allows this because both are
// local types. The impl bodies are identical since both route through `entry`
// and produce the same `ChunkedJsonEntry`.
//
// impl_deserialize_owned_one! emits a single impl for one concrete deserializer
// type. It is called twice per primitive (once for each deserializer) so that
// Rust typechecks each copy independently — sharing a body expression across
// two impls via macro_rules! causes the claim type to resolve to `()` before
// the enclosing async fn return type can constrain it.
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_owned_one {
    ($de:ty; bool) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for bool {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, JsonError> {
                d.entry(|[e]| async move {
                    match e.token {
                        Token::Simple(SimpleToken::Bool(b), tok) => Ok(Probe::Hit((
                            ChunkedJsonClaim { tokenizer: tok, offset: e.offset, handle: e.handle },
                            b,
                        ))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
    ($de:ty; ()) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for () {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, JsonError> {
                d.entry(|[e]| async move {
                    match e.token {
                        Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit((
                            ChunkedJsonClaim { tokenizer: tok, offset: e.offset, handle: e.handle },
                            (),
                        ))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }
    };
    ($de:ty; char) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for char {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, JsonError> {
                d.entry(|[e]| async move {
                    // Open string chunks; accept iff the string contains exactly one scalar value.
                    let chunks = match EntryOwned::deserialize_str_chunks(e).await? {
                        Probe::Hit(c) => c,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    let mut found: Option<char> = None;
                    let mut too_many = false;
                    let mut chunks = chunks;
                    loop {
                        match StrAccessOwned::next_str(chunks, |s: &str| {
                            let mut iter = s.chars();
                            if let Some(c) = iter.next() {
                                if found.is_some() || iter.next().is_some() {
                                    too_many = true;
                                } else {
                                    found = Some(c);
                                }
                            }
                        })
                        .await?
                        {
                            Chunk::Data((next, ())) => chunks = next,
                            Chunk::Done(claim) => {
                                if too_many {
                                    return Ok(Probe::Miss);
                                }
                                return match found {
                                    Some(c) => Ok(Probe::Hit((claim, c))),
                                    None => Ok(Probe::Miss),
                                };
                            }
                        }
                    }
                })
                .await
            }
        }
    };
    ($de:ty; $($t:ty),+) => {
        $(impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, JsonError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_owned_both {
    ($($t:tt),+) => {
        $(
            impl_deserialize_owned_one!(ChunkedJsonDeserializer<'s, B, F>; $t);
            impl_deserialize_owned_one!(ChunkedJsonSubDeserializer<'s, B, F>; $t);
        )+
    };
}

impl_deserialize_owned_both!(bool, (), char);
impl_deserialize_owned_one!(ChunkedJsonDeserializer<'s, B, F>; u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);
impl_deserialize_owned_one!(ChunkedJsonSubDeserializer<'s, B, F>; u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);
