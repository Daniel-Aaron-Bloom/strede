//! Blanket `Deserialize` / `DeserializeOwned` implementations for standard
//! library types beyond the primitives defined in `borrow.rs` and `owned.rs`.

use core::cell::{Cell, RefCell};
use core::cmp::Reverse;
use core::marker::PhantomData;
use core::num::{
    NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize, NonZeroU8,
    NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize, Saturating, Wrapping,
};

use arrayvec::{ArrayString, ArrayVec};
use strede_derive::{Deserialize, DeserializeOwned};

use crate::borrow::{BytesAccess, Deserialize, Deserializer, Entry, SeqAccess, SeqEntry};
use crate::owned::{
    BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, SeqAccessOwned,
    SeqEntryOwned, StrAccessOwned,
};
use crate::{Chunk, DeserializeError, Probe, StrAccess, hit, or_miss, select_probe};

// ===========================================================================
// Core impls — no features required
// ===========================================================================

// --- Unit ---

impl<'de> Deserialize<'de> for () {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, ())))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for () {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, ())))
        })
        .await
    }
}

// --- usize / isize ---

impl<'de> Deserialize<'de> for usize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_u32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_u64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as usize)))
        })
        .await
    }
}

impl<'de> Deserialize<'de> for isize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_i32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_i64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as isize)))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for usize {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_u32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_u64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as usize)))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for isize {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_i32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_i64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as isize)))
        })
        .await
    }
}

// --- PhantomData ---

impl<'de, T: ?Sized> Deserialize<'de> for PhantomData<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, PhantomData)))
        })
        .await
    }
}

impl<'s, T: ?Sized> DeserializeOwned<'s> for PhantomData<T> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, PhantomData)))
        })
        .await
    }
}

// --- Skip ---

/// Deserializes by discarding the current token unconditionally.
///
/// Always returns `Probe::Hit(Skip)` on well-formed input — it never misses,
/// since [`Entry::skip`](crate::Entry::skip) accepts any token type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Skip;

impl<'de> Deserialize<'de> for Skip {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = e.skip().await?;
            Ok(Probe::Hit((claim, Skip)))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for Skip {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = e.skip().await?;
            Ok(Probe::Hit((claim, Skip)))
        })
        .await
    }
}

// --- MatchVals ---

/// Deserializes a string or byte token, matches it against a list of
/// `(key, value)` pairs supplied as `Extra`, and returns the associated value
/// for the first matching key.
///
/// `T` must be `Copy` because the value must be copied out of the const-generic
/// `Extra` array (which is itself `Copy` when `T: Copy`).
///
/// Returns `Probe::Hit(MatchVals(t))` for the first matching key,
/// `Probe::Miss` when no key matches or the token is the wrong type.
///
/// ```rust,ignore
/// d.entry(|[e]| async {
///     let (claim, MatchVals(status)) = hit!(
///         e.deserialize_value::<MatchVals<u8>, [(&str, u8); 3]>([
///             ("ok",      0),
///             ("warn",    1),
///             ("error",   2),
///         ]).await
///     );
///     Ok(Probe::Hit((claim, status)))
/// })
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchVals<T>(pub T);

impl<'de, 'a, T: Copy, const N: usize> Deserialize<'de, [(&'a str, T); N]> for MatchVals<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [(&'a str, T); N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let str_miss = Cell::new(false);
        let str_miss = &str_miss;
        d.entry(async |[e1, e2]| {
            select_probe!(
                async move {
                    let (claim, s) = hit!(e1.deserialize_str().await);
                    for (key, val) in extra {
                        if s == key {
                            return Ok(Probe::Hit((claim, MatchVals(val))));
                        }
                    }
                    str_miss.set(true);
                    Ok(Probe::Miss)
                },
                async move {
                    let mut acc = hit!(e2.deserialize_str_chunks().await);
                    let mut remaining: [&'a str; N] = extra.map(|(k, _)| k);
                    let mut alive = [true; N];
                    let claim = loop {
                        match acc
                            .next_str(|chunk| {
                                for i in 0..N {
                                    if alive[i] {
                                        if remaining[i].starts_with(chunk) {
                                            remaining[i] = &remaining[i][chunk.len()..];
                                        } else {
                                            alive[i] = false;
                                        }
                                    }
                                }
                            })
                            .await?
                        {
                            Chunk::Data((new_acc, ())) => {
                                if str_miss.get() {
                                    alive = [false; N];
                                }
                                acc = new_acc;
                            }
                            Chunk::Done(claim) => break claim,
                        }
                    };
                    for i in 0..N {
                        if alive[i] && remaining[i].is_empty() {
                            return Ok(Probe::Hit((claim, MatchVals(extra[i].1))));
                        }
                    }
                    Ok(Probe::Miss)
                }
            )
        })
        .await
    }
}

impl<'de, 'a, T: Copy, const N: usize> Deserialize<'de, [(&'a [u8], T); N]> for MatchVals<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [(&'a [u8], T); N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let bytes_miss = Cell::new(false);
        let bytes_miss = &bytes_miss;
        d.entry(async |[e1, e2]| {
            select_probe!(
                async move {
                    let (claim, b) = hit!(e1.deserialize_bytes().await);
                    for (key, val) in extra {
                        if b == key {
                            return Ok(Probe::Hit((claim, MatchVals(val))));
                        }
                    }
                    bytes_miss.set(true);
                    Ok(Probe::Miss)
                },
                async move {
                    let mut acc = hit!(e2.deserialize_bytes_chunks().await);
                    let mut remaining: [&'a [u8]; N] = extra.map(|(k, _)| k);
                    let mut alive = [true; N];
                    let claim = loop {
                        match acc
                            .next_bytes(|chunk| {
                                for i in 0..N {
                                    if alive[i] {
                                        if remaining[i].starts_with(chunk) {
                                            remaining[i] = &remaining[i][chunk.len()..];
                                        } else {
                                            alive[i] = false;
                                        }
                                    }
                                }
                            })
                            .await?
                        {
                            Chunk::Data((new_acc, ())) => {
                                if bytes_miss.get() {
                                    alive = [false; N];
                                }
                                acc = new_acc;
                            }
                            Chunk::Done(claim) => break claim,
                        }
                    };
                    for i in 0..N {
                        if alive[i] && remaining[i].is_empty() {
                            return Ok(Probe::Hit((claim, MatchVals(extra[i].1))));
                        }
                    }
                    Ok(Probe::Miss)
                }
            )
        })
        .await
    }
}

impl<'s, 'a, T: Copy, const N: usize> DeserializeOwned<'s, [(&'a str, T); N]> for MatchVals<T> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: [(&'a str, T); N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let mut acc = hit!(e.deserialize_str_chunks().await);
            let mut remaining: [&'a str; N] = extra.map(|(k, _)| k);
            let mut alive = [true; N];
            let claim = loop {
                match acc
                    .next_str(|chunk| {
                        for i in 0..N {
                            if alive[i] {
                                if remaining[i].starts_with(chunk) {
                                    remaining[i] = &remaining[i][chunk.len()..];
                                } else {
                                    alive[i] = false;
                                }
                            }
                        }
                    })
                    .await?
                {
                    Chunk::Data((new_acc, ())) => acc = new_acc,
                    Chunk::Done(claim) => break claim,
                }
            };
            for i in 0..N {
                if alive[i] && remaining[i].is_empty() {
                    return Ok(Probe::Hit((claim, MatchVals(extra[i].1))));
                }
            }
            Ok(Probe::Miss)
        })
        .await
    }
}

impl<'s, 'a, T: Copy, const N: usize> DeserializeOwned<'s, [(&'a [u8], T); N]> for MatchVals<T> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: [(&'a [u8], T); N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let mut acc = hit!(e.deserialize_bytes_chunks().await);
            let mut remaining: [&'a [u8]; N] = extra.map(|(k, _)| k);
            let mut alive = [true; N];
            let claim = loop {
                match acc
                    .next_bytes(|chunk| {
                        for i in 0..N {
                            if alive[i] {
                                if remaining[i].starts_with(chunk) {
                                    remaining[i] = &remaining[i][chunk.len()..];
                                } else {
                                    alive[i] = false;
                                }
                            }
                        }
                    })
                    .await?
                {
                    Chunk::Data((new_acc, ())) => acc = new_acc,
                    Chunk::Done(claim) => break claim,
                }
            };
            for i in 0..N {
                if alive[i] && remaining[i].is_empty() {
                    return Ok(Probe::Hit((claim, MatchVals(extra[i].1))));
                }
            }
            Ok(Probe::Miss)
        })
        .await
    }
}

impl<'de, 'a, const N: usize> Deserialize<'de, [&'a str; N]> for MatchVals<usize> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [&'a str; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut i = 0usize;
        let pairs = extra.map(|k| { let idx = i; i += 1; (k, idx) });
        let probe = <MatchVals<usize> as Deserialize<'de, [(&'a str, usize); N]>>::deserialize(d, pairs).await?;
        Ok(probe)
    }
}

impl<'de, 'a, const N: usize> Deserialize<'de, [&'a [u8]; N]> for MatchVals<usize> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut i = 0usize;
        let pairs = extra.map(|k| { let idx = i; i += 1; (k, idx) });
        let probe = <MatchVals<usize> as Deserialize<'de, [(&'a [u8], usize); N]>>::deserialize(d, pairs).await?;
        Ok(probe)
    }
}

impl<'s, 'a, const N: usize> DeserializeOwned<'s, [&'a str; N]> for MatchVals<usize> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: [&'a str; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut i = 0usize;
        let pairs = extra.map(|k| { let idx = i; i += 1; (k, idx) });
        let probe = <MatchVals<usize> as DeserializeOwned<'s, [(&'a str, usize); N]>>::deserialize_owned(d, pairs).await?;
        Ok(probe)
    }
}

impl<'s, 'a, const N: usize> DeserializeOwned<'s, [&'a [u8]; N]> for MatchVals<usize> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut i = 0usize;
        let pairs = extra.map(|k| { let idx = i; i += 1; (k, idx) });
        let probe = <MatchVals<usize> as DeserializeOwned<'s, [(&'a [u8], usize); N]>>::deserialize_owned(d, pairs).await?;
        Ok(probe)
    }
}

// --- UnwrapOrElse ---

/// Deserializes `T` normally on `Probe::Hit`, or calls a fallback on `Probe::Miss`
/// to produce a `T` unconditionally. The fallback receives the skipped token's
/// `Claim`, so the stream is always advanced exactly once.
///
/// `Extra` is `(F, InnerExtra)` where `F: AsyncFnOnce() -> T`. Sync closures
/// can be wrapped: `async move || value`.
///
/// This is the building block for "match-or-default" map key dispatch: pair it
/// with `MatchVals<usize>` to replace bespoke chunk-matcher code generation with a
/// single call-site array of `(wire_name, index)` pairs.
///
/// ```rust,ignore
/// // Returns the matched index, or SENTINEL for any unrecognized key.
/// e.deserialize_value::<UnwrapOrElse<MatchVals<usize>>, _>(
///     (async || MatchVals(SENTINEL), [("foo", 0usize), ("bar", 1)])
/// )
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnwrapOrElse<T>(pub T);

// Borrow family
impl<'de, T, F, Extra> Deserialize<'de, (F, Extra)> for UnwrapOrElse<T>
where
    T: Deserialize<'de, Extra>,
    F: AsyncFnOnce() -> T,
    Extra: Copy,
{
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        (fallback, extra): (F, Extra),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let fallback = Cell::new(Some(fallback));
        let fallback = &fallback;
        d.entry(|[e1, e2]| async move {
            select_probe!(
                async move {
                    Ok(e1
                        .deserialize_value::<T, Extra>(extra)
                        .await?
                        .map(|(c, v)| (c, UnwrapOrElse(v))))
                },
                async move {
                    let c = e2.skip().await?;
                    let fallback = fallback.take().unwrap();

                    Ok(Probe::Hit((c, UnwrapOrElse(fallback().await))))
                }
            )
        })
        .await
    }
}

// Owned family
impl<'s, T, F, Extra> DeserializeOwned<'s, (F, Extra)> for UnwrapOrElse<T>
where
    T: DeserializeOwned<'s, Extra>,
    F: AsyncFnOnce() -> T,
    Extra: Copy,
{
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        (fallback, extra): (F, Extra),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let fallback = Cell::new(Some(fallback));
        let fallback = &fallback;
        d.entry(|[e1, e2]| async move {
            select_probe!(
                async move {
                    Ok(e1
                        .deserialize_value::<T, Extra>(extra)
                        .await?
                        .map(|(c, v)| (c, UnwrapOrElse(v))))
                },
                async move {
                    let c = e2.skip().await?;
                    let fallback = fallback.take().unwrap();
                    Ok(Probe::Hit((c, UnwrapOrElse(fallback().await))))
                }
            )
        })
        .await
    }
}

// --- Match ---

/// Deserializes a string or byte token and checks it for an exact match
/// against a caller-supplied value passed as `Extra`.
///
/// The `Extra` can be:
/// - `&str` / `&[u8]` — single string/bytes to match.
/// - `[&str; N]` / `[&[u8]; N]` — match any of N strings/bytes.
///
/// Returns `Probe::Hit(Match)` if the token's content equals `extra` (or any
/// element of `extra`), `Probe::Miss` if the token is the wrong type or the
/// content differs.
///
/// Useful for discriminated dispatch in `select_probe!`:
/// ```rust,ignore
/// d.entry(|[e1, e2]| select_probe! {
///     e1.deserialize_value::<Match, &str>("ok"),
///     e2.deserialize_value::<Match, &str>("err"),
///     miss => Ok(Probe::Miss),
/// })
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match;

impl<'de, 'a> Deserialize<'de, &'a str> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: &'a str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a str, Match); 1]>>::deserialize(d, [(extra, Match)]).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'de, 'a> Deserialize<'de, &'a [u8]> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: &'a [u8],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a [u8], Match); 1]>>::deserialize(d, [(extra, Match)]).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'de, 'a, const N: usize> Deserialize<'de, [&'a str; N]> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [&'a str; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a str, Match); N]>>::deserialize(d, extra.map(|k| (k, Match))).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'de, 'a, const N: usize> Deserialize<'de, [&'a [u8]; N]> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a [u8], Match); N]>>::deserialize(d, extra.map(|k| (k, Match))).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'s, 'a> DeserializeOwned<'s, &'a str> for Match {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: &'a str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as DeserializeOwned<'s, [(&'a str, Match); 1]>>::deserialize_owned(d, [(extra, Match)]).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'s, 'a> DeserializeOwned<'s, &'a [u8]> for Match {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: &'a [u8],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as DeserializeOwned<'s, [(&'a [u8], Match); 1]>>::deserialize_owned(d, [(extra, Match)]).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'s, 'a, const N: usize> DeserializeOwned<'s, [&'a str; N]> for Match {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: [&'a str; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as DeserializeOwned<'s, [(&'a str, Match); N]>>::deserialize_owned(d, extra.map(|k| (k, Match))).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'s, 'a, const N: usize> DeserializeOwned<'s, [&'a [u8]; N]> for Match {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as DeserializeOwned<'s, [(&'a [u8], Match); N]>>::deserialize_owned(d, extra.map(|k| (k, Match))).await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

// --- Newtype wrappers (delegate to inner) ---

macro_rules! impl_newtype_wrapper {
    ($wrapper:ident) => {
        impl<'de, T: Deserialize<'de>> Deserialize<'de> for $wrapper<T> {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, $wrapper(v))))
            }
        }

        impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for $wrapper<T> {
            async fn deserialize_owned<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(T::deserialize_owned(d, ()).await?.map(|(c, v)| (c, $wrapper(v))))
            }
        }
    };
}

impl_newtype_wrapper!(Wrapping);
impl_newtype_wrapper!(Saturating);
impl_newtype_wrapper!(Reverse);

// --- Cell<T> ---

impl<'de, T: Deserialize<'de> + Copy> Deserialize<'de> for Cell<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, Cell::new(v))))
    }
}

impl<'s, T: DeserializeOwned<'s> + Copy> DeserializeOwned<'s> for Cell<T> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize_owned(d, ()).await?.map(|(c, v)| (c, Cell::new(v))))
    }
}

// --- RefCell<T> ---

impl<'de, T: Deserialize<'de>> Deserialize<'de> for RefCell<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize(d, ())
            .await?
            .map(|(c, v)| (c, RefCell::new(v))))
    }
}

impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for RefCell<T> {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize_owned(d, ())
            .await?
            .map(|(c, v)| (c, RefCell::new(v))))
    }
}

// --- NonZero types ---

macro_rules! impl_nonzero {
    ($nonzero:ty, $method:ident) => {
        impl<'de> Deserialize<'de> for $nonzero {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let (claim, v) = hit!(e.$method().await);
                    let nz = or_miss!(<$nonzero>::new(v));
                    Ok(Probe::Hit((claim, nz)))
                })
                .await
            }
        }

        impl<'s> DeserializeOwned<'s> for $nonzero {
            async fn deserialize_owned<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let (claim, v) = hit!(e.$method().await);
                    let nz = or_miss!(<$nonzero>::new(v));
                    Ok(Probe::Hit((claim, nz)))
                })
                .await
            }
        }
    };
}

impl_nonzero!(NonZeroU8, deserialize_u8);
impl_nonzero!(NonZeroU16, deserialize_u16);
impl_nonzero!(NonZeroU32, deserialize_u32);
impl_nonzero!(NonZeroU64, deserialize_u64);
impl_nonzero!(NonZeroU128, deserialize_u128);
impl_nonzero!(NonZeroI8, deserialize_i8);
impl_nonzero!(NonZeroI16, deserialize_i16);
impl_nonzero!(NonZeroI32, deserialize_i32);
impl_nonzero!(NonZeroI64, deserialize_i64);
impl_nonzero!(NonZeroI128, deserialize_i128);

// NonZeroUsize / NonZeroIsize — delegate to usize/isize

impl<'de> Deserialize<'de> for NonZeroUsize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<usize as Deserialize>::deserialize(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

impl<'s> DeserializeOwned<'s> for NonZeroUsize {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<usize as DeserializeOwned>::deserialize_owned(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

impl<'de> Deserialize<'de> for NonZeroIsize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<isize as Deserialize>::deserialize(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

impl<'s> DeserializeOwned<'s> for NonZeroIsize {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<isize as DeserializeOwned>::deserialize_owned(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

// --- Arrays [T; N] (const generic) ---

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for [T; N] {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            let mut seq = hit!(e.deserialize_seq().await);
            let mut buf = ArrayVec::<T, N>::new();
            for _ in 0..N {
                let v = hit!(seq.next(|[elem]| elem.get::<T, _>(())).await);
                let (n_seq, v) = or_miss!(v.data());
                seq = n_seq;
                buf.push(v);
            }
            let v = hit!(seq.next(|[elem]| { elem.get::<T, _>(()) }).await);
            let claim = or_miss!(v.done());
            Ok(Probe::Hit((claim, buf.into_inner().unwrap_or_else(|_| panic!("all N elements pushed")))))
        })
        .await
    }
}

impl<'s, T: DeserializeOwned<'s>, const N: usize> DeserializeOwned<'s> for [T; N] {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            let mut seq = hit!(e.deserialize_seq().await);
            let mut buf = ArrayVec::<T, N>::new();
            for _ in 0..N {
                let v = hit!(seq.next(|[elem]| { elem.get::<T, _>(()) }).await);
                let (s, v) = or_miss!(v.data());
                buf.push(v);
                seq = s;
            }
            let v = hit!(
                seq.next::<1, _, _, T>(|[elem]| { elem.get::<T, _>(()) })
                    .await
            );
            let claim = or_miss!(v.done());
            Ok(Probe::Hit((claim, buf.into_inner().unwrap_or_else(|_| panic!("all N elements pushed")))))
        })
        .await
    }
}

// --- Tuples up to 16 ---

macro_rules! impl_tuple {
    ($($T:ident $v:ident),+ $(,)?) => {
        impl<'de, $($T: Deserialize<'de>),+> Deserialize<'de> for ($($T,)+) {
            async fn deserialize<D: Deserializer<'de>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let seq = hit!(e.deserialize_seq().await);
                    $(
                        let v = hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await);
                        let (seq, $v) = or_miss!(v.data());
                    )+
                    let v = hit!(seq.next(|[elem]| async {
                        elem.get::<impl_tuple!(@first $($T),+), _>(()).await
                    }).await);
                    let claim = or_miss!(v.done());
                    Ok(Probe::Hit((claim, ($($v,)+))))
                }).await
            }
        }

        impl<'s, $($T: DeserializeOwned<'s>),+> DeserializeOwned<'s> for ($($T,)+) {
            async fn deserialize_owned<D: DeserializerOwned<'s>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let seq = hit!(e.deserialize_seq().await);
                    $(
                        let v = hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await);
                        let (seq, $v) = or_miss!(v.data());
                    )+
                    let v = hit!(seq.next::<1, _, _, impl_tuple!(@first $($T),+)>(|[elem]| async {
                        elem.get::<impl_tuple!(@first $($T),+), _>(()).await
                    }).await);
                    let claim = or_miss!(v.done());
                    Ok(Probe::Hit((claim, ($($v,)+))))
                }).await
            }
        }
    };

    (@first $first:ident $(, $rest:ident)*) => { $first };
}

impl_tuple!(T0 v0);
impl_tuple!(T0 v0, T1 v1);
impl_tuple!(T0 v0, T1 v1, T2 v2);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15);

// ===========================================================================
// IP address and socket address types — string-parsed
// ===========================================================================

macro_rules! impl_from_str {
    ($ty:ty, $max_len:expr) => {
        impl<'de> Deserialize<'de> for $ty {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e1, e2]| async {
                    select_probe! {
                        async move {
                            let (claim, s) = hit!(e1.deserialize_str().await);
                            let v = or_miss!(s.parse::<$ty>().ok());
                            Ok(Probe::Hit((claim, v)))
                        },
                        async move {
                            let mut chunks = hit!(e2.deserialize_str_chunks().await);
                            let mut buf = ArrayString::<$max_len>::new();
                            let mut overflow = false;
                            let claim = loop {
                                match chunks.next_str(|s| {
                                    if !overflow && buf.try_push_str(s).is_err() {
                                        overflow = true;
                                    }
                                }).await? {
                                    Chunk::Data((new, ())) => chunks = new,
                                    Chunk::Done(claim) => break claim,
                                }
                            };
                            if overflow {
                                return Ok(Probe::Miss);
                            }
                            let v = or_miss!(buf.as_str().parse::<$ty>().ok());
                            Ok(Probe::Hit((claim, v)))
                        },
                        miss => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }

        impl<'s> DeserializeOwned<'s> for $ty {
            async fn deserialize_owned<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let mut chunks = hit!(e.deserialize_str_chunks().await);
                    let mut buf = ArrayString::<$max_len>::new();
                    let mut overflow = false;
                    let claim = loop {
                        match chunks
                            .next_str(|s| {
                                if !overflow {
                                    if buf.try_push_str(s).is_err() {
                                        overflow = true;
                                    }
                                }
                            })
                            .await?
                        {
                            Chunk::Data((c, ())) => chunks = c,
                            Chunk::Done(claim) => break claim,
                        }
                    };
                    if overflow {
                        return Ok(Probe::Miss);
                    }
                    let v = or_miss!(buf.as_str().parse::<$ty>().ok());
                    Ok(Probe::Hit((claim, v)))
                })
                .await
            }
        }
    };
}

impl_from_str!(core::net::IpAddr, 45);
impl_from_str!(core::net::Ipv4Addr, 15);
impl_from_str!(core::net::Ipv6Addr, 45);
impl_from_str!(core::net::SocketAddr, 51);
impl_from_str!(core::net::SocketAddrV4, 21);
impl_from_str!(core::net::SocketAddrV6, 51);

// --- Duration ---
#[derive(DeserializeOwned, Deserialize)]
#[strede(crate = "crate")]
struct Duration {
    secs: u64,
    nanos: u32,
}

impl<'de> Deserialize<'de> for core::time::Duration {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, Duration { secs, nanos }) =
            hit!(<Duration as Deserialize>::deserialize(d, ()).await);
        Ok(Probe::Hit((claim, Self::new(secs, nanos))))
    }
}

impl<'s> DeserializeOwned<'s> for core::time::Duration {
    async fn deserialize_owned<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, Duration { secs, nanos }) =
            hit!(<Duration as DeserializeOwned>::deserialize_owned(d, ()).await);
        Ok(Probe::Hit((claim, Self::new(secs, nanos))))
    }
}

// ===========================================================================
// alloc impls
// ===========================================================================

#[cfg(feature = "alloc")]
mod alloc_impls {
    use crate::borrow::{
        Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry, SeqAccess,
        SeqEntry,
    };
    use crate::owned::{
        BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned,
        MapKeyEntryOwned, MapValueEntryOwned, SeqAccessOwned, SeqEntryOwned, StrAccessOwned,
    };
    use crate::{BytesAccess, Chunk, DeserializeError, Probe, StrAccess, hit, or_miss};

    extern crate alloc;
    use alloc::{
        borrow::{Cow, ToOwned},
        boxed::Box,
        collections::{BTreeMap, BTreeSet, BinaryHeap, LinkedList, VecDeque},
        ffi::CString,
        string::String,
        vec::Vec,
    };

    // --- String ---

    impl<'de> Deserialize<'de> for String {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_str_chunks().await);
                let mut out = String::new();
                let claim = loop {
                    match chunks.next_str(|s| out.push_str(s)).await? {
                        Chunk::Data((new, ())) => chunks = new,
                        Chunk::Done(claim) => break claim,
                    }
                };
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    impl<'s> DeserializeOwned<'s> for String {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_str_chunks().await);
                let mut out = String::new();
                let claim = loop {
                    match chunks
                        .next_str(|s| {
                            out.push_str(s);
                        })
                        .await?
                    {
                        Chunk::Data((c, ())) => chunks = c,
                        Chunk::Done(claim) => break claim,
                    }
                };
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    // --- Vec<T> ---

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for Vec<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = Vec::new();
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((n_seq, v)) => {
                            out.push(v);
                            seq = n_seq;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for Vec<T> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = Vec::new();
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((s, v)) => {
                            out.push(v);
                            seq = s;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- Box<T>, Box<str>, Box<[T]> ---

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for Box<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, Box::new(v))))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for Box<T> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize_owned(d, ()).await?.map(|(c, v)| (c, Box::new(v))))
        }
    }

    impl<'de> Deserialize<'de> for Box<str> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_str())))
        }
    }

    impl<'s> DeserializeOwned<'s> for Box<str> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as DeserializeOwned>::deserialize_owned(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_str())))
        }
    }

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for Box<[T]> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            Ok(Vec::<T>::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, v.into_boxed_slice())))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for Box<[T]> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            Ok(Vec::<T>::deserialize_owned(d, ())
                .await?
                .map(|(c, v)| (c, v.into_boxed_slice())))
        }
    }

    // --- Sequence-collected types (BTreeSet, BinaryHeap, LinkedList, VecDeque) ---

    macro_rules! impl_seq_collect {
        ($ty:ident < $T:ident $(: $($bound:path),+)? >, $new:expr, $add:ident) => {
            impl<'de, $T: Deserialize<'de> $($(+ $bound)+)?> Deserialize<'de> for $ty<$T> {
                async fn deserialize<D: Deserializer<'de>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
                where
                    D::Error: DeserializeError,
                {
                    d.entry(|[e]| async {
                        let mut seq = hit!(e.deserialize_seq().await);
                        let mut out: $ty<$T> = $new;
                        loop {
                            match hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await) {
                                Chunk::Data((n_seq, v)) => {
                                    out.$add(v);
                                    seq = n_seq;
                                }
                                Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                            }
                        }
                    }).await
                }
            }

            impl<'s, $T: DeserializeOwned<'s> $($(+ $bound)+)?> DeserializeOwned<'s> for $ty<$T> {
                async fn deserialize_owned<D: DeserializerOwned<'s>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
                where
                    D::Error: DeserializeError,
                {
                    d.entry(|[e]| async {
                        let mut seq = hit!(e.deserialize_seq().await);
                        let mut out: $ty<$T> = $new;
                        loop {
                            match hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await) {
                                Chunk::Data((s, v)) => { out.$add(v); seq = s; }
                                Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                            }
                        }
                    }).await
                }
            }
        };
    }

    impl_seq_collect!(BTreeSet<T: Eq, Ord>, BTreeSet::new(), insert);
    impl_seq_collect!(BinaryHeap<T: Ord>, BinaryHeap::new(), push);
    impl_seq_collect!(LinkedList<T>, LinkedList::new(), push_back);
    impl_seq_collect!(VecDeque<T>, VecDeque::new(), push_back);

    // --- BTreeMap<K, V> ---

    impl<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>> Deserialize<'de> for BTreeMap<K, V> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = BTreeMap::new();
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) =
                                hit!(ke.key((), |_k, [ve]| ve.value::<V, _>(())).await);
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, K: DeserializeOwned<'s> + Ord, V: DeserializeOwned<'s>> DeserializeOwned<'s>
        for BTreeMap<K, V>
    {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = BTreeMap::new();
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) =
                                hit!(ke.key((), |_k, [ve]| ve.value::<V, _>(())).await);
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- Cow<'a, T> (borrow: try zero-copy first, then Owned) ---

    impl<'de: 'a, 'a, T: ?Sized + ToOwned> Deserialize<'de> for Cow<'a, T>
    where
        &'a T: Deserialize<'de>,
        T::Owned: Deserialize<'de>,
    {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e1, e2]| async {
                if let Probe::Hit((claim, t)) = e1.deserialize_value::<&'a T, ()>(()).await? {
                    return Ok(Probe::Hit((claim, Cow::Borrowed(t))));
                }
                match e2.deserialize_value::<T::Owned, ()>(()).await? {
                    Probe::Hit((claim, owned)) => Ok(Probe::Hit((claim, Cow::Owned(owned)))),
                    Probe::Miss => Ok(Probe::Miss),
                }
            })
            .await
        }
    }

    // --- Cow<'a, T> (owned family: always Cow::Owned) ---

    impl<'s, 'a, T: ?Sized + ToOwned> DeserializeOwned<'s> for Cow<'a, T>
    where
        T::Owned: DeserializeOwned<'s>,
    {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::Owned::deserialize_owned(d, ())
                .await?
                .map(|(c, v)| (c, Cow::Owned(v))))
        }
    }

    // --- CString ---

    impl<'de> Deserialize<'de> for CString {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_bytes_chunks().await);
                let mut out = Vec::new();
                let claim = loop {
                    match chunks.next_bytes(|b| out.extend_from_slice(b)).await? {
                        Chunk::Data((new, ())) => chunks = new,
                        Chunk::Done(claim) => break claim,
                    }
                };
                let cs = or_miss!(CString::new(out).ok());
                Ok(Probe::Hit((claim, cs)))
            })
            .await
        }
    }

    impl<'s> DeserializeOwned<'s> for CString {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_bytes_chunks().await);
                let mut out = Vec::new();
                let claim = loop {
                    match chunks
                        .next_bytes(|b| {
                            out.extend_from_slice(b);
                        })
                        .await?
                    {
                        Chunk::Data((c, ())) => chunks = c,
                        Chunk::Done(claim) => break claim,
                    }
                };
                let cs = or_miss!(CString::new(out).ok());
                Ok(Probe::Hit((claim, cs)))
            })
            .await
        }
    }

    // --- Rc<T>, Arc<T> (behind rc feature) ---

    #[cfg(feature = "rc")]
    mod rc_impls {
        use crate::Probe;
        use crate::borrow::{Deserialize, Deserializer};
        use crate::owned::{DeserializeOwned, DeserializerOwned};
        extern crate alloc;
        use alloc::{boxed::Box, rc::Rc, sync::Arc};

        impl<'de, T: ?Sized> Deserialize<'de> for Rc<T>
        where
            Box<T>: Deserialize<'de>,
        {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Rc::from(b))))
            }
        }

        impl<'s, T: ?Sized> DeserializeOwned<'s> for Rc<T>
        where
            Box<T>: DeserializeOwned<'s>,
        {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Rc::from(b))))
            }
        }

        impl<'de, T: ?Sized> Deserialize<'de> for Arc<T>
        where
            Box<T>: Deserialize<'de>,
        {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Arc::from(b))))
            }
        }

        impl<'s, T: ?Sized> DeserializeOwned<'s> for Arc<T>
        where
            Box<T>: DeserializeOwned<'s>,
        {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Arc::from(b))))
            }
        }
    }
}

// ===========================================================================
// std impls
// ===========================================================================

#[cfg(feature = "std")]
mod std_impls {
    extern crate std;

    use crate::borrow::{
        Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry, SeqAccess,
        SeqEntry,
    };
    use crate::owned::{
        DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapKeyEntryOwned,
        MapValueEntryOwned, SeqAccessOwned, SeqEntryOwned,
    };
    use crate::{Chunk, DeserializeError, Probe, hit, or_miss};
    use core::hash::{BuildHasher, Hash};
    use std::collections::{HashMap, HashSet};

    extern crate alloc;
    use alloc::{boxed::Box, string::String};

    // --- HashMap<K, V, S> ---

    impl<'de, K, V, S> Deserialize<'de> for HashMap<K, V, S>
    where
        K: Deserialize<'de> + Eq + Hash,
        V: Deserialize<'de>,
        S: BuildHasher + Default,
    {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = HashMap::with_hasher(S::default());
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) = hit!(
                                ke.key((), |_k, [ve]| async { ve.value::<V, _>(()).await })
                                    .await
                            );
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, K, V, S> DeserializeOwned<'s> for HashMap<K, V, S>
    where
        K: DeserializeOwned<'s> + Eq + Hash,
        V: DeserializeOwned<'s>,
        S: BuildHasher + Default,
    {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = HashMap::with_hasher(S::default());
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) = hit!(
                                ke.key((), |_k, [ve]| async { ve.value::<V, _>(()).await })
                                    .await
                            );
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- HashSet<T, S> ---

    impl<'de, T, S> Deserialize<'de> for HashSet<T, S>
    where
        T: Deserialize<'de> + Eq + Hash,
        S: BuildHasher + Default,
    {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = HashSet::with_hasher(S::default());
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((n_seq, v)) => {
                            out.insert(v);
                            seq = n_seq;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, T, S> DeserializeOwned<'s> for HashSet<T, S>
    where
        T: DeserializeOwned<'s> + Eq + Hash,
        S: BuildHasher + Default,
    {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = HashSet::with_hasher(S::default());
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((s, v)) => {
                            out.insert(v);
                            seq = s;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- Mutex<T>, RwLock<T> ---

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for std::sync::Mutex<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::Mutex::new(v))))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for std::sync::Mutex<T> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize_owned(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::Mutex::new(v))))
        }
    }

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for std::sync::RwLock<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::RwLock::new(v))))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for std::sync::RwLock<T> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize_owned(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::RwLock::new(v))))
        }
    }

    // --- PathBuf, OsString ---

    impl<'de> Deserialize<'de> for std::path::PathBuf {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::path::PathBuf::from(s))))
        }
    }

    impl<'s> DeserializeOwned<'s> for std::path::PathBuf {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as DeserializeOwned>::deserialize_owned(d, ())
                .await?
                .map(|(c, s)| (c, std::path::PathBuf::from(s))))
        }
    }

    impl<'de> Deserialize<'de> for std::ffi::OsString {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::ffi::OsString::from(s))))
        }
    }

    impl<'s> DeserializeOwned<'s> for std::ffi::OsString {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as DeserializeOwned>::deserialize_owned(d, ())
                .await?
                .map(|(c, s)| (c, std::ffi::OsString::from(s))))
        }
    }

    // --- Box<Path>, Box<OsStr> ---

    impl<'de> Deserialize<'de> for Box<std::path::Path> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::path::PathBuf as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, p)| (c, p.into_boxed_path())))
        }
    }

    impl<'s> DeserializeOwned<'s> for Box<std::path::Path> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::path::PathBuf as DeserializeOwned>::deserialize_owned(d, ())
                .await?
                .map(|(c, p)| (c, p.into_boxed_path())))
        }
    }

    impl<'de> Deserialize<'de> for Box<std::ffi::OsStr> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::ffi::OsString as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_os_str())))
        }
    }

    impl<'s> DeserializeOwned<'s> for Box<std::ffi::OsStr> {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::ffi::OsString as DeserializeOwned>::deserialize_owned(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_os_str())))
        }
    }

    // --- &'a Path (borrow only) ---

    impl<'de: 'a, 'a> Deserialize<'de> for &'a std::path::Path {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<&'de str>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::path::Path::new(s))))
        }
    }

    // --- SystemTime ---

    enum SysTimeField {
        Secs(u64),
        Nanos(u32),
    }

    impl<'de> Deserialize<'de> for std::time::SystemTime {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut secs: Option<u64> = None;
                let mut nanos: Option<u32> = None;
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, _k, r) = hit!(
                                ke.key((), |k, [ve]| {
                                    let field = match *k {
                                        "secs_since_epoch" => 0u8,
                                        "nanos_since_epoch" => 1u8,
                                        _ => 2u8,
                                    };
                                    async move {
                                        match field {
                                            0 => {
                                                let (claim, v) = hit!(ve.value::<u64, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Secs(v))))
                                            }
                                            1 => {
                                                let (claim, v) = hit!(ve.value::<u32, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Nanos(v))))
                                            }
                                            _ => Ok(Probe::Miss),
                                        }
                                    }
                                })
                                .await
                            );
                            Ok(Probe::Hit((claim, r)))
                        })
                        .await
                    ) {
                        Chunk::Data((m, SysTimeField::Secs(v))) => {
                            secs = Some(v);
                            map = m;
                        }
                        Chunk::Data((m, SysTimeField::Nanos(v))) => {
                            nanos = Some(v);
                            map = m;
                        }
                        Chunk::Done(claim) => {
                            let s = or_miss!(secs);
                            let n = or_miss!(nanos);
                            let dur = core::time::Duration::new(s, n);
                            return Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + dur)));
                        }
                    }
                }
            })
            .await
        }
    }

    impl<'s> DeserializeOwned<'s> for std::time::SystemTime {
        async fn deserialize_owned<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut secs: Option<u64> = None;
                let mut nanos: Option<u32> = None;
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, _k, r) = hit!(
                                ke.key((), |k: &String, [ve]| {
                                    let field = match k.as_str() {
                                        "secs_since_epoch" => 0u8,
                                        "nanos_since_epoch" => 1u8,
                                        _ => 2u8,
                                    };
                                    async move {
                                        match field {
                                            0 => {
                                                let (claim, v) = hit!(ve.value::<u64, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Secs(v))))
                                            }
                                            1 => {
                                                let (claim, v) = hit!(ve.value::<u32, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Nanos(v))))
                                            }
                                            _ => Ok(Probe::Miss),
                                        }
                                    }
                                })
                                .await
                            );
                            Ok(Probe::Hit((claim, r)))
                        })
                        .await
                    ) {
                        Chunk::Data((m, SysTimeField::Secs(v))) => {
                            secs = Some(v);
                            map = m;
                        }
                        Chunk::Data((m, SysTimeField::Nanos(v))) => {
                            nanos = Some(v);
                            map = m;
                        }
                        Chunk::Done(claim) => {
                            let s = or_miss!(secs);
                            let n = or_miss!(nanos);
                            let dur = core::time::Duration::new(s, n);
                            return Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + dur)));
                        }
                    }
                }
            })
            .await
        }
    }

    // --- Atomic types ---

    macro_rules! impl_atomic {
        ($atomic:path, $inner:ty) => {
            impl<'de> Deserialize<'de> for $atomic {
                async fn deserialize<D: Deserializer<'de>>(
                    d: D,
                    _extra: (),
                ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                    Ok(<$inner as Deserialize>::deserialize(d, ())
                        .await?
                        .map(|(c, v)| (c, <$atomic>::new(v))))
                }
            }

            impl<'s> DeserializeOwned<'s> for $atomic {
                async fn deserialize_owned<D: DeserializerOwned<'s>>(
                    d: D,
                    _extra: (),
                ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                    Ok(<$inner as DeserializeOwned>::deserialize_owned(d, ())
                        .await?
                        .map(|(c, v)| (c, <$atomic>::new(v))))
                }
            }
        };
    }

    #[cfg(target_has_atomic = "8")]
    impl_atomic!(core::sync::atomic::AtomicBool, bool);
    #[cfg(target_has_atomic = "8")]
    impl_atomic!(core::sync::atomic::AtomicI8, i8);
    #[cfg(target_has_atomic = "8")]
    impl_atomic!(core::sync::atomic::AtomicU8, u8);
    #[cfg(target_has_atomic = "16")]
    impl_atomic!(core::sync::atomic::AtomicI16, i16);
    #[cfg(target_has_atomic = "16")]
    impl_atomic!(core::sync::atomic::AtomicU16, u16);
    #[cfg(target_has_atomic = "32")]
    impl_atomic!(core::sync::atomic::AtomicI32, i32);
    #[cfg(target_has_atomic = "32")]
    impl_atomic!(core::sync::atomic::AtomicU32, u32);
    #[cfg(target_has_atomic = "64")]
    impl_atomic!(core::sync::atomic::AtomicI64, i64);
    #[cfg(target_has_atomic = "64")]
    impl_atomic!(core::sync::atomic::AtomicU64, u64);
    #[cfg(target_has_atomic = "ptr")]
    impl_atomic!(core::sync::atomic::AtomicIsize, isize);
    #[cfg(target_has_atomic = "ptr")]
    impl_atomic!(core::sync::atomic::AtomicUsize, usize);
}

// ===========================================================================
// Generic map-as-deserializer facade
// ===========================================================================

/// Wraps any `MapAccess` (or `MapAccessOwned`) as a `Deserializer` /
/// `DeserializerOwned` whose only supported probe is `deserialize_map`.
/// All other probes return `Miss`.  Used by enum derives to hand an
/// already-opened map to a variant's `Deserialize` impl.
pub mod map_facade {
    use crate::{
        Chunk, Probe,
        borrow::{Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry},
        hit,
        owned::{
            DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapKeyEntryOwned,
            MapValueEntryOwned,
        },
    };
    use core::{array, future::Future, marker::PhantomData};

    // -----------------------------------------------------------------------
    // Borrow family
    // -----------------------------------------------------------------------

    /// A `Deserializer` that presents a single already-opened `MapAccess` as
    /// if it were a fresh entry delivering a map.  Used by internally-tagged
    /// enum derives to hand the in-progress map to a variant's `Deserialize`
    /// impl after the tag key has been consumed.
    ///
    /// When the inner type calls `d.entry(|[e]| e.deserialize_map())` the
    /// facade returns the wrapped map directly.  Any other probe returns Miss.
    pub struct MapDeserializer<'de, M: MapAccess<'de>> {
        pub map: Option<M>,
        pub _marker: core::marker::PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> MapDeserializer<'de, M> {
        pub fn new(map: M) -> Self {
            Self {
                map: Some(map),
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'de, M: MapAccess<'de>> Deserializer<'de> for MapDeserializer<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type Entry = MapDeserializerEntry<'de, M>;

        async fn entry<const N: usize, F, Fut, R>(
            mut self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let entries: [_; N] = array::from_fn(|i| {
                let map = if i == N - 1 {
                    self.map.as_mut().map(|m| m.fork())
                } else {
                    self.map.take()
                };
                MapDeserializerEntry {
                    map,
                    _marker: PhantomData,
                }
            });
            f(entries).await
        }
    }

    pub struct MapDeserializerEntry<'de, M: MapAccess<'de>> {
        pub(crate) map: Option<M>,
        pub(crate) _marker: PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> Entry<'de> for MapDeserializerEntry<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type StrChunks = crate::Never<'de, M::Claim, M::Error>;
        type BytesChunks = crate::Never<'de, M::Claim, M::Error>;
        type Map = M;
        type Seq = crate::Never<'de, M::Claim, M::Error>;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            match self.map {
                Some(m) => Ok(Probe::Hit(m)),
                None => Ok(Probe::Miss),
            }
        }

        async fn deserialize_option<T: Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_value<T: Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        fn fork(&mut self) -> Self {
            Self {
                map: self.map.as_mut().map(|v| v.fork()),
                _marker: PhantomData,
            }
        }

        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            match self.map {
                Some(mut m) => loop {
                    match m
                        .next_kv::<1, _, _, ()>(|[ke]| async {
                            let (c, _, ()) = hit!(
                                ke.key((), |_k: &&'de str, [ve]| async {
                                    let c = ve.skip().await?;
                                    Ok(Probe::Hit((c, ())))
                                })
                                .await
                            );
                            Ok(Probe::Hit((c, ())))
                        })
                        .await?
                    {
                        Probe::Hit(Chunk::Done(c)) => return Ok(c),
                        Probe::Hit(Chunk::Data((new_m, ()))) => {
                            m = new_m;
                        }
                        Probe::Miss => panic!("unexpected Miss draining map"),
                    }
                },
                None => panic!("MapDeserializerEntry::skip called with no map"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Owned family
    // -----------------------------------------------------------------------

    /// Owned counterpart to `MapDeserializer`.
    pub struct MapDeserializerOwned<'s, M: MapAccessOwned<'s>> {
        pub map: M,
        pub _marker: core::marker::PhantomData<&'s ()>,
    }

    impl<'s, M: MapAccessOwned<'s>> MapDeserializerOwned<'s, M> {
        pub fn new(map: M) -> Self {
            Self {
                map,
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'s, M: MapAccessOwned<'s>> DeserializerOwned<'s> for MapDeserializerOwned<'s, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type Entry = MapDeserializerEntryOwned<'s, M>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<
                Output = Result<Probe<(<Self::Entry as EntryOwned<'s>>::Claim, R)>, Self::Error>,
            >,
        {
            let mut map = Some(self.map);
            let entries: [_; N] = array::from_fn(|i| {
                let map = if i == N - 1 {
                    map.as_mut().unwrap().fork()
                } else {
                    map.take().unwrap()
                };
                MapDeserializerEntryOwned {
                    map,
                    _marker: core::marker::PhantomData,
                }
            });
            f(entries).await
        }
    }

    pub struct MapDeserializerEntryOwned<'s, M: MapAccessOwned<'s>> {
        pub map: M,
        pub _marker: core::marker::PhantomData<&'s ()>,
    }

    impl<'s, M: MapAccessOwned<'s>> EntryOwned<'s> for MapDeserializerEntryOwned<'s, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type StrChunks = crate::Never<'s, M::Claim, M::Error>;
        type BytesChunks = crate::Never<'s, M::Claim, M::Error>;
        type Map = M;
        type Seq = crate::Never<'s, M::Claim, M::Error>;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_option<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_value<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            Ok(Probe::Hit(self.map))
        }

        fn fork(&mut self) -> Self {
            Self {
                map: self.map.fork(),
                _marker: PhantomData,
            }
        }

        async fn skip(mut self) -> Result<Self::Claim, Self::Error> {
            loop {
                match self
                    .map
                    .next_kv::<1, _, _, ()>(|[ke]| async {
                        let (c, _, ()) = hit!(
                            ke.key((), |_k: &crate::Skip, [ve]| async {
                                let c = ve.skip().await?;
                                Ok(Probe::Hit((c, ())))
                            })
                            .await
                        );
                        Ok(Probe::Hit((c, ())))
                    })
                    .await?
                {
                    Probe::Hit(Chunk::Done(c)) => return Ok(c),
                    Probe::Hit(Chunk::Data((new_m, ()))) => {
                        self.map = new_m;
                    }
                    Probe::Miss => panic!("unexpected Miss"),
                }
            }
        }
    }
}

// ===========================================================================
// Key facade — feed a real MapKeyEntryOwned into a DeserializeOwned expecting a map
// ===========================================================================

pub mod key_facade {
    //! Inversion-of-control bridge between a `MapAccessOwned` key loop and a
    //! `DeserializeOwned` impl that wants to drive a *full* deserializer.
    //!
    //! # Pattern
    //!
    //! ```ignore
    //! let comms = FutureComms::new();
    //! let fut = pin!(async {
    //!     let kd = KeyDeserializer::new(&comms);
    //!     let (c, v) = hit!(Foo::deserialize(kd, ()).await);
    //!     Ok(Probe::Hit((c, v)))
    //! });
    //! loop {
    //!     match hit!(map.next_kv(|[ke]| async {
    //!         let claim = hit!(comms.input(ke, &mut fut).await);
    //!         Ok(Probe::Hit((claim, ())))
    //!     }).await) {
    //!         Chunk::Data((m, ())) => map = m,
    //!         Chunk::Done(real_claim) => return comms.finish(real_claim, &mut fut).await,
    //!     }
    //! }
    //! ```

    use crate::{
        Chunk, Never, Probe,
        owned::{
            DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapKeyEntryOwned,
            MapValueEntryOwned,
        },
    };
    use core::{
        array,
        cell::Cell,
        future::{Future, poll_fn},
        marker::PhantomData,
        pin::Pin,
        task::{Poll, Waker},
    };
    use std::cell::RefCell;

    // -----------------------------------------------------------------------
    // State shared between KeyDeserializer (inner) and the outer loop
    // -----------------------------------------------------------------------

    /// State machine for the KE/Claim handshake channel.
    ///
    /// Transitions (outer = FutureComms, inner = CommsMap::next_kv):
    ///
    ///   Started → Awaiting          inner drives itself to Awaiting before parking
    ///   Awaiting → Proceed(KE)      outer deposits a KE and wakes inner
    ///   Proceed(KE) → Running       inner takes the KE (replaces with Running)
    ///   Running → Ran(Claim)        inner deposits the claim and wakes outer
    ///   Ran(Claim) → Awaiting       outer takes the claim (replaces with Awaiting)
    ///   Awaiting → Done(Claim)      outer signals exhaustion (finish)
    ///   Done(Claim) → …             inner sees Done, takes claim, returns Chunk::Done
    enum SlotState<KE, Claim> {
        /// Initial state; inner will drive to Awaiting before first park.
        Started,
        /// Inner is parked, waiting for a KE or Done.
        Awaiting,
        /// Outer deposited a KE; inner should take it.
        Proceed(KE),
        /// Inner took the KE and is currently processing it.
        Running,
        /// Inner deposited a Claim; outer should take it.
        Ran(Claim),
        /// Outer signaled map exhaustion with the final Claim.
        Done(Claim),
    }

    struct Inner<'s, KE: MapKeyEntryOwned<'s>> {
        /// Combined KE/Claim handshake slot.
        slot: Cell<SlotState<KE, KE::Claim>>,
        /// Waker for the inner task (waiting for Proceed or Done).
        inner_waker: Cell<Option<Waker>>,
        /// Waker for the outer task (waiting for Ran).
        outer_waker: Cell<Option<Waker>>,
        _marker: PhantomData<&'s ()>,
    }

    impl<'s, KE: MapKeyEntryOwned<'s>> Inner<'s, KE> {
        fn new() -> Self {
            Self {
                slot: Cell::new(SlotState::Started),
                inner_waker: Cell::new(None),
                outer_waker: Cell::new(None),
                _marker: PhantomData,
            }
        }
        
        fn get_slot_state(&self) -> SlotState<(), ()> {
            // This gets compiled down to a single CMP instruction, so there really isn't a window
            // where the intermediary state is visible.
            let s = self.slot.replace(SlotState::Started);
            let r = match s {
                SlotState::Started => SlotState::Started,
                SlotState::Awaiting => SlotState::Awaiting,
                SlotState::Proceed(_) => SlotState::Proceed(()),
                SlotState::Running => SlotState::Running,
                SlotState::Ran(_) => SlotState::Ran(()),
                SlotState::Done(_) => SlotState::Done(()),
            };
            self.slot.set(s);
            r
        }

    }

    // -----------------------------------------------------------------------
    // FutureComms — the bridge object held by the outer loop
    // -----------------------------------------------------------------------

    /// Bridge between the outer `next_kv` loop and the inner `DeserializeOwned`
    /// future.  Create one, then build a [`KeyDeserializer`] from a shared
    /// reference to it.
    pub struct FutureCommsStorage<'s, KE: MapKeyEntryOwned<'s>> {
        inner: Inner<'s, KE>,
    }

    impl<'s, KE: MapKeyEntryOwned<'s>> FutureCommsStorage<'s, KE> {
        pub fn new() -> Self {
            Self { inner: Inner::new() }
        }

        fn inner(&self) -> &Inner<'s, KE> {
            &self.inner
        }
    }

    pub struct FutureComms<'f, 's, KE: MapKeyEntryOwned<'s>, Fut> {
        inner: &'f Inner<'s, KE>,
        fut: &'f RefCell<Pin<&'f mut Option<Fut>>>,
        killed: Cell<bool>,
    }

    impl<'f, 's, KE: MapKeyEntryOwned<'s>, Fut> FutureComms<'f, 's, KE, Fut> {
        /// # Safety
        ///
        /// `fut` must be the future that was constructed using the
        /// [`KeyDeserializer`] built from `storage`.  Passing a future that
        /// communicates through a different `FutureCommsStorage` is undefined
        /// behavior: the two sides will read and write unrelated slots,
        /// leading to data races on the interior-mutable channel state.
        pub fn new_unsafe(storage: &'f FutureCommsStorage<'s, KE>, fut: &'f RefCell<Pin<&'f mut Option<Fut>>>) -> Self {
            Self { inner: storage.inner(), fut, killed: Cell::new(false) }
        }

        /// Mark this future to be dropped the next time it is safe to do so.
        pub fn kill(&self) {
            self.killed.set(true);
        }

        /// Returns `true` if [`kill`] was called and the future has been dropped
        /// (or is about to be).
        /// 
        /// [`Self::input`] checks this and returns a miss if true.
        pub fn is_killed(&self) -> bool {
            self.killed.get()
        }

        fn drop_fut(&self) {
            self.fut.borrow_mut().as_mut().set(None);
        }

        /// Feed one real `KE` into the inner future and wait for the inner
        /// future to consume it and return a `Claim`.
        ///
        /// Drive `fut` forward on each poll (it is the pinned inner future).
        /// Returns the `Claim` produced by the inner `next_kv` arm, which
        /// the outer caller should thread back to the real `next_kv`.
        pub async fn input<R, E>(
            &self,
            ke: KE,
        ) -> Result<Probe<KE::Claim>, E>
        where
            Fut: Future<Output = Result<Probe<(KE::Claim, R)>, E>>,
        {
            if self.killed.get() {
                self.drop_fut();
                return Ok(Probe::Miss);
            }

            // Drive inner from Started → Awaiting before depositing a KE, so
            // we never leave a stale KE in the slot from a prior half-step.
            if matches!(self.inner.get_slot_state(), SlotState::Started) {
                let early: Option<Result<Probe<(KE::Claim, R)>, E>> = poll_fn(|cx| {
                    self.inner.outer_waker.set(Some(cx.waker().clone()));
                    match self.fut.borrow_mut().as_mut().as_pin_mut().unwrap().poll(cx) {
                        Poll::Ready(res) => Poll::Ready(Some(res)),
                        Poll::Pending => {
                            if matches!(self.inner.get_slot_state(), SlotState::Awaiting) {
                                Poll::Ready(None)
                            } else {
                                Poll::Pending
                            }
                        }
                    }
                })
                .await;
                if let Some(res) = early {
                    return res.map(|p| p.map(|(c, _)| c));
                }
            }

            // If inner already ran and deposited a claim, return it immediately
            // without overwriting anything.
            if matches!(self.inner.get_slot_state(), SlotState::Ran(())) {
                if let SlotState::Ran(claim) = self.inner.slot.replace(SlotState::Awaiting) {
                    return Ok(Probe::Hit(claim));
                }
            }

            // Deposit the KE and wake inner.
            self.inner.slot.replace(SlotState::Proceed(ke));
            if let Some(w) = self.inner.inner_waker.take() {
                w.wake();
            }

            // Poll fut until inner deposits Ran(claim).
            poll_fn(|cx| {
                self.inner.outer_waker.set(Some(cx.waker().clone()));

                // Drive the inner future forward.
                match self.fut.borrow_mut().as_mut().as_pin_mut().unwrap().poll(cx) {
                    Poll::Ready(res) => {
                        return Poll::Ready(res.map(|p| p.map(|(c, _)| c)));
                    }
                    Poll::Pending => {}
                }

                // Check if inner deposited a claim.
                if matches!(self.inner.get_slot_state(), SlotState::Ran(())) {
                    if let SlotState::Ran(claim) = self.inner.slot.replace(SlotState::Awaiting) {
                        return Poll::Ready(Ok(Probe::Hit(claim)));
                    }
                }
                Poll::Pending
            })
            .await
        }

        /// Signal that the real map is exhausted (outer got `Chunk::Done`),
        /// then drive `fut` to completion and return its final output.
        pub async fn finish<R, E>(
            &self,
            done_claim: KE::Claim,
        ) -> Result<Probe<(KE::Claim, R)>, E>
        where
            Fut: Future<Output = Result<Probe<(KE::Claim, R)>, E>>,
        {
            self.inner.slot.replace(SlotState::Done(done_claim));
            if let Some(w) = self.inner.inner_waker.take() {
                w.wake();
            }

            // Drive fut to completion.
            poll_fn(|cx| self.fut.borrow_mut().as_mut().as_pin_mut().unwrap().poll(cx)).await
        }
    }

    // -----------------------------------------------------------------------
    // CommsMap — fake MapAccessOwned backed by FutureComms
    // -----------------------------------------------------------------------

    /// A `MapAccessOwned` whose `next_kv` suspends until the outer loop feeds
    /// it a real `KE` via `FutureComms::input`, or returns `Chunk::Done` when
    /// `FutureComms::finish` is called.
    pub struct CommsMap<'comms, 's, KE: MapKeyEntryOwned<'s>> {
        inner: &'comms Inner<'s, KE>,
    }

    impl<'comms, 's, KE: MapKeyEntryOwned<'s>> CommsMap<'comms, 's, KE> {
        fn new(inner: &'comms Inner<'s, KE>) -> Self {
            Self { inner }
        }
    }

    impl<'comms, 's, KE: MapKeyEntryOwned<'s>> MapAccessOwned<'s> for CommsMap<'comms, 's, KE> {
        type Error = KE::Error;
        type Claim = KE::Claim;
        type KeyEntry = CommsKeyEntry<'comms, 's, KE>;

        fn fork(&mut self) -> Self {
            Self { inner: self.inner }
        }

        async fn next_kv<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
        where
            F: FnMut([Self::KeyEntry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            // Signal to outer that we are ready, then wait for Proceed or Done.
            let slot: SlotState<KE, KE::Claim> = poll_fn(|cx| {
                match self.inner.get_slot_state() {
                    // Already have work to do — take it immediately.
                    SlotState::Proceed(()) | SlotState::Done(()) => {
                        Poll::Ready(self.inner.slot.replace(SlotState::Running))
                    }
                    // Transitional states that need us to signal Awaiting.
                    SlotState::Started | SlotState::Running => {
                        self.inner.slot.replace(SlotState::Awaiting);
                        if let Some(w) = self.inner.outer_waker.take() {
                            w.wake();
                        }
                        self.inner.inner_waker.set(Some(cx.waker().clone()));
                        Poll::Pending
                    }
                    // Claim deposited or already Awaiting — just park.
                    SlotState::Ran(()) | SlotState::Awaiting => {
                        self.inner.inner_waker.set(Some(cx.waker().clone()));
                        Poll::Pending
                    }
                }
            })
            .await;

            match slot {
                SlotState::Done(claim) => {
                    return Ok(Probe::Hit(Chunk::Done(claim)));
                }
                SlotState::Proceed(ke) => {
                    // Build N entry handles (all backed by the same ke via fork).
                    let mut ke_opt = Some(ke);
                    let entries: [CommsKeyEntry<'comms, 's, KE>; N] = array::from_fn(|i| {
                        let k = if i == N - 1 {
                            ke_opt.as_mut().unwrap().fork()
                        } else {
                            ke_opt.take().unwrap()
                        };
                        CommsKeyEntry { ke: k, inner: self.inner }
                    });

                    match f(entries).await? {
                        Probe::Hit((claim, r)) => {
                            // Deposit claim and wake outer.
                            self.inner.slot.replace(SlotState::Ran(claim));
                            if let Some(w) = self.inner.outer_waker.take() {
                                w.wake();
                            }
                            // Suspend until next Proceed or Done arrives.
                            poll_fn(|cx| {
                                match self.inner.get_slot_state() {
                                    SlotState::Proceed(()) | SlotState::Done(()) => Poll::Ready(()),
                                    _ => {
                                        self.inner.inner_waker.set(Some(cx.waker().clone()));
                                        Poll::Pending
                                    }
                                }
                            })
                            .await;
                            Ok(Probe::Hit(Chunk::Data((CommsMap { inner: self.inner }, r))))
                        }
                        Probe::Miss => Ok(Probe::Miss),
                    }
                }
                // Reached only if outer called finish before inner first ran —
                // inner transitioned Awaiting → Running but slot was Done.
                SlotState::Awaiting | SlotState::Running | SlotState::Ran(_) | SlotState::Started => {
                    unreachable!()
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // CommsKeyEntry — forwards ke probes to the real KE
    // -----------------------------------------------------------------------

    pub struct CommsKeyEntry<'comms, 's, KE: MapKeyEntryOwned<'s>> {
        ke: KE,
        inner: &'comms Inner<'s, KE>,
    }

    impl<'comms, 's, KE: MapKeyEntryOwned<'s>> MapKeyEntryOwned<'s> for CommsKeyEntry<'comms, 's, KE> {
        type Error = KE::Error;
        type Claim = KE::Claim;
        type ValueEntry = CommsValueEntry<'comms, 's, KE>;

        fn fork(&mut self) -> Self {
            Self { ke: self.ke.fork(), inner: self.inner }
        }

        async fn key<K: DeserializeOwned<'s, KExtra>, KExtra, const N: usize, F, Fut, R>(
            self,
            extra: KExtra,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, K, R)>, Self::Error>
        where
            F: FnMut(&K, [Self::ValueEntry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let inner = self.inner;
            self.ke
                .key::<K, KExtra, N, _, _, R>(extra, |k, ves| {
                    let wrapped: [CommsValueEntry<'comms, 's, KE>; N] =
                        ves.map(|ve| CommsValueEntry { ve, inner });
                    f(k, wrapped)
                })
                .await
        }
    }

    pub struct CommsValueEntry<'comms, 's, KE: MapKeyEntryOwned<'s>> {
        ve: KE::ValueEntry,
        inner: &'comms Inner<'s, KE>,
    }

    impl<'comms, 's, KE: MapKeyEntryOwned<'s>> MapValueEntryOwned<'s>
        for CommsValueEntry<'comms, 's, KE>
    {
        type Error = KE::Error;
        type Claim = KE::Claim;

        fn fork(&mut self) -> Self {
            Self { ve: self.ve.fork(), inner: self.inner }
        }

        async fn value<V: DeserializeOwned<'s, Extra>, Extra>(
            self,
            extra: Extra,
        ) -> Result<Probe<(Self::Claim, V)>, Self::Error> {
            self.ve.value::<V, Extra>(extra).await
        }

        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            self.ve.skip().await
        }
    }

    // -----------------------------------------------------------------------
    // KeyDeserializer — top-level Deserializer wrapping FutureComms
    // -----------------------------------------------------------------------

    /// The `DeserializerOwned` handed to the inner `DeserializeOwned` impl.
    /// All entry probes return `Miss` except `deserialize_map`, which returns
    /// the `CommsMap` fake map.
    pub struct KeyDeserializer<'comms, 's, KE: MapKeyEntryOwned<'s>> {
        comms: &'comms FutureCommsStorage<'s, KE>,
    }

    impl<'comms, 's, KE: MapKeyEntryOwned<'s>> KeyDeserializer<'comms, 's, KE> {
        pub fn new(comms: &'comms FutureCommsStorage<'s, KE>) -> Self {
            Self { comms }
        }
    }

    impl<'comms, 's, KE: MapKeyEntryOwned<'s>> DeserializerOwned<'s>
        for KeyDeserializer<'comms, 's, KE>
    {
        type Error = KE::Error;
        type Claim = KE::Claim;
        type Entry = KeyDeserializerEntry<'comms, 's, KE>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<
                Output = Result<
                    Probe<(<Self::Entry as EntryOwned<'s>>::Claim, R)>,
                    Self::Error,
                >,
            >,
        {
            let comms = self.comms;
            let entries: [KeyDeserializerEntry<'comms, 's, KE>; N] =
                array::from_fn(|_| KeyDeserializerEntry { comms });
            f(entries).await
        }
    }

    pub struct KeyDeserializerEntry<'comms, 's, KE: MapKeyEntryOwned<'s>> {
        comms: &'comms FutureCommsStorage<'s, KE>,
    }

    impl<'comms, 's, KE: MapKeyEntryOwned<'s>> EntryOwned<'s>
        for KeyDeserializerEntry<'comms, 's, KE>
    {
        type Error = KE::Error;
        type Claim = KE::Claim;
        type StrChunks = Never<'s, KE::Claim, KE::Error>;
        type BytesChunks = Never<'s, KE::Claim, KE::Error>;
        type Map = CommsMap<'comms, 's, KE>;
        type Seq = Never<'s, KE::Claim, KE::Error>;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_option<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_value<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            Ok(Probe::Hit(CommsMap::new(self.comms.inner())))
        }

        fn fork(&mut self) -> Self {
            Self { comms: self.comms }
        }

        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            panic!("KeyDeserializerEntry::skip — no real token to skip")
        }
    }
}

// ===========================================================================
// Tag-filtering map facades (used by #[strede(tag)] derive output)
// ===========================================================================

pub mod tag_facade {
    use super::Match;
    use crate::{
        Chunk, Probe, Skip,
        borrow::{Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry},
        hit, or_miss,
        owned::{
            DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapKeyEntryOwned,
            MapValueEntryOwned,
        },
    };
    use core::{cell::RefCell, future::Future, marker::PhantomData};
    use std::cell::Cell;

    pub use super::map_facade::{
        MapDeserializer, MapDeserializerEntry, MapDeserializerEntryOwned, MapDeserializerOwned,
    };

    // -----------------------------------------------------------------------
    // Borrow family
    // -----------------------------------------------------------------------

    /// A `Deserializer` wrapper that wraps any `Deserializer` and produces
    /// `TagFilteredMapEntry` handles. When those handles' `deserialize_map`
    /// is called the resulting `MapAccess` is wrapped in `TagFilteredMap`.
    pub struct TagFilteredMapDeserializer<'de, D: Deserializer<'de>> {
        pub inner: D,
        pub tag_key: &'static str,
        pub _marker: PhantomData<&'de ()>,
    }

    impl<'de, D: Deserializer<'de>> TagFilteredMapDeserializer<'de, D> {
        pub fn new(inner: D, tag_key: &'static str) -> Self {
            Self { inner, tag_key, _marker: PhantomData }
        }
    }

    impl<'de, D: Deserializer<'de>> Deserializer<'de> for TagFilteredMapDeserializer<'de, D> {
        type Error = D::Error;
        type Claim = D::Claim;
        type Entry = TagFilteredMapEntry<'de, D::Entry>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<
                Output = Result<Probe<(<Self::Entry as Entry<'de>>::Claim, R)>, Self::Error>,
            >,
        {
            let tag_key = self.tag_key;
            self.inner
                .entry::<N, _, _, R>(|entries| {
                    let wrapped = entries.map(|e| TagFilteredMapEntry {
                        inner: e,
                        tag_key,
                        _marker: PhantomData,
                    });
                    f(wrapped)
                })
                .await
        }
    }

    /// An `Entry` wrapper that delegates all probes to the inner entry, but
    /// wraps map accesses in `TagFilteredMap`.
    pub struct TagFilteredMapEntry<'de, E: Entry<'de>> {
        pub inner: E,
        pub tag_key: &'static str,
        pub _marker: PhantomData<&'de ()>,
    }

    impl<'de, E: Entry<'de>> Entry<'de> for TagFilteredMapEntry<'de, E> {
        type Error = E::Error;
        type Claim = E::Claim;
        type StrChunks = E::StrChunks;
        type BytesChunks = E::BytesChunks;
        type Map = TagFilteredMap<'de, E::Map>;
        type Seq = E::Seq;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
            self.inner.deserialize_bool().await
        }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
            self.inner.deserialize_u8().await
        }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
            self.inner.deserialize_u16().await
        }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
            self.inner.deserialize_u32().await
        }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
            self.inner.deserialize_u64().await
        }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
            self.inner.deserialize_u128().await
        }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
            self.inner.deserialize_i8().await
        }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
            self.inner.deserialize_i16().await
        }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
            self.inner.deserialize_i32().await
        }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
            self.inner.deserialize_i64().await
        }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
            self.inner.deserialize_i128().await
        }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
            self.inner.deserialize_f32().await
        }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
            self.inner.deserialize_f64().await
        }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
            self.inner.deserialize_char().await
        }
        async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
            self.inner.deserialize_str().await
        }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
            self.inner.deserialize_str_chunks().await
        }
        async fn deserialize_bytes(
            self,
        ) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
            self.inner.deserialize_bytes().await
        }
        async fn deserialize_bytes_chunks(
            self,
        ) -> Result<Probe<Self::BytesChunks>, Self::Error> {
            self.inner.deserialize_bytes_chunks().await
        }
        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            let tag_key = self.tag_key;
            self.inner.deserialize_map().await.map(|p| match p {
                Probe::Hit(m) => Probe::Hit(TagFilteredMap::new(m, tag_key)),
                Probe::Miss => Probe::Miss,
            })
        }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
            self.inner.deserialize_seq().await
        }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
            self.inner.deserialize_null().await
        }
        async fn deserialize_option<T: crate::Deserialize<'de, Extra>, Extra>(
            self,
            extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            self.inner.deserialize_option::<T, Extra>(extra).await
        }
        async fn deserialize_value<T: crate::Deserialize<'de, Extra>, Extra>(
            self,
            extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            self.inner.deserialize_value::<T, Extra>(extra).await
        }
        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_key: self.tag_key,
                _marker: PhantomData,
            }
        }
        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            self.inner.skip().await
        }
    }

    /// A `MapAccess` wrapper that skips one specific key (the tag field) and
    /// passes everything else through unchanged.
    pub struct TagFilteredMap<'de, M: MapAccess<'de>> {
        pub inner: M,
        pub tag_key: &'static str,
        pub _marker: core::marker::PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> TagFilteredMap<'de, M> {
        pub fn new(inner: M, tag_key: &'static str) -> Self {
            Self {
                inner,
                tag_key,
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'de, M: MapAccess<'de>> MapAccess<'de> for TagFilteredMap<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type KeyEntry = M::KeyEntry;

        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_key: self.tag_key,
                _marker: core::marker::PhantomData,
            }
        }

        async fn next_kv<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
        where
            F: FnMut([Self::KeyEntry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let tag_key = self.tag_key;
            let mut inner = self.inner;
            loop {
                let tag_check = inner
                    .fork()
                    .next_kv::<1, _, _, bool>(|[ke]| async move {
                        let result = ke
                            .key((), |k: &&'de str, [ve]| {
                                let is_tag = *k == tag_key;
                                async move {
                                    let c = ve.skip().await?;
                                    Ok(Probe::Hit((c, is_tag)))
                                }
                            })
                            .await?;
                        match result {
                            Probe::Hit((c, _, is_tag)) => Ok(Probe::Hit((c, is_tag))),
                            Probe::Miss => Ok(Probe::Miss),
                        }
                    })
                    .await?;

                match tag_check {
                    Probe::Hit(Chunk::Done(_)) => {
                        // Map exhausted via fork — drain real inner to get the claim.
                        return inner
                            .next_kv::<1, _, _, ()>(|[_ke]| async move { Ok(Probe::Miss) })
                            .await
                            .map(|p| match p {
                                Probe::Hit(Chunk::Done(c)) => Probe::Hit(Chunk::Done(c)),
                                Probe::Hit(Chunk::Data(_)) => Probe::Miss,
                                Probe::Miss => Probe::Miss,
                            });
                    }
                    Probe::Hit(Chunk::Data((_, true))) => {
                        // This key IS the tag — skip it on the real inner and loop.
                        let skipped = inner
                            .next_kv::<1, _, _, ()>(|[ke]| async {
                                let (c, _, ()) = hit!(
                                    ke.key((), |_k: &&'de str, [ve]| async {
                                        let c = ve.skip().await?;
                                        Ok(Probe::Hit((c, ())))
                                    })
                                    .await
                                );
                                Ok(Probe::Hit((c, ())))
                            })
                            .await?;
                        match skipped {
                            Probe::Hit(Chunk::Data((new_inner, ()))) => {
                                inner = new_inner;
                            }
                            Probe::Hit(Chunk::Done(c)) => return Ok(Probe::Hit(Chunk::Done(c))),
                            Probe::Miss => return Ok(Probe::Miss),
                        }
                    }
                    Probe::Hit(Chunk::Data((_, false))) | Probe::Miss => {
                        // Not the tag — delegate to f on real inner.
                        return inner.next_kv(|keys| f(keys)).await.map(|p| match p {
                            Probe::Hit(Chunk::Data((new_inner, r))) => Probe::Hit(Chunk::Data((
                                TagFilteredMap {
                                    inner: new_inner,
                                    tag_key,
                                    _marker: core::marker::PhantomData,
                                },
                                r,
                            ))),
                            Probe::Hit(Chunk::Done(c)) => Probe::Hit(Chunk::Done(c)),
                            Probe::Miss => Probe::Miss,
                        });
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Owned family
    // -----------------------------------------------------------------------

    /// Owned counterpart to `TagFilteredMap`.
    pub struct TagFilteredMapOwned<'s, 'v, M: MapAccessOwned<'s>, V = Skip, TagExtra: Copy = ()> {
        pub inner: M,
        pub tag_key: &'static str,
        pub tag_value: &'v Cell<Option<V>>,
        pub extra: TagExtra,
        pub _marker: core::marker::PhantomData<fn() -> &'s ()>,
    }

    impl<'s, 'v, M: MapAccessOwned<'s>, V, TagExtra: Copy> TagFilteredMapOwned<'s, 'v, M, V, TagExtra> {
        pub fn new(
            inner: M,
            tag_key: &'static str,
            tag_value: &'v Cell<Option<V>>,
            extra: TagExtra,
        ) -> Self {
            Self {
                inner,
                tag_key,
                tag_value,
                extra,
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'s, 'v, M: MapAccessOwned<'s>> TagFilteredMapOwned<'s, 'v, M, Skip, ()> {
        pub fn new_skip(
            inner: M,
            tag_key: &'static str,
            tag_value: &'v Cell<Option<Skip>>,
        ) -> Self {
            Self::new(inner, tag_key, tag_value, ())
        }
    }

    impl<'s, 'v, M: MapAccessOwned<'s>, V: DeserializeOwned<'s, TagExtra>, TagExtra: Copy>
        MapAccessOwned<'s> for TagFilteredMapOwned<'s, 'v, M, V, TagExtra>
    {
        type Error = M::Error;
        type Claim = M::Claim;
        type KeyEntry = M::KeyEntry;

        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_key: self.tag_key,
                tag_value: self.tag_value,
                extra: self.extra,
                _marker: core::marker::PhantomData,
            }
        }

        async fn next_kv<const N: usize, F, Fut, R>(
            mut self,
            f: F,
        ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
        where
            F: FnMut([Self::KeyEntry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let f = RefCell::new(f);
            let f = &f;
            let extra = self.extra;
            // If we're only looking for the tag we have to do a special case
            if N == 0 {
                let tag_key = self.tag_key;
                let result = hit!(
                    self.inner
                        .next_kv(async |[ke]| {
                            let (c, _, v) = hit!(
                                ke.key(tag_key, |_: &Match, [ve]| async move {
                                    let (c, v) = hit!(ve.value::<V, TagExtra>(extra).await);
                                    Ok(Probe::Hit((c, v)))
                                })
                                .await
                            );
                            Ok(Probe::Hit((c, v)))
                        })
                        .await
                );
                let (next_inner, v) = or_miss!(result.data());
                self.tag_key = "";
                self.tag_value.set(Some(v));
                self.inner = next_inner;
            }

            // Try to read the tag until we do
            while !self.tag_key.is_empty() {
                let tag_key = self.tag_key;
                let result = hit!(self.inner.next_kv(|mut keys| async move {
                    let ke = keys[0].fork();
                    let f_fut = f.borrow_mut()(keys);
                    let (c, outcome) = hit!(crate::select_probe! {
                        // Arm 0: check if this key is the tag on a fork.
                        async move {
                            let (c, _, v) = hit!(ke.key(tag_key, |_: &Match, [ve]| async move {
                                kill!(1);
                                let (c, v) = hit!(ve.value::<V, TagExtra>(extra).await);
                                Ok(Probe::Hit((c, v)))
                            }).await);
                            Ok(Probe::Hit((c, (Some(v), None::<R>))))
                        },
                        // Arm 1: delegate to f.
                        async move {
                            let (c, r) = hit!(f_fut.await);
                            Ok(Probe::Hit((c, (None::<V>, Some(r)))))
                        },
                    });
                    Ok(Probe::Hit((c, outcome)))
                }).await);
                match or_miss!(result.data()) {
                    (next_inner, (Some(v), None)) => {
                        self.tag_key = "";
                        self.tag_value.set(Some(v));
                        self.inner = next_inner;
                    }
                    (next_inner, (None, Some(r))) => {
                        self.inner = next_inner;
                        return Ok(Probe::Hit(Chunk::Data((self, r))));
                    }
                    _ => unreachable!(),
                }
            }

            // Read everything else normally; extract tag_value before consuming self.
            let tag_key = self.tag_key;
            let tag_value = self.tag_value;
            return self
                .inner
                .next_kv(|keys| f.borrow_mut()(keys))
                .await
                .map(|p| match p {
                    Probe::Hit(Chunk::Data((new_inner, r))) => Probe::Hit(Chunk::Data((
                        Self {
                            inner: new_inner,
                            tag_key,
                            tag_value,
                            extra,
                            _marker: PhantomData,
                        },
                        r,
                    ))),
                    Probe::Hit(Chunk::Done(c)) => Probe::Hit(Chunk::Done(c)),
                    Probe::Miss => Probe::Miss,
                });
        }
    }

    /// A `DeserializerOwned` wrapper that wraps any `DeserializerOwned` and produces
    /// `TagFilteredMapEntryOwned` handles. When those handles' `deserialize_map`
    /// is called the resulting `MapAccessOwned` is wrapped in `TagFilteredMapOwned`.
    pub struct TagFilteredMapDeserializerOwned<
        's,
        'v,
        D: DeserializerOwned<'s>,
        V = Skip,
        TagExtra: Copy = (),
    > {
        pub inner: D,
        pub tag_key: &'static str,
        pub tag_value: &'v Cell<Option<V>>,
        pub extra: TagExtra,
        pub _marker: PhantomData<fn() -> &'s ()>,
    }

    impl<'s, 'v, D: DeserializerOwned<'s>, V, TagExtra: Copy>
        TagFilteredMapDeserializerOwned<'s, 'v, D, V, TagExtra>
    {
        pub fn new(
            inner: D,
            tag_key: &'static str,
            tag_value: &'v Cell<Option<V>>,
            extra: TagExtra,
        ) -> Self {
            Self { inner, tag_key, tag_value, extra, _marker: PhantomData }
        }
    }

    impl<'s, 'v, D: DeserializerOwned<'s>> TagFilteredMapDeserializerOwned<'s, 'v, D, Skip, ()> {
        pub fn new_skip(
            inner: D,
            tag_key: &'static str,
            tag_value: &'v Cell<Option<Skip>>,
        ) -> Self {
            Self::new(inner, tag_key, tag_value, ())
        }
    }

    impl<'s, 'v, D: DeserializerOwned<'s>, V: DeserializeOwned<'s, TagExtra>, TagExtra: Copy>
        DeserializerOwned<'s> for TagFilteredMapDeserializerOwned<'s, 'v, D, V, TagExtra>
    {
        type Error = D::Error;
        type Claim = D::Claim;
        type Entry = TagFilteredMapEntryOwned<'s, 'v, D::Entry, V, TagExtra>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<
                Output = Result<
                    Probe<(<Self::Entry as EntryOwned<'s>>::Claim, R)>,
                    Self::Error,
                >,
            >,
        {
            let tag_key = self.tag_key;
            let tag_value = self.tag_value;
            let extra = self.extra;
            self.inner
                .entry::<N, _, _, R>(|entries| {
                    let wrapped = entries.map(|e| TagFilteredMapEntryOwned {
                        inner: e,
                        tag_key,
                        tag_value,
                        extra,
                        _marker: PhantomData,
                    });
                    f(wrapped)
                })
                .await
        }
    }

    /// An `EntryOwned` wrapper that delegates all probes to the inner entry, but
    /// wraps map accesses in `TagFilteredMapOwned`.
    pub struct TagFilteredMapEntryOwned<
        's,
        'v,
        E: EntryOwned<'s>,
        V = Skip,
        TagExtra: Copy = (),
    > {
        pub inner: E,
        pub tag_key: &'static str,
        pub tag_value: &'v Cell<Option<V>>,
        pub extra: TagExtra,
        pub _marker: PhantomData<fn() -> &'s ()>,
    }

    impl<'s, 'v, E: EntryOwned<'s>, V: DeserializeOwned<'s, TagExtra>, TagExtra: Copy>
        EntryOwned<'s> for TagFilteredMapEntryOwned<'s, 'v, E, V, TagExtra>
    {
        type Error = E::Error;
        type Claim = E::Claim;
        type StrChunks = E::StrChunks;
        type BytesChunks = E::BytesChunks;
        type Map = TagFilteredMapOwned<'s, 'v, E::Map, V, TagExtra>;
        type Seq = E::Seq;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
            self.inner.deserialize_bool().await
        }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
            self.inner.deserialize_u8().await
        }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
            self.inner.deserialize_u16().await
        }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
            self.inner.deserialize_u32().await
        }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
            self.inner.deserialize_u64().await
        }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
            self.inner.deserialize_u128().await
        }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
            self.inner.deserialize_i8().await
        }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
            self.inner.deserialize_i16().await
        }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
            self.inner.deserialize_i32().await
        }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
            self.inner.deserialize_i64().await
        }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
            self.inner.deserialize_i128().await
        }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
            self.inner.deserialize_f32().await
        }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
            self.inner.deserialize_f64().await
        }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
            self.inner.deserialize_char().await
        }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
            self.inner.deserialize_str_chunks().await
        }
        async fn deserialize_bytes_chunks(
            self,
        ) -> Result<Probe<Self::BytesChunks>, Self::Error> {
            self.inner.deserialize_bytes_chunks().await
        }
        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            let tag_key = self.tag_key;
            let tag_value = self.tag_value;
            let extra = self.extra;
            self.inner.deserialize_map().await.map(|p| match p {
                Probe::Hit(m) => {
                    Probe::Hit(TagFilteredMapOwned::new(m, tag_key, tag_value, extra))
                }
                Probe::Miss => Probe::Miss,
            })
        }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
            self.inner.deserialize_seq().await
        }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
            self.inner.deserialize_null().await
        }
        async fn deserialize_option<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            self.inner.deserialize_option::<T, Extra>(extra).await
        }
        async fn deserialize_value<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            self.inner.deserialize_value::<T, Extra>(extra).await
        }
        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_key: self.tag_key,
                tag_value: self.tag_value,
                extra: self.extra,
                _marker: PhantomData,
            }
        }
        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            self.inner.skip().await
        }
    }
}
