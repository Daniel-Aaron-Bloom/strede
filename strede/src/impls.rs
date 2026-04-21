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
    BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned,
    SeqAccessOwned, SeqEntryOwned, StrAccessOwned,
};
use crate::{Chunk, DeserializeError, Probe, StrAccess, hit, or_miss, select_probe};

// ===========================================================================
// Core impls - no features required
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

impl DeserializeOwned for () {
    async fn deserialize_owned<D: DeserializerOwned>(
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

impl DeserializeOwned for usize {
    async fn deserialize_owned<D: DeserializerOwned>(
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

impl DeserializeOwned for isize {
    async fn deserialize_owned<D: DeserializerOwned>(
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

impl<T: ?Sized> DeserializeOwned for PhantomData<T> {
    async fn deserialize_owned<D: DeserializerOwned>(
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
/// Always returns `Probe::Hit(Skip)` on well-formed input - it never misses,
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

impl DeserializeOwned for Skip {
    async fn deserialize_owned<D: DeserializerOwned>(
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

impl<'a, T: Copy, const N: usize> DeserializeOwned<[(&'a str, T); N]> for MatchVals<T> {
    async fn deserialize_owned<D: DeserializerOwned>(
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

impl<'a, T: Copy, const N: usize> DeserializeOwned<[(&'a [u8], T); N]> for MatchVals<T> {
    async fn deserialize_owned<D: DeserializerOwned>(
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
        let pairs = extra.map(|k| {
            let idx = i;
            i += 1;
            (k, idx)
        });
        let probe =
            <MatchVals<usize> as Deserialize<'de, [(&'a str, usize); N]>>::deserialize(d, pairs)
                .await?;
        Ok(probe)
    }
}

impl<'de, 'a, const N: usize> Deserialize<'de, [&'a [u8]; N]> for MatchVals<usize> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut i = 0usize;
        let pairs = extra.map(|k| {
            let idx = i;
            i += 1;
            (k, idx)
        });
        let probe =
            <MatchVals<usize> as Deserialize<'de, [(&'a [u8], usize); N]>>::deserialize(d, pairs)
                .await?;
        Ok(probe)
    }
}

impl<'a, const N: usize> DeserializeOwned<[&'a str; N]> for MatchVals<usize> {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: [&'a str; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut i = 0usize;
        let pairs = extra.map(|k| {
            let idx = i;
            i += 1;
            (k, idx)
        });
        let probe =
            <MatchVals<usize> as DeserializeOwned<[(&'a str, usize); N]>>::deserialize_owned(
                d, pairs,
            )
            .await?;
        Ok(probe)
    }
}

impl<'a, const N: usize> DeserializeOwned<[&'a [u8]; N]> for MatchVals<usize> {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut i = 0usize;
        let pairs = extra.map(|k| {
            let idx = i;
            i += 1;
            (k, idx)
        });
        let probe =
            <MatchVals<usize> as DeserializeOwned<[(&'a [u8], usize); N]>>::deserialize_owned(
                d, pairs,
            )
            .await?;
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
impl<T, F, Extra> DeserializeOwned<(F, Extra)> for UnwrapOrElse<T>
where
    T: DeserializeOwned<Extra>,
    F: AsyncFnOnce() -> T,
    Extra: Copy,
{
    async fn deserialize_owned<D: DeserializerOwned>(
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
/// - `&str` / `&[u8]` - single string/bytes to match.
/// - `[&str; N]` / `[&[u8]; N]` - match any of N strings/bytes.
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
///     @miss => Ok(Probe::Miss),
/// })
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Match;

impl<'de, 'a> Deserialize<'de, &'a str> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: &'a str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a str, Match); 1]>>::deserialize(
            d,
            [(extra, Match)],
        )
        .await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'de, 'a> Deserialize<'de, &'a [u8]> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: &'a [u8],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a [u8], Match); 1]>>::deserialize(
            d,
            [(extra, Match)],
        )
        .await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'de, 'a, const N: usize> Deserialize<'de, [&'a str; N]> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [&'a str; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a str, Match); N]>>::deserialize(
            d,
            extra.map(|k| (k, Match)),
        )
        .await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'de, 'a, const N: usize> Deserialize<'de, [&'a [u8]; N]> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe = <MatchVals<Match> as Deserialize<'de, [(&'a [u8], Match); N]>>::deserialize(
            d,
            extra.map(|k| (k, Match)),
        )
        .await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'a> DeserializeOwned<&'a str> for Match {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: &'a str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe =
            <MatchVals<Match> as DeserializeOwned<[(&'a str, Match); 1]>>::deserialize_owned(
                d,
                [(extra, Match)],
            )
            .await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'a> DeserializeOwned<&'a [u8]> for Match {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: &'a [u8],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe =
            <MatchVals<Match> as DeserializeOwned<[(&'a [u8], Match); 1]>>::deserialize_owned(
                d,
                [(extra, Match)],
            )
            .await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'a, const N: usize> DeserializeOwned<[&'a str; N]> for Match {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: [&'a str; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe =
            <MatchVals<Match> as DeserializeOwned<[(&'a str, Match); N]>>::deserialize_owned(
                d,
                extra.map(|k| (k, Match)),
            )
            .await?;
        Ok(probe.map(|(claim, MatchVals(m))| (claim, m)))
    }
}

impl<'a, const N: usize> DeserializeOwned<[&'a [u8]; N]> for Match {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: [&'a [u8]; N],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let probe =
            <MatchVals<Match> as DeserializeOwned<[(&'a [u8], Match); N]>>::deserialize_owned(
                d,
                extra.map(|k| (k, Match)),
            )
            .await?;
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

        impl<T: DeserializeOwned> DeserializeOwned for $wrapper<T> {
            async fn deserialize_owned<D: DeserializerOwned>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(T::deserialize_owned(d, ())
                    .await?
                    .map(|(c, v)| (c, $wrapper(v))))
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

impl<T: DeserializeOwned + Copy> DeserializeOwned for Cell<T> {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize_owned(d, ())
            .await?
            .map(|(c, v)| (c, Cell::new(v))))
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

impl<T: DeserializeOwned> DeserializeOwned for RefCell<T> {
    async fn deserialize_owned<D: DeserializerOwned>(
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

        impl DeserializeOwned for $nonzero {
            async fn deserialize_owned<D: DeserializerOwned>(
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

// NonZeroUsize / NonZeroIsize - delegate to usize/isize

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

impl DeserializeOwned for NonZeroUsize {
    async fn deserialize_owned<D: DeserializerOwned>(
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

impl DeserializeOwned for NonZeroIsize {
    async fn deserialize_owned<D: DeserializerOwned>(
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
            Ok(Probe::Hit((
                claim,
                buf.into_inner()
                    .unwrap_or_else(|_| panic!("all N elements pushed")),
            )))
        })
        .await
    }
}

impl<T: DeserializeOwned, const N: usize> DeserializeOwned for [T; N] {
    async fn deserialize_owned<D: DeserializerOwned>(
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
            Ok(Probe::Hit((
                claim,
                buf.into_inner()
                    .unwrap_or_else(|_| panic!("all N elements pushed")),
            )))
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

        impl<$($T: DeserializeOwned),+> DeserializeOwned for ($($T,)+) {
            async fn deserialize_owned<D: DeserializerOwned>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
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
// IP address and socket address types - string-parsed
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
                        @miss => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }

        impl DeserializeOwned for $ty {
            async fn deserialize_owned<D: DeserializerOwned>(
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

impl DeserializeOwned for core::time::Duration {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        use crate::owned::{self, MapKeyProbeOwned, MapValueProbeOwned};

        d.entry(|[e]| async {
            let map = hit!(e.deserialize_map().await);

            let arms = crate::map_arms! {
                |kp: owned::KP<D>| kp.deserialize_key::<crate::Match, &str>("secs")
                => |vp: owned::VP2<D>, k: crate::Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<u64, _>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
                |kp: owned::KP<D>| kp.deserialize_key::<crate::Match, &str>("nanos")
                => |vp: owned::VP2<D>, k: crate::Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<u32, _>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
            };

            let (claim, crate::map_outputs!(opt_secs, opt_nanos)) = hit!(map.iterate(arms).await);

            // Miss on empty (neither field present).
            if opt_secs.is_none() && opt_nanos.is_none() {
                return Ok(Probe::Miss);
            }

            let secs = opt_secs.map(|(_, v)| v).unwrap_or(0);
            let nanos = opt_nanos.map(|(_, v)| v).unwrap_or(0);

            Ok(Probe::Hit((claim, Self::new(secs, nanos))))
        })
        .await
    }
}

// ===========================================================================
// alloc impls
// ===========================================================================

#[cfg(feature = "alloc")]
mod alloc_impls {
    use crate::borrow::{
        Deserialize, Deserializer, Entry, MapAccess, MapArmStack, MapKeyProbe, MapValueProbe,
        SeqAccess, SeqEntry, VC, VP,
    };
    use crate::owned::{
        BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned,
        MapValueProbeOwned, SeqAccessOwned, SeqEntryOwned, StrAccessOwned,
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

    impl DeserializeOwned for String {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl<T: DeserializeOwned> DeserializeOwned for Vec<T> {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl<T: DeserializeOwned> DeserializeOwned for Box<T> {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize_owned(d, ())
                .await?
                .map(|(c, v)| (c, Box::new(v))))
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

    impl DeserializeOwned for Box<str> {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl<T: DeserializeOwned> DeserializeOwned for Box<[T]> {
        async fn deserialize_owned<D: DeserializerOwned>(
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

            impl<$T: DeserializeOwned $($(+ $bound)+)?> DeserializeOwned for $ty<$T> {
                async fn deserialize_owned<D: DeserializerOwned>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
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

    struct CollectBTreeMapArm<K, V> {
        pending_key: Option<K>,
        out: BTreeMap<K, V>,
    }

    impl<K: Ord, V> CollectBTreeMapArm<K, V> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: BTreeMap::new(),
            }
        }
    }

    impl<'de, KP, K, V> MapArmStack<'de, KP> for CollectBTreeMapArm<K, V>
    where
        KP: MapKeyProbe<'de>,
        K: Deserialize<'de> + Ord,
        V: Deserialize<'de>,
    {
        const SIZE: usize = 1;
        type Outputs = BTreeMap<K, V>;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            0
        }

        type RaceState = ();
        type RaceDone = ();
        fn init_race(&mut self, _kp: KP) -> ((), ()) {
            unreachable!()
        }
        fn race_all_done(_done: &()) -> bool {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _done: &mut (),
            _arm_index: usize,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _arm_index: usize, _vp: VP<'de, KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K, _>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }

        async fn dispatch_value(
            &mut self,
            _arm_index: usize,
            vp: VP<'de, KP>,
        ) -> Result<Probe<(VC<'de, KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch_value without pending key");
            match vp.deserialize_value::<V, _>(()).await? {
                Probe::Hit((vc, v)) => {
                    self.out.insert(k, v);
                    Ok(Probe::Hit((vc, ())))
                }
                Probe::Miss => Ok(Probe::Miss),
            }
        }

        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::take(&mut self.out)
        }
    }

    impl<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>> Deserialize<'de> for BTreeMap<K, V> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let map = hit!(e.deserialize_map().await);
                let collector = CollectBTreeMapArm::<K, V>::new();
                let (claim, out) = hit!(map.iterate(collector).await);
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    impl<K: DeserializeOwned + Ord, V: DeserializeOwned> DeserializeOwned for BTreeMap<K, V> {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let map = hit!(e.deserialize_map().await);
                let collector = CollectMapArm::<K, V>::new();
                let (claim, out) = hit!(map.iterate(collector).await);
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    /// A [`MapArmStackOwned`](crate::owned::MapArmStackOwned) implementation that collects
    /// all KV pairs into a `BTreeMap`. Always hungry - never satisfied, always
    /// matches, accumulates until the map is exhausted.
    struct CollectMapArm<K, V> {
        pending_key: Option<K>,
        out: BTreeMap<K, V>,
    }

    impl<K, V> CollectMapArm<K, V> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: BTreeMap::new(),
            }
        }
    }

    impl<KP, K, V> crate::owned::MapArmStackOwned<KP> for CollectMapArm<K, V>
    where
        KP: crate::owned::MapKeyProbeOwned,
        K: DeserializeOwned + Ord,
        V: DeserializeOwned,
    {
        const SIZE: usize = 1;
        type Outputs = BTreeMap<K, V>;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        } // always hungry

        // CollectMapArm is never used inside StackConcat, so init/poll are
        // not called. We provide panicking stubs and override the async methods.
        type RaceState = ();
        type RaceDone = ();
        fn init_race(&mut self, _kp: KP) -> ((), ()) {
            unreachable!()
        }
        fn race_all_done(_done: &()) -> bool {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _done: &mut (),
            _arm_index: usize,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _arm_index: usize, _vp: crate::owned::VP<KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(crate::owned::VC<KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K, _>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }

        async fn dispatch_value(
            &mut self,
            _arm_index: usize,
            vp: crate::owned::VP<KP>,
        ) -> Result<Probe<(crate::owned::VC<KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch_value without pending key");
            match vp.deserialize_value::<V, _>(()).await? {
                Probe::Hit((vc, v)) => {
                    self.out.insert(k, v);
                    Ok(Probe::Hit((vc, ())))
                }
                Probe::Miss => Ok(Probe::Miss),
            }
        }

        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::take(&mut self.out)
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
                let (claim, owned) = hit!(e2.deserialize_value::<T::Owned, ()>(()).await);
                Ok(Probe::Hit((claim, Cow::Owned(owned))))
            })
            .await
        }
    }

    // --- Cow<'a, T> (owned family: always Cow::Owned) ---

    impl<'a, T: ?Sized + ToOwned> DeserializeOwned for Cow<'a, T>
    where
        T::Owned: DeserializeOwned,
    {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl DeserializeOwned for CString {
        async fn deserialize_owned<D: DeserializerOwned>(
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

        impl<T: ?Sized> DeserializeOwned for Rc<T>
        where
            Box<T>: DeserializeOwned,
        {
            async fn deserialize<D: DeserializerOwned>(
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

        impl<T: ?Sized> DeserializeOwned for Arc<T>
        where
            Box<T>: DeserializeOwned,
        {
            async fn deserialize<D: DeserializerOwned>(
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
        Deserialize, Deserializer, Entry, MapAccess, MapArmStack, MapKeyProbe, MapValueProbe,
        SeqAccess, SeqEntry, VC, VP,
    };
    use crate::owned::{
        DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapValueProbeOwned,
        SeqAccessOwned, SeqEntryOwned,
    };
    use crate::{Chunk, DeserializeError, Probe, hit, or_miss};
    use core::hash::{BuildHasher, Hash};
    use std::collections::{HashMap, HashSet};

    extern crate alloc;
    use alloc::{boxed::Box, string::String};

    // --- HashMap<K, V, S> ---

    struct CollectHashMapBorrowArm<K, V, S> {
        pending_key: Option<K>,
        out: HashMap<K, V, S>,
    }

    impl<K, V, S: BuildHasher + Default> CollectHashMapBorrowArm<K, V, S> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: HashMap::with_hasher(S::default()),
            }
        }
    }

    impl<'de, KP, K, V, S> MapArmStack<'de, KP> for CollectHashMapBorrowArm<K, V, S>
    where
        KP: MapKeyProbe<'de>,
        K: Deserialize<'de> + Eq + core::hash::Hash,
        V: Deserialize<'de>,
        S: BuildHasher + Default,
    {
        const SIZE: usize = 1;
        type Outputs = HashMap<K, V, S>;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            0
        }

        type RaceState = ();
        type RaceDone = ();
        fn init_race(&mut self, _kp: KP) -> ((), ()) {
            unreachable!()
        }
        fn race_all_done(_done: &()) -> bool {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _done: &mut (),
            _arm_index: usize,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _arm_index: usize, _vp: VP<'de, KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K, _>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }

        async fn dispatch_value(
            &mut self,
            _arm_index: usize,
            vp: VP<'de, KP>,
        ) -> Result<Probe<(VC<'de, KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch_value without pending key");
            match vp.deserialize_value::<V, _>(()).await? {
                Probe::Hit((vc, v)) => {
                    self.out.insert(k, v);
                    Ok(Probe::Hit((vc, ())))
                }
                Probe::Miss => Ok(Probe::Miss),
            }
        }

        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::take(&mut self.out)
        }
    }

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
                let map = hit!(e.deserialize_map().await);
                let collector = CollectHashMapBorrowArm::<K, V, S>::new();
                let (claim, out) = hit!(map.iterate(collector).await);
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    impl<K, V, S> DeserializeOwned for HashMap<K, V, S>
    where
        K: DeserializeOwned + Eq + Hash,
        V: DeserializeOwned,
        S: BuildHasher + Default,
    {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let map = hit!(e.deserialize_map().await);
                let collector = CollectHashMapArm::<K, V, S>::new();
                let (claim, out) = hit!(map.iterate(collector).await);
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    struct CollectHashMapArm<K, V, S> {
        pending_key: Option<K>,
        out: HashMap<K, V, S>,
    }

    impl<K, V, S: BuildHasher + Default> CollectHashMapArm<K, V, S> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: HashMap::with_hasher(S::default()),
            }
        }
    }

    impl<KP, K, V, S> crate::owned::MapArmStackOwned<KP> for CollectHashMapArm<K, V, S>
    where
        KP: crate::owned::MapKeyProbeOwned,
        K: DeserializeOwned + Eq + Hash,
        V: DeserializeOwned,
        S: BuildHasher + Default,
    {
        const SIZE: usize = 1;
        type Outputs = HashMap<K, V, S>;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        } // always hungry

        // CollectHashMapArm is never used inside StackConcat.
        type RaceState = ();
        type RaceDone = ();
        fn init_race(&mut self, _kp: KP) -> ((), ()) {
            unreachable!()
        }
        fn race_all_done(_done: &()) -> bool {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _done: &mut (),
            _arm_index: usize,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _arm_index: usize, _vp: crate::owned::VP<KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _state: core::pin::Pin<&mut ()>,
            _cx: &mut core::task::Context<'_>,
        ) -> core::task::Poll<Result<Probe<(crate::owned::VC<KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K, _>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }

        async fn dispatch_value(
            &mut self,
            _arm_index: usize,
            vp: crate::owned::VP<KP>,
        ) -> Result<Probe<(crate::owned::VC<KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch_value without pending key");
            match vp.deserialize_value::<V, _>(()).await? {
                Probe::Hit((vc, v)) => {
                    self.out.insert(k, v);
                    Ok(Probe::Hit((vc, ())))
                }
                Probe::Miss => Ok(Probe::Miss),
            }
        }

        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::take(&mut self.out)
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

    impl<T, S> DeserializeOwned for HashSet<T, S>
    where
        T: DeserializeOwned + Eq + Hash,
        S: BuildHasher + Default,
    {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl<T: DeserializeOwned> DeserializeOwned for std::sync::Mutex<T> {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl<T: DeserializeOwned> DeserializeOwned for std::sync::RwLock<T> {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl DeserializeOwned for std::path::PathBuf {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl DeserializeOwned for std::ffi::OsString {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    impl DeserializeOwned for Box<std::path::Path> {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(
                <std::path::PathBuf as DeserializeOwned>::deserialize_owned(d, ())
                    .await?
                    .map(|(c, p)| (c, p.into_boxed_path())),
            )
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

    impl DeserializeOwned for Box<std::ffi::OsStr> {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(
                <std::ffi::OsString as DeserializeOwned>::deserialize_owned(d, ())
                    .await?
                    .map(|(c, s)| (c, s.into_boxed_os_str())),
            )
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

    impl<'de> Deserialize<'de> for std::time::SystemTime {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            use crate::borrow::{KP, MapKeyProbe as _, MapValueProbe as _};

            d.entry(|[e]| async {
                let map = hit!(e.deserialize_map().await);

                let arms = crate::map_arms! {
                    |kp: KP<'de, D>| kp.deserialize_key::<crate::Match, &str>("secs_since_epoch")
                    => |vp: crate::borrow::VP2<'de, D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u64, _>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                    |kp: KP<'de, D>| kp.deserialize_key::<crate::Match, &str>("nanos_since_epoch")
                    => |vp: crate::borrow::VP2<'de, D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u32, _>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                };

                let (claim, crate::map_outputs!(opt_secs, opt_nanos)) =
                    hit!(map.iterate(arms).await);

                let s = or_miss!(opt_secs.map(|(_, v)| v));
                let n = or_miss!(opt_nanos.map(|(_, v)| v));
                let dur = core::time::Duration::new(s, n);
                Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + dur)))
            })
            .await
        }
    }

    impl DeserializeOwned for std::time::SystemTime {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            use crate::owned::{self, MapKeyProbeOwned, MapValueProbeOwned};

            d.entry(|[e]| async {
                let map = hit!(e.deserialize_map().await);

                let arms = crate::map_arms! {
                    |kp: owned::KP<D>| kp.deserialize_key::<crate::Match, &str>("secs_since_epoch")
                    => |vp: owned::VP2<D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u64, _>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                    |kp: owned::KP<D>| kp.deserialize_key::<crate::Match, &str>("nanos_since_epoch")
                    => |vp: owned::VP2<D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u32, _>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                };

                let (claim, crate::map_outputs!(opt_secs, opt_nanos)) =
                    hit!(map.iterate(arms).await);

                let s = or_miss!(opt_secs.map(|(_, v)| v));
                let n = or_miss!(opt_nanos.map(|(_, v)| v));
                let dur = core::time::Duration::new(s, n);
                Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + dur)))
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

            impl DeserializeOwned for $atomic {
                async fn deserialize_owned<D: DeserializerOwned>(
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
        Probe,
        borrow::{Deserialize, Deserializer, Entry, MapAccess, VP},
        utils::repeat,
    };
    use core::{future::Future, marker::PhantomData};

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
        pub map: M,
        pub _marker: core::marker::PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> MapDeserializer<'de, M> {
        pub fn new(map: M) -> Self {
            Self {
                map,
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'de, M: MapAccess<'de>> Deserializer<'de> for MapDeserializer<'de, M> {
        type Error = M::Error;
        type Claim = M::MapClaim;
        type EntryClaim = M::MapClaim;
        type Entry = MapDeserializerEntry<'de, M>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
        {
            let entries = repeat(self.map, M::fork).map(|map| MapDeserializerEntry {
                map,
                _marker: PhantomData,
            });
            f(entries).await
        }
    }

    pub struct MapDeserializerEntry<'de, M: MapAccess<'de>> {
        pub(crate) map: M,
        pub(crate) _marker: PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> Entry<'de> for MapDeserializerEntry<'de, M> {
        type Error = M::Error;
        type Claim = M::MapClaim;
        type StrChunks = crate::Never<'de, M::MapClaim, M::Error>;
        type BytesChunks = crate::Never<'de, M::MapClaim, M::Error>;
        type Map = M;
        type Seq = crate::Never<'de, M::MapClaim, M::Error>;

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
            Ok(Probe::Hit(self.map))
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
                map: self.map.fork(),
                _marker: PhantomData,
            }
        }

        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            let arms = crate::SkipUnknown!(
                crate::MapArmBase,
                <M as MapAccess<'de>>::KeyProbe,
                VP<'de, <M as MapAccess<'de>>::KeyProbe>
            );
            match self.map.iterate(arms).await? {
                Probe::Hit((claim, ())) => Ok(claim),
                Probe::Miss => panic!("MapDeserializerEntry::skip: iterate returned Miss"),
            }
        }
    }
}

// ===========================================================================
// Tag-filtering map facades (used by #[strede(tag)] derive output)
// ===========================================================================

pub mod tag_facade {
    use crate::{
        Never, Probe,
        borrow::{Deserializer, Entry, MapAccess, MapArmStack},
        owned::{
            DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapArmStackOwned,
        },
        utils::repeat,
    };
    use core::cell::Cell;
    use core::{future::Future, marker::PhantomData};

    pub use super::map_facade::{MapDeserializer, MapDeserializerEntry};

    // -----------------------------------------------------------------------
    // Borrow family - TagAwareMap
    // -----------------------------------------------------------------------
    //
    // Mirrors TagAwareMapOwned from the owned family. When `iterate(arms)`
    // is called, it wraps `arms` with a TagInjectingStackOwned that captures the
    // tag field value, then checks it against `expected_variant`.

    /// A `MapAccess` that injects a tag-capturing arm into any arm stack
    /// before delegating to the real inner map (borrow family).
    pub struct TagAwareMap<'de, 'v, M: MapAccess<'de>, const N: usize> {
        pub inner: M,
        pub tag_field: &'static str,
        pub tag_candidates: [(&'static str, usize); N],
        pub expected_variant: usize,
        pub tag_value: &'v Cell<Option<usize>>,
        pub _marker: PhantomData<&'de ()>,
    }

    impl<'de, 'v, M: MapAccess<'de>, const N: usize> MapAccess<'de> for TagAwareMap<'de, 'v, M, N> {
        type Error = M::Error;
        type MapClaim = M::MapClaim;
        type KeyProbe = M::KeyProbe;

        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_field: self.tag_field,
                tag_candidates: self.tag_candidates,
                expected_variant: self.expected_variant,
                tag_value: self.tag_value,
                _marker: PhantomData,
            }
        }

        async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
            self,
            arms: S,
        ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
            let injected = crate::TagInjectingStack!(
                arms,
                self.tag_field,
                self.tag_candidates,
                self.tag_value,
                Self::KeyProbe,
                crate::borrow::VP<'de, Self::KeyProbe>
            );
            let result = self.inner.iterate(injected).await?;
            match result {
                Probe::Hit((claim, outputs)) => {
                    if self.tag_value.get() == Some(self.expected_variant) {
                        Ok(Probe::Hit((claim, outputs)))
                    } else {
                        Ok(Probe::Miss)
                    }
                }
                Probe::Miss => Ok(Probe::Miss),
            }
        }
    }

    /// Entry type for [`TagAwareDeserializer`]. Only `deserialize_map` is
    /// supported; all other probes return `Miss`.
    pub struct TagAwareEntry<'de, 'v, M: MapAccess<'de>, const N: usize> {
        pub map: M,
        pub tag_field: &'static str,
        pub tag_candidates: [(&'static str, usize); N],
        pub expected_variant: usize,
        pub tag_value: &'v Cell<Option<usize>>,
        pub _marker: PhantomData<&'de ()>,
    }

    impl<'de, 'v, M: MapAccess<'de>, const N: usize> Entry<'de> for TagAwareEntry<'de, 'v, M, N> {
        type Error = M::Error;
        type Claim = M::MapClaim;
        type StrChunks = Never<'de, M::MapClaim, M::Error>;
        type BytesChunks = Never<'de, M::MapClaim, M::Error>;
        type Map = TagAwareMap<'de, 'v, M, N>;
        type Seq = Never<'de, M::MapClaim, M::Error>;

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
        async fn deserialize_option<T: crate::Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_value<T: crate::Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            Ok(Probe::Hit(TagAwareMap {
                inner: self.map,
                tag_field: self.tag_field,
                tag_candidates: self.tag_candidates,
                expected_variant: self.expected_variant,
                tag_value: self.tag_value,
                _marker: PhantomData,
            }))
        }

        fn fork(&mut self) -> Self {
            Self {
                map: self.map.fork(),
                tag_field: self.tag_field,
                tag_candidates: self.tag_candidates,
                expected_variant: self.expected_variant,
                tag_value: self.tag_value,
                _marker: PhantomData,
            }
        }
        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            panic!("TagAwareEntry::skip - no real token to skip")
        }
    }

    /// A `Deserializer` facade backed by a `MapAccess` (borrow family).
    ///
    /// Its entry only supports `deserialize_map`. Create one per non-unit
    /// variant in a `select_probe!`; each has its own `expected_variant`
    /// so only the tag-matching variant wins.
    pub struct TagAwareDeserializer<'de, 'v, M: MapAccess<'de>, const N: usize> {
        pub map: M,
        pub tag_field: &'static str,
        pub tag_candidates: [(&'static str, usize); N],
        pub expected_variant: usize,
        pub tag_value: &'v Cell<Option<usize>>,
        pub _marker: PhantomData<&'de ()>,
    }

    impl<'de, 'v, M: MapAccess<'de>, const N: usize> TagAwareDeserializer<'de, 'v, M, N> {
        pub fn new(
            map: M,
            tag_field: &'static str,
            tag_candidates: [(&'static str, usize); N],
            expected_variant: usize,
            tag_value: &'v Cell<Option<usize>>,
        ) -> Self {
            Self {
                map,
                tag_field,
                tag_candidates,
                expected_variant,
                tag_value,
                _marker: PhantomData,
            }
        }
    }

    impl<'de, 'v, M: MapAccess<'de>, const N: usize> Deserializer<'de>
        for TagAwareDeserializer<'de, 'v, M, N>
    {
        type Error = M::Error;
        type Claim = M::MapClaim;
        type EntryClaim = M::MapClaim;
        type Entry = TagAwareEntry<'de, 'v, M, N>;

        async fn entry<const K: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; K]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
        {
            let mut map = self.map;
            let tag_field = self.tag_field;
            let tag_candidates = self.tag_candidates;
            let expected_variant = self.expected_variant;
            let tag_value = self.tag_value;
            let entries: [TagAwareEntry<'de, 'v, M, N>; K] =
                core::array::from_fn(|_| TagAwareEntry {
                    map: map.fork(),
                    tag_field,
                    tag_candidates,
                    expected_variant,
                    tag_value,
                    _marker: PhantomData,
                });
            f(entries).await
        }
    }

    // -----------------------------------------------------------------------
    // Map tag-aware types (owned family)
    // -----------------------------------------------------------------------
    //
    // Pattern (from the derive macro):
    //
    //   let tag_value: Cell<Option<usize>> = Cell::new(None);
    //   select_probe! {
    //       async move {
    //           let d0 = TagAwareDeserializerOwned::new(map.fork(), "type", CANDS, 0, &tag_value);
    //           hit!(VariantType0::deserialize_owned(d0, ()).await)
    //       },
    //       async move {
    //           let d1 = TagAwareDeserializerOwned::new(map.fork(), "type", CANDS, 1, &tag_value);
    //           hit!(VariantType1::deserialize_owned(d1, ()).await)
    //       },
    //       @miss => Ok(Probe::Miss),
    //   }
    //
    // Each variant's DeserializeOwned calls d.entry(|[e]| e.deserialize_map()).
    // TagAwareEntryOwned::deserialize_map() returns a TagAwareMapOwned.
    // TagAwareMapOwned::iterate(their_arms) wraps with TagInjectingStackOwned,
    // calls inner.iterate, then checks captured tag == expected_variant.

    /// A `MapAccessOwned` that injects a tag-capturing arm into any
    /// arm stack before delegating to the real inner map.
    ///
    /// After `iterate` completes, if the captured tag index equals
    /// `expected_variant`, the result is passed through; otherwise `Miss` is
    /// returned so the outer `select_probe!` can try other variants.
    pub struct TagAwareMapOwned<'v, M: MapAccessOwned, const N: usize> {
        pub inner: M,
        pub tag_field: &'static str,
        pub tag_candidates: [(&'static str, usize); N],
        pub expected_variant: usize,
        pub tag_value: &'v Cell<Option<usize>>,
    }

    impl<'v, M: MapAccessOwned, const N: usize> MapAccessOwned for TagAwareMapOwned<'v, M, N> {
        type Error = M::Error;
        type MapClaim = M::MapClaim;
        type KeyProbe = M::KeyProbe;

        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_field: self.tag_field,
                tag_candidates: self.tag_candidates,
                expected_variant: self.expected_variant,
                tag_value: self.tag_value,
            }
        }

        async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
            self,
            arms: S,
        ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
            let injected = crate::TagInjectingStackOwned!(
                arms,
                self.tag_field,
                self.tag_candidates,
                self.tag_value,
                Self::KeyProbe,
                crate::owned::VP<Self::KeyProbe>
            );
            let result = self.inner.iterate(injected).await?;
            match result {
                Probe::Hit((claim, outputs)) => {
                    if self.tag_value.get() == Some(self.expected_variant) {
                        Ok(Probe::Hit((claim, outputs)))
                    } else {
                        Ok(Probe::Miss)
                    }
                }
                Probe::Miss => Ok(Probe::Miss),
            }
        }
    }

    /// The entry type for [`TagAwareDeserializerOwned`].
    ///
    /// All probes return `Miss` except `deserialize_map`, which wraps
    /// the stored map in a [`TagAwareMapOwned`] that injects the tag arm.
    pub struct TagAwareEntryOwned<'v, M: MapAccessOwned, const N: usize> {
        pub map: M,
        pub tag_field: &'static str,
        pub tag_candidates: [(&'static str, usize); N],
        pub expected_variant: usize,
        pub tag_value: &'v Cell<Option<usize>>,
    }

    impl<'v, M: MapAccessOwned, const N: usize> EntryOwned for TagAwareEntryOwned<'v, M, N> {
        type Error = M::Error;
        type Claim = M::MapClaim;
        type StrChunks = Never<'static, M::MapClaim, M::Error>;
        type BytesChunks = Never<'static, M::MapClaim, M::Error>;
        type Map = TagAwareMapOwned<'v, M, N>;
        type Seq = Never<'static, M::MapClaim, M::Error>;

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
        async fn deserialize_option<T: DeserializeOwned<Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_value<T: DeserializeOwned<Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            Ok(Probe::Hit(TagAwareMapOwned {
                inner: self.map,
                tag_field: self.tag_field,
                tag_candidates: self.tag_candidates,
                expected_variant: self.expected_variant,
                tag_value: self.tag_value,
            }))
        }

        fn fork(&mut self) -> Self {
            Self {
                map: self.map.fork(),
                tag_field: self.tag_field,
                tag_candidates: self.tag_candidates,
                expected_variant: self.expected_variant,
                tag_value: self.tag_value,
            }
        }
        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            panic!("TagAwareEntryOwned::skip - no real token to skip")
        }
    }

    /// A `DeserializerOwned` facade backed by a `MapAccessOwned`.
    ///
    /// Its entry only supports `deserialize_map` (all other probes
    /// return `Miss`). When a variant's `DeserializeOwned` calls
    /// `d.entry(|[e]| e.deserialize_map())`, it gets back a
    /// [`TagAwareMapOwned`] that injects the tag arm and validates
    /// `expected_variant` before returning `Hit`.
    ///
    /// Create one per non-unit variant in the `select_probe!` arms; each has
    /// its own `expected_variant` index so only the tag-matching variant wins.
    pub struct TagAwareDeserializerOwned<'v, M: MapAccessOwned, const N: usize> {
        pub map: M,
        pub tag_field: &'static str,
        pub tag_candidates: [(&'static str, usize); N],
        pub expected_variant: usize,
        pub tag_value: &'v Cell<Option<usize>>,
    }

    impl<'v, M: MapAccessOwned, const N: usize> TagAwareDeserializerOwned<'v, M, N> {
        pub fn new(
            map: M,
            tag_field: &'static str,
            tag_candidates: [(&'static str, usize); N],
            expected_variant: usize,
            tag_value: &'v Cell<Option<usize>>,
        ) -> Self {
            Self {
                map,
                tag_field,
                tag_candidates,
                expected_variant,
                tag_value,
            }
        }
    }

    impl<'v, M: MapAccessOwned, const N: usize> DeserializerOwned
        for TagAwareDeserializerOwned<'v, M, N>
    {
        type Error = M::Error;
        type Claim = M::MapClaim;
        type EntryClaim = M::MapClaim;
        type Entry = TagAwareEntryOwned<'v, M, N>;

        async fn entry<const K: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; K]) -> Fut,
            Fut: Future<
                Output = Result<Probe<(<Self::Entry as EntryOwned>::Claim, R)>, Self::Error>,
            >,
        {
            let tag_field = self.tag_field;
            let tag_candidates = self.tag_candidates;
            let expected_variant = self.expected_variant;
            let tag_value = self.tag_value;
            let entries = repeat(self.map, M::fork).map(|map| TagAwareEntryOwned {
                map,
                tag_field,
                tag_candidates,
                expected_variant,
                tag_value,
            });
            f(entries).await
        }
    }
}
