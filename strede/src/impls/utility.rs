use super::*;
use crate::{hit, select_probe};

// -------------------------------------------------------------------------
// PhantomData — independent of D. Always present.
// -------------------------------------------------------------------------

impl<'de, D: Deserializer<'de>, T: ?Sized> Deserialize<'de, D> for PhantomData<T> {
    type Extra = ();
    async fn deserialize(_d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        // PhantomData has no representation; derive skips PhantomData fields so this
        // path is unreachable in normal use.
        Ok(Probe::Miss)
    }
}

impl<D: DeserializerOwned, T: ?Sized> DeserializeOwned<D> for PhantomData<T> {
    type Extra = ();
    async fn deserialize_owned(_d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(Probe::Miss)
    }
}

// -------------------------------------------------------------------------
// Skip — consume + discard the next value.
// -------------------------------------------------------------------------

pub struct Skip;

impl<'de, D: Deserializer<'de>> Deserialize<'de, D> for Skip {
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move {
            let c = e.skip().await?;
            Ok(Probe::Hit((c, Skip)))
        })
        .await
    }
}

impl<D: DeserializerOwned> DeserializeOwned<D> for Skip {
    type Extra = ();
    async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move {
            let c = e.skip().await?;
            Ok(Probe::Hit((c, Skip)))
        })
        .await
    }
}

// -------------------------------------------------------------------------
// Match — discriminator probe.  Hit if and only if the current token is a
// string whose content equals `extra`.  Universal over any `D: Deserializer`
// via the standard `Entry::deserialize_str` / `deserialize_str_chunks` race.
// -------------------------------------------------------------------------

pub struct Match;

impl<'de, D: Deserializer<'de>> Deserialize<'de, D> for Match {
    type Extra = &'static str;
    async fn deserialize(
        d: D,
        expected: &'static str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e1, e2]| async move {
            select_probe! {
                async move {
                    let (claim, s) = hit!(e1.deserialize_str().await);
                    if s == expected {
                        Ok(Probe::Hit((claim, Match)))
                    } else {
                        Ok(Probe::Miss)
                    }
                },
                async move {
                    let mut chunks = hit!(e2.deserialize_str_chunks().await);
                    let mut consumed: usize = 0;
                    loop {
                        let mut chunk_ok = true;
                        let result = chunks.next_str(|s: &str| {
                            let new_consumed = consumed + s.len();
                            if new_consumed > expected.len()
                                || &expected.as_bytes()[consumed..new_consumed] != s.as_bytes()
                            {
                                chunk_ok = false;
                            } else {
                                consumed = new_consumed;
                            }
                        }).await?;
                        if !chunk_ok {
                            return Ok(Probe::Miss);
                        }
                        match result {
                            Chunk::Data((new, ())) => chunks = new,
                            Chunk::Done(claim) => {
                                return Ok(if consumed == expected.len() {
                                    Probe::Hit((claim, Match))
                                } else {
                                    Probe::Miss
                                });
                            }
                        }
                    }
                }
            }
        })
        .await
    }
}

impl<D: DeserializerOwned> DeserializeOwned<D> for Match {
    type Extra = &'static str;
    async fn deserialize_owned(
        d: D,
        expected: &'static str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move {
            let mut chunks = hit!(e.deserialize_str_chunks().await);
            let mut consumed: usize = 0;
            loop {
                let mut chunk_ok = true;
                let result = chunks
                    .next_str(|s: &str| {
                        let new_consumed = consumed + s.len();
                        if new_consumed > expected.len()
                            || &expected.as_bytes()[consumed..new_consumed] != s.as_bytes()
                        {
                            chunk_ok = false;
                        } else {
                            consumed = new_consumed;
                        }
                    })
                    .await?;
                if !chunk_ok {
                    return Ok(Probe::Miss);
                }
                match result {
                    Chunk::Data((new, ())) => chunks = new,
                    Chunk::Done(claim) => {
                        return Ok(if consumed == expected.len() {
                            Probe::Hit((claim, Match))
                        } else {
                            Probe::Miss
                        });
                    }
                }
            }
        })
        .await
    }
}

// -------------------------------------------------------------------------
// MatchVals<T> — discriminator probe with N candidates returning T on hit.
// Universal over any `D: Deserializer`.  `T: Copy` so the candidate array
// can move into both race arms.
// -------------------------------------------------------------------------

pub struct MatchVals<T, const N: usize>(pub T, pub PhantomData<[(); N]>);

impl<'de, D, T, const N: usize> Deserialize<'de, D> for MatchVals<T, N>
where
    D: Deserializer<'de>,
    T: Copy,
{
    type Extra = [(&'static str, T); N];
    async fn deserialize(
        d: D,
        candidates: Self::Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e1, e2]| async move {
            select_probe! {
                async move {
                    let (claim, s) = hit!(e1.deserialize_str().await);
                    for (k, v) in &candidates {
                        if s == *k {
                            return Ok(Probe::Hit((claim, MatchVals(*v, PhantomData))));
                        }
                    }
                    Ok(Probe::Miss)
                },
                async move {
                    let mut chunks = hit!(e2.deserialize_str_chunks().await);
                    let mut viable = [true; N];
                    let mut consumed: usize = 0;
                    loop {
                        let result = chunks.next_str(|s: &str| {
                            let new_consumed = consumed + s.len();
                            for i in 0..N {
                                if !viable[i] {
                                    continue;
                                }
                                let k = candidates[i].0;
                                if new_consumed > k.len()
                                    || &k.as_bytes()[consumed..new_consumed] != s.as_bytes()
                                {
                                    viable[i] = false;
                                }
                            }
                            consumed = new_consumed;
                        }).await?;
                        if !viable.iter().any(|v| *v) {
                            return Ok(Probe::Miss);
                        }
                        match result {
                            Chunk::Data((new, ())) => chunks = new,
                            Chunk::Done(claim) => {
                                for i in 0..N {
                                    if viable[i] && candidates[i].0.len() == consumed {
                                        return Ok(Probe::Hit((claim, MatchVals(candidates[i].1, PhantomData))));
                                    }
                                }
                                return Ok(Probe::Miss);
                            }
                        }
                    }
                }
            }
        }).await
    }
}

impl<D, T, const N: usize> DeserializeOwned<D> for MatchVals<T, N>
where
    D: DeserializerOwned,
    T: Copy,
{
    type Extra = [(&'static str, T); N];
    async fn deserialize_owned(
        d: D,
        candidates: Self::Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move {
            let mut chunks = hit!(e.deserialize_str_chunks().await);
            let mut viable = [true; N];
            let mut consumed: usize = 0;
            loop {
                let result = chunks
                    .next_str(|s: &str| {
                        let new_consumed = consumed + s.len();
                        for i in 0..N {
                            if !viable[i] {
                                continue;
                            }
                            let k = candidates[i].0;
                            if new_consumed > k.len()
                                || &k.as_bytes()[consumed..new_consumed] != s.as_bytes()
                            {
                                viable[i] = false;
                            }
                        }
                        consumed = new_consumed;
                    })
                    .await?;
                if !viable.iter().any(|v| *v) {
                    return Ok(Probe::Miss);
                }
                match result {
                    Chunk::Data((new, ())) => chunks = new,
                    Chunk::Done(claim) => {
                        for i in 0..N {
                            if viable[i] && candidates[i].0.len() == consumed {
                                return Ok(Probe::Hit((
                                    claim,
                                    MatchVals(candidates[i].1, PhantomData),
                                )));
                            }
                        }
                        return Ok(Probe::Miss);
                    }
                }
            }
        })
        .await
    }
}

// -------------------------------------------------------------------------
// UnwrapOrElse<T, F> — try `T::deserialize`; on `Miss`, skip the value and
// invoke the async fallback `F` to produce a `T`. The stream is consumed
// exactly once in either branch. Used by the owned-family enum derive to
// turn an unknown variant name into the `#[strede(other)]` sentinel index
// while still advancing the stream.
// -------------------------------------------------------------------------

pub struct UnwrapOrElse<T, F>(pub T, pub PhantomData<fn() -> F>);

impl<'de, D, T, F> Deserialize<'de, D> for UnwrapOrElse<T, F>
where
    D: Deserializer<'de>,
    T: Deserialize<'de, <D::Entry as crate::Entry<'de>>::SubDeserializer>,
    F: AsyncFnOnce() -> T,
{
    type Extra = (F, T::Extra);
    async fn deserialize(d: D, extra: Self::Extra) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut slot = Some(extra);
        d.entry(|[e1, e2]| {
            let (fallback, inner_extra) = slot.take().expect("FnMut closure invoked twice");
            async move {
                let sub = <D::Entry as crate::Entry<'de>>::deserialize_value::<T>(e1, inner_extra)
                    .await?;
                match sub {
                    Probe::Hit((c, v)) => Ok(Probe::Hit((c, UnwrapOrElse(v, PhantomData)))),
                    Probe::Miss => {
                        let c = e2.skip().await?;
                        let v = fallback().await;
                        Ok(Probe::Hit((c, UnwrapOrElse(v, PhantomData))))
                    }
                }
            }
        })
        .await
    }
}

impl<D, T, F> DeserializeOwned<D> for UnwrapOrElse<T, F>
where
    D: DeserializerOwned,
    T: DeserializeOwned<<D::Entry as crate::EntryOwned>::SubDeserializer>,
    F: AsyncFnOnce() -> T,
{
    type Extra = (F, T::Extra);
    async fn deserialize_owned(
        d: D,
        extra: Self::Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let mut slot = Some(extra);
        d.entry(|[e1, e2]| {
            let (fallback, inner_extra) = slot.take().expect("FnMut closure invoked twice");
            async move {
                let sub = <D::Entry as crate::EntryOwned>::deserialize_value::<T>(e1, inner_extra)
                    .await?;
                match sub {
                    Probe::Hit((c, v)) => Ok(Probe::Hit((c, UnwrapOrElse(v, PhantomData)))),
                    Probe::Miss => {
                        let c = e2.skip().await?;
                        let v = fallback().await;
                        Ok(Probe::Hit((c, UnwrapOrElse(v, PhantomData))))
                    }
                }
            }
        })
        .await
    }
}
