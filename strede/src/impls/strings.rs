use super::*;
use crate::{hit, select_probe};

// -------------------------------------------------------------------------
// Cow<'de, str> — borrow-family zero-copy fast path + chunked alloc fallback.
// (Owned-family Cow specializes to Cow::Owned only — same as String.)
// -------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<'de, D> Deserialize<'de, D> for alloc::borrow::Cow<'de, str>
where
    D: Deserializer<'de>,
{
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        use alloc::borrow::Cow;
        use alloc::string::String;
        d.entry(|[e1, e2]| async move {
            select_probe! {
                async move {
                    let (claim, s) = hit!(e1.deserialize_str().await);
                    Ok(Probe::Hit((claim, Cow::Borrowed(s))))
                },
                async move {
                    let mut chunks = hit!(e2.deserialize_str_chunks().await);
                    let mut out = String::new();
                    loop {
                        match chunks.next_str(|s| out.push_str(s)).await? {
                            Chunk::Data((new, ())) => chunks = new,
                            Chunk::Done(claim) => {
                                return Ok(Probe::Hit((claim, Cow::Owned(out))));
                            }
                        }
                    }
                }
            }
        })
        .await
    }
}

#[cfg(feature = "alloc")]
impl<D> DeserializeOwned<D> for alloc::string::String
where
    D: DeserializerOwned,
{
    type Extra = ();
    async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        use alloc::string::String;
        d.entry(|[e]| async move {
            let mut chunks = hit!(e.deserialize_str_chunks().await);
            let mut out = String::new();
            loop {
                match chunks.next_str(|s| out.push_str(s)).await? {
                    Chunk::Data((new, ())) => chunks = new,
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                }
            }
        })
        .await
    }
}

// -------------------------------------------------------------------------
// String — borrow-family: always allocates (no zero-copy borrow into &str).
// Owned-family String already impl'd above.
// -------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<'de, D> Deserialize<'de, D> for alloc::string::String
where
    D: Deserializer<'de>,
{
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        use alloc::string::String;
        d.entry(|[e]| async move {
            let mut chunks = hit!(e.deserialize_str_chunks().await);
            let mut out = String::new();
            loop {
                match chunks.next_str(|s| out.push_str(s)).await? {
                    Chunk::Data((new, ())) => chunks = new,
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                }
            }
        })
        .await
    }
}

// -------------------------------------------------------------------------
// Box<str> — both families. Delegates to String.
// -------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<'de, D> Deserialize<'de, D> for alloc::boxed::Box<str>
where
    D: Deserializer<'de>,
{
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let (claim, s) =
            hit!(<alloc::string::String as Deserialize<'de, D>>::deserialize(d, ()).await);
        Ok(Probe::Hit((claim, s.into_boxed_str())))
    }
}

#[cfg(feature = "alloc")]
impl<D> DeserializeOwned<D> for alloc::boxed::Box<str>
where
    D: DeserializerOwned,
{
    type Extra = ();
    async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let (claim, s) =
            hit!(<alloc::string::String as DeserializeOwned<D>>::deserialize_owned(d, ()).await);
        Ok(Probe::Hit((claim, s.into_boxed_str())))
    }
}
