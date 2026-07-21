//! Small building blocks for format authors.
//!
//! `strede` core deliberately has no blanket `Vec<T>: Deserialize` /
//! `DeserializeOwned` impl. Whether `Vec<u8>` can safely race a "raw bytes"
//! interpretation against "a sequence of `u8` elements" depends on whether
//! the format's wire representation can actually tell the two apart - a fact
//! only the format itself knows (self-describing formats tag the two
//! differently on the wire; schema-driven formats like postcard have no such
//! tag, so the two encodings can coincide for some inputs and diverge for
//! others, making the race unsound). Each format crate therefore provides
//! its own `Vec<T>` impl, built out of the pieces below.

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

#[cfg(feature = "alloc")]
mod vec_helpers {
    use crate::borrow::{
        BytesAccess, Deserialize, DeserializeFromSeq, Deserializer, Entry, SeqAccess, SeqEntry,
    };
    use crate::owned::{
        BytesAccessOwned, DeserializeFromSeqOwned, DeserializeOwned, DeserializerOwned, EntryOwned,
        SeqAccessOwned, SeqEntryOwned,
    };
    use crate::{Chunk, Probe, hit, select_probe};
    use alloc::vec::Vec;

    /// Reinterpret a `Vec<u8>` as `Vec<T>` via a raw-parts cast.
    ///
    /// # Safety
    /// Caller must have confirmed `TypeId::of::<T>() == TypeId::of::<u8>()`
    /// (e.g. via the `typeid` crate, which works in `no_std`).
    pub unsafe fn u8_vec_as_t<T>(v: Vec<u8>) -> Vec<T> {
        let mut v = core::mem::ManuallyDrop::new(v);
        // SAFETY: delegated to caller (T == u8, same size/alignment/drop).
        unsafe { Vec::from_raw_parts(v.as_mut_ptr() as *mut T, v.len(), v.capacity()) }
    }

    /// Drain a borrow-family [`BytesAccess`] chunk stream into an owned `Vec<u8>`.
    pub async fn collect_bytes_chunks<B: BytesAccess>(
        mut chunks: B,
    ) -> Result<(B::Claim, Vec<u8>), B::Error> {
        let mut out = Vec::new();
        loop {
            match chunks.next_bytes(|b| out.extend_from_slice(b)).await? {
                Chunk::Data((new, ())) => chunks = new,
                Chunk::Done(claim) => return Ok((claim, out)),
            }
        }
    }

    /// Drain an owned-family [`BytesAccessOwned`] chunk stream into an owned `Vec<u8>`.
    pub async fn collect_bytes_chunks_owned<B: BytesAccessOwned>(
        mut chunks: B,
    ) -> Result<(B::Claim, Vec<u8>), B::Error> {
        let mut out = Vec::new();
        loop {
            match chunks.next_bytes(|b| out.extend_from_slice(b)).await? {
                Chunk::Data((new, ())) => chunks = new,
                Chunk::Done(claim) => return Ok((claim, out)),
            }
        }
    }

    /// Deserialize `Vec<T>` by unconditionally treating the entry as a
    /// sequence - no "raw bytes" fast path. Formats whose wire representation
    /// has no distinct byte-string token, or that would rather not guess
    /// between one and a plain sequence, should build their `Vec<T>:
    /// Deserialize` impl on this alone (see `strede-postcard`).
    pub async fn vec_via_seq<'de, D, T>(d: D) -> Result<Probe<(D::Claim, Vec<T>)>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<
                'de,
                <<<D::Entry as Entry<'de>>::Seq as SeqAccess<'de>>::Elem as SeqEntry<'de>>::SubDeserializer,
                Extra = (),
            >,
    {
        d.entry(|[e]| async move { e.deserialize_seq_into::<Vec<T>>(()).await })
            .await
    }

    /// Owned-family counterpart of [`vec_via_seq`].
    pub async fn vec_via_seq_owned<D, T>(d: D) -> Result<Probe<(D::Claim, Vec<T>)>, D::Error>
    where
        D: DeserializerOwned,
        T: DeserializeOwned<
                <<<D::Entry as EntryOwned>::Seq as SeqAccessOwned>::Elem as SeqEntryOwned>::SubDeserializer,
                Extra = (),
            >,
    {
        d.entry(|[e]| async move { e.deserialize_seq_into::<Vec<T>>(()).await })
            .await
    }

    /// Race "raw bytes" against "a sequence of `u8` elements" for `Vec<u8>`,
    /// borrow family. Sound only when the format's wire representation makes
    /// the two shapes distinguishable from the bytes alone (e.g. a dedicated
    /// byte-string token, as in JSON/CBOR/MessagePack) - a schema-driven
    /// format with no such token must not use this (see `strede-postcard`,
    /// which always calls [`vec_via_seq`] instead).
    pub async fn vec_u8_race<'de, D>(d: D) -> Result<Probe<(D::Claim, Vec<u8>)>, D::Error>
    where
        D: Deserializer<'de>,
        Vec<u8>: DeserializeFromSeq<'de, <D::Entry as Entry<'de>>::Seq, Extra = ()>,
    {
        d.entry(|[e1, e2, e3]| async move {
            select_probe!(biased;
                async move {
                    let (claim, b) = hit!(e1.deserialize_bytes().await);
                    Ok(Probe::Hit((claim, b.to_vec())))
                },
                async move {
                    let chunks = hit!(e2.deserialize_bytes_chunks().await);
                    let (claim, out) = collect_bytes_chunks(chunks).await?;
                    Ok(Probe::Hit((claim, out)))
                },
                e3.deserialize_seq_into::<Vec<u8>>(()),
            )
        })
        .await
    }

    /// Owned-family counterpart of [`vec_u8_race`].
    pub async fn vec_u8_race_owned<D>(d: D) -> Result<Probe<(D::Claim, Vec<u8>)>, D::Error>
    where
        D: DeserializerOwned,
        Vec<u8>: DeserializeFromSeqOwned<<D::Entry as EntryOwned>::Seq, Extra = ()>,
    {
        d.entry(|[e1, e2]| async move {
            select_probe!(biased;
                async move {
                    let chunks = hit!(e1.deserialize_bytes_chunks().await);
                    let (claim, out) = collect_bytes_chunks_owned(chunks).await?;
                    Ok(Probe::Hit((claim, out)))
                },
                e2.deserialize_seq_into::<Vec<u8>>(()),
            )
        })
        .await
    }

    /// Deserialize `Vec<u8>` as raw bytes unconditionally - no seq fallback
    /// at all. For formats where `Vec<u8>`'s wire representation is always
    /// the byte-string form (there is no seq-of-elements alternative to
    /// consider in the first place), e.g. because the format has no wire
    /// tag that could ever make the seq reading the *intended* one for a
    /// `Vec<u8>` specifically (see `strede-postcard`, where a bare `u8`
    /// element is itself a varint, so "seq of u8" and "raw bytes" would
    /// otherwise be two genuinely different, format-ambiguous encodings).
    pub async fn vec_u8_bytes_only<'de, D>(d: D) -> Result<Probe<(D::Claim, Vec<u8>)>, D::Error>
    where
        D: Deserializer<'de>,
    {
        d.entry(|[e1, e2]| async move {
            select_probe!(biased;
                async move {
                    let (claim, b) = hit!(e1.deserialize_bytes().await);
                    Ok(Probe::Hit((claim, b.to_vec())))
                },
                async move {
                    let chunks = hit!(e2.deserialize_bytes_chunks().await);
                    let (claim, out) = collect_bytes_chunks(chunks).await?;
                    Ok(Probe::Hit((claim, out)))
                },
            )
        })
        .await
    }

    /// Owned-family counterpart of [`vec_u8_bytes_only`].
    pub async fn vec_u8_bytes_only_owned<D>(d: D) -> Result<Probe<(D::Claim, Vec<u8>)>, D::Error>
    where
        D: DeserializerOwned,
    {
        d.entry(|[e]| async move {
            let chunks = hit!(e.deserialize_bytes_chunks().await);
            let (claim, out) = collect_bytes_chunks_owned(chunks).await?;
            Ok(Probe::Hit((claim, out)))
        })
        .await
    }
}

#[cfg(feature = "alloc")]
pub use vec_helpers::{
    collect_bytes_chunks, collect_bytes_chunks_owned, u8_vec_as_t, vec_u8_bytes_only,
    vec_u8_bytes_only_owned, vec_u8_race, vec_u8_race_owned, vec_via_seq, vec_via_seq_owned,
};
