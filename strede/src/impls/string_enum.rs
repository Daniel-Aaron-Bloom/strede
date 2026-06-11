//! Helper utilities for string-based enum dispatch.
//!
//! Formats that encode externally-tagged enums as bare strings (unit variants)
//! or single-key maps (non-unit variants) can use these helpers in their
//! [`EnumVariantProbe`] implementation.

use crate::Chunk;
use crate::Probe;
use crate::borrow::{Entry, StrAccess};
use crate::owned::StrAccessOwned;

/// Read all chunks from `chunks` and compare the accumulated string against
/// `candidates`.  Returns `Hit((claim, idx))` on match, `Miss` on no match or
/// names longer than 128 bytes, `Err` on a format error.
pub async fn match_str_chunks_against<SA: StrAccess>(
    mut chunks: SA,
    candidates: &[(&'static str, usize)],
) -> Result<Probe<(SA::Claim, usize)>, SA::Error> {
    let mut buf = [0u8; 128];
    let mut len = 0usize;
    let mut overflow = false;

    let claim = loop {
        match chunks
            .next_str(|s| {
                let end = len + s.len();
                if end <= buf.len() {
                    buf[len..end].copy_from_slice(s.as_bytes());
                    len = end;
                } else {
                    overflow = true;
                }
            })
            .await?
        {
            Chunk::Data((next, ())) => {
                chunks = next;
            }
            Chunk::Done(claim) => break claim,
        }
    };

    if overflow {
        return Ok(Probe::Miss);
    }
    let s = core::str::from_utf8(&buf[..len]).unwrap_or("");
    for &(name, idx) in candidates {
        if s == name {
            return Ok(Probe::Hit((claim, idx)));
        }
    }
    Ok(Probe::Miss)
}

/// Read all chunks from `chunks` (owned family) and compare the accumulated
/// string against `candidates`.  Returns `Hit((claim, idx))` on match, `Miss`
/// on no match or names longer than 128 bytes, `Err` on a format error.
pub async fn match_str_chunks_against_owned<SA: StrAccessOwned>(
    mut chunks: SA,
    candidates: &[(&'static str, usize)],
) -> Result<Probe<(SA::Claim, usize)>, SA::Error> {
    let mut buf = [0u8; 128];
    let mut len = 0usize;
    let mut overflow = false;

    let claim = loop {
        match chunks
            .next_str(|s| {
                let end = len + s.len();
                if end <= buf.len() {
                    buf[len..end].copy_from_slice(s.as_bytes());
                    len = end;
                } else {
                    overflow = true;
                }
            })
            .await?
        {
            Chunk::Data((next, ())) => {
                chunks = next;
            }
            Chunk::Done(claim) => break claim,
        }
    };

    if overflow {
        return Ok(Probe::Miss);
    }
    let s = core::str::from_utf8(&buf[..len]).unwrap_or("");
    for &(name, idx) in candidates {
        if s == name {
            return Ok(Probe::Hit((claim, idx)));
        }
    }
    Ok(Probe::Miss)
}

/// Match a string token in `entry` against `candidates` using both the
/// zero-copy fast path (`deserialize_str`) and the chunked fallback
/// (`deserialize_str_chunks`).
///
/// Returns `Hit((claim, idx))`, `Miss`, or `Err`.
pub async fn match_entry_str_against<'de, E: Entry<'de>>(
    mut entry: E,
    candidates: &[(&'static str, usize)],
) -> Result<Probe<(E::Claim, usize)>, E::Error> {
    let entry2 = entry.fork();

    if let Probe::Hit((claim, s)) = entry.deserialize_str().await? {
        for &(name, idx) in candidates {
            if s == name {
                return Ok(Probe::Hit((claim, idx)));
            }
        }
        return Ok(Probe::Miss);
    }

    let chunks = match entry2.deserialize_str_chunks().await? {
        Probe::Hit(c) => c,
        Probe::Miss => return Ok(Probe::Miss),
    };
    match_str_chunks_against(chunks, candidates).await
}
