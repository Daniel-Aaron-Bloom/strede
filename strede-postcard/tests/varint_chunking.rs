//! Deliberately splits input into tiny chunks (1–3 bytes per loader call) to
//! exercise `chunked::varint::{read_varint, read_zigzag, read_varint_bytes}`
//! resuming across chunk boundaries mid-varint — the one failure mode
//! postcard's chunked family introduces that msgpack/CBOR's fixed-width
//! headers never hit (their headers are always 1/2/4/8 bytes, known in
//! advance; postcard's LEB128 varints are 1–10 bytes, continuation-bit
//! terminated, and can in principle span an arbitrary number of chunk
//! refills).
//!
//! Every other `*_owned.rs` test file in this crate uses `parse_owned!`,
//! which feeds the whole input upfront via a trivial "empty the buffer to
//! signal EOF" loader (matching the existing `strede-msgpack`/`strede-cbor`
//! owned-family test convention) - that convention never actually forces a
//! mid-value refill, so it can't catch a bug at the refill seam itself. This
//! file uses `parse_owned_chunked!` instead, which genuinely feeds the input
//! a few bytes at a time.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Cmd {
    A,
    B(u8),
}

// --- u64::MAX: a full 10-byte varint, split byte-by-byte ---

#[test]
fn u64_max_varint_split_byte_by_byte() {
    let bytes: Vec<u8> = varint(u64::MAX);
    assert_eq!(bytes.len(), 10);
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_owned_chunked!(u64, &bytes, chunk_size),
            Ok(Some(u64::MAX)),
            "chunk_size={chunk_size}"
        );
    }
}

// --- zigzag i64 varint split mid-value ---

#[test]
fn zigzag_varint_split_across_chunks() {
    let bytes: Vec<u8> = zigzag(i64::MIN);
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_owned_chunked!(i64, &bytes, chunk_size),
            Ok(Some(i64::MIN)),
            "chunk_size={chunk_size}"
        );
    }
}

// --- u128's two chained varints, split exactly at the seam and elsewhere ---

#[test]
fn u128_two_varints_split_at_the_seam() {
    let v = u128::MAX;
    let bytes: Vec<u8> = pu128(v);
    for chunk_size in 1..=4 {
        assert_eq!(
            parse_owned_chunked!(u128, &bytes, chunk_size),
            Ok(Some(v)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn i128_two_varints_split_across_chunks() {
    let v = i128::MIN;
    let bytes: Vec<u8> = pi128(v);
    for chunk_size in 1..=4 {
        assert_eq!(
            parse_owned_chunked!(i128, &bytes, chunk_size),
            Ok(Some(v)),
            "chunk_size={chunk_size}"
        );
    }
}

// --- string length-varint split from its payload ---

#[test]
fn string_length_varint_split_from_payload() {
    // A string long enough that its length varint needs 2 bytes (>127),
    // so the split can land inside the length varint itself, at the seam
    // between length and payload, or inside the payload.
    let s = "x".repeat(200);
    let bytes: Vec<u8> = pstr(&s);
    for chunk_size in 1..=5 {
        assert_eq!(
            parse_owned_chunked!(String, &bytes, chunk_size),
            Ok(Some(s.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

// --- bytes length-varint split from its payload ---
//
// Covers the full byte range (0..=255), including the high-bit-set values
// that would misparse if `Vec<u8>` were read as a seq of varint-encoded `u8`
// elements. That interpretation is never attempted here - see
// `strede-postcard/src/vec.rs`, which always treats `Vec<u8>` as raw bytes.
#[cfg(feature = "alloc")]
#[test]
fn bytes_length_varint_split_from_payload() {
    let data: Vec<u8> = (0u8..=255).collect();
    let bytes: Vec<u8> = pbytes(&data);
    for chunk_size in 1..=5 {
        assert_eq!(
            parse_owned_chunked!(Vec<u8>, &bytes, chunk_size),
            Ok(Some(data.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

// --- seq count varint split across chunks ---

#[cfg(feature = "alloc")]
#[test]
fn seq_count_varint_split_across_chunks() {
    // 200 elements forces a 2-byte count varint.
    let elems: Vec<Vec<u8>> = (0..200u32).map(|i| varint(i as u64)).collect();
    let elem_refs: Vec<&[u8]> = elems.iter().map(|e| e.as_slice()).collect();
    let bytes: Vec<u8> = pseq(&elem_refs);
    let expected: Vec<u32> = (0..200u32).collect();
    for chunk_size in 1..=7 {
        assert_eq!(
            parse_owned_chunked!(Vec<u32>, &bytes, chunk_size),
            Ok(Some(expected.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

// --- map count varint split across chunks (requires alloc's HashMap) ---

#[test]
fn map_count_varint_split_for_hashmap() {
    use std::collections::HashMap;

    let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..200u32)
        .map(|i| (varint(i as u64), varint((i * 2) as u64)))
        .collect();
    let pair_refs: Vec<(&[u8], &[u8])> = pairs
        .iter()
        .map(|(k, v)| (k.as_slice(), v.as_slice()))
        .collect();
    let bytes: Vec<u8> = pmap(&pair_refs);
    let expected: HashMap<u32, u32> = (0..200u32).map(|i| (i, i * 2)).collect();
    for chunk_size in 1..=7 {
        assert_eq!(
            parse_owned_chunked!(HashMap<u32, u32>, &bytes, chunk_size),
            Ok(Some(expected.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

// --- enum discriminant varint split across chunks ---

#[test]
fn enum_discriminant_varint_split_across_chunks() {
    // Discriminant 1 (B) with payload 0xAB — single-byte discriminant, but
    // exercise the split at every byte boundary anyway for parity with the
    // other cases.
    let mut bytes = varint(1);
    bytes.extend_from_slice(&varint(0xAB));
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_owned_chunked!(Cmd, &bytes, chunk_size),
            Ok(Some(Cmd::B(0xAB))),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn enum_out_of_range_discriminant_two_bytes_split_across_chunks() {
    // Discriminant 128 requires two bytes: [0x80, 0x01]. `Cmd` only
    // declares 0 and 1, so this must miss cleanly - proving the resumable
    // varint discriminant read itself completes correctly across a chunk
    // boundary even when it doesn't match any arm.
    let bytes: Vec<u8> = varint(128);
    for chunk_size in 1..=2 {
        assert_eq!(
            parse_owned_chunked!(Cmd, &bytes, chunk_size),
            Ok(None),
            "chunk_size={chunk_size}"
        );
    }
}

// --- truncated varint at true EOF ---

#[test]
fn truncated_varint_at_true_eof_errors() {
    // All continuation-bit-set bytes, then EOF with no terminator: must
    // error UnexpectedEnd, not hang or panic.
    let bytes: Vec<u8> = vec![0xff, 0xff, 0xff];
    for chunk_size in 1..=2 {
        assert_eq!(
            parse_owned_chunked!(u64, &bytes, chunk_size).unwrap_err(),
            PostcardError::UnexpectedEnd,
            "chunk_size={chunk_size}"
        );
    }
}
