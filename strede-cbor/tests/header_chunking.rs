//! Chunk-boundary/truncation fuzzing for CBOR's multi-byte argument-length
//! headers (TESTING_GAPS.md item #4).
//!
//! CBOR encodes every major type's length/count/tag-number/value as an
//! "additional info" argument: `info` 0..=23 is the value directly, and
//! `info` 24/25/26/27 mean "1/2/4/8 more bytes follow, big-endian". Those
//! extra bytes are read via `strede-cbor/src/chunked/mod.rs`'s
//! `read_bytes_exact`/`read_argument`, which already has resumable state to
//! accumulate across an arbitrary number of chunk refills - but this was
//! never previously exercised at every possible split point. This file
//! sweeps every chunk size from 1 up to the full input length for each
//! argument width (1/2/4/8 bytes) across bstr (major 2), tstr (major 3),
//! array (major 4), map (major 5), and tag (major 6) - matching the sweep
//! discipline in `strede-json/src/number/decimal_seq.rs`'s `parse_chunked`
//! and `strede-postcard/tests/varint_chunking.rs`.
//!
//! Headers are hand-built (not via `helpers::bstr`/`tstr`, which always pick
//! the *minimal* encoding for a given length) so that each test can force a
//! specific argument width regardless of how small the payload is - the
//! header-splitting behavior under test doesn't depend on payload size at
//! all, so payloads are kept small to keep the sweep fast.

#![cfg(feature = "alloc")]

extern crate std;

mod helpers;
use helpers::*;

use std::collections::BTreeMap;
use std::string::ToString;
use std::vec;
use std::vec::Vec;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_test_util::block_on_loop_bounded;

/// Build a header with a forced argument width (24/25/26/27), regardless of
/// how small `arg` actually is.
fn forced_header(major: u8, info: u8, arg: u64) -> Vec<u8> {
    let mut out = vec![(major << 5) | info];
    match info {
        24 => out.push(arg as u8),
        25 => out.extend_from_slice(&(arg as u16).to_be_bytes()),
        26 => out.extend_from_slice(&(arg as u32).to_be_bytes()),
        27 => out.extend_from_slice(&arg.to_be_bytes()),
        _ => unreachable!("info must be 24/25/26/27"),
    }
    out
}

fn bstr_forced(info: u8, data: &[u8]) -> Vec<u8> {
    let mut out = forced_header(2, info, data.len() as u64);
    out.extend_from_slice(data);
    out
}

fn tstr_forced(info: u8, s: &str) -> Vec<u8> {
    let mut out = forced_header(3, info, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
    out
}

macro_rules! parse_chunked {
    ($ty:ty, $input:expr, $chunk_size:expr) => {{
        let input: &[u8] = $input;
        let chunk_size: usize = $chunk_size;
        let pos = ::core::cell::Cell::new(chunk_size.min(input.len()));
        block_on_loop_bounded(
            SharedBuf::with_async(
                &input[..chunk_size.min(input.len())],
                async |buf: &mut &[u8]| {
                    let start = pos.get();
                    let end = (start + chunk_size).min(input.len());
                    pos.set(end);
                    *buf = &input[start..end];
                },
                async |shared| {
                    let de = ChunkedCborDeserializer::new(shared);
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ()).await {
                        Ok(Probe::Hit((_, v))) => Ok(Some(v)),
                        Ok(Probe::Miss) => Ok(None),
                        Err(e) => Err(format!("{e:?}")),
                    }
                },
            ),
            50_000,
        )
    }};
}

fn assert_sweep_errs<T: core::fmt::Debug>(
    ty_name: &str,
    results: impl IntoIterator<Item = (usize, Result<T, String>)>,
) {
    for (chunk_size, result) in results {
        assert!(
            result.is_err(),
            "{ty_name}: chunk_size={chunk_size}: expected error, got {result:?}"
        );
    }
}

// -----------------------------------------------------------------------
// bstr (major 2) - all four argument widths
// -----------------------------------------------------------------------

#[test]
fn bstr_info24_header_split() {
    let input = bstr_forced(24, &[0xAA, 0xBB, 0xCC]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u8>, &input, chunk_size),
            Ok(Some(vec![0xAA, 0xBB, 0xCC])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bstr_info25_header_split() {
    let input = bstr_forced(25, &[0xAA, 0xBB, 0xCC]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u8>, &input, chunk_size),
            Ok(Some(vec![0xAA, 0xBB, 0xCC])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bstr_info26_header_split() {
    let input = bstr_forced(26, &[0xAA, 0xBB, 0xCC]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u8>, &input, chunk_size),
            Ok(Some(vec![0xAA, 0xBB, 0xCC])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bstr_info27_header_split() {
    let input = bstr_forced(27, &[0xAA, 0xBB, 0xCC]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u8>, &input, chunk_size),
            Ok(Some(vec![0xAA, 0xBB, 0xCC])),
            "chunk_size={chunk_size}"
        );
    }
}

// -----------------------------------------------------------------------
// tstr (major 3) - all four argument widths
// -----------------------------------------------------------------------

#[test]
fn tstr_info24_header_split() {
    let input = tstr_forced(24, "hello");
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, &input, chunk_size),
            Ok(Some("hello".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn tstr_info25_header_split() {
    let input = tstr_forced(25, "hello");
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, &input, chunk_size),
            Ok(Some("hello".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn tstr_info26_header_split() {
    let input = tstr_forced(26, "hello");
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, &input, chunk_size),
            Ok(Some("hello".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn tstr_info27_header_split() {
    let input = tstr_forced(27, "hello");
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, &input, chunk_size),
            Ok(Some("hello".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

// -----------------------------------------------------------------------
// array (major 4) / map (major 5) - wide count headers
// -----------------------------------------------------------------------

#[test]
fn array_info25_header_split() {
    let mut input = forced_header(4, 25, 3);
    input.extend_from_slice(&[uint_small(1), uint_small(2), uint_small(3)]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u32>, &input, chunk_size),
            Ok(Some(vec![1, 2, 3])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn array_info26_header_split() {
    let mut input = forced_header(4, 26, 3);
    input.extend_from_slice(&[uint_small(1), uint_small(2), uint_small(3)]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u32>, &input, chunk_size),
            Ok(Some(vec![1, 2, 3])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn map_info25_header_split() {
    let mut input = forced_header(5, 25, 2);
    input.extend_from_slice(&tstr("a"));
    input.push(uint_small(1));
    input.extend_from_slice(&tstr("b"));
    input.push(uint_small(2));
    let expected: BTreeMap<String, u32> =
        BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 2)]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(BTreeMap<String, u32>, &input, chunk_size),
            Ok(Some(expected.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn map_info26_header_split() {
    let mut input = forced_header(5, 26, 2);
    input.extend_from_slice(&tstr("a"));
    input.push(uint_small(1));
    input.extend_from_slice(&tstr("b"));
    input.push(uint_small(2));
    let expected: BTreeMap<String, u32> =
        BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 2)]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(BTreeMap<String, u32>, &input, chunk_size),
            Ok(Some(expected.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

// -----------------------------------------------------------------------
// tag (major 6) - wide tag-number headers, transparently stripped
// -----------------------------------------------------------------------

#[test]
fn tag_info25_header_split_transparent() {
    // Tag number itself (1000) is irrelevant to the result - only tags 2/3
    // get special bignum handling; every other tag is silently discarded,
    // so the inner tstr should come through untouched regardless of chunk
    // size, exercising the tag argument's own 2-byte header split.
    let mut input = forced_header(6, 25, 1000);
    input.extend_from_slice(&tstr("hi"));
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, &input, chunk_size),
            Ok(Some("hi".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn tag_info26_header_split_transparent() {
    let mut input = forced_header(6, 26, 100_000);
    input.extend_from_slice(&tstr("hi"));
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, &input, chunk_size),
            Ok(Some("hi".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

// -----------------------------------------------------------------------
// uint (major 0) - shares the same read_argument/read_bytes_exact path
// -----------------------------------------------------------------------

#[test]
fn uint16_header_split() {
    let input = uint16(0x1234);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(u16, &input, chunk_size),
            Ok(Some(0x1234u16)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn uint32_header_split() {
    let input = uint32(0x1234_5678);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(u32, &input, chunk_size),
            Ok(Some(0x1234_5678u32)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn uint64_header_split() {
    let input = uint64(0x0123_4567_89AB_CDEF);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(u64, &input, chunk_size),
            Ok(Some(0x0123_4567_89AB_CDEFu64)),
            "chunk_size={chunk_size}"
        );
    }
}

// -----------------------------------------------------------------------
// Truncated headers/payloads at true EOF - must error, never panic or hang
// -----------------------------------------------------------------------

#[test]
fn truncated_bstr_info26_header() {
    // 2 of 4 length bytes.
    let input: &[u8] = &[0x5a, 0x00, 0x00];
    assert_sweep_errs(
        "bstr_info26",
        (1..=input.len()).map(|cs| (cs, parse_chunked!(Vec<u8>, input, cs))),
    );
}

#[test]
fn truncated_tstr_info27_header() {
    // 5 of 8 length bytes.
    let input: &[u8] = &[0x7b, 0x00, 0x00, 0x00, 0x00, 0x00];
    assert_sweep_errs(
        "tstr_info27",
        (1..=input.len()).map(|cs| (cs, parse_chunked!(String, input, cs))),
    );
}

#[test]
fn truncated_map_info25_header() {
    // 1 of 2 count bytes.
    let input: &[u8] = &[0xb9, 0x00];
    assert_sweep_errs(
        "map_info25",
        (1..=input.len()).map(|cs| (cs, parse_chunked!(BTreeMap<String, u32>, input, cs))),
    );
}

#[test]
fn truncated_tag_info26_header() {
    // 2 of 4 tag-number bytes.
    let input: &[u8] = &[0xda, 0x00, 0x00];
    assert_sweep_errs(
        "tag_info26",
        (1..=input.len()).map(|cs| (cs, parse_chunked!(String, input, cs))),
    );
}

#[test]
fn truncated_uint64_header() {
    // 4 of 8 value bytes.
    let input: &[u8] = &[0x1b, 1, 2, 3, 4];
    assert_sweep_errs(
        "uint64",
        (1..=input.len()).map(|cs| (cs, parse_chunked!(u64, input, cs))),
    );
}

#[test]
fn truncated_payload_after_complete_header() {
    // Header complete, declares len=5, but only 2 payload bytes exist.
    let input: &[u8] = &[0x78, 5, b'h', b'i'];
    assert_sweep_errs(
        "tstr_payload",
        (1..=input.len()).map(|cs| (cs, parse_chunked!(String, input, cs))),
    );
}
