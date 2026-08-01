//! Chunk-boundary/truncation fuzzing for MessagePack's multi-byte
//! length-prefix headers (TESTING_GAPS.md item #4).
//!
//! `strede-msgpack/src/chunked/mod.rs`'s `read_bytes_exact` already has
//! resumable state to accumulate a header's bytes across an arbitrary number
//! of refills (see CLAUDE.md-adjacent commentary in that file). This wasn't
//! previously exercised at every possible split point for any of the
//! variable-width headers (str8/16/32, bin8/16/32, array16/32, map16/32,
//! ext8/16/32, uint16/32/64) - every existing owned-family test either feeds
//! the whole input upfront or splits at a small, fixed set of byte-at-a-time
//! offsets incidental to some other feature under test. This file instead
//! sweeps every chunk size from 1 up to the full input length for each
//! header shape, matching the sweep discipline established by
//! `strede-json/src/number/decimal_seq.rs`'s `parse_chunked` and
//! `strede-postcard/tests/varint_chunking.rs`.
//!
//! Deliberately out of scope: fixext1/2/4/8/16 (fixed-width, not
//! length-prefixed - already well covered by `strede-msgpack/tests/
//! ext_owned.rs`), and the fixed-width int/float headers (uint8/int8/
//! float32/float64) already exercised elsewhere. uint16/32/64 are included
//! here anyway since they share the exact same `read_bytes_exact` resumable
//! primitive as the length-prefix headers.

#![cfg(feature = "alloc")]

use std::collections::BTreeMap;

use strede::{BytesAccessOwned, Chunk, DeserializeOwned, Probe, SharedBuf};
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_msgpack::{
    ChunkedMsgpackBytesAccess, ChunkedMsgpackClaim, DeserializeFromExtBytesOwned, ExtWrapper,
    MsgpackError,
};
use strede_test_util::block_on_loop_bounded;

macro_rules! parse_chunked {
    ($ty:ty, $input:expr, $chunk_size:expr) => {{ parse_chunked!($ty, $input, $chunk_size, ()) }};
    ($ty:ty, $input:expr, $chunk_size:expr, $extra:expr) => {{
        let input: &[u8] = $input;
        let chunk_size: usize = $chunk_size;
        let extra = $extra;
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
                    let de = ChunkedMsgpackDeserializer::new(shared);
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, extra).await {
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

// -----------------------------------------------------------------------
// Well-formed headers - sweep every chunk size, compare against expected
// -----------------------------------------------------------------------

#[test]
fn str8_header_split() {
    let input: &[u8] = &[0xd9, 5, b'h', b'e', b'l', b'l', b'o'];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, input, chunk_size),
            Ok(Some("hello".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn str16_header_split() {
    let input: &[u8] = &[0xda, 0x00, 0x05, b'h', b'e', b'l', b'l', b'o'];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, input, chunk_size),
            Ok(Some("hello".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn str32_header_split() {
    let input: &[u8] = &[0xdb, 0x00, 0x00, 0x00, 0x05, b'h', b'e', b'l', b'l', b'o'];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(String, input, chunk_size),
            Ok(Some("hello".to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bin8_header_split() {
    let input: &[u8] = &[0xc4, 3, 0xAA, 0xBB, 0xCC];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u8>, input, chunk_size),
            Ok(Some(vec![0xAA, 0xBB, 0xCC])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bin16_header_split() {
    let input: &[u8] = &[0xc5, 0x00, 0x03, 0xAA, 0xBB, 0xCC];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u8>, input, chunk_size),
            Ok(Some(vec![0xAA, 0xBB, 0xCC])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bin32_header_split() {
    let input: &[u8] = &[0xc6, 0x00, 0x00, 0x00, 0x03, 0xAA, 0xBB, 0xCC];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u8>, input, chunk_size),
            Ok(Some(vec![0xAA, 0xBB, 0xCC])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn array16_header_split() {
    let input: &[u8] = &[0xdc, 0x00, 0x03, 1, 2, 3];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u32>, input, chunk_size),
            Ok(Some(vec![1, 2, 3])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn array32_header_split() {
    let input: &[u8] = &[0xdd, 0x00, 0x00, 0x00, 0x03, 1, 2, 3];
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(Vec<u32>, input, chunk_size),
            Ok(Some(vec![1, 2, 3])),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn map16_header_split() {
    let input: &[u8] = &[0xde, 0x00, 0x02, 0xa1, b'a', 0x01, 0xa1, b'b', 0x02];
    let expected: BTreeMap<String, u32> =
        BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 2)]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(BTreeMap<String, u32>, input, chunk_size),
            Ok(Some(expected.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn map32_header_split() {
    let input: &[u8] = &[
        0xdf, 0x00, 0x00, 0x00, 0x02, 0xa1, b'a', 0x01, 0xa1, b'b', 0x02,
    ];
    let expected: BTreeMap<String, u32> =
        BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 2)]);
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(BTreeMap<String, u32>, input, chunk_size),
            Ok(Some(expected.clone())),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn uint16_header_split() {
    let v: u16 = 0x1234;
    let mut input = vec![0xcd];
    input.extend_from_slice(&v.to_be_bytes());
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(u16, &input, chunk_size),
            Ok(Some(v)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn uint32_header_split() {
    let v: u32 = 0x1234_5678;
    let mut input = vec![0xce];
    input.extend_from_slice(&v.to_be_bytes());
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(u32, &input, chunk_size),
            Ok(Some(v)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn uint64_header_split() {
    let v: u64 = 0x0123_4567_89AB_CDEF;
    let mut input = vec![0xcf];
    input.extend_from_slice(&v.to_be_bytes());
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(u64, &input, chunk_size),
            Ok(Some(v)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn int16_negative_header_split() {
    let v: i16 = -1234;
    let mut input = vec![0xd1];
    input.extend_from_slice(&v.to_be_bytes());
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(i16, &input, chunk_size),
            Ok(Some(v)),
            "chunk_size={chunk_size}"
        );
    }
}

// --- ext8/16/32: length header + type byte, both resumable ---

#[derive(Debug)]
struct MyVarExt(Vec<u8>);

impl<'s, B: strede::Buffer, F: AsyncFnMut(&mut B)>
    DeserializeFromExtBytesOwned<ChunkedMsgpackBytesAccess<'s, B, F>> for MyVarExt
{
    type Extra = i8;

    async fn deserialize_from_ext_bytes_owned(
        type_id: i8,
        _len: usize,
        bytes: ChunkedMsgpackBytesAccess<'s, B, F>,
        extra: i8,
    ) -> Result<Probe<(ChunkedMsgpackClaim<'s, B, F>, Self)>, MsgpackError> {
        if type_id != extra {
            return Ok(Probe::Miss);
        }
        let mut collected = Vec::new();
        let mut acc = bytes;
        loop {
            match acc.next_bytes(|b| b.to_vec()).await? {
                Chunk::Data((next_acc, chunk)) => {
                    collected.extend_from_slice(&chunk);
                    acc = next_acc;
                }
                Chunk::Done(claim) => {
                    return Ok(Probe::Hit((claim, MyVarExt(collected))));
                }
            }
        }
    }
}

#[test]
fn ext8_header_split() {
    let input: &[u8] = &[0xc7, 4, 7, 0x11, 0x22, 0x33, 0x44];
    for chunk_size in 1..=input.len() {
        let result = parse_chunked!(ExtWrapper<MyVarExt>, input, chunk_size, 7i8);
        match result {
            Ok(Some(ExtWrapper(MyVarExt(v)))) => {
                assert_eq!(v, vec![0x11, 0x22, 0x33, 0x44], "chunk_size={chunk_size}")
            }
            Ok(None) => panic!("chunk_size={chunk_size}: expected Hit, got Miss"),
            Err(e) => panic!("chunk_size={chunk_size}: expected Hit, got Err({e})"),
        }
    }
}

#[test]
fn ext16_header_split() {
    let input: &[u8] = &[0xc8, 0x00, 0x04, 7, 0x11, 0x22, 0x33, 0x44];
    for chunk_size in 1..=input.len() {
        let result = parse_chunked!(ExtWrapper<MyVarExt>, input, chunk_size, 7i8);
        match result {
            Ok(Some(ExtWrapper(MyVarExt(v)))) => {
                assert_eq!(v, vec![0x11, 0x22, 0x33, 0x44], "chunk_size={chunk_size}")
            }
            Ok(None) => panic!("chunk_size={chunk_size}: expected Hit, got Miss"),
            Err(e) => panic!("chunk_size={chunk_size}: expected Hit, got Err({e})"),
        }
    }
}

#[test]
fn ext32_header_split() {
    let input: &[u8] = &[0xc9, 0x00, 0x00, 0x00, 0x04, 7, 0x11, 0x22, 0x33, 0x44];
    for chunk_size in 1..=input.len() {
        let result = parse_chunked!(ExtWrapper<MyVarExt>, input, chunk_size, 7i8);
        match result {
            Ok(Some(ExtWrapper(MyVarExt(v)))) => {
                assert_eq!(v, vec![0x11, 0x22, 0x33, 0x44], "chunk_size={chunk_size}")
            }
            Ok(None) => panic!("chunk_size={chunk_size}: expected Hit, got Miss"),
            Err(e) => panic!("chunk_size={chunk_size}: expected Hit, got Err({e})"),
        }
    }
}

// -----------------------------------------------------------------------
// Truncated headers/payloads at true EOF - must error, never panic or hang
// -----------------------------------------------------------------------

fn assert_sweep_errs<T: core::fmt::Debug>(
    results: impl IntoIterator<Item = (usize, Result<T, String>)>,
) {
    for (chunk_size, result) in results {
        assert!(
            result.is_err(),
            "chunk_size={chunk_size}: expected error, got {result:?}"
        );
    }
}

#[test]
fn truncated_str16_header() {
    let input: &[u8] = &[0xda, 0x00]; // only 1 of 2 length bytes
    assert_sweep_errs((1..=input.len()).map(|cs| (cs, parse_chunked!(String, input, cs))));
}

#[test]
fn truncated_str32_header() {
    let input: &[u8] = &[0xdb, 0x00, 0x00]; // only 2 of 4 length bytes
    assert_sweep_errs((1..=input.len()).map(|cs| (cs, parse_chunked!(String, input, cs))));
}

#[test]
fn truncated_bin32_header() {
    let input: &[u8] = &[0xc6, 0x00, 0x00, 0x00]; // only 3 of 4 length bytes
    assert_sweep_errs((1..=input.len()).map(|cs| (cs, parse_chunked!(Vec<u8>, input, cs))));
}

#[test]
fn truncated_array16_header() {
    let input: &[u8] = &[0xdc, 0x00]; // only 1 of 2 length bytes
    assert_sweep_errs((1..=input.len()).map(|cs| (cs, parse_chunked!(Vec<u32>, input, cs))));
}

#[test]
fn truncated_map32_header() {
    let input: &[u8] = &[0xdf, 0x00, 0x00]; // only 2 of 4 count bytes
    assert_sweep_errs(
        (1..=input.len()).map(|cs| (cs, parse_chunked!(BTreeMap<String, u32>, input, cs))),
    );
}

#[test]
fn truncated_uint64_header() {
    let input: &[u8] = &[0xcf, 1, 2, 3, 4]; // only 4 of 8 value bytes
    assert_sweep_errs((1..=input.len()).map(|cs| (cs, parse_chunked!(u64, input, cs))));
}

#[test]
fn truncated_payload_after_complete_header() {
    // Header is complete and declares len=5, but only 2 payload bytes exist.
    let input: &[u8] = &[0xd9, 5, b'h', b'i'];
    assert_sweep_errs((1..=input.len()).map(|cs| (cs, parse_chunked!(String, input, cs))));
}

#[test]
fn truncated_ext16_header() {
    // Only 1 of 2 length bytes, type byte missing too. `ExtWrapper<MyVarExt>`
    // doesn't implement `Debug`, so this can't go through the generic
    // `assert_sweep_errs` helper - just check `is_err()` directly.
    let input: &[u8] = &[0xc8, 0x00];
    for chunk_size in 1..=input.len() {
        let result = parse_chunked!(ExtWrapper<MyVarExt>, input, chunk_size, 7i8);
        assert!(result.is_err(), "chunk_size={chunk_size}: expected error");
    }
}
