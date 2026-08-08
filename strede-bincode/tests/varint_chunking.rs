//! Deliberately splits input into tiny chunks (1-7 bytes per loader call) to
//! exercise the resumable async decode helpers in `chunked::num` across
//! chunk boundaries mid-value — under both `Standard` (varint prefix+tail
//! split) and `Legacy` (fixed-width tail split). `Order` (LE/BE) doesn't
//! affect *where* a chunk boundary can land, only how the bytes are later
//! interpreted, so BE configs aren't repeated here.
//!
//! Every other `*_owned.rs` test file in this crate uses `parse_owned!`,
//! which feeds the whole input upfront via a trivial "empty the buffer to
//! signal EOF" loader — that convention never actually forces a mid-value
//! refill. This file uses `parse_owned_chunked!` instead, which genuinely
//! feeds the input a few bytes at a time.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_bincode::{BincodeError, Legacy, Standard};
use strede_derive::DeserializeOwned;

#[derive(Debug, PartialEq, DeserializeOwned)]
enum Cmd {
    A,
    B(u8),
}

macro_rules! chunking_tests_for_config {
    ($mod_name:ident, $cfg:ty, $enc:expr, $max_chunk:expr) => {
        mod $mod_name {
            use super::*;
            const E: Enc = $enc;

            #[test]
            fn u64_max_split_byte_by_byte() {
                let bytes = E.u64(u64::MAX);
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(u64, $cfg, &bytes, chunk_size),
                        Ok(Some(u64::MAX)),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[test]
            fn i64_min_split_across_chunks() {
                let bytes = E.i64(i64::MIN);
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(i64, $cfg, &bytes, chunk_size),
                        Ok(Some(i64::MIN)),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[test]
            fn u128_max_split_across_chunks() {
                let bytes = E.u128(u128::MAX);
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(u128, $cfg, &bytes, chunk_size),
                        Ok(Some(u128::MAX)),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[cfg(feature = "alloc")]
            #[test]
            fn string_length_split_from_payload() {
                // Long enough that (in Varint mode) the length needs the
                // u16-tail prefix (>250), so the split can land inside the
                // length itself, at the length/payload seam, or mid-payload.
                let s = "x".repeat(300);
                let bytes = E.str(&s);
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(String, $cfg, &bytes, chunk_size),
                        Ok(Some(s.clone())),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[cfg(feature = "alloc")]
            #[test]
            fn bytes_length_split_from_payload() {
                let data: Vec<u8> = (0u8..=255).collect();
                let bytes = E.bytes(&data);
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(Vec<u8>, $cfg, &bytes, chunk_size),
                        Ok(Some(data.clone())),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[cfg(feature = "alloc")]
            #[test]
            fn seq_count_split_across_chunks() {
                // 300 elements forces the length past the single-byte-inline
                // boundary in Varint mode.
                let expected: Vec<u32> = (0..300u32).collect();
                let mut bytes = E.len(expected.len());
                for i in &expected {
                    bytes.extend_from_slice(&E.u32(*i));
                }
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(Vec<u32>, $cfg, &bytes, chunk_size),
                        Ok(Some(expected.clone())),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[test]
            fn enum_discriminant_split_across_chunks() {
                let mut bytes = E.discriminant(1);
                bytes.extend_from_slice(&E.u8(0xAB));
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(Cmd, $cfg, &bytes, chunk_size),
                        Ok(Some(Cmd::B(0xAB))),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[test]
            fn enum_out_of_range_discriminant_split_across_chunks() {
                // `Cmd` only declares 0 and 1 — must miss cleanly, proving
                // the resumable discriminant read completes correctly
                // across a chunk boundary even when it doesn't match any arm.
                let bytes = E.discriminant(1000);
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(Cmd, $cfg, &bytes, chunk_size),
                        Ok(None),
                        "chunk_size={chunk_size}"
                    );
                }
            }

            #[test]
            fn truncated_at_true_eof_errors() {
                // Take the real 8-byte-wide encoding of a u64 value and
                // truncate it — valid framing (a real varint prefix or just
                // the start of a fixed-width read), but cut short before
                // the value's own byte count is satisfied.
                let full = E.u64(u64::MAX);
                let bytes = &full[..full.len() - 1];
                for chunk_size in 1..=$max_chunk {
                    assert_eq!(
                        parse_owned_chunked!(u64, $cfg, bytes, chunk_size).unwrap_err(),
                        BincodeError::UnexpectedEnd,
                        "chunk_size={chunk_size}"
                    );
                }
            }
        }
    };
}

chunking_tests_for_config!(standard, Standard, Enc::STANDARD, 4);
chunking_tests_for_config!(legacy, Legacy, Enc::LEGACY, 3);
