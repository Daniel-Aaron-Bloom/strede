//! Primitive deserialization across bincode's 2x2 wire-config matrix:
//! byte order (little/big) x int encoding (fixed-width/varint).
#![allow(clippy::approx_constant)]

mod helpers;
use helpers::*;

use strede_bincode::{BigLegacy, BigStandard, BincodeError, Legacy, Standard};

/// Numeric round-trip tests that hold across all 4 configs — these exercise
/// exactly what `ByteOrder`/`IntEncoding` control.
macro_rules! numeric_tests_for_config {
    ($mod_name:ident, $cfg:ty, $enc:expr) => {
        mod $mod_name {
            use super::*;
            const E: Enc = $enc;

            #[test]
            fn u8_roundtrip() {
                assert_eq!(parse::<u8, $cfg>(&E.u8(0)), Ok(Some(0)));
                assert_eq!(parse::<u8, $cfg>(&E.u8(255)), Ok(Some(255)));
            }

            #[test]
            fn u16_roundtrip() {
                assert_eq!(parse::<u16, $cfg>(&E.u16(1000)), Ok(Some(1000)));
                assert_eq!(parse::<u16, $cfg>(&E.u16(u16::MAX)), Ok(Some(u16::MAX)));
            }

            #[test]
            fn u32_roundtrip() {
                assert_eq!(parse::<u32, $cfg>(&E.u32(100_000)), Ok(Some(100_000)));
                assert_eq!(parse::<u32, $cfg>(&E.u32(u32::MAX)), Ok(Some(u32::MAX)));
            }

            #[test]
            fn u64_roundtrip() {
                assert_eq!(parse::<u64, $cfg>(&E.u64(u64::MAX)), Ok(Some(u64::MAX)));
            }

            #[test]
            fn u128_roundtrip() {
                assert_eq!(parse::<u128, $cfg>(&E.u128(u128::MAX)), Ok(Some(u128::MAX)));
            }

            #[test]
            fn i16_roundtrip() {
                assert_eq!(parse::<i16, $cfg>(&E.i16(-1000)), Ok(Some(-1000)));
                assert_eq!(parse::<i16, $cfg>(&E.i16(i16::MIN)), Ok(Some(i16::MIN)));
            }

            #[test]
            fn i32_roundtrip() {
                assert_eq!(parse::<i32, $cfg>(&E.i32(-100_000)), Ok(Some(-100_000)));
            }

            #[test]
            fn i64_roundtrip() {
                assert_eq!(parse::<i64, $cfg>(&E.i64(i64::MIN)), Ok(Some(i64::MIN)));
            }

            #[test]
            fn i128_roundtrip() {
                assert_eq!(parse::<i128, $cfg>(&E.i128(i128::MIN)), Ok(Some(i128::MIN)));
            }

            #[test]
            fn enum_discriminant_width() {
                // The enum discriminant is a u32 subject to this config's
                // int encoding — exercised directly here via the raw
                // encoder (full enum derive coverage lives in enums_borrow.rs).
                assert_eq!(E.discriminant(0), E.u32(0));
                assert_eq!(E.discriminant(1000), E.u32(1000));
            }
        }
    };
}

numeric_tests_for_config!(standard, Standard, Enc::STANDARD);
numeric_tests_for_config!(legacy, Legacy, Enc::LEGACY);
numeric_tests_for_config!(big_standard, BigStandard, Enc::BIG_STANDARD);
numeric_tests_for_config!(big_legacy, BigLegacy, Enc::BIG_LEGACY);

/// Varint prefix-boundary edge cases — only meaningful under `Varint`
/// configs (`Fixint` has no notion of "value too wide for the target type,"
/// since it always reads the target's own fixed byte count).
macro_rules! varint_boundary_tests_for_config {
    ($mod_name:ident, $cfg:ty, $enc:expr) => {
        mod $mod_name {
            use super::*;
            const E: Enc = $enc;

            #[test]
            fn u16_inline_boundary() {
                // 250 is the last single-byte-inline value; 251 needs the
                // u16-tail prefix.
                assert_eq!(parse::<u16, $cfg>(&E.u16(250)), Ok(Some(250)));
                assert_eq!(parse::<u16, $cfg>(&E.u16(251)), Ok(Some(251)));
            }

            #[test]
            fn u16_out_of_range_misses() {
                assert_eq!(
                    parse::<u16, $cfg>(&E.u32(u16::MAX as u32 + 1)),
                    Ok(None)
                );
            }

            #[test]
            fn u32_tail_boundary() {
                // 65535 fits the u16 tail (prefix 251); 65536 needs the u32
                // tail (prefix 252).
                assert_eq!(
                    parse::<u32, $cfg>(&E.u32(u16::MAX as u32)),
                    Ok(Some(u16::MAX as u32))
                );
                assert_eq!(
                    parse::<u32, $cfg>(&E.u32(u16::MAX as u32 + 1)),
                    Ok(Some(u16::MAX as u32 + 1))
                );
            }

            #[test]
            fn u32_out_of_range_misses() {
                assert_eq!(
                    parse::<u32, $cfg>(&E.u64(u32::MAX as u64 + 1)),
                    Ok(None)
                );
            }

            // --- Non-canonical varint rejection (real bincode2 parity) ---
            //
            // A value small enough to need only a narrow prefix, but
            // deliberately encoded via a wider tail-announcing prefix, is
            // rejected regardless of whether the value itself would fit the
            // target type. Real bincode2 rejects this as
            // `InvalidIntegerType`; a probe miss is this crate's equivalent
            // for a typed numeric decode with no "try another type"
            // fallback available.

            #[test]
            fn u16_rejects_non_canonical_wide_prefix() {
                // 100 fits a single byte; encoded here via the 16-byte
                // u128-tail prefix (254).
                let data = E.varint_with_prefix(254, 100);
                assert_eq!(parse::<u16, $cfg>(&data), Ok(None));
            }

            #[test]
            fn u32_rejects_non_canonical_wide_prefix() {
                let data = E.varint_with_prefix(253, 100);
                assert_eq!(parse::<u32, $cfg>(&data), Ok(None));
            }

            #[test]
            fn u64_rejects_non_canonical_wide_prefix() {
                let data = E.varint_with_prefix(254, 100);
                assert_eq!(parse::<u64, $cfg>(&data), Ok(None));
            }

            #[test]
            fn i16_rejects_non_canonical_wide_prefix() {
                let data = E.varint_with_prefix(254, 100);
                assert_eq!(parse::<i16, $cfg>(&data), Ok(None));
            }

            #[test]
            fn u128_has_no_narrower_width_to_violate() {
                // u128 is the widest target, so even the 16-byte tail
                // prefix is always canonical for it — this isn't a
                // violation, it's the only form u128 ever uses once a
                // value exceeds a u64.
                let data = E.varint_with_prefix(254, 100);
                assert_eq!(parse::<u128, $cfg>(&data), Ok(Some(100)));
            }

        }
    };
}

varint_boundary_tests_for_config!(standard_boundary, Standard, Enc::STANDARD);
varint_boundary_tests_for_config!(big_standard_boundary, BigStandard, Enc::BIG_STANDARD);

// --- Config-independent primitives: tested once under the default (Standard) config ---

const E: Enc = Enc::STANDARD;

#[test]
fn unit_empty_input() {
    assert_eq!(parse::<(), Standard>(&[]), Ok(Some(())));
}

#[test]
fn unit_trailing_bytes_errors() {
    assert_eq!(
        parse_err::<(), Standard>(&[0x01]),
        BincodeError::ExpectedEnd
    );
}

#[test]
fn bool_false() {
    assert_eq!(parse::<bool, Standard>(&E.bool(false)), Ok(Some(false)));
}

#[test]
fn bool_true() {
    assert_eq!(parse::<bool, Standard>(&E.bool(true)), Ok(Some(true)));
}

#[test]
fn bool_invalid_misses() {
    assert_eq!(parse::<bool, Standard>(&[0x02]), Ok(None));
}

#[test]
fn bool_truncated_errors() {
    assert_eq!(
        parse_err::<bool, Standard>(&[]),
        BincodeError::UnexpectedEnd
    );
}

#[test]
fn f32_value() {
    let result = parse::<f32, Standard>(&E.f32(3.14f32)).unwrap().unwrap();
    assert!((result - 3.14f32).abs() < 1e-6);
}

#[test]
fn f32_truncated_errors() {
    assert_eq!(
        parse_err::<f32, Standard>(&[0x00, 0x00]),
        BincodeError::UnexpectedEnd
    );
}

#[test]
fn f64_value() {
    let result = parse::<f64, Standard>(&E.f64(2.718281828))
        .unwrap()
        .unwrap();
    assert!((result - 2.718281828).abs() < 1e-12);
}

#[test]
fn char_ascii() {
    assert_eq!(parse::<char, Standard>(&E.char('A')), Ok(Some('A')));
}

#[test]
fn char_unicode() {
    assert_eq!(parse::<char, Standard>(&E.char('€')), Ok(Some('€')));
}

#[test]
fn char_invalid_lead_byte_misses() {
    assert_eq!(parse::<char, Standard>(&[0xff]), Ok(None));
}

#[test]
fn char_truncated_multibyte_errors() {
    // 'é' (0xC3 0xA9) with the second byte missing.
    assert_eq!(
        parse_err::<char, Standard>(&[0xC3]),
        BincodeError::UnexpectedEnd
    );
}

#[test]
fn str_empty() {
    assert_eq!(parse::<&str, Standard>(&E.str("")), Ok(Some("")));
}

#[test]
fn str_hello() {
    assert_eq!(parse::<&str, Standard>(&E.str("hello")), Ok(Some("hello")));
}

#[test]
fn str_unicode() {
    assert_eq!(parse::<&str, Standard>(&E.str("héllo")), Ok(Some("héllo")));
}

#[test]
fn str_truncated_errors() {
    let mut data = E.len(5);
    data.extend_from_slice(b"hi");
    assert_eq!(
        parse_err::<&str, Standard>(&data),
        BincodeError::UnexpectedEnd
    );
}

#[test]
fn str_length_non_canonical_prefix_errors() {
    // Length 3 legitimately fits a single byte; encoded here via the
    // 16-byte u128-tail prefix (254). `decode_len` has no "try another
    // type" fallback, so this is a hard error rather than a probe miss.
    let mut data = E.varint_with_prefix(254, 3);
    data.extend_from_slice(b"abc");
    assert_eq!(
        parse_err::<&str, Standard>(&data),
        BincodeError::NonCanonicalVarint
    );
}

#[test]
fn str_invalid_utf8_errors() {
    let mut data = E.len(2);
    data.extend_from_slice(&[0xff, 0xfe]);
    assert_eq!(
        parse_err::<&str, Standard>(&data),
        BincodeError::InvalidUtf8
    );
}

#[test]
fn bytes_empty() {
    assert_eq!(
        parse::<&[u8], Standard>(&E.bytes(&[])),
        Ok(Some(&[] as &[u8]))
    );
}

#[test]
fn bytes_values() {
    let data = &[0x00u8, 0x01, 0xff];
    assert_eq!(
        parse::<&[u8], Standard>(&E.bytes(data)),
        Ok(Some(data.as_slice()))
    );
}

#[test]
fn option_none() {
    assert_eq!(parse::<Option<u32>, Standard>(&E.none()), Ok(Some(None)));
}

#[test]
fn option_some_u32() {
    let data = E.some(&E.u32(42));
    assert_eq!(
        parse::<Option<u32>, Standard>(&data),
        Ok(Some(Some(42u32)))
    );
}

#[test]
fn option_invalid_tag_misses() {
    assert_eq!(parse::<Option<u32>, Standard>(&[0x02]), Ok(None));
}

#[test]
fn varint_prefix_255_errors() {
    // `255` is never emitted by any valid encoder under any config —
    // always a corrupt stream.
    assert_eq!(
        parse_err::<u32, Standard>(&[0xff]),
        BincodeError::InvalidVarint
    );
}

#[test]
fn skip_is_unsupported() {
    // Bincode is schema-driven: `Entry::skip()` always fails, since field
    // positions come from the type, not the wire data.
    use strede::{Deserializer, Entry, Probe};
    use strede_bincode::BincodeDeserializer;
    use strede_test_util::block_on;

    let de = BincodeDeserializer::<Standard>::new(&[0x00]);
    let result = block_on(de.entry(|[e]| async move {
        let claim = e.skip().await?;
        Ok::<_, BincodeError>(Probe::Hit((claim, ())))
    }));
    assert_eq!(result.unwrap_err(), BincodeError::CannotSkip);
}

#[test]
fn trailing_bytes_errors() {
    let mut data = E.u32(42);
    data.push(0x00);
    assert_eq!(
        parse_err::<u32, Standard>(&data),
        BincodeError::ExpectedEnd
    );
}
