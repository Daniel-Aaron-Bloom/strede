//! Primitive deserialization via the owned/chunked family, across bincode's
//! 2x2 wire-config matrix.
//!
//! Mirrors `primitives_borrow.rs`. Zero-copy `&str`/`&[u8]` have no
//! owned-family equivalent — `String`/`Vec<u8>` stand in for them here.
#![recursion_limit = "256"]
#![allow(clippy::approx_constant)]

#[macro_use]
mod helpers;
use helpers::*;

use strede_bincode::{BincodeError, Legacy, Standard};

/// Numeric round-trip tests under both `Standard` (varint) and `Legacy`
/// (fixed-width) — the two configs also exercised by the owned-family
/// chunk-boundary sweep in `varint_chunking.rs`, since `Order` (LE/BE)
/// doesn't affect where a chunk boundary can land, only how the bytes are
/// later interpreted.
macro_rules! numeric_tests_for_config {
    ($mod_name:ident, $cfg:ty, $enc:expr) => {
        mod $mod_name {
            use super::*;
            const E: Enc = $enc;

            #[test]
            fn u8_roundtrip() {
                assert_eq!(parse_owned!(u8, $cfg, &E.u8(255)), Ok(Some(255)));
            }

            #[test]
            fn u16_roundtrip() {
                assert_eq!(parse_owned!(u16, $cfg, &E.u16(u16::MAX)), Ok(Some(u16::MAX)));
            }

            #[test]
            fn u32_roundtrip() {
                assert_eq!(parse_owned!(u32, $cfg, &E.u32(u32::MAX)), Ok(Some(u32::MAX)));
            }

            #[test]
            fn u64_roundtrip() {
                assert_eq!(parse_owned!(u64, $cfg, &E.u64(u64::MAX)), Ok(Some(u64::MAX)));
            }

            #[test]
            fn u128_roundtrip() {
                assert_eq!(
                    parse_owned!(u128, $cfg, &E.u128(u128::MAX)),
                    Ok(Some(u128::MAX))
                );
            }

            #[test]
            fn i16_roundtrip() {
                assert_eq!(parse_owned!(i16, $cfg, &E.i16(i16::MIN)), Ok(Some(i16::MIN)));
            }

            #[test]
            fn i64_roundtrip() {
                assert_eq!(parse_owned!(i64, $cfg, &E.i64(i64::MIN)), Ok(Some(i64::MIN)));
            }

            #[test]
            fn i128_roundtrip() {
                assert_eq!(
                    parse_owned!(i128, $cfg, &E.i128(i128::MIN)),
                    Ok(Some(i128::MIN))
                );
            }
        }
    };
}

numeric_tests_for_config!(standard, Standard, Enc::STANDARD);
numeric_tests_for_config!(legacy, Legacy, Enc::LEGACY);

const E: Enc = Enc::STANDARD;

#[test]
fn unit_empty_input() {
    assert_eq!(parse_owned!((), Standard, &[]), Ok(Some(())));
}

#[test]
fn unit_trailing_bytes_errors() {
    assert_eq!(
        parse_owned!((), Standard, &[0x01]).unwrap_err(),
        BincodeError::ExpectedEnd
    );
}

#[test]
fn bool_false() {
    assert_eq!(parse_owned!(bool, Standard, &E.bool(false)), Ok(Some(false)));
}

#[test]
fn bool_true() {
    assert_eq!(parse_owned!(bool, Standard, &E.bool(true)), Ok(Some(true)));
}

#[test]
fn bool_invalid_misses() {
    assert_eq!(parse_owned!(bool, Standard, &[0x02]), Ok(None));
}

#[test]
fn bool_truncated_errors() {
    assert_eq!(
        parse_owned!(bool, Standard, &[]).unwrap_err(),
        BincodeError::UnexpectedEnd
    );
}

#[test]
fn f32_value() {
    let result = parse_owned!(f32, Standard, &E.f32(3.14f32))
        .unwrap()
        .unwrap();
    assert!((result - 3.14f32).abs() < 1e-6);
}

#[test]
fn f32_truncated_errors() {
    assert_eq!(
        parse_owned!(f32, Standard, &[0x00, 0x00]).unwrap_err(),
        BincodeError::UnexpectedEnd
    );
}

#[test]
fn f64_value() {
    let result = parse_owned!(f64, Standard, &E.f64(2.718281828))
        .unwrap()
        .unwrap();
    assert!((result - 2.718281828).abs() < 1e-12);
}

#[test]
fn char_ascii() {
    assert_eq!(parse_owned!(char, Standard, &E.char('A')), Ok(Some('A')));
}

#[test]
fn char_unicode() {
    assert_eq!(parse_owned!(char, Standard, &E.char('€')), Ok(Some('€')));
}

#[test]
fn char_invalid_lead_byte_misses() {
    assert_eq!(parse_owned!(char, Standard, &[0xff]), Ok(None));
}

#[test]
fn char_truncated_multibyte_errors() {
    assert_eq!(
        parse_owned!(char, Standard, &[0xC3]).unwrap_err(),
        BincodeError::UnexpectedEnd
    );
}

// --- String (owned-family stand-in for &str) ---

#[cfg(feature = "alloc")]
#[test]
fn string_empty() {
    assert_eq!(
        parse_owned!(String, Standard, &E.str("")),
        Ok(Some(String::new()))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn string_hello() {
    assert_eq!(
        parse_owned!(String, Standard, &E.str("hello")),
        Ok(Some("hello".to_string()))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn string_truncated_errors() {
    let mut data = E.len(5);
    data.extend_from_slice(b"hi");
    assert_eq!(
        parse_owned!(String, Standard, &data).unwrap_err(),
        BincodeError::UnexpectedEnd
    );
}

#[cfg(feature = "alloc")]
#[test]
fn string_invalid_utf8_errors() {
    let mut data = E.len(2);
    data.extend_from_slice(&[0xff, 0xfe]);
    assert_eq!(
        parse_owned!(String, Standard, &data).unwrap_err(),
        BincodeError::InvalidUtf8
    );
}

// --- Vec<u8> (owned-family stand-in for &[u8]) ---

#[cfg(feature = "alloc")]
#[test]
fn bytes_empty() {
    assert_eq!(
        parse_owned!(Vec<u8>, Standard, &E.bytes(&[])),
        Ok(Some(vec![]))
    );
}

#[cfg(feature = "alloc")]
#[test]
fn bytes_values() {
    let data = &[0x00u8, 0x01, 0xff];
    assert_eq!(
        parse_owned!(Vec<u8>, Standard, &E.bytes(data)),
        Ok(Some(data.to_vec()))
    );
}

// --- Option ---

#[test]
fn option_none() {
    assert_eq!(parse_owned!(Option<u32>, Standard, &E.none()), Ok(Some(None)));
}

#[test]
fn option_some_u32() {
    let data = E.some(&E.u32(42));
    assert_eq!(
        parse_owned!(Option<u32>, Standard, &data),
        Ok(Some(Some(42u32)))
    );
}

#[test]
fn option_invalid_tag_misses() {
    assert_eq!(parse_owned!(Option<u32>, Standard, &[0x02]), Ok(None));
}

#[test]
fn trailing_bytes_errors() {
    let mut data = E.u32(42);
    data.push(0x00);
    assert_eq!(
        parse_owned!(u32, Standard, &data).unwrap_err(),
        BincodeError::ExpectedEnd
    );
}

// --- Non-canonical varint rejection (mirrors primitives_borrow.rs) ---

#[test]
fn u16_rejects_non_canonical_wide_prefix() {
    let data = E.varint_with_prefix(254, 100);
    assert_eq!(parse_owned!(u16, Standard, &data), Ok(None));
}

#[test]
fn u32_rejects_non_canonical_wide_prefix() {
    let data = E.varint_with_prefix(253, 100);
    assert_eq!(parse_owned!(u32, Standard, &data), Ok(None));
}

#[test]
fn u64_rejects_non_canonical_wide_prefix() {
    let data = E.varint_with_prefix(254, 100);
    assert_eq!(parse_owned!(u64, Standard, &data), Ok(None));
}

#[test]
fn i16_rejects_non_canonical_wide_prefix() {
    let data = E.varint_with_prefix(254, 100);
    assert_eq!(parse_owned!(i16, Standard, &data), Ok(None));
}

#[test]
fn u128_has_no_narrower_width_to_violate() {
    let data = E.varint_with_prefix(254, 100);
    assert_eq!(parse_owned!(u128, Standard, &data), Ok(Some(100)));
}

#[cfg(feature = "alloc")]
#[test]
fn string_length_non_canonical_prefix_errors() {
    let mut data = E.varint_with_prefix(254, 3);
    data.extend_from_slice(b"abc");
    assert_eq!(
        parse_owned!(String, Standard, &data).unwrap_err(),
        BincodeError::NonCanonicalVarint
    );
}

#[test]
fn varint_prefix_255_errors() {
    assert_eq!(
        parse_owned!(u32, Standard, &[0xff]).unwrap_err(),
        BincodeError::InvalidVarint
    );
}

#[test]
fn skip_is_unsupported() {
    use strede::{DeserializerOwned, EntryOwned, Probe, SharedBuf};
    use strede_bincode::chunked::ChunkedBincodeDeserializer;
    use strede_test_util::block_on_loop;

    let input: &[u8] = &[0x00];
    let result: Result<(), BincodeError> = block_on_loop(SharedBuf::with_async(
        input,
        async |buf: &mut &[u8]| {
            *buf = &[];
        },
        async |shared| {
            let de = ChunkedBincodeDeserializer::<Standard, _, _>::new(shared);
            de.entry(|[e]| async move {
                let claim = e.skip().await?;
                Ok::<_, BincodeError>(Probe::Hit((claim, ())))
            })
            .await
            .map(|_| ())
        },
    ));
    assert_eq!(result.unwrap_err(), BincodeError::CannotSkip);
}
