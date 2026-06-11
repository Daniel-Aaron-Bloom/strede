//! Primitive deserialization: bool, integers, floats, str, bytes, char, unit, option.

mod helpers;
use helpers::*;

use strede::Probe;
use strede_postcard::{PostcardDeserializer, PostcardError};
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Result<Option<T>, PostcardError>
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Ok(Some(v)),
        Probe::Miss => Ok(None),
    }
}

fn parse_err<'de, T>(input: &'de [u8]) -> PostcardError
where
    T: strede::Deserialize<'de, PostcardDeserializer<'de>, Extra = ()>,
{
    let de = PostcardDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())) {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    }
}

// --- unit ---

#[test]
fn unit_empty_input() {
    assert_eq!(parse::<()>(&[]), Ok(Some(())));
}

#[test]
fn unit_trailing_bytes_errors() {
    assert_eq!(parse_err::<()>(&[0x01]), PostcardError::ExpectedEnd);
}

// --- bool ---

#[test]
fn bool_false() {
    assert_eq!(parse::<bool>(&[0x00]), Ok(Some(false)));
}

#[test]
fn bool_true() {
    assert_eq!(parse::<bool>(&[0x01]), Ok(Some(true)));
}

#[test]
fn bool_invalid_misses() {
    assert_eq!(parse::<bool>(&[0x02]), Ok(None));
}

#[test]
fn bool_truncated_errors() {
    assert_eq!(parse_err::<bool>(&[]), PostcardError::UnexpectedEnd);
}

// --- u8 ---

#[test]
fn u8_zero() {
    assert_eq!(parse::<u8>(&varint(0)), Ok(Some(0)));
}

#[test]
fn u8_max() {
    assert_eq!(parse::<u8>(&varint(255)), Ok(Some(255)));
}

#[test]
fn u8_out_of_range_misses() {
    assert_eq!(parse::<u8>(&varint(256)), Ok(None));
}

// --- u16 ---

#[test]
fn u16_value() {
    assert_eq!(parse::<u16>(&varint(1000)), Ok(Some(1000)));
}

#[test]
fn u16_max() {
    assert_eq!(parse::<u16>(&varint(65535)), Ok(Some(65535)));
}

#[test]
fn u16_out_of_range_misses() {
    assert_eq!(parse::<u16>(&varint(65536)), Ok(None));
}

// --- u32 ---

#[test]
fn u32_value() {
    assert_eq!(parse::<u32>(&varint(100_000)), Ok(Some(100_000)));
}

#[test]
fn u32_max() {
    assert_eq!(parse::<u32>(&varint(u32::MAX as u64)), Ok(Some(u32::MAX)));
}

// --- u64 ---

#[test]
fn u64_max() {
    assert_eq!(parse::<u64>(&varint(u64::MAX)), Ok(Some(u64::MAX)));
}

// --- u128 ---

#[test]
fn u128_zero() {
    assert_eq!(parse::<u128>(&pu128(0)), Ok(Some(0u128)));
}

#[test]
fn u128_large() {
    let v = u128::MAX;
    assert_eq!(parse::<u128>(&pu128(v)), Ok(Some(v)));
}

// --- i8 ---

#[test]
fn i8_positive() {
    assert_eq!(parse::<i8>(&zigzag(42)), Ok(Some(42)));
}

#[test]
fn i8_negative() {
    assert_eq!(parse::<i8>(&zigzag(-1)), Ok(Some(-1)));
}

#[test]
fn i8_min() {
    assert_eq!(parse::<i8>(&zigzag(-128)), Ok(Some(-128)));
}

#[test]
fn i8_max() {
    assert_eq!(parse::<i8>(&zigzag(127)), Ok(Some(127)));
}

#[test]
fn i8_out_of_range_misses() {
    assert_eq!(parse::<i8>(&zigzag(200)), Ok(None));
}

// --- i16 ---

#[test]
fn i16_negative() {
    assert_eq!(parse::<i16>(&zigzag(-1000)), Ok(Some(-1000)));
}

// --- i32 ---

#[test]
fn i32_negative() {
    assert_eq!(parse::<i32>(&zigzag(-100_000)), Ok(Some(-100_000)));
}

// --- i64 ---

#[test]
fn i64_min() {
    assert_eq!(parse::<i64>(&zigzag(i64::MIN)), Ok(Some(i64::MIN)));
}

// --- i128 ---

#[test]
fn i128_negative() {
    let v = i128::MIN;
    assert_eq!(parse::<i128>(&pi128(v)), Ok(Some(v)));
}

// --- f32 ---

#[test]
#[allow(clippy::approx_constant)]
fn f32_value() {
    let result = parse::<f32>(&pf32(3.14f32)).unwrap().unwrap();
    assert!((result - 3.14f32).abs() < 1e-6);
}

#[test]
fn f32_truncated_errors() {
    assert_eq!(parse_err::<f32>(&[0x00, 0x00]), PostcardError::UnexpectedEnd);
}

#[test]
fn f32_nan() {
    let result = parse::<f32>(&pf32(f32::NAN)).unwrap().unwrap();
    assert!(result.is_nan());
}

#[test]
fn f32_pos_infinity() {
    assert_eq!(parse::<f32>(&pf32(f32::INFINITY)), Ok(Some(f32::INFINITY)));
}

#[test]
fn f32_neg_infinity() {
    assert_eq!(parse::<f32>(&pf32(f32::NEG_INFINITY)), Ok(Some(f32::NEG_INFINITY)));
}

// --- f64 ---

#[test]
fn f64_value() {
    let result = parse::<f64>(&pf64(2.718281828)).unwrap().unwrap();
    assert!((result - 2.718281828).abs() < 1e-12);
}

#[test]
fn f64_truncated_errors() {
    assert_eq!(parse_err::<f64>(&[0x00; 4]), PostcardError::UnexpectedEnd);
}

#[test]
fn f64_nan() {
    let result = parse::<f64>(&pf64(f64::NAN)).unwrap().unwrap();
    assert!(result.is_nan());
}

#[test]
fn f64_pos_infinity() {
    assert_eq!(parse::<f64>(&pf64(f64::INFINITY)), Ok(Some(f64::INFINITY)));
}

#[test]
fn f64_neg_infinity() {
    assert_eq!(parse::<f64>(&pf64(f64::NEG_INFINITY)), Ok(Some(f64::NEG_INFINITY)));
}

// --- char ---

#[test]
fn char_ascii() {
    assert_eq!(parse::<char>(&pchar('A')), Ok(Some('A')));
}

#[test]
fn char_unicode() {
    assert_eq!(parse::<char>(&pchar('€')), Ok(Some('€')));
}

#[test]
fn char_invalid_codepoint_misses() {
    // 0xD800 is a surrogate — not a valid char
    assert_eq!(parse::<char>(&varint(0xD800)), Ok(None));
}

// --- &str ---

#[test]
fn str_empty() {
    assert_eq!(parse::<&str>(&pstr("")), Ok(Some("")));
}

#[test]
fn str_hello() {
    assert_eq!(parse::<&str>(&pstr("hello")), Ok(Some("hello")));
}

#[test]
fn str_unicode() {
    assert_eq!(parse::<&str>(&pstr("héllo")), Ok(Some("héllo")));
}

#[test]
fn str_truncated_errors() {
    // length says 5 but only 2 bytes follow
    let mut data = varint(5);
    data.extend_from_slice(b"hi");
    assert_eq!(parse_err::<&str>(&data), PostcardError::UnexpectedEnd);
}

#[test]
fn str_invalid_utf8_errors() {
    let mut data = varint(2);
    data.extend_from_slice(&[0xff, 0xfe]); // invalid UTF-8
    assert_eq!(parse_err::<&str>(&data), PostcardError::InvalidUtf8);
}

// --- &[u8] ---

#[test]
fn bytes_empty() {
    assert_eq!(parse::<&[u8]>(&pbytes(&[])), Ok(Some(&[] as &[u8])));
}

#[test]
fn bytes_values() {
    let data = &[0x00u8, 0x01, 0xff];
    assert_eq!(parse::<&[u8]>(&pbytes(data)), Ok(Some(data.as_slice())));
}

// --- Option ---

#[test]
fn option_none() {
    assert_eq!(parse::<Option<u32>>(&pnone()), Ok(Some(None)));
}

#[test]
fn option_some_u32() {
    let mut data = psome(&[]);
    data.extend_from_slice(&varint(42));
    assert_eq!(parse::<Option<u32>>(&data), Ok(Some(Some(42u32))));
}

#[test]
fn option_some_str() {
    let mut data = psome(&[]);
    data.extend_from_slice(&pstr("hi"));
    assert_eq!(parse::<Option<&str>>(&data), Ok(Some(Some("hi"))));
}

#[test]
fn option_invalid_tag_misses() {
    assert_eq!(parse::<Option<u32>>(&[0x02]), Ok(None));
}

// --- trailing bytes ---

#[test]
fn trailing_bytes_errors() {
    let mut data = varint(42u64);
    data.push(0x00); // extra byte
    assert_eq!(parse_err::<u32>(&data), PostcardError::ExpectedEnd);
}
