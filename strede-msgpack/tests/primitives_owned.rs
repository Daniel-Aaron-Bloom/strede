//! Owned-family primitive deserialization: nil, bool, integers, floats, strings, bytes.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
                match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                {
                    Probe::Hit((_, v)) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
}

// --- nil / unit ---

#[test]
fn nil_as_unit() {
    assert_eq!(parse!((), &[0xc0]), Some(()));
}

#[test]
fn unit_misses_non_nil() {
    assert_eq!(parse!((), &[0xc3]), None);
    assert_eq!(parse!((), &[42]), None);
}

// --- bool ---

#[test]
fn bool_false() {
    assert_eq!(parse!(bool, &[0xc2]), Some(false));
}

#[test]
fn bool_true() {
    assert_eq!(parse!(bool, &[0xc3]), Some(true));
}

#[test]
fn bool_misses_non_bool() {
    assert_eq!(parse!(bool, &[0xc0]), None);
    assert_eq!(parse!(bool, &[42]), None);
}

// --- integers ---

#[test]
fn positive_fixint() {
    assert_eq!(parse!(u8, &[0x00]), Some(0));
    assert_eq!(parse!(u8, &[42]), Some(42));
    assert_eq!(parse!(u8, &[0x7f]), Some(127));
}

#[test]
fn negative_fixint() {
    assert_eq!(parse!(i8, &[0xff]), Some(-1));
    assert_eq!(parse!(i8, &[0xe0]), Some(-32));
}

#[test]
fn uint8_format() {
    assert_eq!(parse!(u8, &[0xcc, 200]), Some(200));
    assert_eq!(parse!(u64, &[0xcc, 200]), Some(200));
}

#[test]
fn uint16_format() {
    let bytes = uint16(1000);
    assert_eq!(parse!(u16, &bytes), Some(1000));
    assert_eq!(parse!(u32, &bytes), Some(1000));
}

#[test]
fn uint32_format() {
    let bytes = uint32(100_000);
    assert_eq!(parse!(u32, &bytes), Some(100_000));
    assert_eq!(parse!(u64, &bytes), Some(100_000));
}

#[test]
fn int8_format() {
    let bytes = int8(-100);
    assert_eq!(parse!(i8, &bytes), Some(-100));
    assert_eq!(parse!(i64, &bytes), Some(-100));
}

#[test]
fn int16_format() {
    let bytes = int16(-1000);
    assert_eq!(parse!(i16, &bytes), Some(-1000));
    assert_eq!(parse!(i32, &bytes), Some(-1000));
}

#[test]
fn negative_int_misses_unsigned() {
    let bytes = int8(-1);
    assert_eq!(parse!(u8, &bytes), None);
    assert_eq!(parse!(u64, &bytes), None);
}

#[test]
fn uint_out_of_range_misses() {
    let mut b = vec![0xcf];
    b.extend_from_slice(&u64::MAX.to_be_bytes());
    assert_eq!(parse!(u8, &b), None);
}

// --- floats ---

#[test]
#[allow(clippy::approx_constant)]
fn float32_value() {
    let bytes = float32(3.14f32);
    let result = parse!(f32, &bytes).unwrap();
    assert!((result - 3.14f32).abs() < 1e-6);
}

#[test]
#[allow(clippy::approx_constant)]
fn float64_value() {
    let bytes = float64(2.718281828);
    let result = parse!(f64, &bytes).unwrap();
    assert!((result - 2.718281828).abs() < 1e-12);
}

#[test]
fn integer_coerced_to_float() {
    assert_eq!(parse!(f64, &[42]), Some(42.0f64));
    assert_eq!(parse!(f32, &[0xcc, 100]), Some(100.0f32));
}

// --- strings (owned → String) ---

#[test]
fn fixstr_empty() {
    assert_eq!(parse!(String, &[0xa0]), Some(String::new()));
}

#[test]
fn fixstr_hello() {
    let bytes = fixstr("hello");
    assert_eq!(parse!(String, &bytes), Some("hello".into()));
}

#[test]
fn str8_format() {
    let s = "x".repeat(50);
    let mut bytes = vec![0xd9, 50];
    bytes.extend_from_slice(s.as_bytes());
    assert_eq!(parse!(String, &bytes), Some(s.clone()));
}

#[test]
fn str_misses_non_string() {
    assert_eq!(parse!(String, &[42]), None);
    assert_eq!(parse!(String, &[0xc0]), None);
}
