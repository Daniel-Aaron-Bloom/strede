extern crate std;
mod helpers;

use strede::{Deserialize, Deserializer, Entry, Probe};
use strede_cbor::CborDeserializer;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    match block_on(T::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn uint_small() {
    assert_eq!(parse::<u8>(&[0x00]), Some(0));
    assert_eq!(parse::<u8>(&[0x17]), Some(23));
    assert_eq!(parse::<u64>(&[0x0a]), Some(10));
}

#[test]
fn uint8() {
    let enc = helpers::uint8(200);
    assert_eq!(parse::<u8>(&enc), Some(200));
    assert_eq!(parse::<u16>(&enc), Some(200));
    assert_eq!(parse::<u64>(&enc), Some(200));
}

#[test]
fn uint_u8_range() {
    assert_eq!(parse::<u8>(&helpers::uint8(255)), Some(255));
    // 256 doesn't fit u8
    assert_eq!(parse::<u8>(&helpers::uint16(256)), None);
}

#[test]
fn negint_small() {
    // 0x20 = negint(0) → actual = -1
    assert_eq!(parse::<i8>(&[0x20]), Some(-1));
    // 0x37 = negint(23) → actual = -24
    assert_eq!(parse::<i8>(&[0x37]), Some(-24));
    // 0x38 0x63 = negint(99) → actual = -100
    assert_eq!(parse::<i16>(&helpers::negint8(99)), Some(-100));
}

#[test]
fn bool_values() {
    assert_eq!(parse::<bool>(&[helpers::cbor_true()]), Some(true));
    assert_eq!(parse::<bool>(&[helpers::cbor_false()]), Some(false));
    assert_eq!(parse::<bool>(&[helpers::cbor_null()]), None);
}

#[test]
fn null_undefined() {
    assert_eq!(parse::<()>(&[helpers::cbor_null()]), Some(()));
    assert_eq!(parse::<()>(&[helpers::cbor_undefined()]), Some(()));
}

#[test]
fn float32() {
    let enc = helpers::float32(1.5f32);
    let v: f32 = parse(&enc).unwrap();
    assert_eq!(v, 1.5f32);
}

#[test]
fn float64() {
    let enc = helpers::float64(f64::INFINITY);
    let v: f64 = parse(&enc).unwrap();
    assert!(v.is_infinite());
}

#[test]
fn bstr_zero_copy() {
    let enc = helpers::bstr(b"hello");
    let de = CborDeserializer::new(&enc);
    let result = block_on(de.entry(|[e]| async move { e.deserialize_bytes().await })).unwrap();
    match result {
        Probe::Hit((_, b)) => assert_eq!(b, b"hello"),
        Probe::Miss => panic!("expected Hit"),
    }
}

#[test]
fn tstr_zero_copy() {
    let enc = helpers::tstr("hello");
    let de = CborDeserializer::new(&enc);
    let result = block_on(de.entry(|[e]| async move { e.deserialize_str().await })).unwrap();
    match result {
        Probe::Hit((_, s)) => assert_eq!(s, "hello"),
        Probe::Miss => panic!("expected Hit"),
    }
}

// --- char ---

#[test]
fn char_from_single_char_tstr() {
    let enc = helpers::tstr("x");
    assert_eq!(parse::<char>(&enc), Some('x'));
}

#[test]
fn char_misses_multi_char_tstr() {
    let enc = helpers::tstr("xy");
    assert_eq!(parse::<char>(&enc), None);
}

#[test]
fn char_misses_empty_tstr() {
    let enc = helpers::tstr("");
    assert_eq!(parse::<char>(&enc), None);
}

#[test]
fn char_misses_non_string() {
    assert_eq!(parse::<char>(&[0x00]), None);
    assert_eq!(parse::<char>(&[helpers::cbor_null()]), None);
}

#[test]
fn type_mismatch_returns_miss() {
    // uint where bool expected
    assert_eq!(parse::<bool>(&[0x01]), None);
    // bool where u8 expected
    assert_eq!(parse::<u8>(&[helpers::cbor_true()]), None);
}
