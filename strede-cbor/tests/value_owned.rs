//! Owned-family `CborValue` deserialization tests.
#![cfg(feature = "alloc")]

extern crate std;
mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::{CborError, CborValue, ChunkedCborDeserializer};
use strede_test_util::block_on_loop;

fn parse(input: &[u8]) -> Result<Option<CborValue>, CborError> {
    let input: &'static [u8] = std::boxed::Box::leak(input.to_vec().into_boxed_slice());
    block_on_loop(SharedBuf::with_async(
        input,
        async |buf: &mut &[u8]| {
            *buf = &[];
        },
        async |shared| {
            let de = ChunkedCborDeserializer::new(shared);
            match CborValue::deserialize_owned(de, ()).await? {
                Probe::Hit((_, v)) => Ok(Some(v)),
                Probe::Miss => Ok(None),
            }
        },
    ))
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

#[test]
fn null_value() {
    assert!(matches!(
        parse(&[cbor_null()]).unwrap().unwrap(),
        CborValue::Null
    ));
}

#[test]
fn undefined_value() {
    assert!(matches!(
        parse(&[cbor_undefined()]).unwrap().unwrap(),
        CborValue::Undefined
    ));
}

#[test]
fn bool_true_value() {
    assert!(matches!(
        parse(&[cbor_true()]).unwrap().unwrap(),
        CborValue::Bool(true)
    ));
}

#[test]
fn uint_small_value() {
    assert!(matches!(
        parse(&[uint_small(10)]).unwrap().unwrap(),
        CborValue::UInt(10)
    ));
}

#[test]
fn negint_in_i64_range() {
    let bytes = negint16(1000);
    assert!(matches!(
        parse(&bytes).unwrap().unwrap(),
        CborValue::Int(-1001)
    ));
}

#[test]
fn negint_overflow() {
    let mut tok = uint64(u64::MAX);
    tok[0] = 0x3b; // major type 1, same argument encoding as uint64
    assert!(matches!(
        parse(&tok).unwrap().unwrap(),
        CborValue::NegIntOverflow(n) if n == u64::MAX
    ));
}

#[test]
fn float32_value() {
    let bytes = float32(1.5);
    assert!(matches!(parse(&bytes).unwrap().unwrap(), CborValue::Float(v) if v == 1.5));
}

#[test]
fn float64_value() {
    let bytes = float64(2.5);
    assert!(matches!(parse(&bytes).unwrap().unwrap(), CborValue::Float(v) if v == 2.5));
}

// ---------------------------------------------------------------------------
// Simple values (major type 7, not false/true/null/undefined)
// ---------------------------------------------------------------------------

#[test]
fn simple_direct_value() {
    let byte = simple_direct(16);
    assert!(matches!(
        parse(&[byte]).unwrap().unwrap(),
        CborValue::Simple(16)
    ));
}

#[test]
fn simple_ext_value() {
    let bytes = simple_ext(255);
    assert!(matches!(
        parse(&bytes).unwrap().unwrap(),
        CborValue::Simple(255)
    ));
}

#[test]
fn simple_ext_reserved_encoding_errors() {
    let bytes = simple_ext_reserved(16);
    assert!(matches!(
        parse(&bytes),
        Err(CborError::UnexpectedByte { .. })
    ));
}

// ---------------------------------------------------------------------------
// Str / Bstr
// ---------------------------------------------------------------------------

#[test]
fn tstr_value() {
    let bytes = tstr("hello");
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, CborValue::Tstr(s) if s == "hello"));
}

#[test]
fn bstr_value() {
    let bytes = bstr(&[1, 2, 3]);
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, CborValue::Bstr(b) if b == [1, 2, 3]));
}

// ---------------------------------------------------------------------------
// Array / Map
// ---------------------------------------------------------------------------

#[test]
fn empty_array() {
    let bytes = array(0);
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, CborValue::Array(a) if a.is_empty()));
}

#[test]
fn nested_array() {
    let mut bytes = array(2);
    bytes.extend(array(2));
    bytes.push(uint_small(1));
    bytes.push(uint_small(2));
    bytes.extend(array(1));
    bytes.push(uint_small(3));
    let v = parse(&bytes).unwrap().unwrap();
    if let CborValue::Array(outer) = v {
        assert_eq!(outer.len(), 2);
        if let CborValue::Array(inner) = &outer[0] {
            assert_eq!(inner.len(), 2);
        } else {
            panic!("expected inner Array");
        }
    } else {
        panic!("expected Array");
    }
}

#[test]
fn string_key_map() {
    let key = tstr("x");
    let val = [uint_small(7)];
    let bytes = build_map(&[(&key, &val)]);
    let v = parse(&bytes).unwrap().unwrap();
    if let CborValue::Map(pairs) = v {
        assert_eq!(pairs.len(), 1);
        assert!(matches!(&pairs[0].0, CborValue::Tstr(s) if s == "x"));
        assert!(matches!(pairs[0].1, CborValue::UInt(7)));
    } else {
        panic!("expected Map");
    }
}

#[test]
fn indefinite_map_value() {
    let key = tstr("a");
    let val = [uint_small(1)];
    let bytes = build_map_indef(&[(&key, &val)]);
    let v = parse(&bytes).unwrap().unwrap();
    if let CborValue::Map(pairs) = v {
        assert_eq!(pairs.len(), 1);
    } else {
        panic!("expected Map");
    }
}

// ---------------------------------------------------------------------------
// Tag
// ---------------------------------------------------------------------------

#[test]
fn tag_value() {
    let mut bytes = tag(1);
    bytes.push(uint_small(5));
    let v = parse(&bytes).unwrap().unwrap();
    if let CborValue::Tag { number, value } = v {
        assert_eq!(number, 1);
        assert!(matches!(*value, CborValue::UInt(5)));
    } else {
        panic!("expected Tag");
    }
}
