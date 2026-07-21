//! Borrow-family MsgpackValue deserialization tests.
#![cfg(feature = "alloc")]

mod helpers;
use helpers::*;

use strede::{Deserialize, Probe};
use strede_msgpack::{MsgpackDeserializer, MsgpackError, MsgpackValue};
use strede_test_util::block_on;

fn parse(input: &[u8]) -> Result<Option<MsgpackValue>, MsgpackError> {
    let de = MsgpackDeserializer::new(input);
    match block_on(MsgpackValue::deserialize(de, ()))? {
        Probe::Hit((_, v)) => Ok(Some(v)),
        Probe::Miss => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// Scalars
// ---------------------------------------------------------------------------

#[test]
fn nil() {
    assert!(matches!(
        parse(&[0xc0]).unwrap().unwrap(),
        MsgpackValue::Nil
    ));
}

#[test]
fn bool_true() {
    assert!(matches!(
        parse(&[0xc3]).unwrap().unwrap(),
        MsgpackValue::Bool(true)
    ));
}

#[test]
fn bool_false() {
    assert!(matches!(
        parse(&[0xc2]).unwrap().unwrap(),
        MsgpackValue::Bool(false)
    ));
}

#[test]
fn positive_fixint() {
    assert!(matches!(
        parse(&[42]).unwrap().unwrap(),
        MsgpackValue::UInt(42)
    ));
}

#[test]
fn negative_fixint() {
    assert!(matches!(
        parse(&[0xff]).unwrap().unwrap(),
        MsgpackValue::Int(-1)
    ));
}

#[test]
fn uint64() {
    let bytes = [0xcf, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff];
    assert!(matches!(
        parse(&bytes).unwrap().unwrap(),
        MsgpackValue::UInt(u64::MAX)
    ));
}

#[test]
fn float32_value() {
    let bytes = helpers::float32(1.5);
    assert!(matches!(parse(&bytes).unwrap().unwrap(), MsgpackValue::Float32(v) if v == 1.5));
}

#[test]
fn float64_value() {
    let bytes = helpers::float64(2.5);
    assert!(matches!(parse(&bytes).unwrap().unwrap(), MsgpackValue::Float64(v) if v == 2.5));
}

// ---------------------------------------------------------------------------
// Str / Bin
// ---------------------------------------------------------------------------

#[test]
fn fixstr_value() {
    let bytes = helpers::fixstr("hello");
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Str(s) if s == "hello"));
}

#[test]
fn empty_str() {
    let bytes = helpers::fixstr("");
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Str(s) if s.is_empty()));
}

#[test]
fn bin8_value() {
    let bytes = bin8(&[1, 2, 3]);
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Bin(b) if b == [1, 2, 3]));
}

#[test]
fn empty_bin() {
    let bytes = bin8(&[]);
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Bin(b) if b.is_empty()));
}

// ---------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------

#[test]
fn empty_array() {
    let bytes = [fixarray(0)];
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Array(a) if a.is_empty()));
}

#[test]
fn flat_array() {
    let mut bytes = vec![fixarray(3)];
    bytes.push(1); // fixint 1
    bytes.push(2); // fixint 2
    bytes.push(3); // fixint 3
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Array(items) = v {
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0], MsgpackValue::UInt(1)));
        assert!(matches!(items[1], MsgpackValue::UInt(2)));
        assert!(matches!(items[2], MsgpackValue::UInt(3)));
    } else {
        panic!("expected Array");
    }
}

#[test]
fn nested_array() {
    // [[1, 2], [3]]
    let mut bytes = vec![fixarray(2)];
    bytes.push(fixarray(2));
    bytes.push(1);
    bytes.push(2);
    bytes.push(fixarray(1));
    bytes.push(3);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Array(outer) = v {
        assert_eq!(outer.len(), 2);
        if let MsgpackValue::Array(inner) = &outer[0] {
            assert_eq!(inner.len(), 2);
        } else {
            panic!("expected inner Array");
        }
    } else {
        panic!("expected Array");
    }
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

#[test]
fn empty_map() {
    let bytes = [fixmap(0)];
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Map(m) if m.is_empty()));
}

#[test]
fn string_key_map() {
    let key = fixstr("x");
    let val = [7u8]; // fixint 7
    let bytes = build_map(&[(&key, &val)]);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Map(pairs) = v {
        assert_eq!(pairs.len(), 1);
        assert!(matches!(&pairs[0].0, MsgpackValue::Str(s) if s == "x"));
        assert!(matches!(pairs[0].1, MsgpackValue::UInt(7)));
    } else {
        panic!("expected Map");
    }
}

#[test]
fn int_key_map() {
    let key = [1u8]; // fixint 1
    let val = [0xc0u8]; // nil
    let bytes = build_map(&[(&key, &val)]);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Map(pairs) = v {
        assert!(matches!(pairs[0].0, MsgpackValue::UInt(1)));
        assert!(matches!(pairs[0].1, MsgpackValue::Nil));
    } else {
        panic!("expected Map");
    }
}

// ---------------------------------------------------------------------------
// Ext
// ---------------------------------------------------------------------------

#[test]
fn fixext_non_timestamp() {
    let bytes = fixext1(3, 0xab);
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Ext { type_id: 3, ref data } if data == &[0xab]));
}

#[test]
fn ext8_non_timestamp() {
    let bytes = ext8(7, b"data");
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Ext { type_id: 7, ref data } if data == b"data"));
}

// ---------------------------------------------------------------------------
// Timestamp variant
// ---------------------------------------------------------------------------

#[test]
fn timestamp32_variant() {
    let bytes = timestamp32(1_700_000_000);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Timestamp(ts) = v {
        assert_eq!(ts.seconds, 1_700_000_000);
        assert_eq!(ts.nanoseconds, 0);
    } else {
        panic!("expected Timestamp");
    }
}

#[test]
fn timestamp64_variant() {
    let bytes = timestamp64(500_000_000, 9999999999);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Timestamp(ts) = v {
        assert_eq!(ts.nanoseconds, 500_000_000);
        assert_eq!(ts.seconds, 9999999999);
    } else {
        panic!("expected Timestamp");
    }
}

#[test]
fn timestamp96_variant() {
    let bytes = timestamp96(1, -100);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Timestamp(ts) = v {
        assert_eq!(ts.seconds, -100);
        assert_eq!(ts.nanoseconds, 1);
    } else {
        panic!("expected Timestamp");
    }
}
