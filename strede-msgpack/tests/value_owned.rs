//! Owned-family MsgpackValue deserialization tests.
#![cfg(feature = "alloc")]

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_msgpack::{MsgpackError, MsgpackValue};
use strede_test_util::block_on_loop;

fn parse(input: &[u8]) -> Result<Option<MsgpackValue>, MsgpackError> {
    let input: &'static [u8] = Box::leak(input.to_vec().into_boxed_slice());
    block_on_loop(SharedBuf::with_async(
        input,
        async |buf: &mut &[u8]| {
            *buf = &[];
        },
        async |shared| {
            let de = ChunkedMsgpackDeserializer::new(shared);
            match MsgpackValue::deserialize_owned(de, ()).await? {
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
fn float32() {
    let bytes = helpers::float32(1.5);
    assert!(matches!(parse(&bytes).unwrap().unwrap(), MsgpackValue::Float32(v) if v == 1.5));
}

#[test]
fn float64() {
    let bytes = helpers::float64(2.5);
    assert!(matches!(parse(&bytes).unwrap().unwrap(), MsgpackValue::Float64(v) if v == 2.5));
}

// ---------------------------------------------------------------------------
// Str / Bin
// ---------------------------------------------------------------------------

#[test]
fn fixstr() {
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
    let mut bytes = vec![fixarray(2)];
    bytes.push(10); // fixint
    bytes.push(20);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Array(items) = v {
        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], MsgpackValue::UInt(10)));
        assert!(matches!(items[1], MsgpackValue::UInt(20)));
    } else {
        panic!("expected Array");
    }
}

#[test]
fn nested_array() {
    let mut bytes = vec![fixarray(1)];
    bytes.push(fixarray(1));
    bytes.push(0xc0); // nil
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Array(outer) = v {
        if let MsgpackValue::Array(inner) = &outer[0] {
            assert!(matches!(inner[0], MsgpackValue::Nil));
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
    let key = helpers::fixstr("k");
    let val = [99u8];
    let bytes = build_map(&[(&key, &val)]);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Map(pairs) = v {
        assert!(matches!(&pairs[0].0, MsgpackValue::Str(s) if s == "k"));
        assert!(matches!(pairs[0].1, MsgpackValue::UInt(99)));
    } else {
        panic!("expected Map");
    }
}

// ---------------------------------------------------------------------------
// Ext
// ---------------------------------------------------------------------------

#[test]
fn fixext_non_timestamp() {
    let bytes = fixext2(9, [0xde, 0xad]);
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Ext { type_id: 9, ref data } if data == &[0xde, 0xad]));
}

#[test]
fn ext8_non_timestamp() {
    let bytes = ext8(4, b"hello");
    let v = parse(&bytes).unwrap().unwrap();
    assert!(matches!(v, MsgpackValue::Ext { type_id: 4, ref data } if data == b"hello"));
}

// ---------------------------------------------------------------------------
// Timestamp variant
// ---------------------------------------------------------------------------

#[test]
fn timestamp32_variant() {
    let bytes = timestamp32(1_000_000);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Timestamp(ts) = v {
        assert_eq!(ts.seconds, 1_000_000);
        assert_eq!(ts.nanoseconds, 0);
    } else {
        panic!("expected Timestamp");
    }
}

#[test]
fn timestamp64_variant() {
    let bytes = timestamp64(100, 9_999_999_999);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Timestamp(ts) = v {
        assert_eq!(ts.nanoseconds, 100);
        assert_eq!(ts.seconds, 9_999_999_999);
    } else {
        panic!("expected Timestamp");
    }
}

#[test]
fn timestamp96_variant() {
    let bytes = timestamp96(0, -999);
    let v = parse(&bytes).unwrap().unwrap();
    if let MsgpackValue::Timestamp(ts) = v {
        assert_eq!(ts.seconds, -999);
        assert_eq!(ts.nanoseconds, 0);
    } else {
        panic!("expected Timestamp");
    }
}
