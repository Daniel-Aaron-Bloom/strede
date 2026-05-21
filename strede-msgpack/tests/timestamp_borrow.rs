//! Borrow-family MsgpackTimestamp deserialization tests.

mod helpers;
use helpers::*;

use strede::{Deserialize, Probe};
use strede_msgpack::{MsgpackDeserializer, MsgpackError, MsgpackTimestamp};
use strede_test_util::block_on;

fn parse<T>(input: &[u8]) -> Result<Option<T>, MsgpackError>
where
    for<'de> T: Deserialize<'de, MsgpackDeserializer<'de>, Extra = ()>,
{
    let de = MsgpackDeserializer::new(input);
    match block_on(T::deserialize(de, ()))? {
        Probe::Hit((_, v)) => Ok(Some(v)),
        Probe::Miss => Ok(None),
    }
}

#[test]
fn timestamp32_roundtrip() {
    let bytes = timestamp32(1_700_000_000);
    let ts = parse::<MsgpackTimestamp>(&bytes).unwrap().unwrap();
    assert_eq!(ts.seconds, 1_700_000_000);
    assert_eq!(ts.nanoseconds, 0);
}

#[test]
fn timestamp32_zero() {
    let bytes = timestamp32(0);
    let ts = parse::<MsgpackTimestamp>(&bytes).unwrap().unwrap();
    assert_eq!(ts.seconds, 0);
    assert_eq!(ts.nanoseconds, 0);
}

#[test]
fn timestamp64_roundtrip() {
    let bytes = timestamp64(999_999_999, 0x3_FFFF_FFFF);
    let ts = parse::<MsgpackTimestamp>(&bytes).unwrap().unwrap();
    assert_eq!(ts.nanoseconds, 999_999_999);
    assert_eq!(ts.seconds, 0x3_FFFF_FFFF);
}

#[test]
fn timestamp64_zero_nsec() {
    let bytes = timestamp64(0, 1234567890);
    let ts = parse::<MsgpackTimestamp>(&bytes).unwrap().unwrap();
    assert_eq!(ts.seconds, 1234567890);
    assert_eq!(ts.nanoseconds, 0);
}

#[test]
fn timestamp96_roundtrip() {
    let bytes = timestamp96(123_456_789, -62_135_596_800);
    let ts = parse::<MsgpackTimestamp>(&bytes).unwrap().unwrap();
    assert_eq!(ts.seconds, -62_135_596_800);
    assert_eq!(ts.nanoseconds, 123_456_789);
}

#[test]
fn timestamp96_negative_epoch() {
    let bytes = timestamp96(0, -1);
    let ts = parse::<MsgpackTimestamp>(&bytes).unwrap().unwrap();
    assert_eq!(ts.seconds, -1);
    assert_eq!(ts.nanoseconds, 0);
}

#[test]
fn non_timestamp_ext_misses() {
    let bytes = fixext4(5, [0x01, 0x02, 0x03, 0x04]);
    assert!(parse::<MsgpackTimestamp>(&bytes).unwrap().is_none());
}

#[test]
fn non_ext_token_misses() {
    assert!(parse::<MsgpackTimestamp>(&[0xc0]).unwrap().is_none());
    assert!(parse::<MsgpackTimestamp>(&[42]).unwrap().is_none());
}
