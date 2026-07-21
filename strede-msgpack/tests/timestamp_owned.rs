//! Owned-family MsgpackTimestamp deserialization tests.

mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_msgpack::MsgpackTimestamp;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_test_util::block_on_loop;

macro_rules! parse {
    ($input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
                match MsgpackTimestamp::deserialize_owned(de, ()).await.unwrap() {
                    Probe::Hit((_, v)) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
}

#[test]
fn timestamp32_roundtrip() {
    let bytes = timestamp32(1_700_000_000);
    let ts = parse!(&bytes).unwrap();
    assert_eq!(ts.seconds, 1_700_000_000);
    assert_eq!(ts.nanoseconds, 0);
}

#[test]
fn timestamp64_roundtrip() {
    let bytes = timestamp64(999_999_999, 0x3_FFFF_FFFF);
    let ts = parse!(&bytes).unwrap();
    assert_eq!(ts.nanoseconds, 999_999_999);
    assert_eq!(ts.seconds, 0x3_FFFF_FFFF);
}

#[cfg(feature = "alloc")]
#[test]
fn timestamp96_roundtrip() {
    let bytes = timestamp96(123_456_789, -62_135_596_800);
    let ts = parse!(&bytes).unwrap();
    assert_eq!(ts.seconds, -62_135_596_800);
    assert_eq!(ts.nanoseconds, 123_456_789);
}

#[cfg(not(feature = "alloc"))]
#[test]
fn timestamp96_misses_without_alloc() {
    let bytes = timestamp96(0, -1);
    assert!(parse!(&bytes).is_none());
}

#[test]
fn non_timestamp_ext_misses() {
    let bytes = fixext4(5, [0x01, 0x02, 0x03, 0x04]);
    assert!(parse!(&bytes).is_none());
}

#[test]
fn non_ext_token_misses() {
    assert!(parse!(&[0xc0u8]).is_none());
    assert!(parse!(&[42u8]).is_none());
}
