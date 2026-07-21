//! `deserialize_number_chunks` tests — borrow family.
#![allow(clippy::approx_constant)]

mod helpers;
use helpers::*;

use strede::Deserializer;
use strede::{Ascii, BigEndian, Chunk, Entry, LittleEndian, NumberAccess, Probe};
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

/// Parse number chunks with `BigEndian` encoding, collecting all bytes.
/// Returns `None` if `deserialize_number_chunks` returns `Miss`.
fn collect_be(input: &[u8]) -> Option<Vec<u8>> {
    match block_on(async {
        let de = MsgpackDeserializer::new(input);
        de.entry(|[e]| async move {
            let mut acc = match e.deserialize_number_chunks::<BigEndian>().await? {
                Probe::Hit(a) => a,
                Probe::Miss => return Ok(Probe::Miss),
            };
            let mut bytes = Vec::<u8>::new();
            loop {
                match <_ as NumberAccess<BigEndian>>::next_number_chunk(acc, |b: &[u8]| b.to_vec())
                    .await?
                {
                    Chunk::Data((next, chunk)) => {
                        bytes.extend_from_slice(&chunk);
                        acc = next;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, bytes))),
                }
            }
        })
        .await
        .unwrap()
    }) {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn ufixint_be() {
    assert_eq!(collect_be(&[42u8]), Some(vec![42u8]));
}

#[test]
fn uint8_be() {
    assert_eq!(collect_be(&uint8(200)), Some(vec![200u8]));
}

#[test]
fn uint16_be() {
    assert_eq!(collect_be(&uint16(0x0102)), Some(vec![0x01, 0x02]));
}

#[test]
fn uint32_be() {
    assert_eq!(
        collect_be(&uint32(0x01020304)),
        Some(vec![0x01, 0x02, 0x03, 0x04])
    );
}

#[test]
fn uint64_be() {
    let v: u64 = 0x0102030405060708;
    assert_eq!(collect_be(&uint64(v)), Some(v.to_be_bytes().to_vec()));
}

#[test]
fn ifixint_be() {
    // 0xff = negative fixint -1
    assert_eq!(collect_be(&[0xffu8]), Some(vec![0xffu8]));
}

#[test]
fn int8_be() {
    assert_eq!(collect_be(&int8(-5)), Some(vec![(-5i8) as u8]));
}

#[test]
fn int16_be() {
    assert_eq!(collect_be(&int16(-1)), Some(vec![0xff, 0xff]));
}

#[test]
fn float32_misses() {
    assert_eq!(collect_be(&float32(1.5)), None);
}

#[test]
fn float64_misses() {
    assert_eq!(collect_be(&float64(3.14)), None);
}

#[test]
fn fixstr_misses() {
    assert_eq!(collect_be(&fixstr("hi")), None);
}

#[test]
fn ascii_encoding_misses() {
    // Ascii is not a supported encoding for msgpack numbers.
    // We verify by attempting to collect — None means Miss.
    // Since collect_be uses BigEndian, we need a direct check.
    // Use collect_be as proxy: fixint 42 hits BE, so just verify
    // that Ascii returns None via a separate collect_ascii helper.
    assert_eq!(collect_ascii(&[42u8]), None);
}

fn collect_ascii(input: &[u8]) -> Option<Vec<u8>> {
    match block_on(async {
        let de = MsgpackDeserializer::new(input);
        de.entry(|[e]| async move {
            let mut acc = match e.deserialize_number_chunks::<Ascii>().await? {
                Probe::Hit(a) => a,
                Probe::Miss => return Ok(Probe::Miss),
            };
            let mut bytes = Vec::<u8>::new();
            loop {
                match <_ as NumberAccess<Ascii>>::next_number_chunk(acc, |b: &str| {
                    b.as_bytes().to_vec()
                })
                .await?
                {
                    Chunk::Data((next, chunk)) => {
                        bytes.extend_from_slice(&chunk);
                        acc = next;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, bytes))),
                }
            }
        })
        .await
        .unwrap()
    }) {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn little_endian_encoding_misses() {
    assert_eq!(collect_le(&[42u8]), None);
}

fn collect_le(input: &[u8]) -> Option<Vec<u8>> {
    match block_on(async {
        let de = MsgpackDeserializer::new(input);
        de.entry(|[e]| async move {
            let mut acc = match e.deserialize_number_chunks::<LittleEndian>().await? {
                Probe::Hit(a) => a,
                Probe::Miss => return Ok(Probe::Miss),
            };
            let mut bytes = Vec::<u8>::new();
            loop {
                match <_ as NumberAccess<LittleEndian>>::next_number_chunk(acc, |b: &[u8]| {
                    b.to_vec()
                })
                .await?
                {
                    Chunk::Data((next, chunk)) => {
                        bytes.extend_from_slice(&chunk);
                        acc = next;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, bytes))),
                }
            }
        })
        .await
        .unwrap()
    }) {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}
