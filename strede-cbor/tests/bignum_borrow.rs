extern crate std;
mod helpers;

use strede::{Ascii, BigEndian, Chunk, Deserializer, Entry, LittleEndian, NumberAccess, Probe};
use strede_cbor::CborDeserializer;
use strede_test_util::block_on;

fn bignum(tag: u64, magnitude: &[u8]) -> Vec<u8> {
    let mut out = helpers::tag(tag);
    out.extend_from_slice(&helpers::bstr(magnitude));
    out
}

fn bignum_indef(tag: u64, chunks: &[&[u8]]) -> Vec<u8> {
    let mut out = helpers::tag(tag);
    out.push(0x5f); // indefinite bstr header
    for chunk in chunks {
        out.extend_from_slice(&helpers::bstr(chunk));
    }
    out.push(0xff); // break
    out
}

fn collect_bignum_bytes(input: &[u8]) -> Option<Vec<u8>> {
    let de = CborDeserializer::new(input);
    match block_on(async {
        de.entry(|[e]| async {
            let mut acc = Vec::<u8>::new();
            let mut chunks = match e.deserialize_number_chunks::<BigEndian>().await? {
                Probe::Hit(c) => c,
                Probe::Miss => return Ok(Probe::Miss),
            };
            let claim = loop {
                match NumberAccess::<BigEndian>::next_number_chunk(chunks, |b| acc.extend_from_slice(b)).await? {
                    Chunk::Data((next, ())) => chunks = next,
                    Chunk::Done(claim) => break claim,
                }
            };
            Ok(Probe::Hit((claim, acc)))
        })
        .await
    })
    .unwrap()
    {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn bignum_tag2_definite() {
    let input = bignum(2, &[0x01, 0x00]);
    assert_eq!(collect_bignum_bytes(&input), Some(vec![0x01, 0x00]));
}

#[test]
fn bignum_tag3_definite() {
    let input = bignum(3, &[0x00]);
    assert_eq!(collect_bignum_bytes(&input), Some(vec![0x00]));
}

#[test]
fn bignum_tag2_indefinite() {
    let input = bignum_indef(2, &[&[0x01], &[0x00]]);
    assert_eq!(collect_bignum_bytes(&input), Some(vec![0x01, 0x00]));
}

#[test]
fn plain_bstr_without_tag_misses() {
    let input = helpers::bstr(&[0x01, 0x00]);
    assert_eq!(collect_bignum_bytes(&input), None);
}

#[test]
fn non_bignum_tag_misses() {
    let mut input = helpers::tag(0);
    input.extend_from_slice(&helpers::bstr(&[0x01]));
    assert_eq!(collect_bignum_bytes(&input), None);
}

#[test]
fn bignum_empty_magnitude() {
    let input = bignum(2, &[]);
    assert_eq!(collect_bignum_bytes(&input), Some(vec![]));
}

#[test]
fn ascii_encoding_misses() {
    let input = bignum(2, &[0x01, 0x00]);
    let de = CborDeserializer::new(&input);
    let result = block_on(async {
        de.entry(|[e1, e2]| async {
            let ascii_missed = matches!(
                e1.deserialize_number_chunks::<Ascii>().await?,
                Probe::Miss
            );
            let mut acc = Vec::<u8>::new();
            let mut chunks = match e2.deserialize_number_chunks::<BigEndian>().await? {
                Probe::Hit(c) => c,
                Probe::Miss => return Ok(Probe::Miss),
            };
            let claim = loop {
                match NumberAccess::<BigEndian>::next_number_chunk(chunks, |b| acc.extend_from_slice(b)).await? {
                    Chunk::Data((next, ())) => chunks = next,
                    Chunk::Done(claim) => break claim,
                }
            };
            Ok(Probe::Hit((claim, (ascii_missed, acc))))
        })
        .await
    })
    .unwrap();
    let (ascii_missed, bytes) = match result {
        Probe::Hit((_, v)) => v,
        Probe::Miss => panic!("expected Hit"),
    };
    assert!(ascii_missed);
    assert_eq!(bytes, vec![0x01, 0x00]);
}

#[test]
fn little_endian_encoding_misses() {
    let input = bignum(2, &[0x01, 0x00]);
    let de = CborDeserializer::new(&input);
    let result = block_on(async {
        de.entry(|[e1, e2]| async {
            let le_missed = matches!(
                e1.deserialize_number_chunks::<LittleEndian>().await?,
                Probe::Miss
            );
            let mut acc = Vec::<u8>::new();
            let mut chunks = match e2.deserialize_number_chunks::<BigEndian>().await? {
                Probe::Hit(c) => c,
                Probe::Miss => return Ok(Probe::Miss),
            };
            let claim = loop {
                match NumberAccess::<BigEndian>::next_number_chunk(chunks, |b| acc.extend_from_slice(b)).await? {
                    Chunk::Data((next, ())) => chunks = next,
                    Chunk::Done(claim) => break claim,
                }
            };
            Ok(Probe::Hit((claim, (le_missed, acc))))
        })
        .await
    })
    .unwrap();
    let (le_missed, bytes) = match result {
        Probe::Hit((_, v)) => v,
        Probe::Miss => panic!("expected Hit"),
    };
    assert!(le_missed);
    assert_eq!(bytes, vec![0x01, 0x00]);
}
