//! `deserialize_number_chunks` tests — owned/chunked family. Mirrors
//! `number_chunks_borrow.rs`'s identical varint-tag-byte-leak regression
//! coverage.

mod helpers;
use helpers::*;

use strede::{
    Chunk, DeserializerOwned, EntryOwned, LittleEndian, NumberAccessOwned, Probe, SharedBuf,
};
use strede_bincode::{BincodeConfig, Legacy, Standard, chunked::ChunkedBincodeDeserializer};
use strede_test_util::block_on_loop;

/// Parse number chunks with `LittleEndian` encoding, collecting all bytes.
/// Returns `None` if `deserialize_number_chunks` returns `Miss`.
fn collect_le<C: BincodeConfig>(input: &[u8]) -> Option<Vec<u8>> {
    let input: &[u8] = input;
    block_on_loop(SharedBuf::with_async(
        input,
        async |buf: &mut &[u8]| {
            *buf = &[];
        },
        async |shared| {
            let de = ChunkedBincodeDeserializer::<C, _, _>::new(shared);
            match de
                .entry(|[e]| async move {
                    let mut acc = match e.deserialize_number_chunks::<LittleEndian>().await? {
                        Probe::Hit(a) => a,
                        Probe::Miss => return Ok(Probe::Miss),
                    };
                    let mut bytes = Vec::<u8>::new();
                    loop {
                        match <_ as NumberAccessOwned<LittleEndian>>::next_number_chunk(
                            acc,
                            |b: &[u8]| b.to_vec(),
                        )
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
            {
                Probe::Hit((_, v)) => Some(v),
                Probe::Miss => None,
            }
        },
    ))
}

const E: Enc = Enc::STANDARD;

#[test]
fn single_byte_inline_value_is_the_magnitude() {
    let bytes = E.u32(42);
    assert_eq!(bytes, vec![42u8]);
    assert_eq!(collect_le::<Standard>(&bytes), Some(vec![42u8]));
}

#[test]
fn two_byte_tail_excludes_prefix() {
    let bytes = E.u32(300);
    assert_eq!(bytes[0], 251);
    assert_eq!(collect_le::<Standard>(&bytes), Some(vec![0x2C, 0x01]));
}

#[test]
fn four_byte_tail_excludes_prefix() {
    let v: u32 = 100_000;
    let bytes = E.u32(v);
    assert_eq!(bytes[0], 252);
    assert_eq!(collect_le::<Standard>(&bytes), Some(v.to_le_bytes().to_vec()));
}

#[test]
fn eight_byte_tail_excludes_prefix() {
    let v: u64 = u32::MAX as u64 + 1;
    let bytes = E.u64(v);
    assert_eq!(bytes[0], 253);
    assert_eq!(collect_le::<Standard>(&bytes), Some(v.to_le_bytes().to_vec()));
}

#[test]
fn sixteen_byte_tail_excludes_prefix() {
    let v: u128 = u64::MAX as u128 + 1;
    let bytes = E.u128(v);
    assert_eq!(bytes[0], 254);
    assert_eq!(collect_le::<Standard>(&bytes), Some(v.to_le_bytes().to_vec()));
}

#[test]
fn fixint_mode_always_misses() {
    let legacy_enc = Enc::LEGACY;
    let bytes = legacy_enc.u32(42);
    assert_eq!(collect_le::<Legacy>(&bytes), None);
}
