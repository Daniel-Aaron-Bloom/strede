//! Owned-family ext deserialization tests.

mod helpers;
use helpers::*;

use strede::{BytesAccessOwned, Chunk, DeserializeOwned, Probe, SharedBuf};
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;
use strede_msgpack::{
    ChunkedMsgpackBytesAccess, ChunkedMsgpackClaim, DeserializeFromExtBytesOwned,
    DeserializeFromFixExt, ExtWrapper, FixExtWrapper, MsgpackError,
};
use strede_test_util::block_on_loop;

macro_rules! parse_fixext {
    ($T:ty, $input:expr, $extra:expr) => {{
        let input: &[u8] = $input;
        let extra = $extra;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
                match <FixExtWrapper<$T> as DeserializeOwned<_>>::deserialize_owned(de, extra)
                    .await
                    .unwrap()
                {
                    Probe::Hit((_, FixExtWrapper(v))) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
}

macro_rules! parse_ext {
    ($T:ty, $input:expr, $extra:expr) => {{
        let input: &[u8] = $input;
        let extra = $extra;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
                match <ExtWrapper<$T> as DeserializeOwned<_>>::deserialize_owned(de, extra)
                    .await
                    .unwrap()
                {
                    Probe::Hit((_, ExtWrapper(v))) => Some(v),
                    Probe::Miss => None,
                }
            },
        ))
    }};
}

// ---------------------------------------------------------------------------
// FixExt
// ---------------------------------------------------------------------------

struct MyFixExt {
    type_id: i8,
    data: Vec<u8>,
}

impl DeserializeFromFixExt for MyFixExt {
    type Extra = i8;

    fn deserialize_from_fixext(
        type_id: i8,
        data: &[u8],
        extra: i8,
    ) -> Result<Probe<Self>, MsgpackError> {
        if type_id != extra {
            return Ok(Probe::Miss);
        }
        Ok(Probe::Hit(MyFixExt {
            type_id,
            data: data.to_vec(),
        }))
    }
}

#[test]
fn fixext1_hit() {
    let bytes = fixext1(7, 0xab);
    let result = parse_fixext!(MyFixExt, &bytes, 7).unwrap();
    assert_eq!(result.type_id, 7);
    assert_eq!(result.data, &[0xab]);
}

#[test]
fn fixext2_hit() {
    let bytes = fixext2(3, [0x01, 0x02]);
    let result = parse_fixext!(MyFixExt, &bytes, 3).unwrap();
    assert_eq!(result.data, &[0x01, 0x02]);
}

#[test]
fn fixext4_hit() {
    let bytes = fixext4(1, [0x10, 0x20, 0x30, 0x40]);
    let result = parse_fixext!(MyFixExt, &bytes, 1).unwrap();
    assert_eq!(result.data, &[0x10, 0x20, 0x30, 0x40]);
}

#[test]
fn fixext8_hit() {
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let bytes = fixext8(2, data);
    let result = parse_fixext!(MyFixExt, &bytes, 2).unwrap();
    assert_eq!(result.data, &data);
}

#[test]
fn fixext16_hit() {
    let data = [42u8; 16];
    let bytes = fixext16(-1, data);
    let result = parse_fixext!(MyFixExt, &bytes, -1).unwrap();
    assert_eq!(result.data.len(), 16);
    assert!(result.data.iter().all(|&b| b == 42));
}

#[test]
fn fixext_wrong_type_id_misses() {
    let bytes = fixext1(7, 0xab);
    assert!(parse_fixext!(MyFixExt, &bytes, 5).is_none());
}

#[test]
fn fixext_non_ext_token_misses() {
    assert!(parse_fixext!(MyFixExt, &[0xc0], 0).is_none());
    assert!(parse_fixext!(MyFixExt, &[42], 0).is_none());
}

// ---------------------------------------------------------------------------
// Ext (variable-length)
// ---------------------------------------------------------------------------

struct MyVarExt(Vec<u8>);

impl<'s, B: strede::Buffer, F: AsyncFnMut(&mut B)>
    DeserializeFromExtBytesOwned<ChunkedMsgpackBytesAccess<'s, B, F>> for MyVarExt
{
    type Extra = i8;

    async fn deserialize_from_ext_bytes_owned(
        type_id: i8,
        _len: usize,
        bytes: ChunkedMsgpackBytesAccess<'s, B, F>,
        extra: i8,
    ) -> Result<Probe<(ChunkedMsgpackClaim<'s, B, F>, Self)>, MsgpackError> {
        if type_id != extra {
            return Ok(Probe::Miss);
        }
        let mut collected = Vec::new();
        let mut acc = bytes;
        loop {
            match acc.next_bytes(|b| b.to_vec()).await? {
                Chunk::Data((next_acc, chunk)) => {
                    collected.extend_from_slice(&chunk);
                    acc = next_acc;
                }
                Chunk::Done(claim) => {
                    return Ok(Probe::Hit((claim, MyVarExt(collected))));
                }
            }
        }
    }
}

#[test]
fn ext8_hit() {
    let data = b"hello ext";
    let bytes = ext8(10, data);
    let result = parse_ext!(MyVarExt, &bytes, 10).unwrap();
    assert_eq!(result.0, data);
}

#[test]
fn ext8_empty_payload() {
    let bytes = ext8(5, &[]);
    let result = parse_ext!(MyVarExt, &bytes, 5).unwrap();
    assert!(result.0.is_empty());
}

#[test]
fn ext_wrong_type_id_misses() {
    let bytes = ext8(10, b"data");
    assert!(parse_ext!(MyVarExt, &bytes, 99).is_none());
}

#[test]
fn ext_non_ext_token_misses() {
    assert!(parse_ext!(MyVarExt, &[0xc0], 0).is_none());
    assert!(parse_ext!(MyVarExt, &[42], 0).is_none());
}
