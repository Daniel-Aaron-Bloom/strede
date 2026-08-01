//! Regression coverage for TESTING_GAPS.md item #1: the owned family's
//! forked-handle concurrency invariant (CLAUDE.md's "owned family — parallel
//! scanning and deadlock hazard") previously had zero coverage that forces
//! more than one forked handle to make real concurrent progress against an
//! incremental source. Every other `*_owned.rs` test in this crate feeds
//! input via a "dump everything upfront, then EOF" loader
//! (`async |buf| { *buf = &[] }`), so the derive-generated map-key race
//! across a struct's field arms never actually has to suspend on I/O with
//! multiple live forked handles - it always finds the whole map already
//! sitting in the buffer.
//!
//! This file feeds the same kind of input a handful of bytes at a time, so
//! the outer map's key race must genuinely interleave partial progress
//! across several forked `MapKeyProbe` handles. If a regression reintroduces
//! the "await one forked handle to completion, then decide" deadlock hazard
//! CLAUDE.md warns about, the derive code would block waiting for a load
//! that never comes (because no sibling handle ever asks for it) - so this
//! uses `block_on_loop_bounded` instead of `block_on_loop` to turn that
//! failure mode into a fast, clear panic rather than a CI hang.
//!
//! This exact test staggering surfaced a real bug in `race_keys`'s
//! arm-priority tie-break (`strede/src/map_arm/owned.rs`): whichever arm
//! happened to drive a shared-buffer refill could finish out of turn and
//! beat an earlier-declared arm that should always win - even a struct with
//! a single field would spuriously error with `DuplicateField` because the
//! internal duplicate-key-detection arm could "win" the very first
//! occurrence of a key. Fixed by re-checking earlier arms once a later one
//! hits, mirroring `select_probe!`'s `biased` mode.
#![cfg(feature = "alloc")]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop_bounded;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Multi {
    alpha: u32,
    beta: String,
    gamma: bool,
    delta: u32,
}

macro_rules! parse_chunked {
    ($ty:ty, $input:expr, $chunk_size:expr) => {{
        let input: &[u8] = $input;
        let chunk_size: usize = $chunk_size;
        let pos = ::core::cell::Cell::new(chunk_size.min(input.len()));
        block_on_loop_bounded(
            SharedBuf::with_async(
                &input[..chunk_size.min(input.len())],
                async |buf: &mut &[u8]| {
                    let start = pos.get();
                    let end = (start + chunk_size).min(input.len());
                    pos.set(end);
                    *buf = &input[start..end];
                },
                async |shared| {
                    let de = ChunkedJsonDeserializer::new(shared);
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ())
                        .await
                        .unwrap()
                    {
                        Probe::Hit((_, v)) => Some(v),
                        Probe::Miss => None,
                    }
                },
            ),
            20_000,
        )
    }};
}

#[test]
fn multi_field_struct_byte_at_a_time() {
    // Keys deliberately reversed vs. declaration order so no arm can win
    // "for free" by matching the first key it happens to see.
    let input: &[u8] = br#"{"delta": 4, "gamma": true, "alpha": 1, "beta": "hi"}"#;
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_chunked!(Multi, input, chunk_size),
            Some(Multi {
                alpha: 1,
                beta: "hi".into(),
                gamma: true,
                delta: 4,
            }),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn multi_field_struct_missing_field_byte_at_a_time_misses() {
    let input: &[u8] = br#"{"delta": 4, "gamma": true, "alpha": 1}"#;
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_chunked!(Multi, input, chunk_size),
            None,
            "chunk_size={chunk_size}"
        );
    }
}
