//! Regression coverage for TESTING_GAPS.md item #1: the owned family's
//! forked-handle concurrency invariant (CLAUDE.md's "owned family — parallel
//! scanning and deadlock hazard"). Every other `*_owned.rs` test in this
//! crate (other than `varint_chunking.rs`) feeds input via a "dump
//! everything upfront, then EOF" loader, so the derive-generated field-arm
//! race across a struct never actually has to suspend on I/O with multiple
//! live forked handles.
//!
//! This uses the existing `parse_owned_chunked!` harness (see
//! `varint_chunking.rs`) on a multi-field struct instead of a single scalar,
//! so every field arm's key race - including the always-present internal
//! duplicate-key-detection arm - runs concurrently against a real
//! incremental source. The same test shape on strede-json surfaced a real
//! bug in `race_keys`'s arm-priority tie-break (`strede/src/map_arm/owned.rs`):
//! whichever arm happened to drive a shared-buffer refill could finish out
//! of turn and beat an earlier-declared arm that should always win, fixed by
//! re-checking earlier arms once a later one hits (mirrors `select_probe!`'s
//! `biased` mode). Postcard's field arms are positional rather than
//! name-matched, so this is a narrower exposure than JSON/msgpack/CBOR, but
//! it goes through the identical `race_keys` machinery.

#![recursion_limit = "256"]
#![cfg(feature = "alloc")]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Multi {
    alpha: u32,
    beta: String,
    gamma: bool,
    delta: u32,
}

#[test]
fn multi_field_struct_byte_at_a_time() {
    let mut data = varint(1); // alpha
    data.extend_from_slice(&pstr("hi")); // beta
    data.push(0x01); // gamma = true
    data.extend_from_slice(&varint(4)); // delta
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_owned_chunked!(Multi, &data, chunk_size),
            Ok(Some(Multi {
                alpha: 1,
                beta: "hi".into(),
                gamma: true,
                delta: 4,
            })),
            "chunk_size={chunk_size}"
        );
    }
}
