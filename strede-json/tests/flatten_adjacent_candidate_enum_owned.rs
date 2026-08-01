//! `#[strede(flatten)]` on an adjacently-tagged enum field — owned family.
//! See `flatten_adjacent_candidate_enum_borrow.rs` for the full design
//! rationale.
//!
//! Also exercises the owned-family concurrency contract directly: the
//! chunked/byte-at-a-time test below forces the content arm's forked
//! candidate readers to make real concurrent progress against an incremental
//! source before the tag resolves - if a losing candidate's forked reader
//! were ever left un-dropped after `select_probe!` picks a winner, the
//! shared buffer would stall waiting on a reader nothing polls anymore.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::{block_on_loop, block_on_loop_bounded};

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Pagination {
    limit: u32,
    offset: u32,
    total: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
#[strede(tag = "type", content = "data")]
enum Message {
    Ping,
    Request {
        id: String,
        method: String,
    },
    Users {
        users: Vec<String>,
        count: u32,
        #[strede(flatten)]
        pagination: Pagination,
    },
    // Structurally identical to `Request` - used only by the shape-ambiguity
    // test below, to force two candidates to both successfully parse the
    // same `data` value before the tag resolves which one is real.
    Alt {
        id: String,
        method: String,
    },
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Envelope {
    tenant: Vec<String>,
    #[strede(flatten)]
    message: Message,
}

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
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
        ))
    }};
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
fn flatten_adjacent_unit_variant() {
    let e: Envelope = parse!(Envelope, br#"{"tenant": ["a"], "type": "Ping"}"#).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Ping,
        }
    );
}

#[test]
fn flatten_adjacent_unit_variant_tag_before_sibling() {
    let e: Envelope = parse!(Envelope, br#"{"type": "Ping", "tenant": ["a"]}"#).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Ping,
        }
    );
}

#[test]
fn flatten_adjacent_struct_variant_tag_before_content() {
    let e: Envelope = parse!(
        Envelope,
        br#"{"tenant": ["a"], "type": "Request", "data": {"id": "1", "method": "GET"}}"#
    )
    .unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Request {
                id: "1".to_string(),
                method: "GET".to_string(),
            },
        }
    );
}

#[test]
fn flatten_adjacent_struct_variant_content_before_tag() {
    // The single most important test in this file: `data` precedes `type` on
    // the wire, forcing the content arm to race every non-unit candidate
    // type (via `select_probe!`, concurrently - required in the owned family
    // to avoid the deadlock hazard of awaiting one forked reader to
    // completion before touching another) before the tag is even seen.
    let e: Envelope = parse!(
        Envelope,
        br#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}, "type": "Request"}"#
    )
    .unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Request {
                id: "1".to_string(),
                method: "GET".to_string(),
            },
        }
    );
}

#[test]
fn flatten_adjacent_sibling_before_both() {
    let e: Envelope = parse!(
        Envelope,
        br#"{"data": {"id": "1", "method": "GET"}, "tenant": ["a"], "type": "Request"}"#
    )
    .unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Request {
                id: "1".to_string(),
                method: "GET".to_string(),
            },
        }
    );
}

#[test]
fn flatten_adjacent_struct_variant_with_nested_flatten() {
    let e: Envelope = parse!(
        Envelope,
        br#"{
            "tenant": ["a", "b"],
            "type": "Users",
            "data": {
                "users": ["alice", "bob"],
                "count": 2,
                "limit": 10,
                "offset": 0,
                "total": 2
            }
        }"#
    )
    .unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string(), "b".to_string()],
            message: Message::Users {
                users: vec!["alice".to_string(), "bob".to_string()],
                count: 2,
                pagination: Pagination {
                    limit: 10,
                    offset: 0,
                    total: 2,
                },
            },
        }
    );
}

#[test]
fn flatten_adjacent_missing_nested_flatten_field_misses() {
    let v: Option<Envelope> = parse!(
        Envelope,
        br#"{
            "tenant": ["a"],
            "type": "Users",
            "data": {
                "users": ["alice"],
                "count": 1,
                "limit": 10,
                "offset": 0
            }
        }"#
    );
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_tag_misses() {
    let v: Option<Envelope> = parse!(
        Envelope,
        br#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}}"#
    );
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_content_for_nonunit_tag_misses() {
    let v: Option<Envelope> = parse!(Envelope, br#"{"tenant": ["a"], "type": "Request"}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_content_present_for_unit_tag_misses() {
    let v: Option<Envelope> = parse!(
        Envelope,
        br#"{"tenant": ["a"], "type": "Ping", "data": {"id": "1", "method": "GET"}}"#
    );
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_unknown_tag_value_misses() {
    let v: Option<Envelope> = parse!(Envelope, br#"{"tenant": ["a"], "type": "Bogus"}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_multi_candidate_shape_ambiguity_tag_agrees() {
    let e: Envelope = parse!(
        Envelope,
        br#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}, "type": "Request"}"#
    )
    .unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Request {
                id: "1".to_string(),
                method: "GET".to_string(),
            },
        }
    );
}

#[test]
fn flatten_adjacent_multi_candidate_shape_ambiguity_tag_disagrees() {
    let v: Option<Envelope> = parse!(
        Envelope,
        br#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}, "type": "Alt"}"#
    );
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_content_before_tag_byte_at_a_time() {
    // True byte-at-a-time granularity (chunk_size=1): forces the content
    // arm's forked `Request`/`Users`/`Alt` candidate readers to make real
    // concurrent progress against an incremental source before `type`
    // resolves which one is real. If a losing candidate's forked reader
    // were ever left un-dropped once `select_probe!` picks a winner, the
    // shared buffer would stall waiting on a reader nothing polls anymore -
    // `block_on_loop_bounded` turns that into a fast panic rather than a CI
    // hang.
    let input: &[u8] =
        br#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}, "type": "Request"}"#;
    let e: Envelope = parse_chunked!(Envelope, input, 1).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Request {
                id: "1".to_string(),
                method: "GET".to_string(),
            },
        }
    );
}
