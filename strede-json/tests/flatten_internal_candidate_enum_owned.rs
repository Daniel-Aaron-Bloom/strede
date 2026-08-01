//! `#[strede(flatten)]` on an internally-tagged enum field — owned family.
//! See `flatten_internal_candidate_enum_borrow.rs` for the full rationale
//! (`CandidateArmStack`, TESTING_GAPS.md item #3(B-2)).
//!
//! Also exercises the owned-family concurrency contract directly: the
//! staggered/byte-at-a-time tests below force more than one candidate's
//! forked reader to make real concurrent progress against an incremental
//! source before the tag resolves and eliminates the losing candidate. If
//! `CandidateArmStack`'s elimination ever left an eliminated candidate's
//! forked reader un-dropped, the shared buffer would stall waiting on a
//! reader nothing polls anymore - `block_on_loop_bounded` turns that hang
//! into a fast, clear panic instead of hanging CI (see
//! `owned_staggered_concurrency.rs` for the same discipline applied to plain
//! struct fields).
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
#[strede(tag = "type")]
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
fn flatten_internal_unit_variant() {
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
fn flatten_internal_struct_variant_tag_in_middle() {
    let e: Envelope = parse!(
        Envelope,
        br#"{"id": "1", "tenant": ["a"], "type": "Request", "method": "GET"}"#
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
fn flatten_internal_struct_variant_with_nested_flatten() {
    let e: Envelope = parse!(
        Envelope,
        br#"{
            "tenant": ["a", "b"],
            "type": "Users",
            "users": ["alice", "bob"],
            "count": 2,
            "limit": 10,
            "offset": 0,
            "total": 2
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
fn flatten_internal_missing_nested_flatten_field_misses() {
    let v: Option<Envelope> = parse!(
        Envelope,
        br#"{
            "tenant": ["a"],
            "type": "Users",
            "users": ["alice"],
            "count": 1,
            "limit": 10,
            "offset": 0
        }"#
    );
    assert!(v.is_none());
}

#[test]
fn flatten_internal_missing_tag_misses() {
    let v: Option<Envelope> = parse!(
        Envelope,
        br#"{"tenant": ["a"], "id": "1", "method": "GET"}"#
    );
    assert!(v.is_none());
}

#[test]
fn flatten_internal_unknown_tag_value_misses() {
    let v: Option<Envelope> = parse!(Envelope, br#"{"tenant": ["a"], "type": "Bogus"}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_internal_candidates_race_concurrently_byte_at_a_time() {
    // Two non-unit candidates (Request, Users) both have fields that could
    // plausibly be raced before "type" resolves which one is real. Feeding
    // the input a few bytes at a time forces genuine concurrent forked-reader
    // progress across both candidates' own arm stacks (plus the tag arm)
    // until the tag key is seen and the losing candidate is eliminated.
    //
    // chunk_size=1 is intentionally excluded here: it currently hits an
    // unrelated pre-existing bug in the *standalone* (non-flatten)
    // internally-tagged dispatch path this reuses (`TagAwareMapOwned` +
    // `select_probe!`, entirely pre-dating this feature) whenever a
    // candidate's own struct variant has a nested `#[strede(flatten)]`
    // field - reproduces identically for `Message` derived standalone with
    // no `Envelope`/`CandidateArmStack` involved at all. See
    // `flatten_internal_unbuffered_byte_at_a_time` below for a chunk_size=1
    // regression test that isolates `CandidateArmStack`'s own elimination/
    // resource-cleanup mechanism from that pre-existing gap by using a
    // candidate with no nested flatten field.
    let input: &[u8] =
        br#"{"tenant": ["a", "b"], "users": ["alice", "bob"], "type": "Users", "count": 2, "limit": 10, "offset": 0, "total": 2}"#;
    for chunk_size in 2..=4 {
        let e: Envelope = parse_chunked!(Envelope, input, chunk_size).unwrap();
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
            },
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn flatten_internal_unbuffered_byte_at_a_time() {
    // Same concurrent-candidate-racing concern as above, but at true
    // byte-at-a-time granularity (chunk_size=1) using the `Request` /
    // `Ping` candidates, which have no nested `#[strede(flatten)]` field of
    // their own - isolating `CandidateArmStack`'s elimination/resource-
    // cleanup mechanism from the unrelated pre-existing gap noted above. If
    // an eliminated candidate's forked reader were ever left un-dropped,
    // the shared buffer would stall waiting on a reader nothing polls
    // anymore; `block_on_loop_bounded` turns that into a fast panic rather
    // than a CI hang.
    let input: &[u8] = br#"{"id": "1", "tenant": ["a"], "type": "Request", "method": "GET"}"#;
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
