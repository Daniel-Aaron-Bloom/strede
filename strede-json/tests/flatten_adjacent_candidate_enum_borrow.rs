//! `#[strede(flatten)]` on an adjacently-tagged enum field — borrow family.
//!
//! Mirrors `flatten_internal_candidate_enum_borrow.rs`'s `Envelope`/`Message`/
//! `Pagination` fixture, retagging `Message` as adjacently-tagged
//! (`#[strede(tag = "type", content = "data")]`). Unlike internally-tagged
//! flatten (`CandidateArmStack`, one sub-arm-stack per candidate racing a
//! shared field key-space), adjacently-tagged flatten contributes exactly 2
//! fixed arms (tag + content) regardless of variant count; the content arm
//! always races every non-unit candidate's `deserialize_value::<CandidateType>()`
//! against forked copies of the same value, deferring the tag/content
//! cross-check to `from_outputs`. See TESTING_GAPS.md's flatten section and
//! `gen_enum_candidate_map_field_provider_adjacent_borrow` for the full
//! design rationale.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
struct Pagination {
    limit: u64,
    offset: u64,
    total: u64,
}

#[derive(Debug, PartialEq, Deserialize)]
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
    // tests below, to force two candidates to both successfully parse the
    // same `data` value before the tag resolves which one is real.
    Alt {
        id: String,
        method: String,
    },
}

#[derive(Debug, PartialEq, Deserialize)]
struct Envelope {
    tenant: Vec<String>,
    #[strede(flatten)]
    message: Message,
}

fn parse<'de, T>(input: &'de str) -> Option<T>
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn flatten_adjacent_unit_variant() {
    let e: Envelope = parse(r#"{"tenant": ["a"], "type": "Ping"}"#).unwrap();
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
    let e: Envelope = parse(r#"{"type": "Ping", "tenant": ["a"]}"#).unwrap();
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
    let e: Envelope =
        parse(r#"{"tenant": ["a"], "type": "Request", "data": {"id": "1", "method": "GET"}}"#)
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
    // type before the tag is even seen, then `from_outputs` cross-checks the
    // race winner against the later-resolved tag.
    let e: Envelope =
        parse(r#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}, "type": "Request"}"#)
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
    // Three-way interleave: parent's own sibling field arrives first, then
    // content, then tag.
    let e: Envelope =
        parse(r#"{"data": {"id": "1", "method": "GET"}, "tenant": ["a"], "type": "Request"}"#)
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
    let e: Envelope = parse(
        r#"{
            "tenant": ["a", "b"],
            "type": "Users",
            "data": {
                "users": ["alice", "bob"],
                "count": 2,
                "limit": 10,
                "offset": 0,
                "total": 2
            }
        }"#,
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
    let v: Option<Envelope> = parse(
        r#"{
            "tenant": ["a"],
            "type": "Users",
            "data": {
                "users": ["alice"],
                "count": 1,
                "limit": 10,
                "offset": 0
            }
        }"#,
    );
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_tag_misses() {
    let v: Option<Envelope> = parse(r#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_content_for_nonunit_tag_misses() {
    let v: Option<Envelope> = parse(r#"{"tenant": ["a"], "type": "Request"}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_content_present_for_unit_tag_misses() {
    // `type` names a unit variant, but a `data` key is also present - an
    // explicit behavioral choice (not silently ignored): the mismatch
    // between the unit tag and the present content is rejected.
    let v: Option<Envelope> =
        parse(r#"{"tenant": ["a"], "type": "Ping", "data": {"id": "1", "method": "GET"}}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_unknown_tag_value_misses() {
    let v: Option<Envelope> = parse(r#"{"tenant": ["a"], "type": "Bogus"}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_multi_candidate_shape_ambiguity_tag_agrees() {
    // `Request` and `Alt` have identical field shapes. With `data` arriving
    // before `type`, the content race can't yet know which one is real, so
    // declaration order wins (the same accepted tie-break policy used
    // everywhere else in this codebase) - `Request` is declared first. Here
    // the tag agrees with that speculative winner.
    let e: Envelope =
        parse(r#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}, "type": "Request"}"#)
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
    // Same ambiguous content, but the tag names `Alt` instead - the race
    // still speculatively resolves to `Request` (declaration order), and
    // `from_outputs` must catch the disagreement rather than silently
    // keeping the wrong parse.
    let v: Option<Envelope> =
        parse(r#"{"tenant": ["a"], "data": {"id": "1", "method": "GET"}, "type": "Alt"}"#);
    assert!(v.is_none());
}
