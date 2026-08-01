//! `#[strede(flatten)]` on an internally-tagged enum field — borrow family.
//!
//! Unlike `flatten_tagged_enum_borrow.rs` (which flattens a field *inside* a
//! tagged variant), this exercises the enum itself as the flatten *target*:
//! `Message`'s own fields (and the discriminant) are merged into `Envelope`'s
//! map. This needs the `CandidateArmStack` runtime primitive (TESTING_GAPS.md
//! item #3(B-2)) since, unlike external tagging, more than one variant's
//! fields could plausibly be present in the shared map until the tag key
//! resolves and eliminates every other candidate.
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
fn flatten_internal_unit_variant() {
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
fn flatten_internal_unit_variant_tag_before_sibling() {
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
fn flatten_internal_struct_variant_tag_first() {
    let e: Envelope =
        parse(r#"{"tenant": ["a"], "type": "Request", "id": "1", "method": "GET"}"#).unwrap();
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
fn flatten_internal_struct_variant_tag_in_middle() {
    // Both the variant's own fields AND the parent's sibling field arrive
    // before the tag resolves — the tag arm must still eliminate the other
    // candidates once it arrives, and the already-collected fields from the
    // (correct) matching candidate must not be lost.
    let e: Envelope =
        parse(r#"{"id": "1", "tenant": ["a"], "type": "Request", "method": "GET"}"#).unwrap();
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
    let e: Envelope = parse(
        r#"{
            "tenant": ["a", "b"],
            "type": "Users",
            "users": ["alice", "bob"],
            "count": 2,
            "limit": 10,
            "offset": 0,
            "total": 2
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
fn flatten_internal_missing_nested_flatten_field_misses() {
    let v: Option<Envelope> = parse(
        r#"{
            "tenant": ["a"],
            "type": "Users",
            "users": ["alice"],
            "count": 1,
            "limit": 10,
            "offset": 0
        }"#,
    );
    assert!(v.is_none());
}

#[test]
fn flatten_internal_missing_tag_misses() {
    let v: Option<Envelope> = parse(r#"{"tenant": ["a"], "id": "1", "method": "GET"}"#);
    assert!(v.is_none());
}

#[test]
fn flatten_internal_unknown_tag_value_misses() {
    let v: Option<Envelope> = parse(r#"{"tenant": ["a"], "type": "Bogus"}"#);
    assert!(v.is_none());
}
