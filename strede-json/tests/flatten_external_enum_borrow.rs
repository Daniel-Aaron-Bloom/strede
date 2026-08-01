//! `#[strede(flatten)]` on an externally-tagged (default representation)
//! enum field — borrow family. Externally tagging only ever splices in one
//! more key/value pair (the matched variant's wire name -> its payload), so
//! flattening it composes via the same `MapFieldProvider`/`StackConcat`
//! machinery as flattening a struct, rather than needing a new primitive.
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
enum Message {
    Ping,
    Request {
        id: String,
        method: String,
    },
    Users {
        users: Vec<String>,
        id: u32,
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
fn flatten_external_unit_variant_as_null() {
    let e: Envelope = parse(r#"{"tenant": ["a"], "Ping": null}"#).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: vec!["a".to_string()],
            message: Message::Ping,
        }
    );
}

#[test]
fn flatten_external_newtype_like_struct_variant() {
    let e: Envelope =
        parse(r#"{"tenant": ["a"], "Request": {"id": "1", "method": "GET"}}"#).unwrap();
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
fn flatten_external_struct_variant_with_nested_flatten() {
    let e: Envelope = parse(
        r#"{
            "tenant": ["a", "b"],
            "Users": {
                "users": ["alice", "bob"],
                "id": 69,
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
                id: 69,
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
fn flatten_external_missing_nested_flatten_field_misses() {
    let v: Option<Envelope> = parse(
        r#"{
            "tenant": ["a"],
            "Users": {"users": ["alice"], "id": 1, "limit": 10, "offset": 0}
        }"#,
    );
    assert!(v.is_none());
}

#[test]
fn flatten_external_unknown_variant_misses() {
    let v: Option<Envelope> = parse(r#"{"tenant": ["a"], "Bogus": null}"#);
    assert!(v.is_none());
}
