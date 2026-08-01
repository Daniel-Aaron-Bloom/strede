//! `#[strede(flatten)]` on an externally-tagged (default representation)
//! enum field — owned family. See `flatten_external_enum_borrow.rs` for the
//! borrow-family counterpart and design rationale.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Pagination {
    limit: u64,
    offset: u64,
    total: u64,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
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

#[test]
fn flatten_external_unit_variant_as_null() {
    let e: Envelope = parse!(Envelope, br#"{"tenant": ["a"], "Ping": null}"#).unwrap();
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
    let e: Envelope = parse!(
        Envelope,
        br#"{"tenant": ["a"], "Request": {"id": "1", "method": "GET"}}"#
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
fn flatten_external_struct_variant_with_nested_flatten() {
    let e: Envelope = parse!(
        Envelope,
        br#"{
            "tenant": ["a", "b"],
            "Users": {
                "users": ["alice", "bob"],
                "id": 69,
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
    let v: Option<Envelope> = parse!(
        Envelope,
        br#"{
            "tenant": ["a"],
            "Users": {"users": ["alice"], "id": 1, "limit": 10, "offset": 0}
        }"#
    );
    assert!(v.is_none());
}

#[test]
fn flatten_external_unknown_variant_misses() {
    let v: Option<Envelope> = parse!(Envelope, br#"{"tenant": ["a"], "Bogus": null}"#);
    assert!(v.is_none());
}
