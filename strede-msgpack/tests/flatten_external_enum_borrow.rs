//! `#[strede(flatten)]` on an externally-tagged (default representation)
//! enum field — MessagePack, borrow family. See
//! `strede-json/tests/flatten_external_enum_borrow.rs` for the design
//! rationale; this file confirms the same derive-generated `MapFieldProvider`
//! codegen works over a binary, non-JSON format.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]

mod helpers;
use helpers::*;

use strede::Probe;
use strede_derive::Deserialize;
use strede_msgpack::MsgpackDeserializer;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, MsgpackDeserializer<'de>, Extra = ()>,
{
    let de = MsgpackDeserializer::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

fn concat(parts: &[&[u8]]) -> Vec<u8> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

fn str_array(items: &[&str]) -> Vec<u8> {
    let mut out = vec![fixarray(items.len())];
    for s in items {
        out.extend_from_slice(&fixstr(s));
    }
    out
}

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

#[test]
fn flatten_external_unit_variant_as_null() {
    let bytes = concat(&[
        &[fixmap(2)],
        &fixstr("tenant"),
        &str_array(&["a"]),
        &fixstr("Ping"),
        &[0xc0],
    ]);
    let e: Envelope = parse(&bytes).unwrap();
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
    let request_body = build_map(&[
        (fixstr("id").as_slice(), fixstr("1").as_slice()),
        (fixstr("method").as_slice(), fixstr("GET").as_slice()),
    ]);
    let bytes = concat(&[
        &[fixmap(2)],
        &fixstr("tenant"),
        &str_array(&["a"]),
        &fixstr("Request"),
        &request_body,
    ]);
    let e: Envelope = parse(&bytes).unwrap();
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
    let users_body = build_map(&[
        (
            fixstr("users").as_slice(),
            str_array(&["alice", "bob"]).as_slice(),
        ),
        (fixstr("id").as_slice(), &[69u8]),
        (fixstr("limit").as_slice(), &[10u8]),
        (fixstr("offset").as_slice(), &[0u8]),
        (fixstr("total").as_slice(), &[2u8]),
    ]);
    let bytes = concat(&[
        &[fixmap(2)],
        &fixstr("tenant"),
        &str_array(&["a", "b"]),
        &fixstr("Users"),
        &users_body,
    ]);
    let e: Envelope = parse(&bytes).unwrap();
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
    let users_body = build_map(&[
        (fixstr("users").as_slice(), str_array(&["alice"]).as_slice()),
        (fixstr("id").as_slice(), &[1u8]),
        (fixstr("limit").as_slice(), &[10u8]),
        (fixstr("offset").as_slice(), &[0u8]),
    ]);
    let bytes = concat(&[
        &[fixmap(2)],
        &fixstr("tenant"),
        &str_array(&["a"]),
        &fixstr("Users"),
        &users_body,
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_external_unknown_variant_misses() {
    let bytes = concat(&[
        &[fixmap(2)],
        &fixstr("tenant"),
        &str_array(&["a"]),
        &fixstr("Bogus"),
        &[0xc0],
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}
