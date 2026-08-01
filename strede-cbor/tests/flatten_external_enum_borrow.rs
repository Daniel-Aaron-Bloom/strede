//! `#[strede(flatten)]` on an externally-tagged (default representation)
//! enum field — CBOR, borrow family. See
//! `strede-json/tests/flatten_external_enum_borrow.rs` for the design
//! rationale; this file confirms the same derive-generated `MapFieldProvider`
//! codegen works over a binary, non-JSON format.
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]
extern crate std;
mod helpers;
use helpers::*;

use strede::Probe;
use strede_cbor::CborDeserializer;
use strede_derive::Deserialize;
use strede_test_util::block_on;

fn parse<'de, T>(input: &'de [u8]) -> Option<T>
where
    T: strede::Deserialize<'de, CborDeserializer<'de>, Extra = ()>,
{
    let de = CborDeserializer::new(input);
    match block_on(T::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

fn concat(parts: &[&[u8]]) -> std::vec::Vec<u8> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

fn str_array(items: &[&str]) -> std::vec::Vec<u8> {
    let mut out = array(items.len());
    for s in items {
        out.extend_from_slice(&tstr(s));
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
        id: std::string::String,
        method: std::string::String,
    },
    Users {
        users: std::vec::Vec<std::string::String>,
        id: u32,
        #[strede(flatten)]
        pagination: Pagination,
    },
}

#[derive(Debug, PartialEq, Deserialize)]
struct Envelope {
    tenant: std::vec::Vec<std::string::String>,
    #[strede(flatten)]
    message: Message,
}

#[test]
fn flatten_external_unit_variant_as_null() {
    let tenant = tstr("tenant");
    let bytes = concat(&[
        &map(2),
        &tenant,
        &str_array(&["a"]),
        &tstr("Ping"),
        &[cbor_null()],
    ]);
    let e: Envelope = parse(&bytes).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: std::vec![std::string::String::from("a")],
            message: Message::Ping,
        }
    );
}

#[test]
fn flatten_external_newtype_like_struct_variant() {
    let request_body = build_map(&[(&tstr("id"), &tstr("1")), (&tstr("method"), &tstr("GET"))]);
    let bytes = concat(&[
        &map(2),
        &tstr("tenant"),
        &str_array(&["a"]),
        &tstr("Request"),
        &request_body,
    ]);
    let e: Envelope = parse(&bytes).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: std::vec![std::string::String::from("a")],
            message: Message::Request {
                id: std::string::String::from("1"),
                method: std::string::String::from("GET"),
            },
        }
    );
}

#[test]
fn flatten_external_struct_variant_with_nested_flatten() {
    let users_body = build_map(&[
        (&tstr("users"), &str_array(&["alice", "bob"])),
        (&tstr("id"), &uint8(69)),
        (&tstr("limit"), &[uint_small(10)]),
        (&tstr("offset"), &[uint_small(0)]),
        (&tstr("total"), &[uint_small(2)]),
    ]);
    let bytes = concat(&[
        &map(2),
        &tstr("tenant"),
        &str_array(&["a", "b"]),
        &tstr("Users"),
        &users_body,
    ]);
    let e: Envelope = parse(&bytes).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: std::vec![
                std::string::String::from("a"),
                std::string::String::from("b")
            ],
            message: Message::Users {
                users: std::vec![
                    std::string::String::from("alice"),
                    std::string::String::from("bob")
                ],
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
        (&tstr("users"), &str_array(&["alice"])),
        (&tstr("id"), &[uint_small(1)]),
        (&tstr("limit"), &[uint_small(10)]),
        (&tstr("offset"), &[uint_small(0)]),
    ]);
    let bytes = concat(&[
        &map(2),
        &tstr("tenant"),
        &str_array(&["a"]),
        &tstr("Users"),
        &users_body,
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_external_unknown_variant_misses() {
    let bytes = concat(&[
        &map(2),
        &tstr("tenant"),
        &str_array(&["a"]),
        &tstr("Bogus"),
        &[cbor_null()],
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}
