//! `#[strede(flatten)]` on an internally-tagged enum field — MessagePack,
//! borrow family. See `strede-json/tests/flatten_internal_candidate_enum_borrow.rs`
//! for the design rationale (`CandidateArmStack`, TESTING_GAPS.md item #3(B-2));
//! this file confirms the same derive-generated codegen works over a binary,
//! non-JSON format.
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

fn str_array(items: &[&str]) -> Vec<u8> {
    let mut out = vec![fixarray(items.len())];
    for s in items {
        out.extend_from_slice(&fixstr(s));
    }
    out
}

#[derive(Debug, PartialEq, Deserialize)]
struct Pagination {
    limit: u32,
    offset: u32,
    total: u32,
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

#[test]
fn flatten_internal_unit_variant_tag_first() {
    let bytes = build_map(&[
        (fixstr("type").as_slice(), fixstr("Ping").as_slice()),
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
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
fn flatten_internal_struct_variant_tag_in_middle() {
    let bytes = build_map(&[
        (fixstr("id").as_slice(), fixstr("1").as_slice()),
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Request").as_slice()),
        (fixstr("method").as_slice(), fixstr("GET").as_slice()),
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
fn flatten_internal_struct_variant_with_nested_flatten() {
    let bytes = build_map(&[
        (
            fixstr("tenant").as_slice(),
            str_array(&["a", "b"]).as_slice(),
        ),
        (fixstr("type").as_slice(), fixstr("Users").as_slice()),
        (
            fixstr("users").as_slice(),
            str_array(&["alice", "bob"]).as_slice(),
        ),
        (fixstr("count").as_slice(), &[2u8]),
        (fixstr("limit").as_slice(), &[10u8]),
        (fixstr("offset").as_slice(), &[0u8]),
        (fixstr("total").as_slice(), &[2u8]),
    ]);
    let e: Envelope = parse(&bytes).unwrap();
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
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Users").as_slice()),
        (fixstr("users").as_slice(), str_array(&["alice"]).as_slice()),
        (fixstr("count").as_slice(), &[1u8]),
        (fixstr("limit").as_slice(), &[10u8]),
        (fixstr("offset").as_slice(), &[0u8]),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_internal_missing_tag_misses() {
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("id").as_slice(), fixstr("1").as_slice()),
        (fixstr("method").as_slice(), fixstr("GET").as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_internal_unknown_tag_value_misses() {
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Bogus").as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}
