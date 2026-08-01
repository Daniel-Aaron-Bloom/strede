//! `#[strede(flatten)]` on an adjacently-tagged enum field — CBOR, borrow
//! family. See `strede-json/tests/flatten_adjacent_candidate_enum_borrow.rs`
//! for the design rationale; this file confirms the same derive-generated
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

fn str_array(items: &[&str]) -> std::vec::Vec<u8> {
    let mut out = array(items.len());
    for s in items {
        out.extend_from_slice(&tstr(s));
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
#[strede(tag = "type", content = "data")]
enum Message {
    Ping,
    Request {
        id: std::string::String,
        method: std::string::String,
    },
    Users {
        users: std::vec::Vec<std::string::String>,
        count: u32,
        #[strede(flatten)]
        pagination: Pagination,
    },
    // Structurally identical to `Request` - used only by the shape-ambiguity
    // test below, to force two candidates to both successfully parse the
    // same `data` value before the tag resolves which one is real.
    Alt {
        id: std::string::String,
        method: std::string::String,
    },
}

#[derive(Debug, PartialEq, Deserialize)]
struct Envelope {
    tenant: std::vec::Vec<std::string::String>,
    #[strede(flatten)]
    message: Message,
}

fn request_content() -> std::vec::Vec<u8> {
    build_map(&[
        (tstr("id").as_slice(), tstr("1").as_slice()),
        (tstr("method").as_slice(), tstr("GET").as_slice()),
    ])
}

#[test]
fn flatten_adjacent_unit_variant() {
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Ping").as_slice()),
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
fn flatten_adjacent_unit_variant_tag_before_sibling() {
    let bytes = build_map(&[
        (tstr("type").as_slice(), tstr("Ping").as_slice()),
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
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
fn flatten_adjacent_struct_variant_tag_before_content() {
    let content = request_content();
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Request").as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
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
fn flatten_adjacent_struct_variant_content_before_tag() {
    // The single most important test in this file: `data` precedes `type` on
    // the wire, forcing the content arm to race every non-unit candidate
    // type before the tag is even seen.
    let content = request_content();
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
        (tstr("type").as_slice(), tstr("Request").as_slice()),
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
fn flatten_adjacent_sibling_before_both() {
    let content = request_content();
    let bytes = build_map(&[
        (tstr("data").as_slice(), content.as_slice()),
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Request").as_slice()),
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
fn flatten_adjacent_struct_variant_with_nested_flatten() {
    let content = build_map(&[
        (
            tstr("users").as_slice(),
            str_array(&["alice", "bob"]).as_slice(),
        ),
        (tstr("count").as_slice(), &[2u8]),
        (tstr("limit").as_slice(), &[10u8]),
        (tstr("offset").as_slice(), &[0u8]),
        (tstr("total").as_slice(), &[2u8]),
    ]);
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a", "b"]).as_slice()),
        (tstr("type").as_slice(), tstr("Users").as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
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
    let content = build_map(&[
        (tstr("users").as_slice(), str_array(&["alice"]).as_slice()),
        (tstr("count").as_slice(), &[1u8]),
        (tstr("limit").as_slice(), &[10u8]),
        (tstr("offset").as_slice(), &[0u8]),
    ]);
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Users").as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_tag_misses() {
    let content = request_content();
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_content_for_nonunit_tag_misses() {
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Request").as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_content_present_for_unit_tag_misses() {
    let content = request_content();
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Ping").as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_unknown_tag_value_misses() {
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Bogus").as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_multi_candidate_shape_ambiguity_tag_agrees() {
    // `Request` and `Alt` have identical field shapes. With `data` arriving
    // before `type`, the content race can't yet know which one is real, so
    // declaration order wins - `Request` is declared first. Here the tag
    // agrees with that speculative winner.
    let content = request_content();
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
        (tstr("type").as_slice(), tstr("Request").as_slice()),
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
fn flatten_adjacent_multi_candidate_shape_ambiguity_tag_disagrees() {
    // Same ambiguous content, but the tag names `Alt` instead - the race
    // still speculatively resolves to `Request` (declaration order), and
    // `from_outputs` must catch the disagreement.
    let content = request_content();
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("data").as_slice(), content.as_slice()),
        (tstr("type").as_slice(), tstr("Alt").as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}
