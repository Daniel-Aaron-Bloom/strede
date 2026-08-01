//! `#[strede(flatten)]` on an adjacently-tagged enum field — MessagePack,
//! borrow family. See `strede-json/tests/flatten_adjacent_candidate_enum_borrow.rs`
//! for the design rationale; this file confirms the same derive-generated
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

#[derive(Debug, PartialEq, Deserialize)]
struct Envelope {
    tenant: Vec<String>,
    #[strede(flatten)]
    message: Message,
}

fn request_content() -> Vec<u8> {
    build_map(&[
        (fixstr("id").as_slice(), fixstr("1").as_slice()),
        (fixstr("method").as_slice(), fixstr("GET").as_slice()),
    ])
}

#[test]
fn flatten_adjacent_unit_variant() {
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Ping").as_slice()),
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
fn flatten_adjacent_unit_variant_tag_before_sibling() {
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
fn flatten_adjacent_struct_variant_tag_before_content() {
    let content = request_content();
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Request").as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
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
fn flatten_adjacent_struct_variant_content_before_tag() {
    // The single most important test in this file: `data` precedes `type` on
    // the wire, forcing the content arm to race every non-unit candidate
    // type before the tag is even seen.
    let content = request_content();
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
        (fixstr("type").as_slice(), fixstr("Request").as_slice()),
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
fn flatten_adjacent_sibling_before_both() {
    let content = request_content();
    let bytes = build_map(&[
        (fixstr("data").as_slice(), content.as_slice()),
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Request").as_slice()),
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
fn flatten_adjacent_struct_variant_with_nested_flatten() {
    let content = build_map(&[
        (
            fixstr("users").as_slice(),
            str_array(&["alice", "bob"]).as_slice(),
        ),
        (fixstr("count").as_slice(), &[2u8]),
        (fixstr("limit").as_slice(), &[10u8]),
        (fixstr("offset").as_slice(), &[0u8]),
        (fixstr("total").as_slice(), &[2u8]),
    ]);
    let bytes = build_map(&[
        (
            fixstr("tenant").as_slice(),
            str_array(&["a", "b"]).as_slice(),
        ),
        (fixstr("type").as_slice(), fixstr("Users").as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
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
fn flatten_adjacent_missing_nested_flatten_field_misses() {
    let content = build_map(&[
        (fixstr("users").as_slice(), str_array(&["alice"]).as_slice()),
        (fixstr("count").as_slice(), &[1u8]),
        (fixstr("limit").as_slice(), &[10u8]),
        (fixstr("offset").as_slice(), &[0u8]),
    ]);
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Users").as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_tag_misses() {
    let content = request_content();
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_missing_content_for_nonunit_tag_misses() {
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Request").as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_content_present_for_unit_tag_misses() {
    let content = request_content();
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Ping").as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_adjacent_unknown_tag_value_misses() {
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("type").as_slice(), fixstr("Bogus").as_slice()),
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
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
        (fixstr("type").as_slice(), fixstr("Request").as_slice()),
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
fn flatten_adjacent_multi_candidate_shape_ambiguity_tag_disagrees() {
    // Same ambiguous content, but the tag names `Alt` instead - the race
    // still speculatively resolves to `Request` (declaration order), and
    // `from_outputs` must catch the disagreement.
    let content = request_content();
    let bytes = build_map(&[
        (fixstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (fixstr("data").as_slice(), content.as_slice()),
        (fixstr("type").as_slice(), fixstr("Alt").as_slice()),
    ]);
    let v: Option<Envelope> = parse(&bytes);
    assert!(v.is_none());
}
