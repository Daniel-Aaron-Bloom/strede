//! `#[strede(flatten)]` on an internally-tagged enum field — CBOR, owned
//! family. See `strede-json/tests/flatten_internal_candidate_enum_owned.rs`
//! for the design rationale (`CandidateArmStack`, TESTING_GAPS.md item #3(B-2)).
#![recursion_limit = "512"]
#![cfg(feature = "alloc")]
extern crate std;
mod helpers;
use helpers::*;

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_cbor::ChunkedCborDeserializer;
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_test_util::block_on_loop;

macro_rules! parse {
    ($ty:ty, $input:expr) => {{
        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedCborDeserializer::new(shared);
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

fn str_array(items: &[&str]) -> std::vec::Vec<u8> {
    let mut out = array(items.len());
    for s in items {
        out.extend_from_slice(&tstr(s));
    }
    out
}

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
        id: std::string::String,
        method: std::string::String,
    },
    Users {
        users: std::vec::Vec<std::string::String>,
        count: u32,
        #[strede(flatten)]
        pagination: Pagination,
    },
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Envelope {
    tenant: std::vec::Vec<std::string::String>,
    #[strede(flatten)]
    message: Message,
}

#[test]
fn flatten_internal_unit_variant() {
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Ping").as_slice()),
    ]);
    let e: Envelope = parse!(Envelope, &bytes).unwrap();
    assert_eq!(
        e,
        Envelope {
            tenant: std::vec![std::string::String::from("a")],
            message: Message::Ping,
        }
    );
}

#[test]
fn flatten_internal_struct_variant_tag_in_middle() {
    let bytes = build_map(&[
        (tstr("id").as_slice(), tstr("1").as_slice()),
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Request").as_slice()),
        (tstr("method").as_slice(), tstr("GET").as_slice()),
    ]);
    let e: Envelope = parse!(Envelope, &bytes).unwrap();
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
fn flatten_internal_struct_variant_with_nested_flatten() {
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a", "b"]).as_slice()),
        (tstr("type").as_slice(), tstr("Users").as_slice()),
        (
            tstr("users").as_slice(),
            str_array(&["alice", "bob"]).as_slice(),
        ),
        (tstr("count").as_slice(), &[2u8]),
        (tstr("limit").as_slice(), &[10u8]),
        (tstr("offset").as_slice(), &[0u8]),
        (tstr("total").as_slice(), &[2u8]),
    ]);
    let e: Envelope = parse!(Envelope, &bytes).unwrap();
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
fn flatten_internal_missing_tag_misses() {
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("id").as_slice(), tstr("1").as_slice()),
        (tstr("method").as_slice(), tstr("GET").as_slice()),
    ]);
    let v: Option<Envelope> = parse!(Envelope, &bytes);
    assert!(v.is_none());
}

#[test]
fn flatten_internal_unknown_tag_value_misses() {
    let bytes = build_map(&[
        (tstr("tenant").as_slice(), str_array(&["a"]).as_slice()),
        (tstr("type").as_slice(), tstr("Bogus").as_slice()),
    ]);
    let v: Option<Envelope> = parse!(Envelope, &bytes);
    assert!(v.is_none());
}
