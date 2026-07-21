//! Collection deserialization via the owned/chunked family: Vec<T>,
//! Option<Vec<T>>, HashMap/BTreeMap, nested collections. Mirrors
//! `collections_borrow.rs`; `Vec<&str>` becomes `Vec<String>`.

#![recursion_limit = "256"]

#[macro_use]
mod helpers;
use helpers::*;

use strede_derive::DeserializeOwned;
use strede_postcard::PostcardError;

// --- Vec<u32> ---

#[test]
fn vec_u32_empty() {
    assert_eq!(parse_owned!(Vec<u32>, &pseq(&[])), Ok(Some(vec![])));
}

#[test]
fn vec_u32_single() {
    let elem = varint(42);
    let data = pseq(&[elem.as_slice()]);
    assert_eq!(parse_owned!(Vec<u32>, &data), Ok(Some(vec![42u32])));
}

#[test]
fn vec_u32_multiple() {
    let e1 = varint(1);
    let e2 = varint(2);
    let e3 = varint(3);
    let data = pseq(&[e1.as_slice(), e2.as_slice(), e3.as_slice()]);
    assert_eq!(parse_owned!(Vec<u32>, &data), Ok(Some(vec![1u32, 2, 3])));
}

#[test]
fn vec_u32_truncated_errors() {
    // Count says 2 but only 1 element follows.
    let mut data = varint(2);
    data.extend_from_slice(&varint(99));
    assert_eq!(
        parse_owned!(Vec<u32>, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}

// --- Vec<String> (owned-family stand-in for Vec<&str>) ---

#[test]
fn vec_str_empty() {
    assert_eq!(parse_owned!(Vec<String>, &pseq(&[])), Ok(Some(vec![])));
}

#[test]
fn vec_str_values() {
    let s1 = pstr("foo");
    let s2 = pstr("bar");
    let data = pseq(&[s1.as_slice(), s2.as_slice()]);
    assert_eq!(
        parse_owned!(Vec<String>, &data),
        Ok(Some(vec!["foo".to_string(), "bar".to_string()]))
    );
}

// --- Vec<Struct> ---

#[derive(Debug, PartialEq, DeserializeOwned)]
struct Point {
    x: u32,
    y: u32,
}

fn encode_point(x: u64, y: u64) -> Vec<u8> {
    let mut v = varint(x);
    v.extend_from_slice(&varint(y));
    v
}

#[test]
fn vec_struct_empty() {
    assert_eq!(parse_owned!(Vec<Point>, &pseq(&[])), Ok(Some(vec![])));
}

#[test]
fn vec_struct_single() {
    let p = encode_point(3, 7);
    let data = pseq(&[p.as_slice()]);
    assert_eq!(
        parse_owned!(Vec<Point>, &data),
        Ok(Some(vec![Point { x: 3, y: 7 }]))
    );
}

#[test]
fn vec_struct_multiple() {
    let p1 = encode_point(1, 2);
    let p2 = encode_point(3, 4);
    let data = pseq(&[p1.as_slice(), p2.as_slice()]);
    assert_eq!(
        parse_owned!(Vec<Point>, &data),
        Ok(Some(vec![Point { x: 1, y: 2 }, Point { x: 3, y: 4 }]))
    );
}

// --- Vec<u8> as bytes (pbytes wire format) ---

#[test]
fn vec_u8_bytes_empty() {
    assert_eq!(parse_owned!(Vec<u8>, &pbytes(&[])), Ok(Some(vec![])));
}

#[test]
fn vec_u8_bytes_values() {
    assert_eq!(
        parse_owned!(Vec<u8>, &pbytes(&[0x00, 0x01, 0xff])),
        Ok(Some(vec![0x00u8, 0x01, 0xff]))
    );
}

#[test]
fn vec_u8_bytes_high_values() {
    // Values 128-255: the seq path would misparse these (varint != raw byte).
    let data: Vec<u8> = (0u8..=255).collect();
    assert_eq!(parse_owned!(Vec<u8>, &pbytes(&data)), Ok(Some(data)));
}

// --- Option<Vec<u32>> ---

#[test]
fn option_vec_none() {
    assert_eq!(parse_owned!(Option<Vec<u32>>, &pnone()), Ok(Some(None)));
}

#[test]
fn option_vec_some_empty() {
    let inner = pseq(&[]);
    let data = psome(inner.as_slice());
    assert_eq!(
        parse_owned!(Option<Vec<u32>>, &data),
        Ok(Some(Some(vec![])))
    );
}

#[test]
fn option_vec_some_values() {
    let e1 = varint(10);
    let e2 = varint(20);
    let inner = pseq(&[e1.as_slice(), e2.as_slice()]);
    let data = psome(inner.as_slice());
    assert_eq!(
        parse_owned!(Option<Vec<u32>>, &data),
        Ok(Some(Some(vec![10u32, 20])))
    );
}

// --- BTreeMap<u32, u32> / HashMap<u32, u32> ---
//
// Maps use the same varint-count-prefix framing as Vec, but structs (no
// framing at all, matched positionally) share the same key-probe type — see
// `ChunkedPostcardMapKeyProbe::dynamic` in strede-postcard/src/chunked/access.rs.

use std::collections::{BTreeMap, HashMap};

#[test]
fn btreemap_u32_u32_empty() {
    assert_eq!(
        parse_owned!(BTreeMap<u32, u32>, &pmap(&[])),
        Ok(Some(BTreeMap::new()))
    );
}

#[test]
fn btreemap_u32_u32_entries() {
    let k1 = varint(1);
    let v1 = varint(10);
    let k2 = varint(2);
    let v2 = varint(20);
    let data = pmap(&[(&k1, &v1), (&k2, &v2)]);
    let expected = BTreeMap::from([(1u32, 10u32), (2u32, 20u32)]);
    assert_eq!(parse_owned!(BTreeMap<u32, u32>, &data), Ok(Some(expected)));
}

#[test]
fn hashmap_u32_u32_entries() {
    let k1 = varint(1);
    let v1 = varint(10);
    let k2 = varint(2);
    let v2 = varint(20);
    let data = pmap(&[(&k1, &v1), (&k2, &v2)]);
    let expected = HashMap::from([(1u32, 10u32), (2u32, 20u32)]);
    assert_eq!(parse_owned!(HashMap<u32, u32>, &data), Ok(Some(expected)));
}

#[test]
fn hashmap_u32_u32_truncated_errors() {
    // Count says 2 but only 1 pair follows.
    let k1 = varint(1);
    let v1 = varint(10);
    let mut data = varint(2);
    data.extend_from_slice(&k1);
    data.extend_from_slice(&v1);
    assert_eq!(
        parse_owned!(HashMap<u32, u32>, &data).unwrap_err(),
        PostcardError::UnexpectedEnd
    );
}
