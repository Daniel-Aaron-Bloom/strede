#![allow(dead_code)]
extern crate std;
use std::vec::Vec;

/// Encode a small uint (0..=23)
pub fn uint_small(v: u8) -> u8 {
    assert!(v <= 23);
    v
}

/// Encode uint8
pub fn uint8(v: u8) -> Vec<u8> {
    vec![0x18, v]
}

/// Encode uint16
pub fn uint16(v: u16) -> Vec<u8> {
    let mut out = vec![0x19];
    out.extend_from_slice(&v.to_be_bytes());
    out
}

/// Encode uint32
pub fn uint32(v: u32) -> Vec<u8> {
    let mut out = vec![0x1a];
    out.extend_from_slice(&v.to_be_bytes());
    out
}

/// Encode uint64
pub fn uint64(v: u64) -> Vec<u8> {
    let mut out = vec![0x1b];
    out.extend_from_slice(&v.to_be_bytes());
    out
}

/// Encode negint (actual = -1 - n where n is the raw value)
pub fn negint_small(n: u8) -> u8 {
    assert!(n <= 23);
    0x20 | n
}

pub fn negint8(n: u8) -> Vec<u8> {
    vec![0x38, n]
}

pub fn negint16(n: u16) -> Vec<u8> {
    let mut out = vec![0x39];
    out.extend_from_slice(&n.to_be_bytes());
    out
}

/// Encode a definite-length byte string
pub fn bstr(data: &[u8]) -> Vec<u8> {
    let mut out = encode_head(2, data.len());
    out.extend_from_slice(data);
    out
}

/// Encode a definite-length text string
pub fn tstr(s: &str) -> Vec<u8> {
    let mut out = encode_head(3, s.len());
    out.extend_from_slice(s.as_bytes());
    out
}

/// Encode a definite-length array header
pub fn array(count: usize) -> Vec<u8> {
    encode_head(4, count)
}

/// Encode an indefinite-length array header + break
pub fn array_indef(items: &[Vec<u8>]) -> Vec<u8> {
    let mut out = vec![0x9f];
    for item in items {
        out.extend_from_slice(item);
    }
    out.push(0xff);
    out
}

/// Encode a definite-length map header (n pairs follow)
pub fn map(count: usize) -> Vec<u8> {
    encode_head(5, count)
}

/// Build a definite map from alternating key-value byte slices
pub fn build_map(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
    let n = pairs.len();
    let mut out = encode_head(5, n);
    for (k, v) in pairs {
        out.extend_from_slice(k);
        out.extend_from_slice(v);
    }
    out
}

/// Encode a semantic tag
pub fn tag(number: u64) -> Vec<u8> {
    encode_head(6, number as usize)
}

/// CBOR simple values
pub fn cbor_false() -> u8 {
    0xf4
}
pub fn cbor_true() -> u8 {
    0xf5
}
pub fn cbor_null() -> u8 {
    0xf6
}
pub fn cbor_undefined() -> u8 {
    0xf7
}

/// Encode float16 (raw bits)
pub fn float16(bits: u16) -> Vec<u8> {
    let mut out = vec![0xf9];
    out.extend_from_slice(&bits.to_be_bytes());
    out
}

/// Encode float32
pub fn float32(v: f32) -> Vec<u8> {
    let mut out = vec![0xfa];
    out.extend_from_slice(&v.to_bits().to_be_bytes());
    out
}

/// Encode float64
pub fn float64(v: f64) -> Vec<u8> {
    let mut out = vec![0xfb];
    out.extend_from_slice(&v.to_bits().to_be_bytes());
    out
}

fn encode_head(major: u8, value: usize) -> Vec<u8> {
    let prefix = major << 5;
    if value <= 23 {
        vec![prefix | value as u8]
    } else if value <= 0xff {
        vec![prefix | 24, value as u8]
    } else if value <= 0xffff {
        let mut v = vec![prefix | 25];
        v.extend_from_slice(&(value as u16).to_be_bytes());
        v
    } else if value <= 0xffff_ffff {
        let mut v = vec![prefix | 26];
        v.extend_from_slice(&(value as u32).to_be_bytes());
        v
    } else {
        let mut v = vec![prefix | 27];
        v.extend_from_slice(&(value as u64).to_be_bytes());
        v
    }
}
