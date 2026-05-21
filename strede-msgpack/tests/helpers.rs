#![allow(dead_code)]

/// Helper: encode a fixstr (string ≤ 31 bytes)
pub fn fixstr(s: &str) -> Vec<u8> {
    assert!(s.len() <= 31);
    let mut v = vec![0xa0 | s.len() as u8];
    v.extend_from_slice(s.as_bytes());
    v
}

/// Helper: encode a fixmap header (≤ 15 pairs)
pub fn fixmap(count: usize) -> u8 {
    assert!(count <= 15);
    0x80 | count as u8
}

/// Helper: encode a fixarray header (≤ 15 elements)
pub fn fixarray(count: usize) -> u8 {
    assert!(count <= 15);
    0x90 | count as u8
}

/// Helper: encode a uint16 value
pub fn uint16(v: u16) -> Vec<u8> {
    let mut out = vec![0xcd];
    out.extend_from_slice(&v.to_be_bytes());
    out
}

/// Helper: encode a uint32 value
pub fn uint32(v: u32) -> Vec<u8> {
    let mut out = vec![0xce];
    out.extend_from_slice(&v.to_be_bytes());
    out
}

/// Helper: encode an int8 value
pub fn int8(v: i8) -> Vec<u8> {
    vec![0xd0, v as u8]
}

/// Helper: encode an int16 value
pub fn int16(v: i16) -> Vec<u8> {
    let mut out = vec![0xd1];
    out.extend_from_slice(&v.to_be_bytes());
    out
}

/// Helper: encode a float32 value
pub fn float32(v: f32) -> Vec<u8> {
    let mut out = vec![0xca];
    out.extend_from_slice(&v.to_bits().to_be_bytes());
    out
}

/// Helper: encode a float64 value
pub fn float64(v: f64) -> Vec<u8> {
    let mut out = vec![0xcb];
    out.extend_from_slice(&v.to_bits().to_be_bytes());
    out
}

/// Helper: encode a bin8 value
pub fn bin8(data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= 255);
    let mut out = vec![0xc4, data.len() as u8];
    out.extend_from_slice(data);
    out
}

/// Helper: encode a fixext1 value
pub fn fixext1(type_id: i8, data: u8) -> Vec<u8> {
    vec![0xd4, type_id as u8, data]
}

/// Helper: encode a fixext2 value
pub fn fixext2(type_id: i8, data: [u8; 2]) -> Vec<u8> {
    vec![0xd5, type_id as u8, data[0], data[1]]
}

/// Helper: encode a fixext4 value
pub fn fixext4(type_id: i8, data: [u8; 4]) -> Vec<u8> {
    let mut v = vec![0xd6, type_id as u8];
    v.extend_from_slice(&data);
    v
}

/// Helper: encode a fixext8 value
pub fn fixext8(type_id: i8, data: [u8; 8]) -> Vec<u8> {
    let mut v = vec![0xd7, type_id as u8];
    v.extend_from_slice(&data);
    v
}

/// Helper: encode a fixext16 value
pub fn fixext16(type_id: i8, data: [u8; 16]) -> Vec<u8> {
    let mut v = vec![0xd8, type_id as u8];
    v.extend_from_slice(&data);
    v
}

/// Helper: encode an ext8 value
pub fn ext8(type_id: i8, data: &[u8]) -> Vec<u8> {
    assert!(data.len() <= 255);
    let mut v = vec![0xc7, data.len() as u8, type_id as u8];
    v.extend_from_slice(data);
    v
}

/// Helper: encode a Timestamp 32 (fixext4, type_id = -1)
pub fn timestamp32(sec: u32) -> Vec<u8> {
    let mut v = vec![0xd6, 0xffu8]; // fixext4, type_id = -1
    v.extend_from_slice(&sec.to_be_bytes());
    v
}

/// Helper: encode a Timestamp 64 (fixext8, type_id = -1)
pub fn timestamp64(nsec: u32, sec: u64) -> Vec<u8> {
    let val: u64 = ((nsec as u64) << 34) | sec;
    let mut v = vec![0xd7, 0xffu8]; // fixext8, type_id = -1
    v.extend_from_slice(&val.to_be_bytes());
    v
}

/// Helper: encode a Timestamp 96 (ext8 len=12, type_id = -1)
pub fn timestamp96(nsec: u32, sec: i64) -> Vec<u8> {
    let mut v = vec![0xc7, 0x0c, 0xffu8]; // ext8, len=12, type_id = -1
    v.extend_from_slice(&nsec.to_be_bytes());
    v.extend_from_slice(&sec.to_be_bytes());
    v
}

/// Build a map from alternating key-value byte slices.
pub fn build_map(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
    let n = pairs.len();
    let mut out = if n <= 15 {
        vec![fixmap(n)]
    } else {
        let mut h = vec![0xde];
        h.extend_from_slice(&(n as u16).to_be_bytes());
        h
    };
    for (k, v) in pairs {
        out.extend_from_slice(k);
        out.extend_from_slice(v);
    }
    out
}
