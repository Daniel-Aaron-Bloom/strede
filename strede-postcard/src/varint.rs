use crate::PostcardError;

/// Decode a postcard variable-length unsigned integer from `src`.
///
/// Postcard uses a variant of LEB128 where the continuation bit is the MSB
/// of each byte (0x80). Returns (value, bytes_consumed).
#[inline]
pub fn decode_varint(src: &[u8]) -> Result<(u64, usize), PostcardError> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    for (i, &byte) in src.iter().enumerate() {
        let low7 = (byte & 0x7f) as u64;
        result |= low7 << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return Err(PostcardError::UnexpectedEnd);
        }
    }
    Err(PostcardError::UnexpectedEnd)
}

/// Decode a postcard variable-length signed integer (zigzag + varint).
///
/// Zigzag encoding: n → (n << 1) ^ (n >> 63). Inverse: (n >> 1) ^ -(n & 1).
#[inline]
pub fn decode_zigzag(src: &[u8]) -> Result<(i64, usize), PostcardError> {
    let (n, consumed) = decode_varint(src)?;
    let decoded = ((n >> 1) as i64) ^ (-((n & 1) as i64));
    Ok((decoded, consumed))
}

/// Decode a varint and return just the raw bytes that encoded it (for NumberAccess).
/// Returns a slice of `src` containing exactly the varint bytes.
#[inline]
pub fn varint_bytes(src: &[u8]) -> Result<(&[u8], usize), PostcardError> {
    for (i, &byte) in src.iter().enumerate() {
        if byte & 0x80 == 0 {
            return Ok((&src[..=i], i + 1));
        }
        if i >= 9 {
            return Err(PostcardError::UnexpectedEnd);
        }
    }
    Err(PostcardError::UnexpectedEnd)
}
