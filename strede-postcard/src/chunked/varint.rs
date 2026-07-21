//! Resumable LEB128 varint decoding for chunked postcard input.
//!
//! Mirrors `crate::varint`'s continuation-bit/shift logic, but reads one byte
//! at a time from a `Handle`, refilling via [`super::refill`] whenever the
//! current chunk is exhausted mid-varint. Kept separate from the sync
//! `crate::varint` module rather than unified with it: the sync functions
//! assume the whole varint is already contiguous in one `&[u8]` and have no
//! accumulator to resume from, so sharing the I/O shape isn't possible - only
//! the per-byte shift/overflow arithmetic is duplicated, and that's small
//! enough not to be worth an awkward sync/async-generic abstraction.
//!
//! `u128`/`i128` are not handled here: like the sync `crate::impls` module,
//! callers chain two [`read_varint`] calls (low then high) at the use site.

use crate::PostcardError;
use strede::{Buffer, Handle};

use super::refill;

/// Read one resumable LEB128 varint. Returns the decoded `u64` value.
///
/// Errors with `UnexpectedEnd` if 10 bytes are read without a terminating
/// byte (continuation bit clear), matching `crate::varint::decode_varint`'s
/// bound (`shift >= 64`).
pub(crate) async fn read_varint<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, u64), PostcardError> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        let buf = handle.buf();
        if *offset >= buf.len() {
            handle = refill(handle, offset).await?;
            continue;
        }
        let byte = buf[*offset];
        *offset += 1;
        let low7 = (byte & 0x7f) as u64;
        result |= low7 << shift;
        if byte & 0x80 == 0 {
            return Ok((handle, result));
        }
        shift += 7;
        if shift >= 64 {
            return Err(PostcardError::UnexpectedEnd);
        }
    }
}

/// Zigzag-decode wrapper: [`read_varint`] followed by the same inversion
/// formula as `crate::varint::decode_zigzag`.
pub(crate) async fn read_zigzag<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, i64), PostcardError> {
    let (handle, n) = read_varint(handle, offset).await?;
    let decoded = ((n >> 1) as i64) ^ (-((n & 1) as i64));
    Ok((handle, decoded))
}

/// Read a varint and also return its raw encoded bytes, for
/// [`strede::NumberAccessOwned`] which must hand out `&[u8]` (matching
/// `LittleEndian::Data`). Buffers into a fixed-size stack array since the
/// bytes may be scattered across chunk boundaries and can't be borrowed from
/// any single chunk. Returns `(handle, bytes, len)`; caller slices `[..len]`.
///
/// Bound matches `crate::varint::varint_bytes`: up to 10 bytes are
/// permitted, erroring only when an 11th byte would be needed with no
/// terminator seen yet.
pub(crate) async fn read_varint_bytes<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, [u8; 10], usize), PostcardError> {
    let mut out = [0u8; 10];
    let mut len = 0usize;
    loop {
        let buf = handle.buf();
        if *offset >= buf.len() {
            handle = refill(handle, offset).await?;
            continue;
        }
        let byte = buf[*offset];
        *offset += 1;
        if len == 10 {
            return Err(PostcardError::UnexpectedEnd);
        }
        out[len] = byte;
        len += 1;
        if byte & 0x80 == 0 {
            return Ok((handle, out, len));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strede_test_util::block_on_loop;

    fn decode(input: &'static [u8]) -> Result<u64, PostcardError> {
        block_on_loop(strede::SharedBuf::with_async(
            input,
            async |buf: &mut &'static [u8]| {
                *buf = &[];
            },
            async |shared| {
                let handle = shared.fork();
                let mut offset = 0usize;
                let (_, v) = read_varint(handle, &mut offset).await?;
                Ok(v)
            },
        ))
    }

    #[test]
    fn single_byte() {
        assert_eq!(decode(&[0x01]).unwrap(), 1);
        assert_eq!(decode(&[0x7f]).unwrap(), 127);
    }

    #[test]
    fn multi_byte() {
        // 300 = 0b1_0010_1100 -> [0xac, 0x02]
        assert_eq!(decode(&[0xac, 0x02]).unwrap(), 300);
    }

    #[test]
    fn max_width_u64() {
        let bytes: &'static [u8] = &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert_eq!(decode(bytes).unwrap(), u64::MAX);
    }

    #[test]
    fn truncated_errors() {
        assert_eq!(decode(&[0x80]), Err(PostcardError::UnexpectedEnd));
    }

    #[test]
    fn eleven_continuation_bytes_errors() {
        let bytes: &'static [u8] = &[0xff; 11];
        assert_eq!(decode(bytes), Err(PostcardError::UnexpectedEnd));
    }
}
