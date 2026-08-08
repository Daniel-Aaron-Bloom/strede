//! Async (owned-family) config-generic wire decode helpers.
//!
//! Mirrors `crate::num`'s sync decode functions, but reads from a
//! `Handle`+`offset` pair instead of a `&[u8]` slice, refilling across
//! chunk boundaries via [`super::read_bytes_exact`]. Bincode's varint
//! scheme (unlike postcard's LEB128) makes this simpler than a
//! byte-at-a-time loop: the single prefix byte immediately determines the
//! entire remaining tail length (0, 2, 4, 8, or 16 more bytes — never
//! ambiguous), so this reduces to "one `read_bytes_exact::<1>` for the
//! prefix, then at most one more `read_bytes_exact::<N>` for the fixed
//! tail" — the same idiom `strede-msgpack`/`strede-cbor` already use for
//! their own multi-byte headers, not postcard's dedicated byte-at-a-time
//! state machine.

use strede::{Buffer, Handle};

use super::read_bytes_exact;
use crate::BincodeError;
use crate::config::{BincodeConfig, ByteOrder, Fixint, IntEncoding};

#[inline]
pub(crate) fn is_fixint<C: BincodeConfig>() -> bool {
    C::Int::NAME == Fixint::NAME
}

/// Bincode's own unsigned varint scheme (async, chunk-boundary-resumable
/// counterpart to `crate::num`'s sync `decode_varint_u128`). `max_prefix`
/// carries the same canonical-form-enforcement meaning as the sync version:
/// `Ok(None)` when the wire prefix announces a tail wider than the calling
/// decoder's own target width could ever need.
async fn decode_varint_u128<'s, B: Buffer, F: AsyncFnMut(&mut B), O: ByteOrder>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
    max_prefix: u8,
) -> Result<(Handle<'s, B, F>, Option<u128>), BincodeError> {
    let (handle, [prefix]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
    if (251..=254).contains(&prefix) && prefix > max_prefix {
        return Ok((handle, None));
    }
    match prefix {
        0..=250 => Ok((handle, Some(prefix as u128))),
        251 => {
            let (handle, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((handle, Some(O::read_u16(b) as u128)))
        }
        252 => {
            let (handle, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((handle, Some(O::read_u32(b) as u128)))
        }
        253 => {
            let (handle, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
            Ok((handle, Some(O::read_u64(b) as u128)))
        }
        254 => {
            let (handle, b) = read_bytes_exact::<_, _, 16>(handle, offset).await?;
            Ok((handle, Some(O::read_u128(b))))
        }
        255 => Err(BincodeError::InvalidVarint),
    }
}

#[inline]
fn zigzag_decode(raw: u128) -> i128 {
    ((raw >> 1) as i128) ^ (-((raw & 1) as i128))
}

/// `Ok((handle, None))` means the wire prefix was wider than `$name`'s own
/// target width could ever need — see `decode_varint_u128`. Callers map this
/// to `Probe::Miss` (typed numeric decode) or a hard `Err` (`decode_len`,
/// `decode_discriminant`).
macro_rules! decode_unsigned {
    ($name:ident, $read:ident, $n:literal, $max_prefix:literal) => {
        pub(crate) async fn $name<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
            handle: Handle<'s, B, F>,
            offset: &mut usize,
        ) -> Result<(Handle<'s, B, F>, Option<u128>), BincodeError> {
            if is_fixint::<C>() {
                let (handle, b) = read_bytes_exact::<_, _, $n>(handle, offset).await?;
                Ok((handle, Some(C::Order::$read(b) as u128)))
            } else {
                decode_varint_u128::<_, _, C::Order>(handle, offset, $max_prefix).await
            }
        }
    };
}

decode_unsigned!(decode_u16, read_u16, 2, 251);
decode_unsigned!(decode_u32, read_u32, 4, 252);
decode_unsigned!(decode_u64, read_u64, 8, 253);

/// No narrower width to violate, so every prefix (`max_prefix = 254`) is
/// legal — `decode_varint_u128` never returns `None` here.
pub(crate) async fn decode_u128<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, u128), BincodeError> {
    if is_fixint::<C>() {
        let (handle, b) = read_bytes_exact::<_, _, 16>(handle, offset).await?;
        Ok((handle, C::Order::read_u128(b)))
    } else {
        let (handle, v) = decode_varint_u128::<_, _, C::Order>(handle, offset, 254).await?;
        Ok((handle, v.expect("254 permits every prefix")))
    }
}

macro_rules! decode_signed {
    ($name:ident, $read:ident, $n:literal, $ty:ty, $max_prefix:literal) => {
        pub(crate) async fn $name<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
            handle: Handle<'s, B, F>,
            offset: &mut usize,
        ) -> Result<(Handle<'s, B, F>, Option<i128>), BincodeError> {
            if is_fixint::<C>() {
                let (handle, b) = read_bytes_exact::<_, _, $n>(handle, offset).await?;
                Ok((handle, Some(C::Order::$read(b) as $ty as i128)))
            } else {
                let (handle, raw) =
                    decode_varint_u128::<_, _, C::Order>(handle, offset, $max_prefix).await?;
                Ok((handle, raw.map(zigzag_decode)))
            }
        }
    };
}

decode_signed!(decode_i16, read_u16, 2, i16, 251);
decode_signed!(decode_i32, read_u32, 4, i32, 252);
decode_signed!(decode_i64, read_u64, 8, i64, 253);

/// No narrower width to violate, mirrors [`decode_u128`].
pub(crate) async fn decode_i128<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, i128), BincodeError> {
    if is_fixint::<C>() {
        let (handle, b) = read_bytes_exact::<_, _, 16>(handle, offset).await?;
        Ok((handle, C::Order::read_u128(b) as i128))
    } else {
        let (handle, raw) = decode_varint_u128::<_, _, C::Order>(handle, offset, 254).await?;
        Ok((
            handle,
            zigzag_decode(raw.expect("254 permits every prefix")),
        ))
    }
}

/// `u8`/`i8` are always exactly 1 raw byte, regardless of config.
pub(crate) async fn decode_u8<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, u8), BincodeError> {
    let (handle, [b]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
    Ok((handle, b))
}
pub(crate) async fn decode_i8<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, i8), BincodeError> {
    let (handle, [b]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
    Ok((handle, b as i8))
}

/// Floats are always fixed-width IEEE754 — varies only by byte order, never
/// by `IntEncoding`.
pub(crate) async fn decode_f32<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, f32), BincodeError> {
    let (handle, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
    Ok((handle, f32::from_bits(C::Order::read_u32(b))))
}
pub(crate) async fn decode_f64<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, f64), BincodeError> {
    let (handle, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
    Ok((handle, f64::from_bits(C::Order::read_u64(b))))
}

/// `bool` is always exactly 1 raw byte, 0/1, regardless of config. An
/// out-of-range byte is `None` (caller maps to `Probe::Miss`), matching
/// `crate::num::decode_bool`'s identical convention.
pub(crate) async fn decode_bool<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, Option<bool>), BincodeError> {
    let (handle, [tag]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
    match tag {
        0 => Ok((handle, Some(false))),
        1 => Ok((handle, Some(true))),
        _ => Ok((handle, None)),
    }
}

#[inline]
fn utf8_len(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead & 0xE0 == 0xC0 {
        2
    } else if lead & 0xF0 == 0xE0 {
        3
    } else if lead & 0xF8 == 0xF0 {
        4
    } else {
        0
    }
}

/// Bincode encodes `char` as its own literal UTF-8 byte sequence (1-4
/// bytes), independent of both `ByteOrder` and `IntEncoding` — see
/// `crate::num::decode_char`'s doc comment for the full rationale. `None`
/// covers any invalid sequence (caller maps to `Probe::Miss`); running out
/// of bytes for a claimed multi-byte lead surfaces as `Err(UnexpectedEnd)`
/// from `read_bytes_exact` itself.
pub(crate) async fn decode_char<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, Option<char>), BincodeError> {
    let (handle, [lead]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
    let len = utf8_len(lead);
    if len == 0 {
        return Ok((handle, None));
    }
    let mut buf = [0u8; 4];
    buf[0] = lead;
    let (handle, filled) = match len {
        1 => (handle, 1),
        2 => {
            let (h, b) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
            buf[1] = b[0];
            (h, 2)
        }
        3 => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            buf[1..3].copy_from_slice(&b);
            (h, 3)
        }
        4 => {
            let (h, b) = read_bytes_exact::<_, _, 3>(handle, offset).await?;
            buf[1..4].copy_from_slice(&b);
            (h, 4)
        }
        _ => unreachable!(),
    };
    match core::str::from_utf8(&buf[..filled])
        .ok()
        .and_then(|s| s.chars().next())
    {
        Some(c) if c.len_utf8() == filled => Ok((handle, Some(c))),
        _ => Ok((handle, None)),
    }
}

/// Length prefix (`Vec<T>`/`String`/`HashMap`/slices): cast to `u64`,
/// encoded via `u64`'s own rule. Truncating cast, matching
/// `crate::num::decode_len`'s identical permissiveness. A non-canonical
/// prefix has no "try another type" fallback here, so it surfaces as `Err`
/// rather than a probe miss — mirrors `crate::num::decode_len`.
pub(crate) async fn decode_len<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, usize), BincodeError> {
    let (handle, v) = decode_u64::<_, _, C>(handle, offset).await?;
    let v = v.ok_or(BincodeError::NonCanonicalVarint)?;
    Ok((handle, v as usize))
}

/// Enum discriminant: `u32`, encoded via `u32`'s own rule (respects
/// `IntEncoding`). Mirrors `crate::num::decode_discriminant`: `decode_u32`'s
/// own canonical-width check already rejects any prefix wide enough to
/// carry a value exceeding `u32::MAX`, so the `as usize` cast below can no
/// longer silently wrap an out-of-range discriminant into a valid-looking
/// small index.
pub(crate) async fn decode_discriminant<'s, B: Buffer, F: AsyncFnMut(&mut B), C: BincodeConfig>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, usize), BincodeError> {
    let (handle, v) = decode_u32::<_, _, C>(handle, offset).await?;
    let v = v.ok_or(BincodeError::NonCanonicalVarint)?;
    Ok((handle, v as usize))
}

/// Reads the raw *magnitude* byte span of one varint token — tail bytes
/// only, excluding the 1-byte length-tag prefix (for the `0..=250`
/// single-byte case there is no separate tag to strip: the prefix byte *is*
/// the value) — into a fixed-size stack buffer, for `NumberAccessOwned`
/// (which must hand out `&[u8]`/`&str` matching the caller's requested
/// `NumberEncoding`), since the bytes may be scattered across chunk
/// boundaries and can't be borrowed from any single chunk. Mirrors
/// `strede-postcard`'s `read_varint_bytes`. Max magnitude is 16 bytes (the
/// `u128` tail). Returns `(handle, magnitude_bytes, magnitude_len)`.
pub(crate) async fn varint_span<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, [u8; 16], usize), BincodeError> {
    let (handle, [prefix]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
    let mut buf = [0u8; 16];
    match prefix {
        0..=250 => {
            buf[0] = prefix;
            Ok((handle, buf, 1))
        }
        251 => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            buf[..2].copy_from_slice(&b);
            Ok((h, buf, 2))
        }
        252 => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            buf[..4].copy_from_slice(&b);
            Ok((h, buf, 4))
        }
        253 => {
            let (h, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
            buf[..8].copy_from_slice(&b);
            Ok((h, buf, 8))
        }
        254 => {
            let (h, b) = read_bytes_exact::<_, _, 16>(handle, offset).await?;
            buf.copy_from_slice(&b);
            Ok((h, buf, 16))
        }
        255 => Err(BincodeError::InvalidVarint),
    }
}
