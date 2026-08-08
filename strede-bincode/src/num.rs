//! Sync (borrow-family) config-generic wire decode helpers.
//!
//! Unsigned/signed integer decoders are widened uniformly to `u128`/`i128`
//! regardless of the target Rust type's width, so the caller
//! ([`crate::impls`]'s `ParseNum<C>` impls) can range-check via `TryFrom`
//! and fall back to `Probe::Miss` on overflow. This can genuinely happen
//! even for a well-formed stream in `Varint` mode: the decode algorithm's
//! byte count is determined entirely by the wire prefix byte, not by which
//! Rust type the caller asked for (see [`decode_varint_u128`]).
//!
//! [`decode_varint_u128`] also enforces real bincode2's canonical-form rule:
//! a decode targeting a given width rejects a wire prefix wider than that
//! width could ever legitimately need (e.g. decoding a `u16` rejects the
//! 4/8/16-byte tail prefixes outright, regardless of what value the tail
//! bytes decode to) — verified against the vendored `bincode` 2.0.1 source
//! (`varint_decode_u16`/`u32`/`u64`, each of which errors on a discriminant
//! wider than their own target width). Each of [`decode_u16`], [`decode_u32`],
//! [`decode_u64`] (and their signed/zigzag counterparts) passes the narrowest
//! prefix its own width could ever use; [`decode_u128`]/[`decode_i128`] have
//! no narrower width to violate, so every prefix is legal for them.
//!
//! Following the convention already established by `strede-postcard`'s own
//! `ParseNum` impls: `Err` is reserved for stream truncation and structural
//! corruption (ran out of bytes, a varint prefix byte of `255` which no
//! valid encoder ever emits); anything else that fails to decode into the
//! requested shape (range overflow, a non-canonical prefix width, an invalid
//! bool tag, an invalid `char` byte sequence) is a probe miss, not an error
//! — except where noted (`decode_len`/`decode_discriminant`, which have no
//! "wrong type, try another" fallback available to their callers and so
//! surface a non-canonical prefix as an `Err` instead).

use crate::BincodeError;
use crate::config::{BincodeConfig, ByteOrder, Fixint, IntEncoding};

#[inline]
fn read_array<const N: usize>(src: &[u8]) -> Result<[u8; N], BincodeError> {
    if src.len() < N {
        return Err(BincodeError::UnexpectedEnd);
    }
    Ok(src[..N].try_into().unwrap())
}

#[inline]
pub(crate) fn is_fixint<C: BincodeConfig>() -> bool {
    C::Int::NAME == Fixint::NAME
}

/// Bincode's own unsigned varint scheme: `0..=250` is a single raw byte
/// equal to the value; `251`/`252`/`253`/`254` announce a 2/4/8/16-byte
/// tail (in the configured byte order) holding the value; `255` is never
/// emitted by any valid encoder and is always a corrupt stream.
///
/// `max_prefix` is the widest tail-announcing prefix (251/252/253/254) legal
/// for the calling decoder's own target width — e.g. a `u16` decode passes
/// `251` (its own width), so a `252`/`253`/`254` prefix returns `Ok(None)`
/// (canonical-form violation: the tail is wider than a `u16` could ever
/// need) rather than being decoded and only range-checked afterward. `254`
/// (passed by `u128`/`i128`) imposes no restriction, since nothing is wider.
fn decode_varint_u128<O: ByteOrder>(
    src: &[u8],
    max_prefix: u8,
) -> Result<Option<(u128, usize)>, BincodeError> {
    let &prefix = src.first().ok_or(BincodeError::UnexpectedEnd)?;
    match prefix {
        0..=250 => Ok(Some((prefix as u128, 1))),
        251..=254 if prefix > max_prefix => Ok(None),
        251 => {
            let b = read_array::<2>(&src[1..])?;
            Ok(Some((O::read_u16(b) as u128, 3)))
        }
        252 => {
            let b = read_array::<4>(&src[1..])?;
            Ok(Some((O::read_u32(b) as u128, 5)))
        }
        253 => {
            let b = read_array::<8>(&src[1..])?;
            Ok(Some((O::read_u64(b) as u128, 9)))
        }
        254 => {
            let b = read_array::<16>(&src[1..])?;
            Ok(Some((O::read_u128(b), 17)))
        }
        255 => Err(BincodeError::InvalidVarint),
    }
}

/// Zigzag-decode: inverse of bincode's `n>=0 -> 2n; n<0 -> -2n-1` (same
/// formula `strede-postcard` uses for its own zigzag varints).
#[inline]
fn zigzag_decode(raw: u128) -> i128 {
    ((raw >> 1) as i128) ^ (-((raw & 1) as i128))
}

/// `Ok(None)` means the wire prefix was wider than `$name`'s own target
/// width could ever need (canonical-form violation, see
/// [`decode_varint_u128`]) — callers map this to `Probe::Miss` (typed
/// numeric decode) or a hard `Err` (`decode_len`/`decode_discriminant`,
/// which have no "try another type" fallback).
macro_rules! decode_unsigned {
    ($name:ident, $read:ident, $n:literal, $max_prefix:literal) => {
        pub(crate) fn $name<C: BincodeConfig>(
            src: &[u8],
        ) -> Result<Option<(u128, usize)>, BincodeError> {
            if is_fixint::<C>() {
                let b = read_array::<$n>(src)?;
                Ok(Some((C::Order::$read(b) as u128, $n)))
            } else {
                decode_varint_u128::<C::Order>(src, $max_prefix)
            }
        }
    };
}

decode_unsigned!(decode_u16, read_u16, 2, 251);
decode_unsigned!(decode_u32, read_u32, 4, 252);
decode_unsigned!(decode_u64, read_u64, 8, 253);

/// No narrower width to violate, so every prefix (`max_prefix = 254`) is
/// legal — `decode_varint_u128` never returns `None` here.
pub(crate) fn decode_u128<C: BincodeConfig>(src: &[u8]) -> Result<(u128, usize), BincodeError> {
    if is_fixint::<C>() {
        let b = read_array::<16>(src)?;
        Ok((C::Order::read_u128(b), 16))
    } else {
        Ok(decode_varint_u128::<C::Order>(src, 254)?.expect("254 permits every prefix"))
    }
}

macro_rules! decode_signed {
    ($name:ident, $read:ident, $n:literal, $ty:ty, $max_prefix:literal) => {
        pub(crate) fn $name<C: BincodeConfig>(
            src: &[u8],
        ) -> Result<Option<(i128, usize)>, BincodeError> {
            if is_fixint::<C>() {
                let b = read_array::<$n>(src)?;
                Ok(Some((C::Order::$read(b) as $ty as i128, $n)))
            } else {
                match decode_varint_u128::<C::Order>(src, $max_prefix)? {
                    Some((raw, c)) => Ok(Some((zigzag_decode(raw), c))),
                    None => Ok(None),
                }
            }
        }
    };
}

decode_signed!(decode_i16, read_u16, 2, i16, 251);
decode_signed!(decode_i32, read_u32, 4, i32, 252);
decode_signed!(decode_i64, read_u64, 8, i64, 253);

/// No narrower width to violate, mirrors [`decode_u128`].
pub(crate) fn decode_i128<C: BincodeConfig>(src: &[u8]) -> Result<(i128, usize), BincodeError> {
    if is_fixint::<C>() {
        let b = read_array::<16>(src)?;
        Ok((C::Order::read_u128(b) as i128, 16))
    } else {
        let (raw, c) = decode_varint_u128::<C::Order>(src, 254)?.expect("254 permits every prefix");
        Ok((zigzag_decode(raw), c))
    }
}

/// `u8`/`i8` are always exactly 1 raw byte, regardless of config — real
/// bincode never routes 8-bit widths through `IntEncoding` at all.
pub(crate) fn decode_u8(src: &[u8]) -> Result<(u8, usize), BincodeError> {
    let &b = src.first().ok_or(BincodeError::UnexpectedEnd)?;
    Ok((b, 1))
}
pub(crate) fn decode_i8(src: &[u8]) -> Result<(i8, usize), BincodeError> {
    let &b = src.first().ok_or(BincodeError::UnexpectedEnd)?;
    Ok((b as i8, 1))
}

/// Floats are always fixed-width IEEE754 — varies only by byte order, never
/// by `IntEncoding`.
pub(crate) fn decode_f32<C: BincodeConfig>(src: &[u8]) -> Result<(f32, usize), BincodeError> {
    let b = read_array::<4>(src)?;
    Ok((f32::from_bits(C::Order::read_u32(b)), 4))
}
pub(crate) fn decode_f64<C: BincodeConfig>(src: &[u8]) -> Result<(f64, usize), BincodeError> {
    let b = read_array::<8>(src)?;
    Ok((f64::from_bits(C::Order::read_u64(b)), 8))
}

/// `bool` is always exactly 1 raw byte, 0/1, regardless of config. A wire
/// byte outside `{0, 1}` is a miss (matches `strede-postcard`'s own
/// `ParseNum for bool`), not a stream-corruption error.
pub(crate) fn decode_bool(src: &[u8]) -> Result<Option<(bool, usize)>, BincodeError> {
    match src.first() {
        None => Err(BincodeError::UnexpectedEnd),
        Some(0) => Ok(Some((false, 1))),
        Some(1) => Ok(Some((true, 1))),
        Some(_) => Ok(None),
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
/// bytes) — completely independent of both `ByteOrder` and `IntEncoding`.
/// `Ok(None)` (caller maps to `Probe::Miss`) covers any invalid sequence —
/// bad lead byte, bad continuation bytes, overlong or surrogate-range
/// encodings — matching `strede-postcard`'s `ParseNum for char` treating an
/// invalid codepoint as a miss rather than an error. Running out of bytes
/// for a claimed multi-byte lead is still `Err(UnexpectedEnd)` (truncation).
pub(crate) fn decode_char(src: &[u8]) -> Result<Option<(char, usize)>, BincodeError> {
    let &lead = src.first().ok_or(BincodeError::UnexpectedEnd)?;
    let len = utf8_len(lead);
    if len == 0 {
        return Ok(None);
    }
    if src.len() < len {
        return Err(BincodeError::UnexpectedEnd);
    }
    match core::str::from_utf8(&src[..len])
        .ok()
        .and_then(|s| s.chars().next())
    {
        Some(c) if c.len_utf8() == len => Ok(Some((c, len))),
        _ => Ok(None),
    }
}

/// Length prefix (`Vec<T>`/`String`/`HashMap`/slices): cast to `u64`,
/// encoded via `u64`'s own rule. Truncating `as usize` cast, matching
/// `strede-postcard`'s equally permissive `len as usize` — this is a
/// deserialization library for trusted-schema data, not a hardened decoder
/// against adversarial length prefixes. A non-canonical prefix (wider than a
/// `u64` length could ever need) has no "try another type" fallback here, so
/// it surfaces as `Err` rather than a probe miss.
pub(crate) fn decode_len<C: BincodeConfig>(src: &[u8]) -> Result<(usize, usize), BincodeError> {
    let (v, c) = decode_u64::<C>(src)?.ok_or(BincodeError::NonCanonicalVarint)?;
    Ok((v as usize, c))
}

/// Enum discriminant: `u32`, encoded via `u32`'s own rule (respects
/// `IntEncoding`). A non-canonical prefix is `Err` for the same reason as
/// `decode_len`. Note this is what closes the previous truncation hole: `v`
/// is guaranteed `<= u32::MAX` here — `decode_u32`'s own canonical-width
/// check (`max_prefix = 252`) rejects any prefix wide enough to carry a
/// larger value (e.g. the 8/16-byte tail, prefix `253`/`254`) as `None`
/// before this function ever sees it, so the `as usize` cast below can no
/// longer silently wrap an out-of-range discriminant into a valid-looking
/// small index.
pub(crate) fn decode_discriminant<C: BincodeConfig>(
    src: &[u8],
) -> Result<(usize, usize), BincodeError> {
    let (v, c) = decode_u32::<C>(src)?.ok_or(BincodeError::NonCanonicalVarint)?;
    Ok((v as usize, c))
}

/// Returns the raw *magnitude* byte span of one varint token — the tail
/// bytes only, excluding the 1-byte length-tag prefix — without
/// interpreting it. Used by `deserialize_number_chunks`, which exposes a
/// number's raw wire bytes to callers (e.g. an arbitrary-precision number
/// type) without committing to a specific target width. For the `0..=250`
/// single-byte case there is no separate tag to strip: the prefix byte *is*
/// the value. Only meaningful in `Varint` mode, where the span length is
/// self-describing from the prefix byte alone; `Fixint` mode has no such
/// self-describing span (see the caller in `full.rs`/`chunked/mod.rs`,
/// which never invokes this in `Fixint` mode). Returns
/// `(magnitude_bytes, total_consumed)` — `total_consumed` includes the
/// prefix byte and is what callers use to advance the cursor.
pub(crate) fn varint_span(src: &[u8]) -> Result<(&[u8], usize), BincodeError> {
    let &prefix = src.first().ok_or(BincodeError::UnexpectedEnd)?;
    let (tail_len, total) = match prefix {
        0..=250 => (0, 1),
        251 => (2, 3),
        252 => (4, 5),
        253 => (8, 9),
        254 => (16, 17),
        255 => return Err(BincodeError::InvalidVarint),
    };
    if src.len() < total {
        return Err(BincodeError::UnexpectedEnd);
    }
    let bytes = if tail_len == 0 {
        &src[..1]
    } else {
        &src[total - tail_len..total]
    };
    Ok((bytes, total))
}
