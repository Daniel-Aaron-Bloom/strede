use crate::CborError;

/// A decoded CBOR header token.
///
/// For `Bstr`/`Tstr` (definite): carries the byte length of the payload, which
/// follows immediately after the header in the stream.
///
/// For `BstrIndef`/`TstrIndef`: payload is a sequence of definite-length chunks
/// each prefixed by their own header, terminated by a break byte (0xff).
///
/// For `Array`/`Map`: `Some(n)` means definite-length with n elements/pairs;
/// `None` means indefinite-length, terminated by break (0xff).
///
/// Numbers, bool, null, and undefined carry their values inline.
///
/// `Tag` carries the tag number; the tagged value immediately follows in the
/// stream and must be decoded with a subsequent token call.
#[derive(Clone, Copy, Debug)]
pub enum CborToken {
    // Major type 0: unsigned integer
    UInt(u64),
    // Major type 1: negative integer, stored as raw additional value N.
    //   actual value = -1 - N.  ParseNum converts this.
    NegInt(u64),
    // Major type 2: byte string (definite-length)
    Bstr(usize),
    // Major type 2: byte string (indefinite-length, chunks follow)
    BstrIndef,
    // Major type 3: text string (definite-length)
    Tstr(usize),
    // Major type 3: text string (indefinite-length, chunks follow)
    TstrIndef,
    // Major type 4: array. Some(n) = definite, None = indefinite
    Array(Option<usize>),
    // Major type 5: map. Some(n) = definite, None = indefinite
    Map(Option<usize>),
    // Major type 6: semantic tag number
    Tag(u64),
    // Major type 7: simple / float
    Bool(bool),
    Null,
    Undefined,
    /// Half-precision float, decoded to f32.
    Float16(f32),
    Float32(f32),
    Float64(f64),
    /// Break stop-code (0xff). Only valid inside indefinite-length collections
    /// or indefinite-length bstr/tstr. Returns an error if encountered outside.
    Break,
}

/// Read the next CBOR token from `src`, advancing it past the header bytes.
///
/// For `Bstr`/`Tstr` (definite): payload bytes are NOT consumed; caller reads them.
/// For `Array`/`Map`: element/pair count returned; caller iterates them.
/// For scalars: value is inline in the token.
pub fn next_token(src: &mut &[u8]) -> Result<CborToken, CborError> {
    let byte = match src.first() {
        Some(&b) => {
            *src = &src[1..];
            b
        }
        None => return Err(CborError::UnexpectedEnd),
    };
    parse_token(byte, src)
}

/// Read the additional value given the lower 5 bits of the initial byte.
/// CBOR additional info encoding:
///   0-23  → value directly
///   24    → 1-byte uint follows
///   25    → 2-byte uint follows
///   26    → 4-byte uint follows
///   27    → 8-byte uint follows
///   31    → indefinite-length (for applicable major types)
macro_rules! read_be {
    ($src:expr, $n:literal, $t:ty) => {{
        if $src.len() < $n {
            return Err(CborError::UnexpectedEnd);
        }
        let mut b = [0u8; $n];
        b.copy_from_slice(&$src[..$n]);
        *$src = &$src[$n..];
        <$t>::from_be_bytes(b)
    }};
}

fn read_argument(src: &mut &[u8], info: u8) -> Result<u64, CborError> {
    match info {
        0..=23 => Ok(info as u64),
        24 => Ok(read_be!(src, 1, u8) as u64),
        25 => Ok(read_be!(src, 2, u16) as u64),
        26 => Ok(read_be!(src, 4, u32) as u64),
        27 => Ok(read_be!(src, 8, u64)),
        28..=30 => Err(CborError::UnexpectedByte {
            byte: info, // actual leading byte context lost; info bits shown
        }),
        31 => Err(CborError::UnexpectedByte { byte: 31 }), // indefinite — handled by caller
        _ => unreachable!(),
    }
}

/// Decode half-precision float (binary16) to f32.
pub(crate) fn decode_f16(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal: normalize
            let mut e = 0u32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            m &= 0x3ff;
            ((sign << 31) | ((127 - 15 - e + 1) << 23)) | (m << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Inner decode given the leading format byte.
/// Called by both `next_token` (borrow) and the chunked owned-family decoder.
pub fn parse_token(byte: u8, src: &mut &[u8]) -> Result<CborToken, CborError> {
    let major = byte >> 5;
    let info = byte & 0x1f;

    match major {
        // Major type 0: unsigned integer
        0 => {
            let v = read_argument(src, info)?;
            Ok(CborToken::UInt(v))
        }
        // Major type 1: negative integer
        1 => {
            let v = read_argument(src, info)?;
            Ok(CborToken::NegInt(v))
        }
        // Major type 2: byte string
        2 => {
            if info == 31 {
                return Ok(CborToken::BstrIndef);
            }
            let len = read_argument(src, info)? as usize;
            Ok(CborToken::Bstr(len))
        }
        // Major type 3: text string
        3 => {
            if info == 31 {
                return Ok(CborToken::TstrIndef);
            }
            let len = read_argument(src, info)? as usize;
            Ok(CborToken::Tstr(len))
        }
        // Major type 4: array
        4 => {
            if info == 31 {
                return Ok(CborToken::Array(None));
            }
            let count = read_argument(src, info)? as usize;
            Ok(CborToken::Array(Some(count)))
        }
        // Major type 5: map
        5 => {
            if info == 31 {
                return Ok(CborToken::Map(None));
            }
            let count = read_argument(src, info)? as usize;
            Ok(CborToken::Map(Some(count)))
        }
        // Major type 6: tag
        6 => {
            let n = read_argument(src, info)?;
            Ok(CborToken::Tag(n))
        }
        // Major type 7: float / simple
        7 => match info {
            20 => Ok(CborToken::Bool(false)),
            21 => Ok(CborToken::Bool(true)),
            22 => Ok(CborToken::Null),
            23 => Ok(CborToken::Undefined),
            24 => {
                // simple value in next byte (only 32..=255 are valid)
                let _v = read_be!(src, 1, u8);
                Err(CborError::UnexpectedByte { byte })
            }
            25 => {
                let bits = read_be!(src, 2, u16);
                Ok(CborToken::Float16(decode_f16(bits)))
            }
            26 => {
                let bits = read_be!(src, 4, u32);
                Ok(CborToken::Float32(f32::from_bits(bits)))
            }
            27 => {
                let bits = read_be!(src, 8, u64);
                Ok(CborToken::Float64(f64::from_bits(bits)))
            }
            31 => Ok(CborToken::Break),
            _ => Err(CborError::UnexpectedByte { byte }),
        },
        _ => unreachable!(),
    }
}

/// Skip one complete CBOR value given its leading token.
///
/// Indefinite-length collections and strings are handled iteratively.
/// No recursion — uses a depth stack to stay no_std and avoid async layout cycles.
pub fn skip_value(src: &mut &[u8], tok: CborToken) -> Result<(), CborError> {
    // Use a simple counter stack: each entry is either
    //   - (count, false) for a definite array of `count` remaining items
    //   - (count, true)  for a definite map of `count` remaining *values*
    //     (key+value = 2 items, so map(n) seeds 2*n)
    //   - (usize::MAX, _) signals "indefinite, look for Break"
    //
    // We flatten to a single stack of remaining-item counts.
    // Indefinite is flagged as usize::MAX.

    const INDEF: usize = usize::MAX;

    let mut stack: [usize; 64] = [0; 64];
    let mut depth: usize = 0;

    macro_rules! push {
        ($v:expr) => {
            if depth >= 64 {
                return Err(CborError::SkipDepthExceeded);
            }
            stack[depth] = $v;
            depth += 1;
        };
    }

    // Seed the stack from the root token.
    match tok {
        CborToken::UInt(_)
        | CborToken::NegInt(_)
        | CborToken::Bool(_)
        | CborToken::Null
        | CborToken::Undefined
        | CborToken::Float16(_)
        | CborToken::Float32(_)
        | CborToken::Float64(_) => return Ok(()),
        CborToken::Break => return Err(CborError::InvalidBreak),
        CborToken::Bstr(len) | CborToken::Tstr(len) => {
            if src.len() < len {
                return Err(CborError::UnexpectedEnd);
            }
            *src = &src[len..];
            return Ok(());
        }
        CborToken::BstrIndef | CborToken::TstrIndef => {
            // Skip definite chunks until Break
            loop {
                let chunk_tok = next_token(src)?;
                match chunk_tok {
                    CborToken::Break => return Ok(()),
                    CborToken::Bstr(len) | CborToken::Tstr(len) => {
                        if src.len() < len {
                            return Err(CborError::UnexpectedEnd);
                        }
                        *src = &src[len..];
                    }
                    _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                }
            }
        }
        CborToken::Tag(_) => {
            // Just skip the tagged value
            let inner = next_token(src)?;
            return skip_value(src, inner);
        }
        CborToken::Array(Some(0)) | CborToken::Map(Some(0)) => return Ok(()),
        CborToken::Array(Some(n)) => {
            push!(n);
        }
        CborToken::Map(Some(n)) => {
            push!(n.checked_mul(2).ok_or(CborError::SkipDepthExceeded)?);
        }
        CborToken::Array(None) => {
            push!(INDEF);
        }
        CborToken::Map(None) => {
            push!(INDEF);
        }
    }

    loop {
        let remaining = match depth.checked_sub(1) {
            None => break,
            Some(top) => stack[top],
        };

        if remaining == INDEF {
            // Indefinite: read next token; if Break, pop and continue
            let next = next_token(src)?;
            match next {
                CborToken::Break => {
                    depth -= 1;
                }
                other => {
                    skip_token_payload(src, other, &mut stack, &mut depth)?;
                }
            }
        } else {
            // Definite: consume one item, decrement
            depth -= 1;
            if remaining > 1 {
                if depth >= 64 {
                    return Err(CborError::SkipDepthExceeded);
                }
                stack[depth] = remaining - 1;
                depth += 1;
            }
            let next = next_token(src)?;
            skip_token_payload(src, next, &mut stack, &mut depth)?;
        }
    }

    Ok(())
}

/// Handle the payload (and possibly stack push) for a single token encountered
/// during iterative skip. Does not handle Break (caller handles that).
fn skip_token_payload(
    src: &mut &[u8],
    tok: CborToken,
    stack: &mut [usize; 64],
    depth: &mut usize,
) -> Result<(), CborError> {
    const INDEF: usize = usize::MAX;

    macro_rules! push {
        ($v:expr) => {
            if *depth >= 64 {
                return Err(CborError::SkipDepthExceeded);
            }
            stack[*depth] = $v;
            *depth += 1;
        };
    }

    match tok {
        CborToken::UInt(_)
        | CborToken::NegInt(_)
        | CborToken::Bool(_)
        | CborToken::Null
        | CborToken::Undefined
        | CborToken::Float16(_)
        | CborToken::Float32(_)
        | CborToken::Float64(_) => {}
        CborToken::Break => return Err(CborError::InvalidBreak),
        CborToken::Bstr(len) | CborToken::Tstr(len) => {
            if src.len() < len {
                return Err(CborError::UnexpectedEnd);
            }
            *src = &src[len..];
        }
        CborToken::BstrIndef | CborToken::TstrIndef => loop {
            let chunk_tok = next_token(src)?;
            match chunk_tok {
                CborToken::Break => break,
                CborToken::Bstr(len) | CborToken::Tstr(len) => {
                    if src.len() < len {
                        return Err(CborError::UnexpectedEnd);
                    }
                    *src = &src[len..];
                }
                _ => return Err(CborError::UnexpectedByte { byte: 0 }),
            }
        },
        CborToken::Tag(_) => {
            // Tagged value: push 1 more item to process
            push!(1);
        }
        CborToken::Array(Some(0)) | CborToken::Map(Some(0)) => {}
        CborToken::Array(Some(n)) => {
            push!(n);
        }
        CborToken::Map(Some(n)) => {
            push!(n.checked_mul(2).ok_or(CborError::SkipDepthExceeded)?);
        }
        CborToken::Array(None) => {
            push!(INDEF);
        }
        CborToken::Map(None) => {
            push!(INDEF);
        }
    }
    Ok(())
}
