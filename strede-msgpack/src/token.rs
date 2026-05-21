use crate::MsgpackError;

/// A decoded MessagePack header token.
///
/// For `Str`/`Bin`: carries the byte length of the payload, which follows
/// immediately after the header in the stream.  Callers are responsible for
/// consuming exactly that many bytes.
///
/// For `Array`/`Map`: carries the element/pair count.  Callers iterate that
/// many elements/pairs from the stream.
///
/// Numbers, bool, and nil carry their values inline — no additional bytes to
/// consume from the stream.
#[derive(Clone, Copy, Debug)]
pub enum MsgpackToken {
    Nil,
    Bool(bool),
    /// Positive integer (fixint 0x00–0x7f, uint8/16/32/64).
    UInt(u64),
    /// Negative or signed integer (negative fixint 0xe0–0xff, int8/16/32/64).
    Int(i64),
    Float32(f32),
    Float64(f64),
    /// UTF-8 string; payload is `len` bytes immediately following the header.
    Str(usize),
    /// Raw bytes; payload is `len` bytes immediately following the header.
    Bin(usize),
    /// Array; contains `count` elements.
    Array(usize),
    /// Map; contains `count` key-value pairs.
    Map(usize),
    /// fixext1/2/4/8/16 — all header bytes consumed, payload embedded.
    /// `len` is one of {1, 2, 4, 8, 16}; payload is `data[..len as usize]`.
    FixExt {
        type_id: i8,
        data: [u8; 16],
        len: u8,
    },
    /// ext8/16/32 — type byte consumed, `len` data bytes follow in the stream.
    Ext {
        type_id: i8,
        len: usize,
    },
}

/// Read a multi-byte big-endian integer of size `N` from `src`, advancing it.
/// Returns `Err(UnexpectedEnd)` if fewer than `N` bytes remain.
macro_rules! read_be {
    ($src:expr, $n:literal, $t:ty) => {{
        if $src.len() < $n {
            return Err(MsgpackError::UnexpectedEnd);
        }
        let mut b = [0u8; $n];
        b.copy_from_slice(&$src[..$n]);
        *$src = &$src[$n..];
        <$t>::from_be_bytes(b)
    }};
}

/// Read the next MessagePack token from `src`, advancing it past the header
/// (and any inline payload bytes — numbers, bool, nil).
///
/// For `Str`/`Bin` tokens the payload bytes are **not** consumed; the caller
/// reads them via the accessor.  For `Array`/`Map` tokens the element/pair
/// count is returned; the caller iterates them.
pub fn next_token(src: &mut &[u8]) -> Result<MsgpackToken, MsgpackError> {
    let byte = match src.first() {
        Some(&b) => {
            *src = &src[1..];
            b
        }
        None => return Err(MsgpackError::UnexpectedEnd),
    };
    parse_token(byte, src)
}

/// Decode the token given the leading format byte, consuming any inline bytes
/// from `src`.  Called by both `next_token` (borrow family) and the chunked
/// owned-family `next_dispatch` (which handles buffer splits separately).
pub fn parse_token(byte: u8, src: &mut &[u8]) -> Result<MsgpackToken, MsgpackError> {
    match byte {
        // positive fixint 0x00–0x7f
        0x00..=0x7f => Ok(MsgpackToken::UInt(byte as u64)),
        // fixmap 0x80–0x8f
        0x80..=0x8f => Ok(MsgpackToken::Map((byte & 0x0f) as usize)),
        // fixarray 0x90–0x9f
        0x90..=0x9f => Ok(MsgpackToken::Array((byte & 0x0f) as usize)),
        // fixstr 0xa0–0xbf
        0xa0..=0xbf => Ok(MsgpackToken::Str((byte & 0x1f) as usize)),
        // nil
        0xc0 => Ok(MsgpackToken::Nil),
        // 0xc1 — never used
        0xc1 => Err(MsgpackError::UnexpectedByte { byte }),
        // false / true
        0xc2 => Ok(MsgpackToken::Bool(false)),
        0xc3 => Ok(MsgpackToken::Bool(true)),
        // bin8 / bin16 / bin32
        0xc4 => {
            let len = read_be!(src, 1, u8) as usize;
            Ok(MsgpackToken::Bin(len))
        }
        0xc5 => {
            let len = read_be!(src, 2, u16) as usize;
            Ok(MsgpackToken::Bin(len))
        }
        0xc6 => {
            let len = read_be!(src, 4, u32) as usize;
            Ok(MsgpackToken::Bin(len))
        }
        // ext8/16/32 — 1 type byte + N data bytes follow in stream
        0xc7 => {
            let len = read_be!(src, 1, u8) as usize;
            if src.is_empty() {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            *src = &src[1..];
            Ok(MsgpackToken::Ext { type_id, len })
        }
        0xc8 => {
            let len = read_be!(src, 2, u16) as usize;
            if src.is_empty() {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            *src = &src[1..];
            Ok(MsgpackToken::Ext { type_id, len })
        }
        0xc9 => {
            let len = read_be!(src, 4, u32) as usize;
            if src.is_empty() {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            *src = &src[1..];
            Ok(MsgpackToken::Ext { type_id, len })
        }
        // float32 / float64
        0xca => {
            let bits = read_be!(src, 4, u32);
            Ok(MsgpackToken::Float32(f32::from_bits(bits)))
        }
        0xcb => {
            let bits = read_be!(src, 8, u64);
            Ok(MsgpackToken::Float64(f64::from_bits(bits)))
        }
        // uint8 / uint16 / uint32 / uint64
        0xcc => Ok(MsgpackToken::UInt(read_be!(src, 1, u8) as u64)),
        0xcd => Ok(MsgpackToken::UInt(read_be!(src, 2, u16) as u64)),
        0xce => Ok(MsgpackToken::UInt(read_be!(src, 4, u32) as u64)),
        0xcf => Ok(MsgpackToken::UInt(read_be!(src, 8, u64))),
        // int8 / int16 / int32 / int64
        0xd0 => Ok(MsgpackToken::Int(read_be!(src, 1, i8) as i64)),
        0xd1 => Ok(MsgpackToken::Int(read_be!(src, 2, i16) as i64)),
        0xd2 => Ok(MsgpackToken::Int(read_be!(src, 4, i32) as i64)),
        0xd3 => Ok(MsgpackToken::Int(read_be!(src, 8, i64))),
        // fixext1/2/4/8/16 — type byte + N data bytes, all embedded in token
        0xd4 => {
            if src.len() < 2 {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            let mut data = [0u8; 16];
            data[0] = src[1];
            *src = &src[2..];
            Ok(MsgpackToken::FixExt {
                type_id,
                data,
                len: 1,
            })
        }
        0xd5 => {
            if src.len() < 3 {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            let mut data = [0u8; 16];
            data[..2].copy_from_slice(&src[1..3]);
            *src = &src[3..];
            Ok(MsgpackToken::FixExt {
                type_id,
                data,
                len: 2,
            })
        }
        0xd6 => {
            if src.len() < 5 {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            let mut data = [0u8; 16];
            data[..4].copy_from_slice(&src[1..5]);
            *src = &src[5..];
            Ok(MsgpackToken::FixExt {
                type_id,
                data,
                len: 4,
            })
        }
        0xd7 => {
            if src.len() < 9 {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            let mut data = [0u8; 16];
            data[..8].copy_from_slice(&src[1..9]);
            *src = &src[9..];
            Ok(MsgpackToken::FixExt {
                type_id,
                data,
                len: 8,
            })
        }
        0xd8 => {
            if src.len() < 17 {
                return Err(MsgpackError::UnexpectedEnd);
            }
            let type_id = src[0] as i8;
            let mut data = [0u8; 16];
            data.copy_from_slice(&src[1..17]);
            *src = &src[17..];
            Ok(MsgpackToken::FixExt {
                type_id,
                data,
                len: 16,
            })
        }
        // str8 / str16 / str32
        0xd9 => {
            let len = read_be!(src, 1, u8) as usize;
            Ok(MsgpackToken::Str(len))
        }
        0xda => {
            let len = read_be!(src, 2, u16) as usize;
            Ok(MsgpackToken::Str(len))
        }
        0xdb => {
            let len = read_be!(src, 4, u32) as usize;
            Ok(MsgpackToken::Str(len))
        }
        // array16 / array32
        0xdc => {
            let count = read_be!(src, 2, u16) as usize;
            Ok(MsgpackToken::Array(count))
        }
        0xdd => {
            let count = read_be!(src, 4, u32) as usize;
            Ok(MsgpackToken::Array(count))
        }
        // map16 / map32
        0xde => {
            let count = read_be!(src, 2, u16) as usize;
            Ok(MsgpackToken::Map(count))
        }
        0xdf => {
            let count = read_be!(src, 4, u32) as usize;
            Ok(MsgpackToken::Map(count))
        }
        // negative fixint 0xe0–0xff
        0xe0..=0xff => Ok(MsgpackToken::Int(byte as i8 as i64)),
    }
}

/// Skip one complete msgpack value given its leading token.
///
/// For scalar tokens (nil, bool, int, float) nothing further is consumed —
/// the header bytes were already read by `next_token`.  For `Str`/`Bin`,
/// `len` payload bytes are skipped.  For `Array`/`Map`, elements/pairs are
/// recursively skipped.
pub fn skip_value(src: &mut &[u8], tok: MsgpackToken) -> Result<(), MsgpackError> {
    match tok {
        MsgpackToken::Nil
        | MsgpackToken::Bool(_)
        | MsgpackToken::UInt(_)
        | MsgpackToken::Int(_)
        | MsgpackToken::Float32(_)
        | MsgpackToken::Float64(_)
        | MsgpackToken::FixExt { .. } => Ok(()),
        MsgpackToken::Ext { len, .. } => {
            if src.len() < len {
                return Err(MsgpackError::UnexpectedEnd);
            }
            *src = &src[len..];
            Ok(())
        }
        MsgpackToken::Str(len) | MsgpackToken::Bin(len) => {
            if src.len() < len {
                return Err(MsgpackError::UnexpectedEnd);
            }
            *src = &src[len..];
            Ok(())
        }
        MsgpackToken::Array(count) => {
            for _ in 0..count {
                let t = next_token(src)?;
                skip_value(src, t)?;
            }
            Ok(())
        }
        MsgpackToken::Map(count) => {
            for _ in 0..count {
                let kt = next_token(src)?;
                skip_value(src, kt)?;
                let vt = next_token(src)?;
                skip_value(src, vt)?;
            }
            Ok(())
        }
    }
}
