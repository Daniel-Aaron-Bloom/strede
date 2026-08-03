use core::fmt;
use strede::DeserializeError;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BincodeError {
    UnexpectedEnd,
    InvalidUtf8,
    ExpectedEnd,
    DuplicateField(&'static str),
    /// A varint prefix byte was `255`, which bincode never emits under any
    /// configuration — always a corrupt stream, never a type mismatch.
    InvalidVarint,
    /// A varint's tail-announcing prefix (`251`/`252`/`253`/`254`) was wider
    /// than the value it decoded to could ever need — e.g. a length or enum
    /// discriminant encoded via the 16-byte tail when the value fits in a
    /// single byte. Real bincode2 rejects this as `InvalidIntegerType`
    /// regardless of what the tail bytes decode to; raised only where the
    /// caller has no "try another type" fallback (`decode_len`,
    /// `decode_discriminant`) — the same violation surfaces as a probe miss
    /// for an ordinary typed numeric decode (`u16`/`u32`/`u64`/`i16`/`i32`/`i64`).
    NonCanonicalVarint,
    /// skip() is not supported — bincode is schema-driven; field positions are
    /// determined by the type, not the wire data, so skipping a value of unknown
    /// type is impossible. allow_unknown_fields and flatten are incompatible with bincode.
    CannotSkip,
}

impl fmt::Display for BincodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8"),
            Self::ExpectedEnd => write!(f, "trailing bytes after top-level value"),
            Self::DuplicateField(name) => write!(f, "duplicate field `{name}`"),
            Self::InvalidVarint => write!(f, "invalid varint prefix byte"),
            Self::NonCanonicalVarint => write!(f, "non-canonical varint: prefix wider than the decoded value needs"),
            Self::CannotSkip => write!(f, "cannot skip a value: bincode is schema-driven"),
        }
    }
}

impl core::error::Error for BincodeError {}

impl DeserializeError for BincodeError {
    fn duplicate_field(field: &'static str) -> Self {
        Self::DuplicateField(field)
    }
}
