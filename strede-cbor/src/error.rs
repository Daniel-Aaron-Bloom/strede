use core::fmt;
use strede::DeserializeError;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CborError {
    UnexpectedEnd,
    UnexpectedByte {
        byte: u8,
    },
    InvalidUtf8,
    ExpectedEnd,
    DuplicateField(&'static str),
    SkipDepthExceeded,
    /// Break code (0xff) encountered outside an indefinite-length context.
    InvalidBreak,
}

impl fmt::Display for CborError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::UnexpectedByte { byte } => write!(f, "unexpected byte {byte:#x}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8"),
            Self::ExpectedEnd => write!(f, "trailing garbage after top-level value"),
            Self::DuplicateField(name) => write!(f, "duplicate field `{name}`"),
            Self::SkipDepthExceeded => write!(f, "skip depth limit exceeded"),
            Self::InvalidBreak => write!(f, "unexpected break code"),
        }
    }
}

impl core::error::Error for CborError {}

impl DeserializeError for CborError {
    fn duplicate_field(field: &'static str) -> Self {
        Self::DuplicateField(field)
    }
}
