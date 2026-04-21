use core::fmt;
use strede::DeserializeError;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum JsonError {
    UnsupportedSkip,
    UnexpectedEnd,
    UnexpectedByte { byte: u8 },
    InvalidNumber,
    InvalidEscape,
    InvalidUtf8,
    ExpectedEnd,
    DuplicateField(&'static str),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSkip => write!(f, "unsupported skip"),
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::UnexpectedByte { byte } => {
                write!(f, "unexpected byte {byte:#x}")
            }
            Self::InvalidNumber => write!(f, "invalid number"),
            Self::InvalidEscape => write!(f, "invalid escape sequence"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8"),
            Self::ExpectedEnd => write!(f, "trailing garbage after top-level value"),
            Self::DuplicateField(name) => write!(f, "duplicate field `{name}`"),
        }
    }
}

impl core::error::Error for JsonError {}

impl DeserializeError for JsonError {
    fn duplicate_field(field: &'static str) -> Self {
        Self::DuplicateField(field)
    }
}
