use core::fmt;
use strede::DeserializeError;

#[derive(Debug, PartialEq, Eq)]
pub enum JsonError {
    UnexpectedEnd,
    UnexpectedByte { byte: u8 },
    InvalidNumber,
    InvalidEscape,
    InvalidUtf8,
    TrailingGarbage,
    DuplicateField(&'static str),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::UnexpectedByte { byte } => {
                write!(f, "unexpected byte {byte:#x}")
            }
            Self::InvalidNumber => write!(f, "invalid number"),
            Self::InvalidEscape => write!(f, "invalid escape sequence"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8"),
            Self::TrailingGarbage => write!(f, "trailing garbage after top-level value"),
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
