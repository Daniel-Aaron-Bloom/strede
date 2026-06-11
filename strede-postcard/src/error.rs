use core::fmt;
use strede::DeserializeError;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PostcardError {
    UnexpectedEnd,
    InvalidUtf8,
    ExpectedEnd,
    DuplicateField(&'static str),
    /// skip() is not supported — postcard is schema-driven; field positions are
    /// determined by the type, not the wire data, so skipping a value of unknown
    /// type is impossible. allow_unknown_fields and flatten are incompatible with postcard.
    CannotSkip,
}

impl fmt::Display for PostcardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8"),
            Self::ExpectedEnd => write!(f, "trailing bytes after top-level value"),
            Self::DuplicateField(name) => write!(f, "duplicate field `{name}`"),
            Self::CannotSkip => write!(f, "cannot skip a value: postcard is schema-driven"),
        }
    }
}

impl core::error::Error for PostcardError {}

impl DeserializeError for PostcardError {
    fn duplicate_field(field: &'static str) -> Self {
        Self::DuplicateField(field)
    }
}
