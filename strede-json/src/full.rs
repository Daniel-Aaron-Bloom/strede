//! In-memory borrow-family JSON deserializer.
//!
//! Implements [`strede::Deserializer`] over a borrowed `&[u8]` source.
//! Entry probe futures resolve immediately - `Ok(Probe::Hit(...))` when the
//! token type matches, `Ok(Probe::Miss)` when it does not.  `Pending` means
//! only "no data available yet" (never reached in in-memory use).
//!
//! # Zero-copy vs. chunked strings
//!
//! [`strede::Entry::deserialize_str`] returns `Ok(Probe::Miss)` for any string that
//! contains escape sequences, because it can only hand back a zero-copy
//! `&'de str` pointing directly into the source bytes.  Use
//! [`strede::Entry::deserialize_str_chunks`] (or race both with `select_probe!`) for
//! strings that may contain `\n`, `\"`, `\uXXXX`, etc.
//!
//! This makes `Cow<str>` / `Cow<[u8]>` callers easy to write: race
//! `deserialize_str` against `deserialize_str_chunks`; the former hits only
//! when the borrow is free, the latter handles the rest via owned allocation.

use core::mem;

use crate::{
    JsonError,
    token::{SimpleToken, StrChunk, Token, Tokenizer},
};

use strede::{
    BytesAccess, Chunk, Deserialize, Deserializer, Entry, MapAccess, MapArmStack, MapKeyClaim,
    MapKeyProbe, MapValueClaim, MapValueProbe, Probe, SeqAccess, SeqEntry, StrAccess, hit,
    utils::repeat,
};

// ---------------------------------------------------------------------------
// JsonClaim - unified proof-of-consumption type
// ---------------------------------------------------------------------------

/// Carries the post-consumption tokenizer and source state.
/// Returned from every probe method and threaded back to [`Deserializer::entry`]
/// (or [`MapAccess::iterate`] / [`SeqAccess::next`]) to advance the stream.
pub struct JsonClaim<'de> {
    tokenizer: Tokenizer,
    src: &'de [u8],
}

// ---------------------------------------------------------------------------
// JsonDeserializer
// ---------------------------------------------------------------------------

pub struct JsonDeserializer<'de> {
    tokenizer: Tokenizer,
    src: &'de [u8],
}

impl<'de> JsonDeserializer<'de> {
    #[inline(always)]
    pub fn new(src: &'de [u8]) -> Self {
        Self {
            tokenizer: Tokenizer::new(),
            src,
        }
    }

    /// Read the next dispatch token.
    ///
    /// `NoTokens` from the tokenizer means the in-memory buffer is exhausted -
    /// for this deserializer that always means the input is truncated.
    fn next_dispatch(&mut self) -> Result<Token, JsonError> {
        while !self.src.is_empty() {
            let old = mem::replace(&mut self.tokenizer, Tokenizer::new());
            match old.next_token(&mut self.src) {
                Ok(Token::Simple(SimpleToken::PartialLiteral, new_tok)) => {
                    self.tokenizer = new_tok; // resume on next iteration
                }
                Ok(Token::NoTokens(new_tok)) => self.tokenizer = new_tok,
                Ok(t) => return Ok(t),
                Err(e) => return Err(e),
            }
        }
        Err(JsonError::UnexpectedEnd)
    }
}

// ---------------------------------------------------------------------------
// JsonSubDeserializer — internal sub-deserializer (no trailing-garbage check)
// ---------------------------------------------------------------------------

/// Sub-deserializer created internally for map keys, map values, seq elements,
/// `deserialize_option`, and `deserialize_value`. Has a pre-loaded token and
/// skips the trailing-garbage check performed by [`JsonDeserializer`].
struct JsonSubDeserializer<'de> {
    src: &'de [u8],
    pending_tok: Token,
}

impl<'de> JsonSubDeserializer<'de> {
    #[inline(always)]
    fn new(src: &'de [u8], tok: Token) -> Self {
        Self {
            src,
            pending_tok: tok,
        }
    }
}

#[inline(always)]
async fn json_root_next<'de, const N: usize, F, Fut, R>(
    mut de: JsonDeserializer<'de>,
    mut f: F,
) -> Result<Probe<(JsonClaim<'de>, R)>, JsonError>
where
    F: FnMut([JsonEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(JsonClaim<'de>, R)>, JsonError>>,
{
    let token = de.next_dispatch()?;
    let entry = JsonEntry {
        token: token.clone(),
        src: de.src,
    };
    match f(repeat(entry, |e| e.clone())).await? {
        Probe::Hit((claim, r)) => {
            // Root deserializer: reject trailing non-whitespace garbage.
            let mut rest = claim.src;
            while !rest.is_empty() && matches!(rest[0], b' ' | b'\t' | b'\n' | b'\r') {
                rest = &rest[1..];
            }
            if !rest.is_empty() {
                return Err(JsonError::ExpectedEnd);
            }
            Ok(Probe::Hit((claim, r)))
        }
        Probe::Miss => Ok(Probe::Miss),
    }
}

impl<'de> Deserializer<'de> for JsonDeserializer<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;
    type EntryClaim = JsonClaim<'de>;
    type Entry = JsonEntry<'de>;

    #[inline(always)]
    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        json_root_next(self, f).await
    }
}

impl<'de> Deserializer<'de> for JsonSubDeserializer<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;
    type EntryClaim = JsonClaim<'de>;
    type Entry = JsonEntry<'de>;

    #[inline(always)]
    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = JsonEntry {
            token: self.pending_tok,
            src: self.src,
        };
        f(repeat(entry, |e| e.clone())).await
    }
}

// ---------------------------------------------------------------------------
// JsonEntry
// ---------------------------------------------------------------------------

pub struct JsonEntry<'de> {
    token: Token,
    src: &'de [u8],
}

impl<'de> JsonEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            token: self.token.clone(),
            src: self.src,
        }
    }
}

/// Skip one complete JSON value given its leading token.
/// Returns the tokenizer state positioned after the skipped value.
fn skip_value(src: &mut &[u8], tok: Token) -> Result<Tokenizer, JsonError> {
    match tok {
        Token::Simple(SimpleToken::Null | SimpleToken::Bool(_), tok) => Ok(tok),
        Token::Number(mut access) => {
            while access.next_chunk(src)?.is_some() {}
            Ok(Tokenizer::new())
        }
        Token::Str(mut access) => {
            while access.next_chunk(src)?.is_some() {}
            Ok(Tokenizer::new())
        }
        Token::Simple(SimpleToken::ObjectStart, mut tok) => {
            let mut first = true;
            loop {
                // Expect comma (after first pair) or closing brace.
                if !first {
                    match tok.next_token(src)? {
                        Token::Simple(SimpleToken::Comma, next) => tok = next,
                        Token::Simple(SimpleToken::ObjectEnd, next) => return Ok(next),
                        _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
                    }
                }
                first = false;
                // Read key or closing brace (for empty object on first iteration).
                let key_tok = match tok.next_token(src)? {
                    Token::Simple(SimpleToken::ObjectEnd, next) => return Ok(next),
                    t => t,
                };
                // Skip the key (must be a string).
                tok = skip_value(src, key_tok)?;
                // Consume colon.
                match tok.next_token(src)? {
                    Token::Simple(SimpleToken::Colon, next) => tok = next,
                    _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
                }
                // Read and skip the value.
                let val_tok = tok.next_token(src)?;
                tok = skip_value(src, val_tok)?;
            }
        }
        Token::Simple(SimpleToken::ArrayStart, mut tok) => {
            let mut first = true;
            loop {
                if !first {
                    match tok.next_token(src)? {
                        Token::Simple(SimpleToken::Comma, next) => tok = next,
                        Token::Simple(SimpleToken::ArrayEnd, next) => return Ok(next),
                        _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
                    }
                }
                first = false;
                let elem_tok = match tok.next_token(src)? {
                    Token::Simple(SimpleToken::ArrayEnd, next) => return Ok(next),
                    t => t,
                };
                tok = skip_value(src, elem_tok)?;
            }
        }
        _ => Err(JsonError::UnexpectedByte { byte: 0 }),
    }
}

impl<'de> Entry<'de> for JsonEntry<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;
    type StrChunks = JsonStrAccess<'de>;
    type BytesChunks = JsonBytesAccess<'de>;
    type Map = JsonMapAccess<'de>;
    type Seq = JsonSeqAccess<'de>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    // ---- Scalars -----------------------------------------------------------
    #[inline(always)]
    async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::Bool(b), tok) => Ok(Probe::Hit((
                JsonClaim {
                    tokenizer: tok,
                    src: self.src,
                },
                b,
            ))),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Integers ----------------------------------------------------------
    #[inline(always)]
    async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((claim, n as u8)))
    }
    #[inline(always)]
    async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((claim, n as u16)))
    }
    #[inline(always)]
    async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((claim, n as u32)))
    }
    #[inline(always)]
    async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
        self.parse_num::<u64>().await
    }
    #[inline(always)]
    async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
        self.parse_num::<u128>().await
    }
    #[inline(always)]
    async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((claim, n as i8)))
    }
    #[inline(always)]
    async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((claim, n as i16)))
    }
    #[inline(always)]
    async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((claim, n as i32)))
    }
    #[inline(always)]
    async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
        self.parse_num::<i64>().await
    }
    #[inline(always)]
    async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
        self.parse_num::<i128>().await
    }
    #[inline(always)]
    async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
        self.parse_num::<f32>().await
    }
    #[inline(always)]
    async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
        self.parse_num::<f64>().await
    }

    // ---- Char --------------------------------------------------------------

    #[inline(always)]
    async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
        // V1: deserialize a single-char JSON string
        let (claim, s) = hit!(self.deserialize_str().await);
        if s.len() != 1 {
            return Ok(Probe::Miss);
        }
        Ok(Probe::Hit((claim, s.chars().next().unwrap())))
    }

    // ---- Strings -----------------------------------------------------------

    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
        match self.token {
            Token::Str(mut access) => {
                // Snapshot src before reading so we can return Miss without consuming
                // the stream if the string contains escape sequences.  Callers can race
                // deserialize_str against deserialize_str_chunks via select_probe!:
                // deserialize_str hits only when the entire value is a single zero-copy
                // slice (no escapes); deserialize_str_chunks handles the general case.
                // This makes Cow<str> / Cow<[u8]> callers free to borrow when possible
                // and fall back to an owned allocation only when actually needed.
                let mut src = self.src;
                let s = match access.next_chunk(&mut src)? {
                    Some(StrChunk::Slice(s)) => {
                        // Consume the closing `"` (next chunk must be None).
                        if access.next_chunk(&mut src)?.is_some() {
                            // More chunks means there were escape sequences; Miss so that
                            // a concurrent deserialize_str_chunks arm can take over.
                            return Ok(Probe::Miss);
                        }
                        s
                    }
                    // First chunk is already an escaped char - not zero-copy; Miss.
                    Some(StrChunk::Char(_)) => return Ok(Probe::Miss),
                    None => "",
                };
                Ok(Probe::Hit((
                    JsonClaim {
                        tokenizer: Tokenizer::new(),
                        src,
                    },
                    s,
                )))
            }
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.token {
            Token::Str(access) => Ok(Probe::Hit(JsonStrAccess {
                access,
                src: self.src,
                char_buf: [0; 4],
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Bytes -------------------------------------------------------------
    #[inline(always)]
    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
        let (claim, s) = hit!(self.deserialize_str().await);
        Ok(Probe::Hit((claim, s.as_bytes())))
    }

    #[inline(always)]
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.token {
            Token::Str(access) => Ok(Probe::Hit(JsonBytesAccess {
                access,
                src: self.src,
                char_buf: [0; 4],
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Map ---------------------------------------------------------------
    #[inline(always)]
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::ObjectStart, tok) => Ok(Probe::Hit(JsonMapAccess {
                tokenizer: tok,
                src: self.src,
                first: true,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Seq ---------------------------------------------------------------
    #[inline(always)]
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::ArrayStart, tok) => Ok(Probe::Hit(JsonSeqAccess {
                tokenizer: tok,
                src: self.src,
                first: true,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Option ------------------------------------------------------------
    #[inline(always)]
    async fn deserialize_option<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit((
                JsonClaim {
                    tokenizer: tok,
                    src: self.src,
                },
                None,
            ))),
            other => {
                let sub = JsonSubDeserializer::new(self.src, other);
                let (claim, v) = hit!(T::deserialize(sub, extra).await);
                Ok(Probe::Hit((claim, Some(v))))
            }
        }
    }

    // ---- Null ---------------------------------------------------------------
    #[inline(always)]
    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit(JsonClaim {
                tokenizer: tok,
                src: self.src,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Value (delegate to T::deserialize) ---------------------------------
    #[inline(always)]
    async fn deserialize_value<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        let sub = JsonSubDeserializer::new(self.src, self.token);
        let (claim, v) = hit!(T::deserialize(sub, extra).await);
        Ok(Probe::Hit((claim, v)))
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut src = self.src;
        let tok = skip_value(&mut src, self.token)?;
        Ok(JsonClaim {
            tokenizer: tok,
            src,
        })
    }
}

// Number parsing helper - not a trait method so it can be called generically.
impl<'de> JsonEntry<'de> {
    async fn parse_num<T: ParseNum>(self) -> Result<Probe<(JsonClaim<'de>, T)>, JsonError> {
        match self.token {
            Token::Number(mut access) => {
                let mut src = self.src;
                let s = match access.next_chunk(&mut src) {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => return Err(JsonError::InvalidNumber),
                    Err(e) => return Err(e),
                };
                let value = T::parse(s)?;
                Ok(Probe::Hit((
                    JsonClaim {
                        tokenizer: Tokenizer::new(),
                        src,
                    },
                    value,
                )))
            }
            _ => Ok(Probe::Miss),
        }
    }
}

trait ParseNum: Sized {
    fn parse(s: &str) -> Result<Self, JsonError>;
}

macro_rules! impl_parse_num {
    ($($t:ty),*) => {
        $(impl ParseNum for $t {
            fn parse(s: &str) -> Result<Self, JsonError> {
                s.parse().map_err(|_| JsonError::InvalidNumber)
            }
        })*
    };
}
impl_parse_num!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

// ---------------------------------------------------------------------------
// JsonStrAccess / JsonBytesAccess
// ---------------------------------------------------------------------------

pub struct JsonStrAccess<'de> {
    access: crate::token::StrAccess,
    src: &'de [u8],
    char_buf: [u8; 4],
}

impl<'de> StrAccess for JsonStrAccess<'de> {
    type Claim = JsonClaim<'de>;
    type Error = JsonError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            access: self.access,
            src: self.src,
            char_buf: self.char_buf,
        }
    }

    async fn next_str<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.access.next_chunk(&mut self.src) {
            Ok(Some(StrChunk::Slice(s))) => Ok(Chunk::Data((self, f(s)))),
            Ok(Some(StrChunk::Char(c))) => {
                let r = f(c.encode_utf8(&mut self.char_buf));
                Ok(Chunk::Data((self, r)))
            }
            Ok(None) => Ok(Chunk::Done(JsonClaim {
                tokenizer: Tokenizer::new(),
                src: self.src,
            })),
            Err(e) => Err(e),
        }
    }
}

pub struct JsonBytesAccess<'de> {
    access: crate::token::StrAccess,
    src: &'de [u8],
    char_buf: [u8; 4],
}

impl<'de> BytesAccess for JsonBytesAccess<'de> {
    type Claim = JsonClaim<'de>;
    type Error = JsonError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            access: self.access,
            src: self.src,
            char_buf: self.char_buf,
        }
    }

    async fn next_bytes<R>(
        mut self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.access.next_chunk(&mut self.src) {
            Ok(Some(StrChunk::Slice(s))) => Ok(Chunk::Data((self, f(s.as_bytes())))),
            Ok(Some(StrChunk::Char(c))) => {
                let r = f(c.encode_utf8(&mut self.char_buf).as_bytes());
                Ok(Chunk::Data((self, r)))
            }
            Ok(None) => Ok(Chunk::Done(JsonClaim {
                tokenizer: Tokenizer::new(),
                src: self.src,
            })),
            Err(e) => Err(e),
        }
    }
}

// ---------------------------------------------------------------------------
// JsonMapAccess / JsonMapKeyProbe / JsonMapKeyClaim / JsonMapValueProbe
// ---------------------------------------------------------------------------

pub struct JsonMapAccess<'de> {
    tokenizer: Tokenizer,
    src: &'de [u8],
    first: bool,
}

/// Key probe: holds the start-of-key token and remaining source bytes.
pub struct JsonMapKeyProbe<'de> {
    src: &'de [u8],
    key_tok: Token,
}

impl<'de> JsonMapKeyProbe<'de> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            key_tok: self.key_tok.clone(),
        }
    }
}

impl<'de> MapKeyProbe<'de> for JsonMapKeyProbe<'de> {
    type Error = JsonError;
    type KeyClaim = JsonMapKeyClaim<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn deserialize_key<K: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error> {
        let key_deser = JsonSubDeserializer::new(self.src, self.key_tok);
        let (key_claim, k) = match K::deserialize(key_deser, extra).await? {
            Probe::Hit(v) => v,
            Probe::Miss => return Ok(Probe::Miss),
        };
        // key_claim positions us right after the key string.
        Ok(Probe::Hit((
            JsonMapKeyClaim {
                tokenizer: key_claim.tokenizer,
                src: key_claim.src,
            },
            k,
        )))
    }
}

/// Proof that a key was consumed. Positioned right after the key, before the colon.
pub struct JsonMapKeyClaim<'de> {
    tokenizer: Tokenizer,
    src: &'de [u8],
}

impl<'de> MapKeyClaim<'de> for JsonMapKeyClaim<'de> {
    type Error = JsonError;
    type MapClaim = JsonClaim<'de>;
    type ValueProbe = JsonMapValueProbe<'de>;

    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        // Consume the colon.
        let mut src = self.src;
        let colon_new_tok = match self.tokenizer.next_token(&mut src) {
            Ok(Token::Simple(SimpleToken::Colon, new_tok)) => new_tok,
            Ok(_) => return Err(JsonError::UnexpectedByte { byte: 0 }),
            Err(e) => return Err(e),
        };
        // Read the first token of the value.
        let value_tok = match colon_new_tok.next_token(&mut src) {
            Ok(t) => t,
            Err(e) => return Err(e),
        };
        Ok(JsonMapValueProbe { src, value_tok })
    }
}

/// Value probe: positioned at the start of the value token.
pub struct JsonMapValueProbe<'de> {
    src: &'de [u8],
    value_tok: Token,
}

impl<'de> JsonMapValueProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            value_tok: self.value_tok.clone(),
        }
    }
}

impl<'de> MapValueProbe<'de> for JsonMapValueProbe<'de> {
    type Error = JsonError;
    type MapClaim = JsonClaim<'de>;
    type ValueClaim = JsonMapValueClaim<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    #[inline(always)]
    async fn deserialize_value<V: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error> {
        let value_deser = JsonSubDeserializer::new(self.src, self.value_tok);
        let (claim, v) = hit!(V::deserialize(value_deser, extra).await);
        Ok(Probe::Hit((
            JsonMapValueClaim {
                tokenizer: claim.tokenizer,
                src: claim.src,
            },
            v,
        )))
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let mut src = self.src;
        let tok = skip_value(&mut src, self.value_tok)?;
        Ok(JsonMapValueClaim {
            tokenizer: tok,
            src,
        })
    }
}

/// Proof that a value was consumed. Positioned right after the value.
pub struct JsonMapValueClaim<'de> {
    tokenizer: Tokenizer,
    src: &'de [u8],
}

impl<'de> MapValueClaim<'de> for JsonMapValueClaim<'de> {
    type Error = JsonError;
    type KeyProbe = JsonMapKeyProbe<'de>;
    type MapClaim = JsonClaim<'de>;

    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<strede::NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        let mut src = self.src;
        // Expect comma or closing brace.
        match self.tokenizer.next_token(&mut src) {
            Ok(Token::Simple(SimpleToken::Comma, new_tok)) => {
                // Read the next key start token (or end of map).
                match new_tok.next_token(&mut src) {
                    Ok(Token::Simple(SimpleToken::ObjectEnd, end_tok)) => {
                        Ok(strede::NextKey::Done(JsonClaim {
                            tokenizer: end_tok,
                            src,
                        }))
                    }
                    Ok(key_tok) => Ok(strede::NextKey::Entry(JsonMapKeyProbe { src, key_tok })),
                    Err(e) => Err(e),
                }
            }
            Ok(Token::Simple(SimpleToken::ObjectEnd, new_tok)) => {
                Ok(strede::NextKey::Done(JsonClaim {
                    tokenizer: new_tok,
                    src,
                }))
            }
            Ok(_) => Err(JsonError::UnexpectedByte { byte: 0 }),
            Err(e) => Err(e),
        }
    }
}

impl<'de> MapAccess<'de> for JsonMapAccess<'de> {
    type Error = JsonError;
    type MapClaim = JsonClaim<'de>;
    type KeyProbe = JsonMapKeyProbe<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            src: self.src,
            first: self.first,
        }
    }

    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        mut arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        // Get the first key probe (or end-of-map for empty maps).
        let mut key_probe_opt: Option<JsonMapKeyProbe<'de>> = {
            let mut src = self.src;
            let tok = self.tokenizer;
            match tok.next_token(&mut src) {
                Ok(Token::Simple(SimpleToken::ObjectEnd, new_tok)) => {
                    // Empty map.
                    return Ok(Probe::Hit((
                        JsonClaim {
                            tokenizer: new_tok,
                            src,
                        },
                        arms.take_outputs(),
                    )));
                }
                Ok(key_tok) => Some(JsonMapKeyProbe { src, key_tok }),
                Err(e) => return Err(e),
            }
        };

        loop {
            let key_probe = match key_probe_opt.take() {
                Some(kp) => kp,
                None => unreachable!(),
            };

            match arms.race_keys(key_probe).await? {
                Probe::Miss => return Ok(Probe::Miss),
                Probe::Hit((arm_index, key_claim)) => {
                    let value_probe = key_claim.into_value_probe().await?;
                    match arms.dispatch_value(arm_index, value_probe).await? {
                        Probe::Miss => return Ok(Probe::Miss),
                        Probe::Hit((value_claim, ())) => {
                            match value_claim
                                .next_key(arms.unsatisfied_count(), arms.open_count())
                                .await?
                            {
                                strede::NextKey::Done(map_claim) => {
                                    return Ok(Probe::Hit((map_claim, arms.take_outputs())));
                                }
                                strede::NextKey::Entry(next_kp) => {
                                    key_probe_opt = Some(next_kp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JsonSeqAccess / JsonSeqEntry
// ---------------------------------------------------------------------------

pub struct JsonSeqAccess<'de> {
    tokenizer: Tokenizer,
    src: &'de [u8],
    first: bool,
}

enum SeqNextState<'de> {
    Done(JsonClaim<'de>),
    Elem(JsonSeqAccess<'de>, JsonSeqEntry<'de>),
}

// Separated from the async call site so that `json_seq_next` can be `#[inline(always)]`
// (inlining the `f` call), while keeping the token-advance logic out-of-line.
fn json_seq_advance<'de>(mut seq: JsonSeqAccess<'de>) -> Result<SeqNextState<'de>, JsonError> {
    // After first element, expect comma or closing bracket.
    if !seq.first {
        let old = mem::replace(&mut seq.tokenizer, Tokenizer::new());
        match old.next_token(&mut seq.src) {
            Ok(Token::Simple(SimpleToken::Comma, new_tok)) => {
                seq.tokenizer = new_tok;
            }
            Ok(Token::Simple(SimpleToken::ArrayEnd, new_tok)) => {
                return Ok(SeqNextState::Done(JsonClaim {
                    tokenizer: new_tok,
                    src: seq.src,
                }));
            }
            Ok(_) => return Err(JsonError::UnexpectedByte { byte: 0 }),
            Err(e) => return Err(e),
        }
    }
    seq.first = false;

    // Read element start token (or closing bracket for empty seq).
    let elem_tok = {
        let old = mem::replace(&mut seq.tokenizer, Tokenizer::new());
        match old.next_token(&mut seq.src) {
            Ok(Token::Simple(SimpleToken::ArrayEnd, new_tok)) => {
                return Ok(SeqNextState::Done(JsonClaim {
                    tokenizer: new_tok,
                    src: seq.src,
                }));
            }
            Ok(t) => t,
            Err(e) => return Err(e),
        }
    };

    let se = JsonSeqEntry {
        src: seq.src,
        elem_tok,
    };
    Ok(SeqNextState::Elem(seq, se))
}

// Free function to work around a rustc ICE (triggered by RPITIT lifetime checking
// in `compare_impl_item` for local DefId), by placing the async body here.
// `#[inline(always)]` so the call to `f` is inlined at each call site.
#[inline(always)]
async fn json_seq_next<'de, const N: usize, F, Fut, R>(
    seq: JsonSeqAccess<'de>,
    mut f: F,
) -> Result<Probe<Chunk<(JsonSeqAccess<'de>, R), JsonClaim<'de>>>, JsonError>
where
    F: FnMut([JsonSeqEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(JsonClaim<'de>, R)>, JsonError>>,
{
    let (mut seq, se) = match json_seq_advance(seq)? {
        SeqNextState::Done(claim) => return Ok(Probe::Hit(Chunk::Done(claim))),
        SeqNextState::Elem(seq, se) => (seq, se),
    };
    let (claim, r) = hit!(f(repeat(se, |se| se.clone())).await);
    seq.tokenizer = claim.tokenizer;
    seq.src = claim.src;
    Ok(Probe::Hit(Chunk::Data((seq, r))))
}

impl<'de> SeqAccess<'de> for JsonSeqAccess<'de> {
    type Error = JsonError;
    type SeqClaim = JsonClaim<'de>;
    type ElemClaim = JsonClaim<'de>;

    type Elem = JsonSeqEntry<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            src: self.src,
            first: self.first,
        }
    }

    #[inline(always)]
    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        json_seq_next(self, f).await
    }
}

pub struct JsonSeqEntry<'de> {
    src: &'de [u8],
    elem_tok: Token,
}

impl<'de> JsonSeqEntry<'de> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            elem_tok: self.elem_tok.clone(),
        }
    }
}

impl<'de> SeqEntry<'de> for JsonSeqEntry<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    #[inline(always)]
    async fn get<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        let elem_deser = JsonSubDeserializer::new(self.src, self.elem_tok);
        let (claim, v) = hit!(T::deserialize(elem_deser, extra).await);
        Ok(Probe::Hit((claim, v)))
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut src = self.src;
        let tok = skip_value(&mut src, self.elem_tok)?;
        Ok(JsonClaim {
            tokenizer: tok,
            src,
        })
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::string::String;
    use strede::{
        Deserialize, Deserializer, Match, MatchVals, Probe, SeqAccess, SeqEntry, StrAccess,
        UnwrapOrElse,
    };

    use strede_test_util::block_on;

    // Local Deserialize impls for test-only newtypes.
    struct U32(u32);
    impl<'de> Deserialize<'de> for U32 {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let (c, v) = hit!(e.deserialize_u32().await);
                Ok(Probe::Hit((c, U32(v))))
            })
            .await
        }
    }

    // ---- bool ---------------------------------------------------------------

    #[test]
    fn bool_true() {
        let de = JsonDeserializer::new(b"true");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_bool().await }))
            .unwrap()
            .unwrap();
        assert!(v);
    }

    #[test]
    fn bool_false() {
        let de = JsonDeserializer::new(b"false");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_bool().await }))
            .unwrap()
            .unwrap();
        assert!(!v);
    }

    // ---- integers -----------------------------------------------------------

    #[test]
    fn u32_positive() {
        let de = JsonDeserializer::new(b"42");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_u32().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, 42u32);
    }

    #[test]
    fn i64_negative() {
        let de = JsonDeserializer::new(b"-7");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_i64().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, -7i64);
    }

    #[test]
    fn u128_large() {
        // 2^64 - exceeds u64::MAX, exercises the u128 parse path.
        let de = JsonDeserializer::new(b"18446744073709551616");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_u128().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, u64::MAX as u128 + 1);
    }

    // ---- floats -------------------------------------------------------------

    #[test]
    fn f64_decimal() {
        let de = JsonDeserializer::new(b"3.14");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_f64().await }))
            .unwrap()
            .unwrap();
        assert!((v - 3.14f64).abs() < 1e-10);
    }

    #[test]
    fn f32_half() {
        let de = JsonDeserializer::new(b"1.5");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_f32().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, 1.5f32);
    }

    // ---- str / char ---------------------------------------------------------

    #[test]
    fn str_plain() {
        let de = JsonDeserializer::new(b"\"hello\"");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_str().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, "hello");
    }

    #[test]
    fn str_empty() {
        let de = JsonDeserializer::new(b"\"\"");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_str().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, "");
    }

    #[test]
    fn char_single() {
        let de = JsonDeserializer::new(b"\"A\"");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_char().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, 'A');
    }

    // ---- str_chunks (escape sequences) --------------------------------------

    #[test]
    fn deserialize_str_misses_on_escape_sequences_too() {
        // Strings with escapes return Miss, not an error - src is owned so the
        // stream position can be handed off to a deserialize_str_chunks arm.
        let de = JsonDeserializer::new(b"\"\\n\"");
        let result = block_on(de.entry(|[e]| async { e.deserialize_str().await }));
        assert!(matches!(result, Ok(Probe::Miss)));
    }

    #[test]
    fn str_chunks_escape_newline() {
        // JSON "hello\nworld": tokenizer yields "hello", '\n' (Char), "world".
        let de = JsonDeserializer::new(b"\"hello\\nworld\"");
        let (_, total_len) = block_on(de.entry(|[e]| async {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            let mut len = 0usize;
            loop {
                match chunks.next_str(|chunk| chunk.len()).await? {
                    Chunk::Data((new, n)) => {
                        chunks = new;
                        len += n;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, len))),
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(total_len, 11); // "hello"(5) + '\n'(1) + "world"(5)
    }

    #[test]
    fn str_chunks_unicode_escape() {
        // JSON "\u0041" decodes to 'A' (1 byte in UTF-8).
        let de = JsonDeserializer::new(b"\"\\u0041\"");
        let (_, total_len) = block_on(de.entry(|[e]| async {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            let mut len = 0usize;
            loop {
                match chunks.next_str(|chunk| chunk.len()).await? {
                    Chunk::Data((new, n)) => {
                        chunks = new;
                        len += n;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, len))),
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(total_len, 1);
    }

    // ---- surrogate pairs (str_chunks) ----------------------------------------

    /// Helper: collect all str_chunks into a String.
    fn collect_str_chunks(input: &[u8]) -> String {
        let de = JsonDeserializer::new(input);
        block_on(de.entry(|[e]| async {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            let mut out = String::new();
            loop {
                match chunks.next_str(|chunk| out.push_str(chunk)).await? {
                    Chunk::Data((new, ())) => chunks = new,
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                }
            }
        }))
        .unwrap()
        .unwrap()
        .1
    }

    #[test]
    fn str_chunks_surrogate_pair() {
        // \uD834\uDD1E = U+1D11E = 𝄞 (musical symbol G clef), 4 bytes in UTF-8
        assert_eq!(collect_str_chunks(b"\"\\uD834\\uDD1E\""), "\u{1D11E}");
    }

    #[test]
    fn str_chunks_surrogate_pair_min() {
        // \uD800\uDC00 = U+10000, first supplementary character
        assert_eq!(collect_str_chunks(b"\"\\uD800\\uDC00\""), "\u{10000}");
    }

    #[test]
    fn str_chunks_surrogate_pair_max() {
        // \uDBFF\uDFFF = U+10FFFF, last valid codepoint
        assert_eq!(collect_str_chunks(b"\"\\uDBFF\\uDFFF\""), "\u{10FFFF}");
    }

    #[test]
    fn str_chunks_surrogate_pair_with_text() {
        // Surrogate pair embedded in normal text
        assert_eq!(
            collect_str_chunks(b"\"abc\\uD834\\uDD1Exyz\""),
            "abc\u{1D11E}xyz",
        );
    }

    #[test]
    fn str_chunks_lone_high_surrogate_err() {
        let de = JsonDeserializer::new(b"\"\\uD834\"");
        let result = block_on(de.entry(|[e]| async {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            loop {
                match chunks.next_str(|_| ()).await? {
                    Chunk::Data((new, ())) => chunks = new,
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, ()))),
                }
            }
        }));
        assert!(result.is_err());
    }

    #[test]
    fn str_chunks_lone_low_surrogate_err() {
        let de = JsonDeserializer::new(b"\"\\uDD1E\"");
        let result = block_on(de.entry(|[e]| async {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            loop {
                match chunks.next_str(|_| ()).await? {
                    Chunk::Data((new, ())) => chunks = new,
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, ()))),
                }
            }
        }));
        assert!(result.is_err());
    }

    // ---- seq ----------------------------------------------------------------

    #[test]
    fn seq_sum_of_numbers() {
        let de = JsonDeserializer::new(b"[1, 2, 3]");
        let (_, sum) = block_on(de.entry(|[e]| async {
            let mut seq = match e.deserialize_seq().await? {
                Probe::Hit(s) => s,
                Probe::Miss => panic!("expected seq"),
            };
            let mut sum = 0u32;
            loop {
                match seq
                    .next(|[se]| async {
                        let (claim, v) = hit!(se.get::<U32, ()>(()).await);
                        Ok(Probe::Hit((claim, v.0)))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data((n_seq, v)) => {
                        sum += v;
                        seq = n_seq;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, sum))),
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(sum, 6);
    }

    #[test]
    fn seq_empty() {
        let de = JsonDeserializer::new(b"[]");
        let (_, count) = block_on(de.entry(|[e]| async {
            let mut seq = match e.deserialize_seq().await? {
                Probe::Hit(s) => s,
                Probe::Miss => panic!("expected seq"),
            };
            let mut count = 0usize;
            loop {
                match seq
                    .next(|[se]| async {
                        let (claim, v) = hit!(se.get::<U32, ()>(()).await);
                        Ok(Probe::Hit((claim, v.0)))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data((n_seq, _)) => {
                        count += 1;
                        seq = n_seq;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, count))),
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(count, 0);
    }

    // ---- map ----------------------------------------------------------------

    #[test]
    fn map_single_pair() {
        // Verify a single-entry map round-trips through BTreeMap deserialization.
        let de = JsonDeserializer::new(b"{\"x\": 10}");
        let v = block_on(
            <alloc::collections::BTreeMap<String, u32> as strede::Deserialize<'_>>::deserialize(
                de,
                (),
            ),
        )
        .unwrap()
        .unwrap();
        let (_, map) = v;
        assert_eq!(map.get("x"), Some(&10));
    }

    #[test]
    fn map_empty() {
        let de = JsonDeserializer::new(b"{}");
        let v = block_on(
            <alloc::collections::BTreeMap<String, u32> as strede::Deserialize<'_>>::deserialize(
                de,
                (),
            ),
        )
        .unwrap()
        .unwrap();
        let (_, map) = v;
        assert!(map.is_empty());
    }

    #[test]
    fn map_multiple_pairs() {
        let de = JsonDeserializer::new(b"{\"a\": 1, \"b\": 2, \"c\": 3}");
        let v = block_on(
            <alloc::collections::BTreeMap<String, u32> as strede::Deserialize<'_>>::deserialize(
                de,
                (),
            ),
        )
        .unwrap()
        .unwrap();
        let (_, map) = v;
        assert_eq!(map.values().sum::<u32>(), 6);
    }

    // ---- option -------------------------------------------------------------

    struct Bool(bool);
    impl<'de> Deserialize<'de> for Bool {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let (c, v) = hit!(e.deserialize_bool().await);
                Ok(Probe::Hit((c, Bool(v))))
            })
            .await
        }
    }

    #[test]
    fn option_null_is_none() {
        let de = JsonDeserializer::new(b"null");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_option::<Bool, ()>(()).await }))
            .unwrap()
            .unwrap();
        assert!(v.is_none());
    }

    #[test]
    fn option_bool_is_some() {
        let de = JsonDeserializer::new(b"true");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_option::<Bool, ()>(()).await }))
            .unwrap()
            .unwrap();
        assert!(v.unwrap().0);
    }

    // ---- error handling -----------------------------------------------------

    #[test]
    fn error_truncated_literal() {
        let de = JsonDeserializer::new(b"tru");
        let result = block_on(de.entry(|[e]| async { e.deserialize_bool().await }));
        assert!(matches!(result, Err(JsonError::UnexpectedEnd)));
    }

    #[test]
    fn error_invalid_number() {
        let de = JsonDeserializer::new(b"1.}");
        let result = block_on(de.entry(|[e]| async { e.deserialize_f64().await }));
        assert!(matches!(result, Err(JsonError::InvalidNumber)));
    }

    #[test]
    fn deserialize_str_misses_on_escape_sequences() {
        // deserialize_str only hits for zero-copy slices; strings with escape
        // sequences return Miss so callers can fall through to deserialize_str_chunks.
        let de = JsonDeserializer::new(b"\"\\n\"");
        let result = block_on(de.entry(|[e]| async { e.deserialize_str().await }));
        assert!(matches!(result, Ok(Probe::Miss)));
    }

    // ---- trailing garbage ---------------------------------------------------

    #[test]
    fn trailing_garbage_after_bool() {
        let de = JsonDeserializer::new(b"true garbage");
        let result = block_on(de.entry(|[e]| async { e.deserialize_bool().await }));
        assert!(matches!(result, Err(JsonError::ExpectedEnd)));
    }

    #[test]
    fn trailing_garbage_after_number() {
        let de = JsonDeserializer::new(b"42 extra");
        let result = block_on(de.entry(|[e]| async { e.deserialize_u32().await }));
        assert!(matches!(result, Err(JsonError::ExpectedEnd)));
    }

    #[test]
    fn trailing_whitespace_is_ok() {
        let de = JsonDeserializer::new(b"true   \n");
        let (_, v) = block_on(de.entry(|[e]| async { e.deserialize_bool().await }))
            .unwrap()
            .unwrap();
        assert!(v);
    }

    // ---- derive tests -------------------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct Point {
        x: i64,
        y: i64,
    }

    #[test]
    fn derive_basic() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"y\": -2}");
        assert_eq!(
            block_on(Point::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Point { x: 1, y: -2 })),
        );
    }

    #[test]
    fn derive_fields_out_of_order() {
        let de = JsonDeserializer::new(b"{\"y\": 7, \"x\": 3}");
        assert_eq!(
            block_on(Point::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Point { x: 3, y: 7 })),
        );
    }

    #[test]
    fn derive_duplicate_field() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"x\": 2, \"y\": 3}");
        assert_eq!(
            block_on(Point::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Err(JsonError::DuplicateField("x")),
        );
    }

    #[test]
    fn derive_unknown_field_is_miss() {
        // "z" is an unknown field - derive returns Miss.
        let de = JsonDeserializer::new(b"{\"x\": 1, \"z\": 99, \"y\": 2}");
        assert_eq!(
            block_on(Point::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_missing_field_is_miss() {
        // "y" is missing - derive returns Miss.
        let de = JsonDeserializer::new(b"{\"x\": 5}");
        assert_eq!(
            block_on(Point::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: non-object input → Miss ------------------------------------

    #[test]
    fn derive_non_object_is_miss() {
        let de = JsonDeserializer::new(b"42");
        assert_eq!(
            block_on(Point::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_array_is_miss() {
        let de = JsonDeserializer::new(b"[1, 2]");
        assert_eq!(
            block_on(Point::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: mixed field types ------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct Mixed<'de> {
        name: &'de str,
        count: u32,
        active: bool,
    }

    #[test]
    fn derive_mixed_types() {
        let de = JsonDeserializer::new(br#"{"name": "hello", "count": 7, "active": true}"#);
        assert_eq!(
            block_on(Mixed::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Mixed {
                name: "hello",
                count: 7,
                active: true
            })),
        );
    }

    #[test]
    fn derive_mixed_types_reordered() {
        let de = JsonDeserializer::new(br#"{"active": false, "name": "world", "count": 0}"#);
        assert_eq!(
            block_on(Mixed::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Mixed {
                name: "world",
                count: 0,
                active: false
            })),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct MixedOwned {
        score: f64,
        count: u32,
        active: bool,
    }

    #[test]
    fn derive_mixed_owned_types() {
        let de = JsonDeserializer::new(br#"{"score": 3.14, "count": 7, "active": true}"#);
        assert_eq!(
            block_on(MixedOwned::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedOwned {
                score: 3.14,
                count: 7,
                active: true
            })),
        );
    }

    // ---- derive: single field -----------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct Wrapper {
        value: i64,
    }

    #[test]
    fn derive_single_field() {
        let de = JsonDeserializer::new(br#"{"value": -99}"#);
        assert_eq!(
            block_on(Wrapper::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Wrapper { value: -99 })),
        );
    }

    // ---- derive: nested structs ---------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct Rect {
        origin: Point,
        size: Point,
    }

    #[test]
    fn derive_nested_struct() {
        let de =
            JsonDeserializer::new(br#"{"origin": {"x": 1, "y": 2}, "size": {"x": 10, "y": 20}}"#);
        assert_eq!(
            block_on(Rect::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Rect {
                origin: Point { x: 1, y: 2 },
                size: Point { x: 10, y: 20 },
            })),
        );
    }

    // ---- derive: optional fields --------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct OptFields {
        required: i64,
        maybe: Option<i64>,
    }

    #[test]
    fn derive_option_present() {
        let de = JsonDeserializer::new(br#"{"required": 1, "maybe": 42}"#);
        assert_eq!(
            block_on(OptFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(OptFields {
                required: 1,
                maybe: Some(42)
            })),
        );
    }

    #[test]
    fn derive_option_null() {
        let de = JsonDeserializer::new(br#"{"required": 1, "maybe": null}"#);
        assert_eq!(
            block_on(OptFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(OptFields {
                required: 1,
                maybe: None
            })),
        );
    }

    // ---- derive: generic type parameters -------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct Pair<A, B> {
        first: A,
        second: B,
    }

    #[test]
    fn derive_generic_two_type_params() {
        let de = JsonDeserializer::new(br#"{"first": 10, "second": true}"#);
        assert_eq!(
            block_on(<Pair<i64, bool>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Pair {
                first: 10,
                second: true
            })),
        );
    }

    #[test]
    fn derive_generic_different_instantiation() {
        let de = JsonDeserializer::new(br#"{"first": false, "second": 42}"#);
        assert_eq!(
            block_on(<Pair<bool, u32>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Pair {
                first: false,
                second: 42
            })),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct GenericWrapper<T> {
        inner: T,
    }

    #[test]
    fn derive_generic_single_param() {
        let de = JsonDeserializer::new(br#"{"inner": 99}"#);
        assert_eq!(
            block_on(<GenericWrapper<i64>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(GenericWrapper { inner: 99 })),
        );
    }

    #[test]
    fn derive_generic_nested() {
        // GenericWrapper<Point> - generic struct containing a non-generic derived struct
        let de = JsonDeserializer::new(br#"{"inner": {"x": 5, "y": 6}}"#);
        assert_eq!(
            block_on(<GenericWrapper<Point>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(GenericWrapper {
                inner: Point { x: 5, y: 6 }
            })),
        );
    }

    #[test]
    fn derive_generic_nested_generic() {
        // GenericWrapper<GenericWrapper<bool>> - nested generics
        let de = JsonDeserializer::new(br#"{"inner": {"inner": true}}"#);
        assert_eq!(
            block_on(<GenericWrapper<GenericWrapper<bool>>>::deserialize(de, ()))
                .map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(GenericWrapper {
                inner: GenericWrapper { inner: true }
            })),
        );
    }

    #[test]
    fn derive_generic_with_option() {
        let de = JsonDeserializer::new(br#"{"inner": null}"#);
        assert_eq!(
            block_on(<GenericWrapper<Option<i64>>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(GenericWrapper { inner: None })),
        );

        let de = JsonDeserializer::new(br#"{"inner": 7}"#);
        assert_eq!(
            block_on(<GenericWrapper<Option<i64>>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(GenericWrapper { inner: Some(7) })),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct WithLifetimeAndGeneric<'de, T> {
        label: &'de str,
        value: T,
    }

    #[test]
    fn derive_generic_with_lifetime() {
        let de = JsonDeserializer::new(br#"{"label": "test", "value": 42}"#);
        assert_eq!(
            block_on(<WithLifetimeAndGeneric<i64>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithLifetimeAndGeneric {
                label: "test",
                value: 42
            })),
        );
    }

    // ---- derive: enum - unit-only --------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    #[test]
    fn derive_enum_unit_variant() {
        let de = JsonDeserializer::new(br#""Red""#);
        assert_eq!(
            block_on(Color::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Color::Red)),
        );
    }

    #[test]
    fn derive_enum_unit_variant_other() {
        let de = JsonDeserializer::new(br#""Blue""#);
        assert_eq!(
            block_on(Color::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Color::Blue)),
        );
    }

    #[test]
    fn derive_enum_unit_unknown_variant_is_miss() {
        let de = JsonDeserializer::new(br#""Yellow""#);
        assert_eq!(
            block_on(Color::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_enum_unit_non_string_is_miss() {
        let de = JsonDeserializer::new(b"42");
        assert_eq!(
            block_on(Color::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: enum - mixed (unit + newtype + struct) ----------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Shape {
        Circle,
        Square(Point),
        Rect { origin: Point, size: Point },
    }

    #[test]
    fn derive_enum_mixed_unit_variant() {
        let de = JsonDeserializer::new(br#""Circle""#);
        assert_eq!(
            block_on(Shape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Shape::Circle)),
        );
    }

    #[test]
    fn derive_enum_newtype_variant() {
        let de = JsonDeserializer::new(br#"{"Square": {"x": 1, "y": 2}}"#);
        assert_eq!(
            block_on(Shape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Shape::Square(Point { x: 1, y: 2 }))),
        );
    }

    #[test]
    fn derive_enum_struct_variant() {
        let de = JsonDeserializer::new(
            br#"{"Rect": {"origin": {"x": 0, "y": 0}, "size": {"x": 10, "y": 20}}}"#,
        );
        assert_eq!(
            block_on(Shape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Shape::Rect {
                origin: Point { x: 0, y: 0 },
                size: Point { x: 10, y: 20 },
            })),
        );
    }

    #[test]
    fn derive_enum_unknown_variant_map_is_miss() {
        let de = JsonDeserializer::new(br#"{"Triangle": {"x": 1}}"#);
        assert_eq!(
            block_on(Shape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_enum_empty_map_is_miss() {
        let de = JsonDeserializer::new(br#"{}"#);
        assert_eq!(
            block_on(Shape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: enum - newtype-only (no unit variants) ---------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Value {
        Int(i64),
        Bool(bool),
    }

    #[test]
    fn derive_enum_newtype_only() {
        let de = JsonDeserializer::new(br#"{"Int": 42}"#);
        assert_eq!(
            block_on(Value::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Value::Int(42))),
        );
    }

    #[test]
    fn derive_enum_newtype_only_other() {
        let de = JsonDeserializer::new(br#"{"Bool": true}"#);
        assert_eq!(
            block_on(Value::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Value::Bool(true))),
        );
    }

    // ---- derive: enum - generics --------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Either<A, B> {
        Left(A),
        Right(B),
    }

    #[test]
    fn derive_enum_generic_left() {
        let de = JsonDeserializer::new(br#"{"Left": 42}"#);
        assert_eq!(
            block_on(<Either<i64, bool>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Either::Left(42))),
        );
    }

    #[test]
    fn derive_enum_generic_right() {
        let de = JsonDeserializer::new(br#"{"Right": true}"#);
        assert_eq!(
            block_on(<Either<i64, bool>>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Either::Right(true))),
        );
    }

    // ---- derive: enum - tuple variants ---------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Geom {
        Point(i64, i64),
        Color(u8, u8, u8),
        Single(bool),
        Unit,
    }

    #[test]
    fn derive_enum_tuple_variant_two_fields() {
        let de = JsonDeserializer::new(br#"{"Point": [3, 4]}"#);
        assert_eq!(
            block_on(Geom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Geom::Point(3, 4))),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_three_fields() {
        let de = JsonDeserializer::new(br#"{"Color": [255, 128, 0]}"#);
        assert_eq!(
            block_on(Geom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Geom::Color(255, 128, 0))),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_mixed_with_unit() {
        let de = JsonDeserializer::new(br#""Unit""#);
        assert_eq!(
            block_on(Geom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Geom::Unit)),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_mixed_with_newtype() {
        let de = JsonDeserializer::new(br#"{"Single": true}"#);
        assert_eq!(
            block_on(Geom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Geom::Single(true))),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_wrong_length_is_miss() {
        // Array with too many elements
        let de = JsonDeserializer::new(br#"{"Point": [1, 2, 3]}"#);
        assert_eq!(
            block_on(Geom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: default attribute ---------------------------------------------

    fn default_count() -> i64 {
        99
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct WithDefaults {
        required: i64,
        #[strede(default = "default_count")]
        count: i64,
        #[strede(default)]
        flag: bool,
    }

    #[test]
    fn derive_default_fields_missing() {
        let de = JsonDeserializer::new(br#"{"required": 1}"#);
        assert_eq!(
            block_on(WithDefaults::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithDefaults {
                required: 1,
                count: 99,
                flag: false
            })),
        );
    }

    #[test]
    fn derive_default_fields_present_overrides() {
        let de = JsonDeserializer::new(br#"{"required": 1, "count": 5, "flag": true}"#);
        assert_eq!(
            block_on(WithDefaults::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithDefaults {
                required: 1,
                count: 5,
                flag: true
            })),
        );
    }

    #[test]
    fn derive_default_required_missing_is_miss() {
        let de = JsonDeserializer::new(br#"{"count": 5}"#);
        assert_eq!(
            block_on(WithDefaults::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: default expression attribute -----------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct WithDefaultExpr {
        required: i64,
        #[strede(default = "99i64")]
        count: i64,
        #[strede(default = "String::from(\"hello\")")]
        greeting: String,
    }

    #[test]
    fn derive_default_expr_missing() {
        let de = JsonDeserializer::new(br#"{"required": 1}"#);
        assert_eq!(
            block_on(WithDefaultExpr::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithDefaultExpr {
                required: 1,
                count: 99,
                greeting: String::from("hello"),
            })),
        );
    }

    #[test]
    fn derive_default_expr_present_overrides() {
        let de = JsonDeserializer::new(br#"{"required": 1, "count": 5, "greeting": "world"}"#);
        assert_eq!(
            block_on(WithDefaultExpr::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithDefaultExpr {
                required: 1,
                count: 5,
                greeting: String::from("world"),
            })),
        );
    }

    // ---- derive: deserialize_with attribute ------------------------------------

    /// Custom deserializer: reads an i64 and doubles it.
    async fn double_i64<'de, D: strede::Deserializer<'de>>(
        d: D,
    ) -> Result<Probe<(D::Claim, i64)>, D::Error>
    where
        D::Error: strede::DeserializeError,
    {
        d.entry(|[e]| async {
            let (c, v) = hit!(e.deserialize_i64().await);
            Ok(Probe::Hit((c, v * 2)))
        })
        .await
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct WithCustomDe {
        normal: i64,
        #[strede(deserialize_with = "double_i64")]
        doubled: i64,
    }

    #[test]
    fn derive_deserialize_with() {
        let de = JsonDeserializer::new(br#"{"normal": 5, "doubled": 3}"#);
        assert_eq!(
            block_on(WithCustomDe::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithCustomDe {
                normal: 5,
                doubled: 6
            })),
        );
    }

    // ---- derive: skip_deserializing attribute ----------------------------------

    fn skipped_default() -> i64 {
        42
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct WithSkip {
        required: i64,
        #[strede(skip_deserializing, default = "skipped_default")]
        skipped: i64,
        #[strede(skip_deserializing, default)]
        skipped_trait: bool,
    }

    #[test]
    fn derive_skip_uses_default_when_absent() {
        let de = JsonDeserializer::new(br#"{"required": 1}"#);
        assert_eq!(
            block_on(WithSkip::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithSkip {
                required: 1,
                skipped: 42,
                skipped_trait: false
            })),
        );
    }

    #[test]
    fn derive_skip_ignores_present_value() {
        // Even though "skipped" is in the data, it should be ignored (treated as unknown → Miss)
        let de = JsonDeserializer::new(br#"{"required": 1, "skipped": 99, "skipped_trait": true}"#);
        assert_eq!(
            block_on(WithSkip::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_skip_mixed_with_required() {
        let de = JsonDeserializer::new(br#"{"required": 7}"#);
        assert_eq!(
            block_on(WithSkip::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithSkip {
                required: 7,
                skipped: 42,
                skipped_trait: false
            })),
        );
    }

    // ---- derive: rename attribute ---------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct RenamedFields {
        #[strede(rename = "type")]
        kind: i64,
        #[strede(rename = "value")]
        val: bool,
    }

    #[test]
    fn derive_rename_struct_fields() {
        let de = JsonDeserializer::new(br#"{"type": 42, "value": true}"#);
        assert_eq!(
            block_on(RenamedFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(RenamedFields {
                kind: 42,
                val: true
            })),
        );
    }

    #[test]
    fn derive_rename_original_name_is_miss() {
        let de = JsonDeserializer::new(br#"{"kind": 42, "val": true}"#);
        assert_eq!(
            block_on(RenamedFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum RenamedVariants {
        #[strede(rename = "circle")]
        Circle,
        #[strede(rename = "rect")]
        Rect(i64),
    }

    #[test]
    fn derive_rename_unit_variant() {
        let de = JsonDeserializer::new(br#""circle""#);
        assert_eq!(
            block_on(RenamedVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(RenamedVariants::Circle)),
        );
    }

    #[test]
    fn derive_rename_newtype_variant() {
        let de = JsonDeserializer::new(br#"{"rect": 5}"#);
        assert_eq!(
            block_on(RenamedVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(RenamedVariants::Rect(5))),
        );
    }

    #[test]
    fn derive_rename_original_variant_name_is_miss() {
        let de = JsonDeserializer::new(br#""Circle""#);
        assert_eq!(
            block_on(RenamedVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: alias attribute ------------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct AliasedFields {
        #[strede(alias = "hostname", alias = "server")]
        host: String,
        port: u16,
    }

    #[test]
    fn derive_alias_primary_name() {
        let de = JsonDeserializer::new(br#"{"host": "a.com", "port": 80}"#);
        assert_eq!(
            block_on(AliasedFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AliasedFields {
                host: String::from("a.com"),
                port: 80
            })),
        );
    }

    #[test]
    fn derive_alias_first_alias() {
        let de = JsonDeserializer::new(br#"{"hostname": "b.com", "port": 443}"#);
        assert_eq!(
            block_on(AliasedFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AliasedFields {
                host: String::from("b.com"),
                port: 443
            })),
        );
    }

    #[test]
    fn derive_alias_second_alias() {
        let de = JsonDeserializer::new(br#"{"server": "c.com", "port": 8080}"#);
        assert_eq!(
            block_on(AliasedFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AliasedFields {
                host: String::from("c.com"),
                port: 8080
            })),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct RenamedWithAlias {
        #[strede(rename = "type", alias = "kind")]
        ty: String,
    }

    #[test]
    fn derive_rename_with_alias_primary() {
        let de = JsonDeserializer::new(br#"{"type": "a"}"#);
        assert_eq!(
            block_on(RenamedWithAlias::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(RenamedWithAlias {
                ty: String::from("a")
            })),
        );
    }

    #[test]
    fn derive_rename_with_alias_alias() {
        let de = JsonDeserializer::new(br#"{"kind": "b"}"#);
        assert_eq!(
            block_on(RenamedWithAlias::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(RenamedWithAlias {
                ty: String::from("b")
            })),
        );
    }

    #[test]
    fn derive_rename_with_alias_original_is_miss() {
        let de = JsonDeserializer::new(br#"{"ty": "c"}"#);
        assert_eq!(
            block_on(RenamedWithAlias::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum AliasedVariants {
        #[strede(alias = "pong")]
        Ping,
        #[strede(alias = "payload")]
        Data(i64),
    }

    #[test]
    fn derive_alias_unit_variant_primary() {
        let de = JsonDeserializer::new(br#""Ping""#);
        assert_eq!(
            block_on(AliasedVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AliasedVariants::Ping)),
        );
    }

    #[test]
    fn derive_alias_unit_variant_alias() {
        let de = JsonDeserializer::new(br#""pong""#);
        assert_eq!(
            block_on(AliasedVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AliasedVariants::Ping)),
        );
    }

    #[test]
    fn derive_alias_newtype_variant_primary() {
        let de = JsonDeserializer::new(br#"{"Data": 42}"#);
        assert_eq!(
            block_on(AliasedVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AliasedVariants::Data(42))),
        );
    }

    #[test]
    fn derive_alias_newtype_variant_alias() {
        let de = JsonDeserializer::new(br#"{"payload": 42}"#);
        assert_eq!(
            block_on(AliasedVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AliasedVariants::Data(42))),
        );
    }

    // ---- derive: untagged attribute -------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(untagged)]
    enum Untagged {
        Num(i64),
        Flag(bool),
        Pt(i64, i64),
        Named { x: i64 },
    }

    #[test]
    fn derive_untagged_newtype_first_match() {
        let de = JsonDeserializer::new(b"42");
        assert_eq!(
            block_on(Untagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Untagged::Num(42))),
        );
    }

    #[test]
    fn derive_untagged_newtype_second_match() {
        let de = JsonDeserializer::new(b"true");
        assert_eq!(
            block_on(Untagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Untagged::Flag(true))),
        );
    }

    #[test]
    fn derive_untagged_tuple_variant() {
        let de = JsonDeserializer::new(b"[1, 2]");
        assert_eq!(
            block_on(Untagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Untagged::Pt(1, 2))),
        );
    }

    #[test]
    fn derive_untagged_struct_variant() {
        let de = JsonDeserializer::new(br#"{"x": 7}"#);
        assert_eq!(
            block_on(Untagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Untagged::Named { x: 7 })),
        );
    }

    #[test]
    fn derive_untagged_all_miss() {
        let de = JsonDeserializer::new(br#""hello""#);
        assert_eq!(
            block_on(Untagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: per-variant untagged -----------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum MixedTagged {
        Ping,
        Data(i64),
        #[strede(untagged)]
        Fallback(bool),
    }

    #[test]
    fn derive_mixed_tagged_unit() {
        let de = JsonDeserializer::new(br#""Ping""#);
        assert_eq!(
            block_on(MixedTagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedTagged::Ping)),
        );
    }

    #[test]
    fn derive_mixed_tagged_newtype() {
        let de = JsonDeserializer::new(br#"{"Data": 42}"#);
        assert_eq!(
            block_on(MixedTagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedTagged::Data(42))),
        );
    }

    #[test]
    fn derive_mixed_untagged_fallback() {
        let de = JsonDeserializer::new(b"true");
        assert_eq!(
            block_on(MixedTagged::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedTagged::Fallback(true))),
        );
    }

    // ---- select_probe! ------------------------------------------------------

    #[test]
    fn select_probe_first_arm_hits() {
        let de = JsonDeserializer::new(b"42");
        let result: Result<Probe<(JsonClaim, i64)>, JsonError> = block_on(async {
            de.entry(|[e1, e2]| async {
                strede::select_probe! {
                    e1.deserialize_i64(),
                    async move {
                        let (claim, v) = strede::hit!(e2.deserialize_bool().await);
                        Ok(Probe::Hit((claim, if v { 1i64 } else { 0 })))
                    },
                }
            })
            .await
        });
        assert_eq!(result.map(|p| p.map(|(_, v)| v)), Ok(Probe::Hit(42)));
    }

    #[test]
    fn select_probe_second_arm_hits() {
        let de = JsonDeserializer::new(b"true");
        let result: Result<Probe<(JsonClaim, i64)>, JsonError> = block_on(async {
            de.entry(|[e1, e2]| async {
                strede::select_probe! {
                    e1.deserialize_i64(),
                    async move {
                        let (claim, v) = strede::hit!(e2.deserialize_bool().await);
                        Ok(Probe::Hit((claim, if v { 1i64 } else { 0 })))
                    },
                }
            })
            .await
        });
        assert_eq!(result.map(|p| p.map(|(_, v)| v)), Ok(Probe::Hit(1)));
    }

    #[test]
    fn select_probe_all_miss_default() {
        let de = JsonDeserializer::new(br#""hello""#);
        let result: Result<Probe<(JsonClaim, i64)>, JsonError> = block_on(async {
            de.entry(|[e1, e2]| async {
                strede::select_probe! {
                    e1.deserialize_i64(),
                    async move {
                        let (claim, v) = strede::hit!(e2.deserialize_bool().await);
                        Ok(Probe::Hit((claim, if v { 1i64 } else { 0 })))
                    },
                }
            })
            .await
        });
        assert_eq!(result.map(|p| p.map(|(_, v)| v)), Ok(Probe::Miss));
    }

    #[test]
    fn select_probe_all_miss_with_handler() {
        let de = JsonDeserializer::new(br#""hello""#);
        let result: Result<Probe<(JsonClaim, i64)>, JsonError> = block_on(async {
            de.entry(|[e1, e2]| async {
                strede::select_probe! {
                    e1.deserialize_i64(),
                    async move {
                        let (claim, v) = strede::hit!(e2.deserialize_bool().await);
                        Ok(Probe::Hit((claim, if v { 1i64 } else { 0 })))
                    },
                    @miss => Ok(Probe::Miss),
                }
            })
            .await
        });
        assert_eq!(result.map(|p| p.map(|(_, v)| v)), Ok(Probe::Miss));
    }

    #[test]
    fn select_probe_single_arm() {
        let de = JsonDeserializer::new(b"false");
        let result: Result<Probe<(JsonClaim, bool)>, JsonError> = block_on(async {
            de.entry(|[e]| async {
                strede::select_probe! {
                    e.deserialize_bool(),
                }
            })
            .await
        });
        assert_eq!(result.map(|p| p.map(|(_, v)| v)), Ok(Probe::Hit(false)));
    }

    #[test]
    fn select_probe_three_arms() {
        let de = JsonDeserializer::new(br#""test""#);
        #[derive(Debug, PartialEq)]
        enum Val<'a> {
            Num(i64),
            Bool(bool),
            Str(&'a str),
        }

        let result: Result<Probe<(JsonClaim, Val)>, JsonError> = block_on(async {
            de.entry(|[e1, e2, e3]| async {
                strede::select_probe! {
                    async move {
                        let (claim, v) = strede::hit!(e1.deserialize_i64().await);
                        Ok(Probe::Hit((claim, Val::Num(v))))
                    },
                    async move {
                        let (claim, v) = strede::hit!(e2.deserialize_bool().await);
                        Ok(Probe::Hit((claim, Val::Bool(v))))
                    },
                    async move {
                        let (claim, v) = strede::hit!(e3.deserialize_str().await);
                        Ok(Probe::Hit((claim, Val::Str(v))))
                    },
                }
            })
            .await
        });
        assert_eq!(
            result.map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Val::Str("test")))
        );
    }

    // ==== skip_value tests ====================================================

    #[test]
    fn skip_null() {
        let mut src: &[u8] = b"null, rest";
        let tok = Tokenizer::new().next_token(&mut src).unwrap();
        let tok = skip_value(&mut src, tok).unwrap();
        // Should be positioned right after "null"
        let next = tok.next_token(&mut src).unwrap();
        assert!(matches!(next, Token::Simple(SimpleToken::Comma, _)));
    }

    #[test]
    fn skip_bool() {
        let mut src: &[u8] = b"true, rest";
        let tok = Tokenizer::new().next_token(&mut src).unwrap();
        let tok = skip_value(&mut src, tok).unwrap();
        let next = tok.next_token(&mut src).unwrap();
        assert!(matches!(next, Token::Simple(SimpleToken::Comma, _)));
    }

    #[test]
    fn skip_number() {
        let mut src: &[u8] = b"12345, rest";
        let tok = Tokenizer::new().next_token(&mut src).unwrap();
        let tok = skip_value(&mut src, tok).unwrap();
        let next = tok.next_token(&mut src).unwrap();
        assert!(matches!(next, Token::Simple(SimpleToken::Comma, _)));
    }

    #[test]
    fn skip_string() {
        let mut src: &[u8] = b"\"hello\\nworld\", rest";
        let tok = Tokenizer::new().next_token(&mut src).unwrap();
        let tok = skip_value(&mut src, tok).unwrap();
        let next = tok.next_token(&mut src).unwrap();
        assert!(matches!(next, Token::Simple(SimpleToken::Comma, _)));
    }

    #[test]
    fn skip_empty_object() {
        let mut src: &[u8] = b"{}, rest";
        let tok = Tokenizer::new().next_token(&mut src).unwrap();
        let tok = skip_value(&mut src, tok).unwrap();
        let next = tok.next_token(&mut src).unwrap();
        assert!(matches!(next, Token::Simple(SimpleToken::Comma, _)));
    }

    #[test]
    fn skip_empty_array() {
        let mut src: &[u8] = b"[], rest";
        let tok = Tokenizer::new().next_token(&mut src).unwrap();
        let tok = skip_value(&mut src, tok).unwrap();
        let next = tok.next_token(&mut src).unwrap();
        assert!(matches!(next, Token::Simple(SimpleToken::Comma, _)));
    }

    #[test]
    fn skip_nested_structure() {
        let mut src: &[u8] = b"{\"a\": [1, {\"b\": 2}], \"c\": null}, rest";
        let tok = Tokenizer::new().next_token(&mut src).unwrap();
        let tok = skip_value(&mut src, tok).unwrap();
        let next = tok.next_token(&mut src).unwrap();
        assert!(matches!(next, Token::Simple(SimpleToken::Comma, _)));
    }

    // ==== allow_unknown_fields (borrow family) ================================

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(allow_unknown_fields)]
    struct Lenient {
        x: u32,
        y: u32,
    }

    #[test]
    fn allow_unknown_fields_basic() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"extra\": 99, \"y\": 2}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 1, y: 2 })),
        );
    }

    #[test]
    fn allow_unknown_fields_string_value() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"name\": \"hello\", \"y\": 2}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 1, y: 2 })),
        );
    }

    #[test]
    fn allow_unknown_fields_nested_object_value() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"nested\": {\"a\": [1,2,3]}, \"y\": 2}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 1, y: 2 })),
        );
    }

    #[test]
    fn allow_unknown_fields_null_value() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"gone\": null, \"y\": 2}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 1, y: 2 })),
        );
    }

    #[test]
    fn allow_unknown_fields_array_value() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"list\": [1, \"a\", true], \"y\": 2}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 1, y: 2 })),
        );
    }

    #[test]
    fn allow_unknown_fields_bool_value() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"flag\": false, \"y\": 2}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 1, y: 2 })),
        );
    }

    #[test]
    fn allow_unknown_fields_no_extra() {
        let de = JsonDeserializer::new(b"{\"x\": 10, \"y\": 20}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 10, y: 20 })),
        );
    }

    #[test]
    fn allow_unknown_fields_multiple_unknowns() {
        let de = JsonDeserializer::new(b"{\"a\": 1, \"x\": 5, \"b\": 2, \"y\": 6, \"c\": 3}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(Lenient { x: 5, y: 6 })),
        );
    }

    #[test]
    fn allow_unknown_fields_missing_required_still_misses() {
        // y is missing - even with allow_unknown_fields, required fields must be present.
        let de = JsonDeserializer::new(b"{\"x\": 1, \"extra\": 99}");
        assert_eq!(
            block_on(Lenient::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss)
        );
    }

    #[test]
    fn allow_unknown_fields_duplicate_still_errors() {
        let de = JsonDeserializer::new(b"{\"x\": 1, \"y\": 2, \"x\": 3}");
        assert!(
            block_on(Lenient::deserialize(de, ()))
                .map(|p| p.map(|(_, v)| v))
                .is_err()
        );
    }

    // ==== tuple struct derive =================================================

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct TuplePair(u32, u32);

    #[test]
    fn derive_tuple_struct_basic() {
        let de = JsonDeserializer::new(b"[10, 20]");
        assert_eq!(
            block_on(TuplePair::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TuplePair(10, 20))),
        );
    }

    #[test]
    fn derive_tuple_struct_wrong_length_is_miss() {
        let de = JsonDeserializer::new(b"[1, 2, 3]");
        assert_eq!(
            block_on(TuplePair::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss)
        );
    }

    #[test]
    fn derive_tuple_struct_too_short_is_miss() {
        let de = JsonDeserializer::new(b"[1]");
        assert_eq!(
            block_on(TuplePair::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss)
        );
    }

    #[test]
    fn derive_tuple_struct_non_array_is_miss() {
        let de = JsonDeserializer::new(b"42");
        assert_eq!(
            block_on(TuplePair::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss)
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct TupleTriple(u32, bool, u32);

    #[test]
    fn derive_tuple_struct_mixed_types() {
        let de = JsonDeserializer::new(b"[1, true, 3]");
        assert_eq!(
            block_on(TupleTriple::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TupleTriple(1, true, 3))),
        );
    }

    // ==== transparent derive ==================================================

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(transparent)]
    struct TransparentNamed {
        inner: u32,
    }

    #[test]
    fn derive_transparent_named() {
        let de = JsonDeserializer::new(b"42");
        assert_eq!(
            block_on(TransparentNamed::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TransparentNamed { inner: 42 })),
        );
    }

    #[test]
    fn derive_transparent_named_miss() {
        let de = JsonDeserializer::new(b"\"hello\"");
        assert_eq!(
            block_on(TransparentNamed::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(transparent)]
    struct TransparentNewtype(u32);

    #[test]
    fn derive_transparent_tuple() {
        let de = JsonDeserializer::new(b"99");
        assert_eq!(
            block_on(TransparentNewtype::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TransparentNewtype(99))),
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(transparent)]
    struct TransparentBool(bool);

    #[test]
    fn derive_transparent_bool() {
        let de = JsonDeserializer::new(b"true");
        assert_eq!(
            block_on(TransparentBool::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TransparentBool(true))),
        );
    }

    // ==== unit struct derive ==================================================

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct UnitStruct;

    #[test]
    fn derive_unit_struct_null() {
        let de = JsonDeserializer::new(b"null");
        assert_eq!(
            block_on(UnitStruct::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(UnitStruct)),
        );
    }

    #[test]
    fn derive_unit_struct_non_null_is_miss() {
        let de = JsonDeserializer::new(b"42");
        assert_eq!(
            block_on(UnitStruct::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ==== bound attribute =====================================================

    // A supertrait used to test bound replacement: anything that implements
    // Deserialize also implements MyDeserialize, so the generated body
    // (which calls T::deserialize) still compiles.
    trait MyDeserialize<'de>: strede::Deserialize<'de> {}
    impl<'de, T: strede::Deserialize<'de>> MyDeserialize<'de> for T {}

    // -- container-level bound = "T: Copy" (harmless extra bound) -------------

    // The auto-bound would be `T: Deserialize<'de>`. With `bound = "T: Copy"`,
    // only `T: Copy` appears in the impl - the body still compiles because
    // Copy is only needed at the call site to prove the type is acceptable,
    // not inside the generated deserialization body. Here we instantiate with
    // u32 (Copy + Deserialize) so everything is satisfied.
    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(bound = "T: Copy + strede::Deserialize<'de>")]
    struct BoundedCopy<T> {
        value: T,
    }

    #[test]
    fn derive_container_bound_copy_hit() {
        let de = JsonDeserializer::new(br#"{"value": 7}"#);
        assert_eq!(
            block_on(BoundedCopy::<u32>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(BoundedCopy { value: 7u32 })),
        );
    }

    #[test]
    fn derive_container_bound_copy_miss() {
        let de = JsonDeserializer::new(br#""oops""#);
        assert_eq!(
            block_on(BoundedCopy::<u32>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // -- container-level bound replacing T: Deserialize with MyDeserialize ----

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(bound = "T: MyDeserialize<'de>")]
    struct BoundedMyDeserialize<T> {
        a: T,
        b: u32,
    }

    #[test]
    fn derive_container_bound_custom_trait_hit() {
        let de = JsonDeserializer::new(br#"{"a": true, "b": 9}"#);
        assert_eq!(
            block_on(BoundedMyDeserialize::<bool>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(BoundedMyDeserialize { a: true, b: 9 })),
        );
    }

    #[test]
    fn derive_container_bound_custom_trait_miss() {
        let de = JsonDeserializer::new(br#""nope""#);
        assert_eq!(
            block_on(BoundedMyDeserialize::<bool>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // -- field-level bound: extra Copy on one field, auto-bound on the other --

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct FieldBoundCopy<T, U> {
        #[strede(bound = "T: Copy + strede::Deserialize<'de>")]
        first: T,
        second: U, // auto-bound: U: Deserialize<'de>
    }

    #[test]
    fn derive_field_bound_copy_hit() {
        let de = JsonDeserializer::new(br#"{"first": 3, "second": false}"#);
        assert_eq!(
            block_on(FieldBoundCopy::<u32, bool>::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(FieldBoundCopy {
                first: 3u32,
                second: false
            })),
        );
    }

    // -- field-level bound replacing T: Deserialize with MyDeserialize --------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct FieldBoundMyDeserialize<T> {
        #[strede(bound = "T: MyDeserialize<'de>")]
        inner: T,
        tag: u32,
    }

    #[test]
    fn derive_field_bound_custom_trait_hit() {
        let de = JsonDeserializer::new(br#"{"inner": 1, "tag": 2}"#);
        assert_eq!(
            block_on(FieldBoundMyDeserialize::<u32>::deserialize(de, ()))
                .map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(FieldBoundMyDeserialize {
                inner: 1u32,
                tag: 2
            })),
        );
    }

    // ---- derive: rename_all attribute ------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(rename_all = "camelCase")]
    struct CamelCaseFields {
        first_name: String,
        last_name: String,
        age_years: u32,
    }

    #[test]
    fn derive_rename_all_camel_case_hit() {
        let de = JsonDeserializer::new(
            br#"{"firstName": "Alice", "lastName": "Smith", "ageYears": 30}"#,
        );
        assert_eq!(
            block_on(CamelCaseFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(CamelCaseFields {
                first_name: String::from("Alice"),
                last_name: String::from("Smith"),
                age_years: 30,
            })),
        );
    }

    #[test]
    fn derive_rename_all_original_name_is_miss() {
        let de = JsonDeserializer::new(
            br#"{"first_name": "Alice", "last_name": "Smith", "age_years": 30}"#,
        );
        assert_eq!(
            block_on(CamelCaseFields::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // rename_all with an explicit rename on one field - explicit wins

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(rename_all = "camelCase")]
    struct RenameAllWithExplicit {
        first_name: String,
        #[strede(rename = "custom_key")]
        last_name: String,
    }

    #[test]
    fn derive_rename_all_explicit_rename_wins() {
        let de = JsonDeserializer::new(br#"{"firstName": "Bob", "custom_key": "Jones"}"#);
        assert_eq!(
            block_on(RenameAllWithExplicit::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(RenameAllWithExplicit {
                first_name: String::from("Bob"),
                last_name: String::from("Jones"),
            })),
        );
    }

    // rename_all on enum variants

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(rename_all = "snake_case")]
    enum SnakeCaseVariants {
        MyVariant,
        AnotherOne(i64),
        WithStruct { x: u32, y: u32 },
    }

    #[test]
    fn derive_rename_all_unit_variant() {
        let de = JsonDeserializer::new(br#""my_variant""#);
        assert_eq!(
            block_on(SnakeCaseVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(SnakeCaseVariants::MyVariant)),
        );
    }

    #[test]
    fn derive_rename_all_newtype_variant() {
        let de = JsonDeserializer::new(br#"{"another_one": 7}"#);
        assert_eq!(
            block_on(SnakeCaseVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(SnakeCaseVariants::AnotherOne(7))),
        );
    }

    #[test]
    fn derive_rename_all_struct_variant() {
        let de = JsonDeserializer::new(br#"{"with_struct": {"x": 1, "y": 2}}"#);
        assert_eq!(
            block_on(SnakeCaseVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(SnakeCaseVariants::WithStruct { x: 1, y: 2 })),
        );
    }

    #[test]
    fn derive_rename_all_original_variant_name_is_miss() {
        let de = JsonDeserializer::new(br#""MyVariant""#);
        assert_eq!(
            block_on(SnakeCaseVariants::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: other attribute ------------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum WithOther {
        Known,
        #[strede(other)]
        Unknown,
    }

    #[test]
    fn derive_other_known_variant_hits() {
        let de = JsonDeserializer::new(br#""Known""#);
        assert_eq!(
            block_on(WithOther::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithOther::Known)),
        );
    }

    #[test]
    fn derive_other_unknown_string_returns_other() {
        let de = JsonDeserializer::new(br#""anything_else""#);
        assert_eq!(
            block_on(WithOther::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithOther::Unknown)),
        );
    }

    // other in a mixed enum (unit + non-unit variants)

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum MixedWithOther {
        Unit,
        Pair(i64),
        #[strede(other)]
        Unknown,
    }

    #[test]
    fn derive_other_mixed_known_unit() {
        let de = JsonDeserializer::new(br#""Unit""#);
        assert_eq!(
            block_on(MixedWithOther::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedWithOther::Unit)),
        );
    }

    #[test]
    fn derive_other_mixed_known_nonunit() {
        let de = JsonDeserializer::new(br#"{"Pair": 42}"#);
        assert_eq!(
            block_on(MixedWithOther::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedWithOther::Pair(42))),
        );
    }

    #[test]
    fn derive_other_mixed_unknown_string() {
        let de = JsonDeserializer::new(br#""nope""#);
        assert_eq!(
            block_on(MixedWithOther::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedWithOther::Unknown)),
        );
    }

    #[test]
    fn derive_other_mixed_unknown_map_key() {
        let de = JsonDeserializer::new(br#"{"nope": 99}"#);
        assert_eq!(
            block_on(MixedWithOther::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MixedWithOther::Unknown)),
        );
    }

    // ---- derive: from attribute (field level) ---------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct WithFrom {
        name: String,
        /// Deserializes as u32, then widens to u64 via From.
        #[strede(from = "u32")]
        count: u64,
    }

    #[test]
    fn derive_field_from_converts() {
        let de = JsonDeserializer::new(br#"{"name": "hi", "count": 7}"#);
        assert_eq!(
            block_on(WithFrom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithFrom {
                name: String::from("hi"),
                count: 7
            })),
        );
    }

    // ---- derive: try_from attribute (field level) ----------------------------

    /// Newtype that only accepts positive integers.
    #[derive(Debug, PartialEq)]
    struct Positive(i64);

    impl TryFrom<i64> for Positive {
        type Error = ();
        fn try_from(v: i64) -> Result<Self, ()> {
            if v > 0 { Ok(Positive(v)) } else { Err(()) }
        }
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct WithTryFrom {
        #[strede(try_from = "i64")]
        value: Positive,
    }

    #[test]
    fn derive_field_try_from_hit() {
        let de = JsonDeserializer::new(br#"{"value": 5}"#);
        assert_eq!(
            block_on(WithTryFrom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(WithTryFrom { value: Positive(5) })),
        );
    }

    #[test]
    fn derive_field_try_from_miss_on_conversion_failure() {
        let de = JsonDeserializer::new(br#"{"value": -3}"#);
        assert_eq!(
            block_on(WithTryFrom::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: from attribute (container level) ----------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(from = "f64")]
    struct MetersWrapper(#[allow(dead_code)] f64);

    impl From<f64> for MetersWrapper {
        fn from(v: f64) -> Self {
            MetersWrapper(v)
        }
    }

    #[test]
    fn derive_container_from_converts() {
        let de = JsonDeserializer::new(b"3.14");
        assert_eq!(
            block_on(MetersWrapper::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(MetersWrapper(3.14))),
        );
    }

    // ---- derive: try_from attribute (container level) ------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(try_from = "String")]
    struct NonEmptyStringWrapper(String);

    impl TryFrom<String> for NonEmptyStringWrapper {
        type Error = ();
        fn try_from(s: String) -> Result<Self, ()> {
            if s.is_empty() {
                Err(())
            } else {
                Ok(NonEmptyStringWrapper(s))
            }
        }
    }

    #[test]
    fn derive_container_try_from_hit() {
        let de = JsonDeserializer::new(br#""hello""#);
        assert_eq!(
            block_on(NonEmptyStringWrapper::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(NonEmptyStringWrapper(String::from("hello")))),
        );
    }

    #[test]
    fn derive_container_try_from_miss_on_conversion_failure() {
        let de = JsonDeserializer::new(br#""""#);
        assert_eq!(
            block_on(NonEmptyStringWrapper::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- fork ---------------------------------------------------------------

    #[test]
    fn str_access_fork_reads_same_content() {
        // Fork a StrAccess at the start; both forks must read the same full string.
        let (_, (a, b)) = block_on(JsonDeserializer::new(b"\"hello\\nworld\"").entry(
            |[e]| async {
                let mut chunks = match e.deserialize_str_chunks().await? {
                    Probe::Hit(c) => c,
                    Probe::Miss => panic!("expected str chunks"),
                };
                let mut fork = chunks.fork();

                let mut a = String::new();
                while let Chunk::Data((new, ())) = chunks.next_str(|s| a.push_str(s)).await? {
                    chunks = new
                }

                let mut b = String::new();
                let claim = loop {
                    match fork.next_str(|s| b.push_str(s)).await? {
                        Chunk::Data((new, ())) => fork = new,
                        Chunk::Done(claim) => break claim,
                    }
                };
                Ok(Probe::Hit((claim, (a, b))))
            },
        ))
        .unwrap()
        .unwrap();
        assert_eq!(a, "hello\nworld");
        assert_eq!(b, "hello\nworld");
    }

    #[test]
    fn seq_access_fork_computes_same_sum() {
        // Fork a SeqAccess at the start; both forks must see all elements.
        let (_, (s1, s2)) = block_on(JsonDeserializer::new(b"[1, 2, 3]").entry(|[e]| async {
            let mut seq = match e.deserialize_seq().await? {
                Probe::Hit(s) => s,
                Probe::Miss => panic!("expected seq"),
            };
            let mut fork = seq.fork();

            let mut s1 = 0u32;
            while let Chunk::Data((n_seq, v)) = seq
                .next(|[se]| async {
                    let (claim, v) = hit!(se.get::<U32, ()>(()).await);
                    Ok(Probe::Hit((claim, v.0)))
                })
                .await?
                .unwrap()
            {
                s1 += v;
                seq = n_seq;
            }

            let mut s2 = 0u32;
            let claim = loop {
                match fork
                    .next(|[se]| async {
                        let (claim, v) = hit!(se.get::<U32, ()>(()).await);
                        Ok(Probe::Hit((claim, v.0)))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data((n_seq, v)) => {
                        s2 += v;
                        fork = n_seq;
                    }
                    Chunk::Done(claim) => break claim,
                }
            };
            Ok(Probe::Hit((claim, (s1, s2))))
        }))
        .unwrap()
        .unwrap();
        assert_eq!(s1, 6);
        assert_eq!(s2, 6);
    }

    // ---- Match (borrow family) -----------------------------------------------

    #[test]
    fn match_str_hits_plain() {
        let de = JsonDeserializer::new(b"\"hello\"");
        let v = block_on(<Match as Deserialize<&str>>::deserialize(de, "hello")).unwrap();
        assert!(matches!(v, Probe::Hit((_, Match))));
    }

    #[test]
    fn match_str_misses_wrong_content() {
        let de = JsonDeserializer::new(b"\"hello\"");
        let v = block_on(<Match as Deserialize<&str>>::deserialize(de, "world")).unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    #[test]
    fn match_str_hits_escaped() {
        // "hello\nworld" has an escape sequence - deserialize_str misses,
        // deserialize_str_chunks handles it.
        let de = JsonDeserializer::new(b"\"hello\\nworld\"");
        let v = block_on(<Match as Deserialize<&str>>::deserialize(
            de,
            "hello\nworld",
        ))
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, Match))));
    }

    #[test]
    fn match_str_misses_escaped_wrong_content() {
        let de = JsonDeserializer::new(b"\"hello\\nworld\"");
        let v = block_on(<Match as Deserialize<&str>>::deserialize(
            de,
            "hello\tworld",
        ))
        .unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    #[test]
    fn match_str_misses_wrong_type() {
        let de = JsonDeserializer::new(b"42");
        let v = block_on(<Match as Deserialize<&str>>::deserialize(de, "42")).unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    #[test]
    fn match_bytes_hits() {
        let de = JsonDeserializer::new(b"\"hello\"");
        let v = block_on(<Match as Deserialize<&[u8]>>::deserialize(de, b"hello")).unwrap();
        assert!(matches!(v, Probe::Hit((_, Match))));
    }

    #[test]
    fn match_bytes_misses_wrong_content() {
        let de = JsonDeserializer::new(b"\"hello\"");
        let v = block_on(<Match as Deserialize<&[u8]>>::deserialize(de, b"world")).unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    #[test]
    fn match_bytes_misses_wrong_type() {
        let de = JsonDeserializer::new(b"42");
        let v = block_on(<Match as Deserialize<&[u8]>>::deserialize(de, b"42")).unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    // ---- MatchVals (borrow family) -------------------------------------------

    #[test]
    fn match_vals_str_hits_first() {
        let de = JsonDeserializer::new(b"\"ok\"");
        let v = block_on(
            <MatchVals<u8> as Deserialize<[(&str, u8); 3]>>::deserialize(
                de,
                [("ok", 0), ("warn", 1), ("error", 2)],
            ),
        )
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, MatchVals(0)))));
    }

    #[test]
    fn match_vals_str_hits_last() {
        let de = JsonDeserializer::new(b"\"error\"");
        let v = block_on(
            <MatchVals<u8> as Deserialize<[(&str, u8); 3]>>::deserialize(
                de,
                [("ok", 0), ("warn", 1), ("error", 2)],
            ),
        )
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, MatchVals(2)))));
    }

    #[test]
    fn match_vals_str_misses_unknown() {
        let de = JsonDeserializer::new(b"\"unknown\"");
        let v = block_on(
            <MatchVals<u8> as Deserialize<[(&str, u8); 3]>>::deserialize(
                de,
                [("ok", 0), ("warn", 1), ("error", 2)],
            ),
        )
        .unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    #[test]
    fn match_vals_str_hits_escaped() {
        // "hello\nworld" contains an escape - exercises the chunks path.
        let de = JsonDeserializer::new(b"\"hello\\nworld\"");
        let v = block_on(
            <MatchVals<u8> as Deserialize<[(&str, u8); 2]>>::deserialize(
                de,
                [("other", 0), ("hello\nworld", 1)],
            ),
        )
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, MatchVals(1)))));
    }

    #[test]
    fn match_vals_bytes_hits() {
        let de = JsonDeserializer::new(b"\"warn\"");
        let v = block_on(
            <MatchVals<u8> as Deserialize<[(&[u8], u8); 2]>>::deserialize(
                de,
                [(b"ok".as_ref(), 0), (b"warn".as_ref(), 1)],
            ),
        )
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, MatchVals(1)))));
    }

    #[test]
    fn match_vals_bytes_misses() {
        let de = JsonDeserializer::new(b"\"nope\"");
        let v = block_on(
            <MatchVals<u8> as Deserialize<[(&[u8], u8); 2]>>::deserialize(
                de,
                [(b"ok".as_ref(), 0), (b"warn".as_ref(), 1)],
            ),
        )
        .unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    #[test]
    fn match_str_array_hits_any() {
        let de = JsonDeserializer::new(b"\"b\"");
        let v = block_on(<Match as Deserialize<[&str; 3]>>::deserialize(
            de,
            ["a", "b", "c"],
        ))
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, Match))));
    }

    #[test]
    fn match_str_array_misses_none() {
        let de = JsonDeserializer::new(b"\"d\"");
        let v = block_on(<Match as Deserialize<[&str; 3]>>::deserialize(
            de,
            ["a", "b", "c"],
        ))
        .unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    #[test]
    fn match_vals_usize_str_returns_index() {
        let de = JsonDeserializer::new(b"\"c\"");
        let v = block_on(<MatchVals<usize> as Deserialize<[&str; 3]>>::deserialize(
            de,
            ["a", "b", "c"],
        ))
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, MatchVals(2)))));
    }

    #[test]
    fn match_vals_usize_str_misses() {
        let de = JsonDeserializer::new(b"\"z\"");
        let v = block_on(<MatchVals<usize> as Deserialize<[&str; 3]>>::deserialize(
            de,
            ["a", "b", "c"],
        ))
        .unwrap();
        assert!(matches!(v, Probe::Miss));
    }

    // ---- UnwrapOrElse (borrow family) ----------------------------------------

    #[test]
    fn unwrap_or_else_hits_inner() {
        let de = JsonDeserializer::new(b"\"b\"");
        let v = block_on(<UnwrapOrElse<MatchVals<usize>> as Deserialize<(
            _,
            [(&str, usize); 3],
        )>>::deserialize(
            de,
            (async || MatchVals(99usize), [("a", 0), ("b", 1), ("c", 2)]),
        ))
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, UnwrapOrElse(MatchVals(1))))));
    }

    #[test]
    fn unwrap_or_else_falls_back_on_miss() {
        let de = JsonDeserializer::new(b"\"z\"");
        let v = block_on(<UnwrapOrElse<MatchVals<usize>> as Deserialize<(
            _,
            [(&str, usize); 3],
        )>>::deserialize(
            de,
            (async || MatchVals(99usize), [("a", 0), ("b", 1), ("c", 2)]),
        ))
        .unwrap();
        assert!(matches!(v, Probe::Hit((_, UnwrapOrElse(MatchVals(99))))));
    }

    // ---- derive: #[strede(tag)] internally tagged enum (borrow family) --------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(tag = "type")]
    enum TaggedEvent {
        Ping,
        Pong,
    }

    #[test]
    fn derive_internally_tagged_unit_hits_first() {
        let de = JsonDeserializer::new(br#"{"type": "Ping"}"#);
        assert_eq!(
            block_on(TaggedEvent::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TaggedEvent::Ping)),
        );
    }

    #[test]
    fn derive_internally_tagged_unit_hits_second() {
        let de = JsonDeserializer::new(br#"{"type": "Pong"}"#);
        assert_eq!(
            block_on(TaggedEvent::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TaggedEvent::Pong)),
        );
    }

    #[test]
    fn derive_internally_tagged_unit_unknown_variant_misses() {
        let de = JsonDeserializer::new(br#"{"type": "Unknown"}"#);
        assert_eq!(
            block_on(TaggedEvent::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_internally_tagged_unit_missing_tag_misses() {
        let de = JsonDeserializer::new(br#"{"other": "Ping"}"#);
        assert_eq!(
            block_on(TaggedEvent::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_internally_tagged_unit_tag_after_other_key() {
        // Tag is not the first key - should still be found.
        let de = JsonDeserializer::new(br#"{"extra": "ignored", "type": "Ping"}"#);
        assert_eq!(
            block_on(TaggedEvent::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TaggedEvent::Ping)),
        );
    }

    // ---- derive: #[strede(tag)] internally tagged enum - non-unit variants (borrow) ---
    //
    // Note: newtype/tuple variants only work when the inner type deserializes from a
    // map (i.e. a struct). Primitives like i32 are not supported because the tag
    // facade only exposes `deserialize_map`; all other probes return Miss.

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(tag = "kind")]
    struct Point2D {
        x: f64,
        y: f64,
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(tag = "kind")]
    enum TaggedShape {
        Unit,
        Newtype(Point2D),
        Struct { x: f64, y: f64 },
    }

    #[test]
    fn derive_internally_tagged_non_unit_unit_variant() {
        let de = JsonDeserializer::new(br#"{"kind": "Unit"}"#);
        assert_eq!(
            block_on(TaggedShape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TaggedShape::Unit)),
        );
    }

    #[test]
    fn derive_internally_tagged_non_unit_struct() {
        let de = JsonDeserializer::new(br#"{"kind": "Struct", "x": 1.0, "y": 2.0}"#);
        assert_eq!(
            block_on(TaggedShape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TaggedShape::Struct { x: 1.0, y: 2.0 })),
        );
    }

    #[test]
    fn derive_internally_tagged_non_unit_struct_tag_last() {
        // Tag appears after the payload fields.
        let de = JsonDeserializer::new(br#"{"x": 3.0, "y": 4.0, "kind": "Struct"}"#);
        assert_eq!(
            block_on(TaggedShape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TaggedShape::Struct { x: 3.0, y: 4.0 })),
        );
    }

    #[test]
    fn derive_internally_tagged_non_unit_newtype_wrapping_struct() {
        // Newtype wrapping a struct type that itself deserializes from a map.
        // The tag is injected alongside the inner struct's fields.
        let de = JsonDeserializer::new(br#"{"kind": "Newtype", "x": 7.0, "y": 8.0}"#);
        assert_eq!(
            block_on(TaggedShape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(TaggedShape::Newtype(Point2D { x: 7.0, y: 8.0 }))),
        );
    }

    #[test]
    fn derive_internally_tagged_non_unit_unknown_variant_misses() {
        let de = JsonDeserializer::new(br#"{"kind": "Circle", "r": 5.0}"#);
        assert_eq!(
            block_on(TaggedShape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_internally_tagged_non_unit_missing_tag_misses() {
        let de = JsonDeserializer::new(br#"{"x": 1.0, "y": 2.0}"#);
        assert_eq!(
            block_on(TaggedShape::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: #[strede(tag, content)] adjacently tagged enum (borrow) -------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(tag = "t", content = "c")]
    enum AdjacentMsg {
        Ping,
        Move { x: f64, y: f64 },
        Load(i64),
    }

    #[test]
    fn derive_adjacent_tagged_unit_variant() {
        let de = JsonDeserializer::new(br#"{"t": "Ping"}"#);
        assert_eq!(
            block_on(AdjacentMsg::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AdjacentMsg::Ping)),
        );
    }

    #[test]
    fn derive_adjacent_tagged_struct_variant() {
        let de = JsonDeserializer::new(br#"{"t": "Move", "c": {"x": 1.0, "y": 2.0}}"#);
        assert_eq!(
            block_on(AdjacentMsg::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AdjacentMsg::Move { x: 1.0, y: 2.0 })),
        );
    }

    #[test]
    fn derive_adjacent_tagged_newtype_variant() {
        let de = JsonDeserializer::new(br#"{"t": "Load", "c": 99}"#);
        assert_eq!(
            block_on(AdjacentMsg::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AdjacentMsg::Load(99))),
        );
    }

    #[test]
    fn derive_adjacent_tagged_content_before_tag() {
        // Key order reversed - should still work.
        let de = JsonDeserializer::new(br#"{"c": {"x": 5.0, "y": 6.0}, "t": "Move"}"#);
        assert_eq!(
            block_on(AdjacentMsg::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Hit(AdjacentMsg::Move { x: 5.0, y: 6.0 })),
        );
    }

    #[test]
    fn derive_adjacent_tagged_unknown_variant_misses() {
        let de = JsonDeserializer::new(br#"{"t": "Unknown", "c": 1}"#);
        assert_eq!(
            block_on(AdjacentMsg::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_adjacent_tagged_missing_tag_misses() {
        let de = JsonDeserializer::new(br#"{"c": {"x": 1.0, "y": 2.0}}"#);
        assert_eq!(
            block_on(AdjacentMsg::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    #[test]
    fn derive_adjacent_tagged_missing_content_misses() {
        // Non-unit variant with no content field - should miss.
        let de = JsonDeserializer::new(br#"{"t": "Move"}"#);
        assert_eq!(
            block_on(AdjacentMsg::deserialize(de, ())).map(|p| p.map(|(_, v)| v)),
            Ok(Probe::Miss),
        );
    }

    // -------------------------------------------------------------------------
    // flatten - borrow family
    // -------------------------------------------------------------------------

    #[derive(Debug, PartialEq, strede_derive::Deserialize)]
    struct BorrowFlatInner {
        x: f64,
        y: f64,
    }

    #[derive(Debug, PartialEq, strede_derive::Deserialize)]
    struct BorrowFlatOuter {
        name: String,
        #[strede(flatten)]
        pos: BorrowFlatInner,
    }

    // A struct where the flatten field is declared before a regular field.
    #[derive(Debug, PartialEq, strede_derive::Deserialize)]
    struct BorrowFlatLeading {
        #[strede(flatten)]
        pos: BorrowFlatInner,
        label: String,
    }

    #[test]
    fn derive_borrow_flatten_basic() {
        let de = JsonDeserializer::new(br#"{"name": "origin", "x": 1.0, "y": 2.0}"#);
        let (_, v) = block_on(BorrowFlatOuter::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(
            v,
            BorrowFlatOuter {
                name: "origin".into(),
                pos: BorrowFlatInner { x: 1.0, y: 2.0 }
            }
        );
    }

    #[test]
    fn derive_borrow_flatten_inner_keys_first() {
        // Flatten (inner) keys come before the outer key.
        let de = JsonDeserializer::new(br#"{"x": 3.0, "y": 4.0, "name": "point"}"#);
        let (_, v) = block_on(BorrowFlatOuter::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(
            v,
            BorrowFlatOuter {
                name: "point".into(),
                pos: BorrowFlatInner { x: 3.0, y: 4.0 }
            }
        );
    }

    #[test]
    fn derive_borrow_flatten_extra_fields_ignored() {
        // Extra fields not claimed by outer or inner are skipped.
        let de = JsonDeserializer::new(br#"{"name": "p", "x": 5.0, "extra": true, "y": 6.0}"#);
        let (_, v) = block_on(BorrowFlatOuter::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(
            v,
            BorrowFlatOuter {
                name: "p".into(),
                pos: BorrowFlatInner { x: 5.0, y: 6.0 }
            }
        );
    }

    #[test]
    fn derive_borrow_flatten_missing_inner_field_misses() {
        // Inner field missing - should return Miss.
        let missed = block_on(BorrowFlatOuter::deserialize(
            JsonDeserializer::new(br#"{"name": "p", "x": 1.0}"#),
            (),
        ))
        .unwrap()
        .map(|(_, v)| v);
        assert_eq!(missed, Probe::Miss);
    }

    #[test]
    fn derive_borrow_flatten_missing_outer_field_misses() {
        // Outer (non-flatten) field missing - should miss.
        let missed = block_on(BorrowFlatOuter::deserialize(
            JsonDeserializer::new(br#"{"x": 1.0, "y": 2.0}"#),
            (),
        ))
        .unwrap()
        .map(|(_, v)| v);
        assert_eq!(missed, Probe::Miss);
    }

    #[test]
    fn derive_borrow_flatten_field_first() {
        // Flatten field declared first - outer field comes after in JSON.
        let de = JsonDeserializer::new(br#"{"x": 7.0, "y": 8.0, "label": "q"}"#);
        let (_, v) = block_on(BorrowFlatLeading::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(
            v,
            BorrowFlatLeading {
                pos: BorrowFlatInner { x: 7.0, y: 8.0 },
                label: "q".into()
            }
        );
    }

    // ---- derive: borrow lifetime bounds ------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct BorrowedStr<'a> {
        name: &'a str,
    }

    #[test]
    fn derive_borrow_auto_ref_str() {
        // &'a str field auto-generates 'de: 'a.
        let json = br#"{"name": "hello"}"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(BorrowedStr::deserialize(de, ())).unwrap().unwrap();
        assert_eq!(v, BorrowedStr { name: "hello" });
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct BorrowedBytes<'a> {
        data: &'a [u8],
    }

    #[test]
    fn derive_borrow_auto_ref_bytes() {
        let json = br#"{"data": "hello"}"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(BorrowedBytes::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(v, BorrowedBytes { data: b"hello" });
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct BorrowedCow<'a> {
        name: alloc::borrow::Cow<'a, str>,
    }

    #[test]
    fn derive_borrow_auto_cow() {
        // Cow<'a, str> field auto-generates 'de: 'a.
        let json = br#"{"name": "world"}"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(BorrowedCow::deserialize(de, ())).unwrap().unwrap();
        assert_eq!(v.name, "world");
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct TwoLifetimes<'a, 'b> {
        first: &'a str,
        second: &'b str,
    }

    #[test]
    fn derive_borrow_two_lifetimes() {
        let json = br#"{"first": "a", "second": "b"}"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(TwoLifetimes::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(
            v,
            TwoLifetimes {
                first: "a",
                second: "b"
            }
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct MixedGenericAndLifetime<'a, T> {
        name: &'a str,
        value: T,
    }

    #[test]
    fn derive_borrow_mixed_generic_and_lifetime() {
        // Both 'de: 'a (from &'a str) and T: Deserialize<'de> (from generic T).
        let json = br#"{"name": "key", "value": 42}"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(<MixedGenericAndLifetime<i64>>::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(
            v,
            MixedGenericAndLifetime {
                name: "key",
                value: 42
            }
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct ThreeLifetimes<'a, 'b, 'c> {
        first: &'a str,
        second: &'b str,
        #[strede(skip_deserializing, default)]
        _phantom: core::marker::PhantomData<&'c ()>,
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct ExplicitBorrow<'a, 'b, 'c> {
        #[strede(borrow = "'a + 'b")]
        inner: ThreeLifetimes<'a, 'b, 'c>,
    }

    #[test]
    fn derive_borrow_explicit_lifetimes() {
        // Only 'de: 'a and 'de: 'b are emitted, not 'de: 'c.
        let json = br#"{"inner": {"first": "x", "second": "y"}}"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(<ExplicitBorrow<'_, '_, '_>>::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(v.inner.first, "x");
        assert_eq!(v.inner.second, "y");
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct BorrowAll<'a, 'b> {
        #[strede(borrow)]
        pair: TwoLifetimes<'a, 'b>,
    }

    #[test]
    fn derive_borrow_all_lifetimes() {
        // #[strede(borrow)] grabs all lifetimes ('a and 'b).
        let json = br#"{"pair": {"first": "x", "second": "y"}}"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(BorrowAll::deserialize(de, ())).unwrap().unwrap();
        assert_eq!(
            v.pair,
            TwoLifetimes {
                first: "x",
                second: "y"
            }
        );
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    struct BorrowedTuple<'a>(&'a str, i64);

    #[test]
    fn derive_borrow_tuple_struct() {
        // Tuple struct with a borrowed field.
        let json = br#"["hello", 42]"#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(BorrowedTuple::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(v, BorrowedTuple("hello", 42));
    }

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    #[strede(transparent)]
    struct TransparentBorrow<'a>(&'a str);

    #[test]
    fn derive_borrow_transparent() {
        let json = br#""hello""#;
        let de = JsonDeserializer::new(json);
        let (_, v) = block_on(TransparentBorrow::deserialize(de, ()))
            .unwrap()
            .unwrap();
        assert_eq!(v, TransparentBorrow("hello"));
    }
}
