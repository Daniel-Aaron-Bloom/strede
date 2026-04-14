//! `strede-json` — JSON backend for the `strede` deserialization framework.
//!
//! Implements [`strede::Deserializer`] over a borrowed `&[u8]` source.
//! Entry probe futures resolve immediately — `Ok(Probe::Hit(...))` when the
//! token type matches, `Ok(Probe::Miss)` when it does not.  `Pending` means
//! only "no data available yet" (never reached in in-memory use).
//!
//! # Zero-copy vs. chunked strings
//!
//! [`Entry::deserialize_str`] returns `Ok(Probe::Miss)` for any string that
//! contains escape sequences, because it can only hand back a zero-copy
//! `&'de str` pointing directly into the source bytes.  Use
//! [`Entry::deserialize_str_chunks`] (or race both with `select_probe!`) for
//! strings that may contain `\n`, `\"`, `\uXXXX`, etc.
//!
//! This makes `Cow<str>` / `Cow<[u8]>` callers easy to write: race
//! `deserialize_str` against `deserialize_str_chunks`; the former hits only
//! when the borrow is free, the latter handles the rest via owned allocation.
//!
//! # Limitations (V1)
//!
//! - Streaming across buffer boundaries is not supported (in-memory only).

#![no_std]
#![allow(async_fn_in_trait)]

pub mod chunked;
mod error;
mod token;

pub use error::JsonError;
use token::{SimpleToken, StrChunk, Token, Tokenizer};

use strede::{
    BytesAccess, Chunk, Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry,
    Probe, SeqAccess, SeqEntry, StrAccess, hit,
};

// ---------------------------------------------------------------------------
// JsonClaim — unified proof-of-consumption type
// ---------------------------------------------------------------------------

/// Carries the post-consumption tokenizer and source state.
/// Returned from every probe method and threaded back to [`Deserializer::next`]
/// (or [`MapAccess::next`] / [`SeqAccess::next`]) to advance the stream.
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
    /// Pre-read token to replay on the next `next()` call.
    /// Set by sub-deserializers created for map keys, map values, and seq elements,
    /// and also restored when `next()` returns `Probe::Miss`.
    pending_tok: Option<Token>,
    /// True for deserializers created via [`Self::new`]; triggers trailing-garbage
    /// checks after the top-level value is consumed.
    is_root: bool,
}

impl<'de> JsonDeserializer<'de> {
    pub fn new(src: &'de [u8]) -> Self {
        Self {
            tokenizer: Tokenizer::new(),
            src,
            pending_tok: None,
            is_root: true,
        }
    }

    fn sub_with_pending(src: &'de [u8], tok: Token) -> Self {
        Self {
            tokenizer: Tokenizer::new(),
            src,
            pending_tok: Some(tok),
            is_root: false,
        }
    }

    /// Read the next dispatch token, honouring any pre-read `pending_tok`.
    ///
    /// `NoTokens` from the tokenizer means the in-memory buffer is exhausted —
    /// for this deserializer that always means the input is truncated.
    fn next_dispatch(&mut self) -> Result<Token, JsonError> {
        if let Some(tok) = self.pending_tok.take() {
            return Ok(tok);
        }
        while !self.src.is_empty() {
            let old = core::mem::replace(&mut self.tokenizer, Tokenizer::new());
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

async fn json_deserializer_next<'s, 'de: 's, const N: usize, F, Fut, R>(
    de: &'s mut JsonDeserializer<'de>,
    f: F,
) -> Result<Probe<R>, JsonError>
where
    F: FnOnce([JsonEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(JsonClaim<'de>, R)>, JsonError>>,
{
    let token = de.next_dispatch()?;
    // Keep a copy so we can restore pending_tok if the closure returns Miss.
    let entry = JsonEntry {
        token: token.clone(),
        src: de.src,
    };
    match f(core::array::from_fn(|_| entry.clone())).await? {
        Probe::Hit((claim, r)) => {
            de.tokenizer = claim.tokenizer;
            de.src = claim.src;
            if de.is_root {
                let mut rest = de.src;
                while !rest.is_empty() && matches!(rest[0], b' ' | b'\t' | b'\n' | b'\r') {
                    rest = &rest[1..];
                }
                if !rest.is_empty() {
                    return Err(JsonError::TrailingGarbage);
                }
            }
            Ok(Probe::Hit(r))
        }
        Probe::Miss => {
            // Restore: put the token back so the next call can retry.
            de.pending_tok = Some(token);
            Ok(Probe::Miss)
        }
    }
}

impl<'de> Deserializer<'de> for JsonDeserializer<'de> {
    type Error = JsonError;

    type Entry<'a>
        = JsonEntry<'de>
    where
        Self: 'a,
        'de: 'a;

    async fn next<'s, const N: usize, F, Fut, R>(
        &'s mut self,
        f: F,
    ) -> Result<Probe<R>, Self::Error>
    where
        'de: 's,
        F: FnOnce([Self::Entry<'s>; N]) -> Fut,
        Fut: core::future::Future<
                Output = Result<Probe<(<Self::Entry<'s> as Entry<'de>>::Claim, R)>, Self::Error>,
            >,
    {
        json_deserializer_next(self, f).await
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

impl<'de> Entry<'de> for JsonEntry<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;
    type StrChunks = JsonStrAccess<'de>;
    type BytesChunks = JsonBytesAccess<'de>;
    type Map = JsonMapAccess<'de>;
    type Seq = JsonSeqAccess<'de>;

    // ---- Scalars -----------------------------------------------------------

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

    async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((claim, n as u8)))
    }
    async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((claim, n as u16)))
    }
    async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((claim, n as u32)))
    }
    async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
        self.parse_num::<u64>().await
    }
    async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
        self.parse_num::<u128>().await
    }
    async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((claim, n as i8)))
    }
    async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((claim, n as i16)))
    }
    async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
        let (claim, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((claim, n as i32)))
    }
    async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
        self.parse_num::<i64>().await
    }
    async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
        self.parse_num::<i128>().await
    }
    async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
        self.parse_num::<f32>().await
    }
    async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
        self.parse_num::<f64>().await
    }

    // ---- Char --------------------------------------------------------------

    async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
        // V1: deserialize a single-char JSON string
        let (claim, s) = hit!(self.deserialize_str().await);
        let c = s
            .chars()
            .next()
            .ok_or(JsonError::UnexpectedByte { byte: b'"' })?;
        Ok(Probe::Hit((claim, c)))
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
                let s = match access.next_chunk(&mut src) {
                    Ok(Some(StrChunk::Slice(s))) => {
                        // Consume the closing `"` (next chunk must be None).
                        match access.next_chunk(&mut src) {
                            Ok(None) => {}
                            // More chunks means there were escape sequences; Miss so that
                            // a concurrent deserialize_str_chunks arm can take over.
                            Ok(Some(_)) => return Ok(Probe::Miss),
                            Err(e) => return Err(e),
                        }
                        s
                    }
                    // First chunk is already an escaped char — not zero-copy; Miss.
                    Ok(Some(StrChunk::Char(_))) => return Ok(Probe::Miss),
                    Ok(None) => "",
                    Err(e) => return Err(e),
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

    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
        let (claim, s) = hit!(self.deserialize_str().await);
        Ok(Probe::Hit((claim, s.as_bytes())))
    }

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

    async fn deserialize_option<T: Deserialize<'de>>(
        self,
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
                let mut sub = JsonDeserializer::sub_with_pending(self.src, other);
                let v = hit!(T::deserialize(&mut sub).await);
                Ok(Probe::Hit((
                    JsonClaim {
                        tokenizer: sub.tokenizer,
                        src: sub.src,
                    },
                    Some(v),
                )))
            }
        }
    }

    // ---- Null ---------------------------------------------------------------

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

    async fn deserialize_value<T: Deserialize<'de>>(
        self,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        let mut sub = JsonDeserializer::sub_with_pending(self.src, self.token);
        let v = hit!(T::deserialize(&mut sub).await);
        Ok(Probe::Hit((
            JsonClaim {
                tokenizer: sub.tokenizer,
                src: sub.src,
            },
            v,
        )))
    }
}

// Number parsing helper — not a trait method so it can be called generically.
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
    access: token::StrAccess,
    src: &'de [u8],
    char_buf: [u8; 4],
}

impl<'de> StrAccess for JsonStrAccess<'de> {
    type Claim = JsonClaim<'de>;
    type Error = JsonError;

    async fn next(&mut self) -> Result<Chunk<&str, Self::Claim>, Self::Error> {
        match self.access.next_chunk(&mut self.src) {
            Ok(Some(StrChunk::Slice(s))) => Ok(Chunk::Data(s)),
            Ok(Some(StrChunk::Char(c))) => Ok(Chunk::Data(&*c.encode_utf8(&mut self.char_buf))),
            Ok(None) => Ok(Chunk::Done(JsonClaim {
                tokenizer: Tokenizer::new(),
                src: self.src,
            })),
            Err(e) => Err(e),
        }
    }
}

pub struct JsonBytesAccess<'de> {
    access: token::StrAccess,
    src: &'de [u8],
    char_buf: [u8; 4],
}

impl<'de> BytesAccess for JsonBytesAccess<'de> {
    type Claim = JsonClaim<'de>;
    type Error = JsonError;

    async fn next(&mut self) -> Result<Chunk<&[u8], Self::Claim>, Self::Error> {
        match self.access.next_chunk(&mut self.src) {
            Ok(Some(StrChunk::Slice(s))) => Ok(Chunk::Data(s.as_bytes())),
            Ok(Some(StrChunk::Char(c))) => {
                Ok(Chunk::Data(c.encode_utf8(&mut self.char_buf).as_bytes()))
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
// JsonMapAccess
// ---------------------------------------------------------------------------

pub struct JsonMapAccess<'de> {
    tokenizer: Tokenizer,
    src: &'de [u8],
    first: bool,
}

async fn json_map_next<'s, 'de: 's, const N: usize, F, Fut, R>(
    map: &'s mut JsonMapAccess<'de>,
    f: F,
) -> Result<Probe<Chunk<R, JsonClaim<'de>>>, JsonError>
where
    F: FnOnce([JsonMapKeyEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(JsonClaim<'de>, R)>, JsonError>>,
{
    // After first pair, expect comma or closing brace.
    if !map.first {
        let old = core::mem::replace(&mut map.tokenizer, Tokenizer::new());
        match old.next_token(&mut map.src) {
            Ok(Token::Simple(SimpleToken::Comma, new_tok)) => {
                map.tokenizer = new_tok;
            }
            Ok(Token::Simple(SimpleToken::ObjectEnd, new_tok)) => {
                return Ok(Probe::Hit(Chunk::Done(JsonClaim {
                    tokenizer: new_tok,
                    src: map.src,
                })));
            }
            Ok(_) => return Err(JsonError::UnexpectedByte { byte: 0 }),
            Err(e) => return Err(e),
        }
    }
    map.first = false;

    // Read key start token (or closing brace for empty map).
    let key_tok = {
        let old = core::mem::replace(&mut map.tokenizer, Tokenizer::new());
        match old.next_token(&mut map.src) {
            Ok(Token::Simple(SimpleToken::ObjectEnd, new_tok)) => {
                return Ok(Probe::Hit(Chunk::Done(JsonClaim {
                    tokenizer: new_tok,
                    src: map.src,
                })));
            }
            Ok(t) => t,
            Err(e) => return Err(e),
        }
    };

    let key_entry = JsonMapKeyEntry {
        src: map.src,
        key_tok,
    };
    let (claim, r) = hit!(f(core::array::from_fn(|_| key_entry.clone())).await);
    map.tokenizer = claim.tokenizer;
    map.src = claim.src;
    Ok(Probe::Hit(Chunk::Data(r)))
}

impl<'de> MapAccess<'de> for JsonMapAccess<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;

    type KeyEntry<'a>
        = JsonMapKeyEntry<'de>
    where
        Self: 'a,
        'de: 'a;

    async fn next<'s, const N: usize, F, Fut, R>(
        &'s mut self,
        f: F,
    ) -> Result<Probe<Chunk<R, Self::Claim>>, Self::Error>
    where
        'de: 's,
        F: FnOnce([Self::KeyEntry<'s>; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
    {
        json_map_next(self, f).await
    }
}

// ---------------------------------------------------------------------------
// JsonMapKeyEntry / JsonMapValueEntry
// ---------------------------------------------------------------------------

pub struct JsonMapKeyEntry<'de> {
    src: &'de [u8],
    key_tok: Token,
}

impl<'de> JsonMapKeyEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            key_tok: self.key_tok.clone(),
        }
    }
}

impl<'de> MapKeyEntry<'de> for JsonMapKeyEntry<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;
    type ValueEntry = JsonMapValueEntry<'de>;

    async fn key<K: Deserialize<'de>, const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, K, R)>, Self::Error>
    where
        F: FnOnce(&K, [Self::ValueEntry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
    {
        // Deserialize the key through a sub-deserializer that replays key_tok.
        let mut key_deser = JsonDeserializer::sub_with_pending(self.src, self.key_tok);
        let k = hit!(K::deserialize(&mut key_deser).await);

        // Consume the colon separator.
        let old = core::mem::replace(&mut key_deser.tokenizer, Tokenizer::new());
        match old.next_token(&mut key_deser.src) {
            Ok(Token::Simple(SimpleToken::Colon, new_tok)) => {
                key_deser.tokenizer = new_tok;
            }
            Ok(_) => return Err(JsonError::UnexpectedByte { byte: 0 }),
            Err(e) => return Err(e),
        }

        // Read the first token of the value.
        let value_tok = {
            let old = core::mem::replace(&mut key_deser.tokenizer, Tokenizer::new());
            match old.next_token(&mut key_deser.src) {
                Ok(t) => t,
                Err(e) => return Err(e),
            }
        };

        let ve = JsonMapValueEntry {
            src: key_deser.src,
            value_tok,
        };
        let (claim, r) = hit!(f(&k, core::array::from_fn(|_| ve.clone())).await);
        Ok(Probe::Hit((claim, k, r)))
    }
}

pub struct JsonMapValueEntry<'de> {
    src: &'de [u8],
    value_tok: Token,
}

impl<'de> JsonMapValueEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            value_tok: self.value_tok.clone(),
        }
    }
}

impl<'de> MapValueEntry<'de> for JsonMapValueEntry<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;

    async fn value<V: Deserialize<'de>>(self) -> Result<Probe<(Self::Claim, V)>, Self::Error> {
        let mut value_deser = JsonDeserializer::sub_with_pending(self.src, self.value_tok);
        let v = hit!(V::deserialize(&mut value_deser).await);
        Ok(Probe::Hit((
            JsonClaim {
                tokenizer: value_deser.tokenizer,
                src: value_deser.src,
            },
            v,
        )))
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

// Free function to work around a rustc ICE (triggered by RPITIT lifetime checking
// in `compare_impl_item` for cross-crate `async fn` trait implementations).
// By placing the async body here (local DefId), region checking uses local DefIds.
async fn json_seq_next<'s, 'de: 's, const N: usize, F, Fut, R>(
    seq: &'s mut JsonSeqAccess<'de>,
    f: F,
) -> Result<Probe<Chunk<R, JsonClaim<'de>>>, JsonError>
where
    F: FnOnce([JsonSeqEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(JsonClaim<'de>, R)>, JsonError>>,
{
    // After first element, expect comma or closing bracket.
    if !seq.first {
        let old = core::mem::replace(&mut seq.tokenizer, Tokenizer::new());
        match old.next_token(&mut seq.src) {
            Ok(Token::Simple(SimpleToken::Comma, new_tok)) => {
                seq.tokenizer = new_tok;
            }
            Ok(Token::Simple(SimpleToken::ArrayEnd, new_tok)) => {
                return Ok(Probe::Hit(Chunk::Done(JsonClaim {
                    tokenizer: new_tok,
                    src: seq.src,
                })));
            }
            Ok(_) => return Err(JsonError::UnexpectedByte { byte: 0 }),
            Err(e) => return Err(e),
        }
    }
    seq.first = false;

    // Read element start token (or closing bracket for empty seq).
    let elem_tok = {
        let old = core::mem::replace(&mut seq.tokenizer, Tokenizer::new());
        match old.next_token(&mut seq.src) {
            Ok(Token::Simple(SimpleToken::ArrayEnd, new_tok)) => {
                return Ok(Probe::Hit(Chunk::Done(JsonClaim {
                    tokenizer: new_tok,
                    src: seq.src,
                })));
            }
            Ok(t) => t,
            Err(e) => return Err(e),
        }
    };

    let se = JsonSeqEntry {
        src: seq.src,
        elem_tok,
    };
    let (claim, r) = hit!(f(core::array::from_fn(|_| se.clone())).await);
    seq.tokenizer = claim.tokenizer;
    seq.src = claim.src;
    Ok(Probe::Hit(Chunk::Data(r)))
}

impl<'de> SeqAccess<'de> for JsonSeqAccess<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;

    type Elem<'a>
        = JsonSeqEntry<'de>
    where
        Self: 'a;

    async fn next<'s, const N: usize, F, Fut, R>(
        &'s mut self,
        f: F,
    ) -> Result<Probe<Chunk<R, Self::Claim>>, Self::Error>
    where
        'de: 's,
        F: FnOnce([Self::Elem<'s>; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
    {
        json_seq_next(self, f).await
    }
}

pub struct JsonSeqEntry<'de> {
    src: &'de [u8],
    elem_tok: Token,
}

impl<'de> JsonSeqEntry<'de> {
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

    async fn get<T: Deserialize<'de>>(self) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        let mut elem_deser = JsonDeserializer::sub_with_pending(self.src, self.elem_tok);
        let v = hit!(T::deserialize(&mut elem_deser).await);
        Ok(Probe::Hit((
            JsonClaim {
                tokenizer: elem_deser.tokenizer,
                src: elem_deser.src,
            },
            v,
        )))
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::string::String;
    use strede::{
        Deserialize, Deserializer, MapAccess, MapKeyEntry, MapValueEntry, Probe, SeqAccess,
        SeqEntry,
    };

    // Minimal no-op executor.  In-memory input never yields `Pending` so all
    // futures complete in a single poll.
    fn block_on<F: core::future::Future>(f: F) -> F::Output {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        static VTABLE: RawWakerVTable =
            RawWakerVTable::new(|p| RawWaker::new(p, &VTABLE), |_| {}, |_| {}, |_| {});
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = core::pin::pin!(f);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("future pending — unexpected for in-memory input"),
        }
    }

    // Local Deserialize impls for test-only newtypes.
    struct U32(u32);
    impl<'de> Deserialize<'de> for U32 {
        async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error> {
            d.next(|[e]| async move {
                let (c, v) = hit!(e.deserialize_u32().await);
                Ok(Probe::Hit((c, U32(v))))
            })
            .await
        }
    }

    struct Str<'de>(&'de str);
    impl<'de> Deserialize<'de> for Str<'de> {
        async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error> {
            d.next(|[e]| async move {
                let (c, v) = hit!(e.deserialize_str().await);
                Ok(Probe::Hit((c, Str(v))))
            })
            .await
        }
    }

    // ---- bool ---------------------------------------------------------------

    #[test]
    fn bool_true() {
        let mut de = JsonDeserializer::new(b"true");
        let v = block_on(de.next(|[e]| async move { e.deserialize_bool().await }))
            .unwrap()
            .unwrap();
        assert!(v);
    }

    #[test]
    fn bool_false() {
        let mut de = JsonDeserializer::new(b"false");
        let v = block_on(de.next(|[e]| async move { e.deserialize_bool().await }))
            .unwrap()
            .unwrap();
        assert!(!v);
    }

    // ---- integers -----------------------------------------------------------

    #[test]
    fn u32_positive() {
        let mut de = JsonDeserializer::new(b"42");
        let v = block_on(de.next(|[e]| async move { e.deserialize_u32().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, 42u32);
    }

    #[test]
    fn i64_negative() {
        let mut de = JsonDeserializer::new(b"-7");
        let v = block_on(de.next(|[e]| async move { e.deserialize_i64().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, -7i64);
    }

    #[test]
    fn u128_large() {
        // 2^64 — exceeds u64::MAX, exercises the u128 parse path.
        let mut de = JsonDeserializer::new(b"18446744073709551616");
        let v = block_on(de.next(|[e]| async move { e.deserialize_u128().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, u64::MAX as u128 + 1);
    }

    // ---- floats -------------------------------------------------------------

    #[test]
    fn f64_decimal() {
        let mut de = JsonDeserializer::new(b"3.14");
        let v = block_on(de.next(|[e]| async move { e.deserialize_f64().await }))
            .unwrap()
            .unwrap();
        assert!((v - 3.14f64).abs() < 1e-10);
    }

    #[test]
    fn f32_half() {
        let mut de = JsonDeserializer::new(b"1.5");
        let v = block_on(de.next(|[e]| async move { e.deserialize_f32().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, 1.5f32);
    }

    // ---- str / char ---------------------------------------------------------

    #[test]
    fn str_plain() {
        let mut de = JsonDeserializer::new(b"\"hello\"");
        let v = block_on(de.next(|[e]| async move { e.deserialize_str().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, "hello");
    }

    #[test]
    fn str_empty() {
        let mut de = JsonDeserializer::new(b"\"\"");
        let v = block_on(de.next(|[e]| async move { e.deserialize_str().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, "");
    }

    #[test]
    fn char_single() {
        let mut de = JsonDeserializer::new(b"\"A\"");
        let v = block_on(de.next(|[e]| async move { e.deserialize_char().await }))
            .unwrap()
            .unwrap();
        assert_eq!(v, 'A');
    }

    // ---- str_chunks (escape sequences) --------------------------------------

    #[test]
    fn deserialize_str_misses_on_escape_sequences_too() {
        // Strings with escapes return Miss, not an error — src is owned so the
        // stream position can be handed off to a deserialize_str_chunks arm.
        let mut de = JsonDeserializer::new(b"\"\\n\"");
        let result = block_on(de.next(|[e]| async move { e.deserialize_str().await }));
        assert!(matches!(result, Ok(Probe::Miss)));
    }

    #[test]
    fn str_chunks_escape_newline() {
        // JSON "hello\nworld": tokenizer yields "hello", '\n' (Char), "world".
        let mut de = JsonDeserializer::new(b"\"hello\\nworld\"");
        let total_len = block_on(de.next(|[e]| async move {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            let mut len = 0usize;
            loop {
                match chunks.next().await? {
                    Chunk::Data(chunk) => len += chunk.len(),
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
        let mut de = JsonDeserializer::new(b"\"\\u0041\"");
        let total_len = block_on(de.next(|[e]| async move {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            let mut len = 0usize;
            loop {
                match chunks.next().await? {
                    Chunk::Data(chunk) => len += chunk.len(),
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
        let mut de = JsonDeserializer::new(input);
        block_on(de.next(|[e]| async move {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            let mut out = String::new();
            loop {
                match chunks.next().await? {
                    Chunk::Data(chunk) => out.push_str(chunk),
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                }
            }
        }))
        .unwrap()
        .unwrap()
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
        let mut de = JsonDeserializer::new(b"\"\\uD834\"");
        let result = block_on(de.next(|[e]| async move {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            loop {
                match chunks.next().await? {
                    Chunk::Data(_) => {}
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, ()))),
                }
            }
        }));
        assert!(result.is_err());
    }

    #[test]
    fn str_chunks_lone_low_surrogate_err() {
        let mut de = JsonDeserializer::new(b"\"\\uDD1E\"");
        let result = block_on(de.next(|[e]| async move {
            let mut chunks = match e.deserialize_str_chunks().await? {
                Probe::Hit(c) => c,
                Probe::Miss => panic!("expected str chunks"),
            };
            loop {
                match chunks.next().await? {
                    Chunk::Data(_) => {}
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, ()))),
                }
            }
        }));
        assert!(result.is_err());
    }

    // ---- seq ----------------------------------------------------------------

    #[test]
    fn seq_sum_of_numbers() {
        let mut de = JsonDeserializer::new(b"[1, 2, 3]");
        let sum = block_on(de.next(|[e]| async move {
            let mut seq = match e.deserialize_seq().await? {
                Probe::Hit(s) => s,
                Probe::Miss => panic!("expected seq"),
            };
            let mut sum = 0u32;
            loop {
                match seq
                    .next(|[se]| async move {
                        let (claim, v) = hit!(se.get::<U32>().await);
                        Ok(Probe::Hit((claim, v.0)))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data(v) => sum += v,
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
        let mut de = JsonDeserializer::new(b"[]");
        let count = block_on(de.next(|[e]| async move {
            let mut seq = match e.deserialize_seq().await? {
                Probe::Hit(s) => s,
                Probe::Miss => panic!("expected seq"),
            };
            let mut count = 0usize;
            loop {
                match seq
                    .next(|[se]| async move {
                        let (claim, v) = hit!(se.get::<U32>().await);
                        Ok(Probe::Hit((claim, v.0)))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data(_) => count += 1,
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
        let mut de = JsonDeserializer::new(b"{\"x\": 10}");
        let (key, val) = block_on(de.next(|[e]| async move {
            let mut map = match e.deserialize_map().await? {
                Probe::Hit(m) => m,
                Probe::Miss => panic!("expected map"),
            };
            let mut key = "";
            let mut val = 0u32;
            loop {
                match map
                    .next(|[ke]| async move {
                        let (claim, k, v) = hit!(
                            ke.key::<Str<'_>, 1, _, _, _>(|_k, [ve]| async move {
                                let (claim, v) = hit!(ve.value::<U32>().await);
                                Ok(Probe::Hit((claim, v.0)))
                            })
                            .await
                        );
                        Ok(Probe::Hit((claim, (k.0, v))))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data((k, v)) => {
                        key = k;
                        val = v;
                    }
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, (key, val)))),
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(key, "x");
        assert_eq!(val, 10);
    }

    #[test]
    fn map_empty() {
        let mut de = JsonDeserializer::new(b"{}");
        let count = block_on(de.next(|[e]| async move {
            let mut map = match e.deserialize_map().await? {
                Probe::Hit(m) => m,
                Probe::Miss => panic!("expected map"),
            };
            let mut count = 0usize;
            loop {
                match map
                    .next(|[ke]| async move {
                        let (claim, k, v) = hit!(
                            ke.key::<Str<'_>, 1, _, _, _>(|_k, [ve]| async move {
                                let (claim, v) = hit!(ve.value::<U32>().await);
                                Ok(Probe::Hit((claim, v.0)))
                            })
                            .await
                        );
                        Ok(Probe::Hit((claim, (k.0, v))))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data(_) => count += 1,
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, count))),
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn map_multiple_pairs() {
        let mut de = JsonDeserializer::new(b"{\"a\": 1, \"b\": 2, \"c\": 3}");
        let sum = block_on(de.next(|[e]| async move {
            let mut map = match e.deserialize_map().await? {
                Probe::Hit(m) => m,
                Probe::Miss => panic!("expected map"),
            };
            let mut total = 0u32;
            loop {
                match map
                    .next(|[ke]| async move {
                        let (claim, _k, v) = hit!(
                            ke.key::<Str<'_>, 1, _, _, _>(|_k, [ve]| async move {
                                let (claim, v) = hit!(ve.value::<U32>().await);
                                Ok(Probe::Hit((claim, v.0)))
                            })
                            .await
                        );
                        Ok(Probe::Hit((claim, v)))
                    })
                    .await?
                    .unwrap()
                {
                    Chunk::Data(v) => total += v,
                    Chunk::Done(claim) => return Ok(Probe::Hit((claim, total))),
                }
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(sum, 6);
    }

    // ---- option -------------------------------------------------------------

    struct Bool(bool);
    impl<'de> Deserialize<'de> for Bool {
        async fn deserialize<D: Deserializer<'de>>(d: &mut D) -> Result<Probe<Self>, D::Error> {
            d.next(|[e]| async move {
                let (c, v) = hit!(e.deserialize_bool().await);
                Ok(Probe::Hit((c, Bool(v))))
            })
            .await
        }
    }

    #[test]
    fn option_null_is_none() {
        let mut de = JsonDeserializer::new(b"null");
        let v = block_on(de.next(|[e]| async move { e.deserialize_option::<Bool>().await }))
            .unwrap()
            .unwrap();
        assert!(v.is_none());
    }

    #[test]
    fn option_bool_is_some() {
        let mut de = JsonDeserializer::new(b"true");
        let v = block_on(de.next(|[e]| async move { e.deserialize_option::<Bool>().await }))
            .unwrap()
            .unwrap();
        assert!(v.unwrap().0);
    }

    // ---- error handling -----------------------------------------------------

    #[test]
    fn error_truncated_literal() {
        let mut de = JsonDeserializer::new(b"tru");
        let result = block_on(de.next(|[e]| async move { e.deserialize_bool().await }));
        assert!(matches!(result, Err(JsonError::UnexpectedEnd)));
    }

    #[test]
    fn error_invalid_number() {
        let mut de = JsonDeserializer::new(b"1.}");
        let result = block_on(de.next(|[e]| async move { e.deserialize_f64().await }));
        assert!(matches!(result, Err(JsonError::InvalidNumber)));
    }

    #[test]
    fn deserialize_str_misses_on_escape_sequences() {
        // deserialize_str only hits for zero-copy slices; strings with escape
        // sequences return Miss so callers can fall through to deserialize_str_chunks.
        let mut de = JsonDeserializer::new(b"\"\\n\"");
        let result = block_on(de.next(|[e]| async move { e.deserialize_str().await }));
        assert!(matches!(result, Ok(Probe::Miss)));
    }

    // ---- trailing garbage ---------------------------------------------------

    #[test]
    fn trailing_garbage_after_bool() {
        let mut de = JsonDeserializer::new(b"true garbage");
        let result = block_on(de.next(|[e]| async move { e.deserialize_bool().await }));
        assert_eq!(result, Err(JsonError::TrailingGarbage));
    }

    #[test]
    fn trailing_garbage_after_number() {
        let mut de = JsonDeserializer::new(b"42 extra");
        let result = block_on(de.next(|[e]| async move { e.deserialize_u32().await }));
        assert_eq!(result, Err(JsonError::TrailingGarbage));
    }

    #[test]
    fn trailing_whitespace_is_ok() {
        let mut de = JsonDeserializer::new(b"true   \n");
        let v = block_on(de.next(|[e]| async move { e.deserialize_bool().await }))
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
        let mut de = JsonDeserializer::new(b"{\"x\": 1, \"y\": -2}");
        assert_eq!(
            block_on(Point::deserialize(&mut de)),
            Ok(Probe::Hit(Point { x: 1, y: -2 })),
        );
    }

    #[test]
    fn derive_fields_out_of_order() {
        let mut de = JsonDeserializer::new(b"{\"y\": 7, \"x\": 3}");
        assert_eq!(
            block_on(Point::deserialize(&mut de)),
            Ok(Probe::Hit(Point { x: 3, y: 7 })),
        );
    }

    #[test]
    fn derive_duplicate_field() {
        let mut de = JsonDeserializer::new(b"{\"x\": 1, \"x\": 2, \"y\": 3}");
        assert_eq!(
            block_on(Point::deserialize(&mut de)),
            Err(JsonError::DuplicateField("x")),
        );
    }

    #[test]
    fn derive_unknown_field_is_miss() {
        // "z" is an unknown field — derive returns Miss.
        let mut de = JsonDeserializer::new(b"{\"x\": 1, \"z\": 99, \"y\": 2}");
        assert_eq!(block_on(Point::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    #[test]
    fn derive_missing_field_is_miss() {
        // "y" is missing — derive returns Miss.
        let mut de = JsonDeserializer::new(b"{\"x\": 5}");
        assert_eq!(block_on(Point::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    // ---- derive: non-object input → Miss ------------------------------------

    #[test]
    fn derive_non_object_is_miss() {
        let mut de = JsonDeserializer::new(b"42");
        assert_eq!(block_on(Point::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    #[test]
    fn derive_array_is_miss() {
        let mut de = JsonDeserializer::new(b"[1, 2]");
        assert_eq!(block_on(Point::deserialize(&mut de)), Ok(Probe::Miss),);
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
        let mut de = JsonDeserializer::new(br#"{"name": "hello", "count": 7, "active": true}"#);
        assert_eq!(
            block_on(Mixed::deserialize(&mut de)),
            Ok(Probe::Hit(Mixed {
                name: "hello",
                count: 7,
                active: true
            })),
        );
    }

    #[test]
    fn derive_mixed_types_reordered() {
        let mut de = JsonDeserializer::new(br#"{"active": false, "name": "world", "count": 0}"#);
        assert_eq!(
            block_on(Mixed::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"score": 3.14, "count": 7, "active": true}"#);
        assert_eq!(
            block_on(MixedOwned::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"value": -99}"#);
        assert_eq!(
            block_on(Wrapper::deserialize(&mut de)),
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
        let mut de =
            JsonDeserializer::new(br#"{"origin": {"x": 1, "y": 2}, "size": {"x": 10, "y": 20}}"#);
        assert_eq!(
            block_on(Rect::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"required": 1, "maybe": 42}"#);
        assert_eq!(
            block_on(OptFields::deserialize(&mut de)),
            Ok(Probe::Hit(OptFields {
                required: 1,
                maybe: Some(42)
            })),
        );
    }

    #[test]
    fn derive_option_null() {
        let mut de = JsonDeserializer::new(br#"{"required": 1, "maybe": null}"#);
        assert_eq!(
            block_on(OptFields::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"first": 10, "second": true}"#);
        assert_eq!(
            block_on(<Pair<i64, bool>>::deserialize(&mut de)),
            Ok(Probe::Hit(Pair {
                first: 10,
                second: true
            })),
        );
    }

    #[test]
    fn derive_generic_different_instantiation() {
        let mut de = JsonDeserializer::new(br#"{"first": false, "second": 42}"#);
        assert_eq!(
            block_on(<Pair<bool, u32>>::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"inner": 99}"#);
        assert_eq!(
            block_on(<GenericWrapper<i64>>::deserialize(&mut de)),
            Ok(Probe::Hit(GenericWrapper { inner: 99 })),
        );
    }

    #[test]
    fn derive_generic_nested() {
        // GenericWrapper<Point> — generic struct containing a non-generic derived struct
        let mut de = JsonDeserializer::new(br#"{"inner": {"x": 5, "y": 6}}"#);
        assert_eq!(
            block_on(<GenericWrapper<Point>>::deserialize(&mut de)),
            Ok(Probe::Hit(GenericWrapper {
                inner: Point { x: 5, y: 6 }
            })),
        );
    }

    #[test]
    fn derive_generic_nested_generic() {
        // GenericWrapper<GenericWrapper<bool>> — nested generics
        let mut de = JsonDeserializer::new(br#"{"inner": {"inner": true}}"#);
        assert_eq!(
            block_on(<GenericWrapper<GenericWrapper<bool>>>::deserialize(&mut de)),
            Ok(Probe::Hit(GenericWrapper {
                inner: GenericWrapper { inner: true }
            })),
        );
    }

    #[test]
    fn derive_generic_with_option() {
        let mut de = JsonDeserializer::new(br#"{"inner": null}"#);
        assert_eq!(
            block_on(<GenericWrapper<Option<i64>>>::deserialize(&mut de)),
            Ok(Probe::Hit(GenericWrapper { inner: None })),
        );

        let mut de = JsonDeserializer::new(br#"{"inner": 7}"#);
        assert_eq!(
            block_on(<GenericWrapper<Option<i64>>>::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"label": "test", "value": 42}"#);
        assert_eq!(
            block_on(<WithLifetimeAndGeneric<i64>>::deserialize(&mut de)),
            Ok(Probe::Hit(WithLifetimeAndGeneric {
                label: "test",
                value: 42
            })),
        );
    }

    // ---- derive: enum — unit-only --------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    #[test]
    fn derive_enum_unit_variant() {
        let mut de = JsonDeserializer::new(br#""Red""#);
        assert_eq!(
            block_on(Color::deserialize(&mut de)),
            Ok(Probe::Hit(Color::Red)),
        );
    }

    #[test]
    fn derive_enum_unit_variant_other() {
        let mut de = JsonDeserializer::new(br#""Blue""#);
        assert_eq!(
            block_on(Color::deserialize(&mut de)),
            Ok(Probe::Hit(Color::Blue)),
        );
    }

    #[test]
    fn derive_enum_unit_unknown_variant_is_miss() {
        let mut de = JsonDeserializer::new(br#""Yellow""#);
        assert_eq!(block_on(Color::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    #[test]
    fn derive_enum_unit_non_string_is_miss() {
        let mut de = JsonDeserializer::new(b"42");
        assert_eq!(block_on(Color::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    // ---- derive: enum — mixed (unit + newtype + struct) ----------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Shape {
        Circle,
        Square(Point),
        Rect { origin: Point, size: Point },
    }

    #[test]
    fn derive_enum_mixed_unit_variant() {
        let mut de = JsonDeserializer::new(br#""Circle""#);
        assert_eq!(
            block_on(Shape::deserialize(&mut de)),
            Ok(Probe::Hit(Shape::Circle)),
        );
    }

    #[test]
    fn derive_enum_newtype_variant() {
        let mut de = JsonDeserializer::new(br#"{"Square": {"x": 1, "y": 2}}"#);
        assert_eq!(
            block_on(Shape::deserialize(&mut de)),
            Ok(Probe::Hit(Shape::Square(Point { x: 1, y: 2 }))),
        );
    }

    #[test]
    fn derive_enum_struct_variant() {
        let mut de = JsonDeserializer::new(
            br#"{"Rect": {"origin": {"x": 0, "y": 0}, "size": {"x": 10, "y": 20}}}"#,
        );
        assert_eq!(
            block_on(Shape::deserialize(&mut de)),
            Ok(Probe::Hit(Shape::Rect {
                origin: Point { x: 0, y: 0 },
                size: Point { x: 10, y: 20 },
            })),
        );
    }

    #[test]
    fn derive_enum_unknown_variant_map_is_miss() {
        let mut de = JsonDeserializer::new(br#"{"Triangle": {"x": 1}}"#);
        assert_eq!(block_on(Shape::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    #[test]
    fn derive_enum_empty_map_is_miss() {
        let mut de = JsonDeserializer::new(br#"{}"#);
        assert_eq!(block_on(Shape::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    // ---- derive: enum — newtype-only (no unit variants) ---------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Value {
        Int(i64),
        Bool(bool),
    }

    #[test]
    fn derive_enum_newtype_only() {
        let mut de = JsonDeserializer::new(br#"{"Int": 42}"#);
        assert_eq!(
            block_on(Value::deserialize(&mut de)),
            Ok(Probe::Hit(Value::Int(42))),
        );
    }

    #[test]
    fn derive_enum_newtype_only_other() {
        let mut de = JsonDeserializer::new(br#"{"Bool": true}"#);
        assert_eq!(
            block_on(Value::deserialize(&mut de)),
            Ok(Probe::Hit(Value::Bool(true))),
        );
    }

    // ---- derive: enum — generics --------------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Either<A, B> {
        Left(A),
        Right(B),
    }

    #[test]
    fn derive_enum_generic_left() {
        let mut de = JsonDeserializer::new(br#"{"Left": 42}"#);
        assert_eq!(
            block_on(<Either<i64, bool>>::deserialize(&mut de)),
            Ok(Probe::Hit(Either::Left(42))),
        );
    }

    #[test]
    fn derive_enum_generic_right() {
        let mut de = JsonDeserializer::new(br#"{"Right": true}"#);
        assert_eq!(
            block_on(<Either<i64, bool>>::deserialize(&mut de)),
            Ok(Probe::Hit(Either::Right(true))),
        );
    }

    // ---- derive: enum — tuple variants ---------------------------------------

    #[derive(strede_derive::Deserialize, Debug, PartialEq)]
    enum Geom {
        Point(i64, i64),
        Color(u8, u8, u8),
        Single(bool),
        Unit,
    }

    #[test]
    fn derive_enum_tuple_variant_two_fields() {
        let mut de = JsonDeserializer::new(br#"{"Point": [3, 4]}"#);
        assert_eq!(
            block_on(Geom::deserialize(&mut de)),
            Ok(Probe::Hit(Geom::Point(3, 4))),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_three_fields() {
        let mut de = JsonDeserializer::new(br#"{"Color": [255, 128, 0]}"#);
        assert_eq!(
            block_on(Geom::deserialize(&mut de)),
            Ok(Probe::Hit(Geom::Color(255, 128, 0))),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_mixed_with_unit() {
        let mut de = JsonDeserializer::new(br#""Unit""#);
        assert_eq!(
            block_on(Geom::deserialize(&mut de)),
            Ok(Probe::Hit(Geom::Unit)),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_mixed_with_newtype() {
        let mut de = JsonDeserializer::new(br#"{"Single": true}"#);
        assert_eq!(
            block_on(Geom::deserialize(&mut de)),
            Ok(Probe::Hit(Geom::Single(true))),
        );
    }

    #[test]
    fn derive_enum_tuple_variant_wrong_length_is_miss() {
        // Array with too many elements
        let mut de = JsonDeserializer::new(br#"{"Point": [1, 2, 3]}"#);
        assert_eq!(block_on(Geom::deserialize(&mut de)), Ok(Probe::Miss),);
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
        let mut de = JsonDeserializer::new(br#"{"required": 1}"#);
        assert_eq!(
            block_on(WithDefaults::deserialize(&mut de)),
            Ok(Probe::Hit(WithDefaults {
                required: 1,
                count: 99,
                flag: false
            })),
        );
    }

    #[test]
    fn derive_default_fields_present_overrides() {
        let mut de = JsonDeserializer::new(br#"{"required": 1, "count": 5, "flag": true}"#);
        assert_eq!(
            block_on(WithDefaults::deserialize(&mut de)),
            Ok(Probe::Hit(WithDefaults {
                required: 1,
                count: 5,
                flag: true
            })),
        );
    }

    #[test]
    fn derive_default_required_missing_is_miss() {
        let mut de = JsonDeserializer::new(br#"{"count": 5}"#);
        assert_eq!(
            block_on(WithDefaults::deserialize(&mut de)),
            Ok(Probe::Miss),
        );
    }

    // ---- derive: deserialize_with attribute ------------------------------------

    /// Custom deserializer: reads an i64 and doubles it.
    async fn double_i64<'de, D: strede::Deserializer<'de>>(
        d: &mut D,
    ) -> Result<Probe<i64>, D::Error>
    where
        D::Error: strede::DeserializeError,
    {
        d.next(|[e]| async move {
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
        let mut de = JsonDeserializer::new(br#"{"normal": 5, "doubled": 3}"#);
        assert_eq!(
            block_on(WithCustomDe::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"required": 1}"#);
        assert_eq!(
            block_on(WithSkip::deserialize(&mut de)),
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
        let mut de =
            JsonDeserializer::new(br#"{"required": 1, "skipped": 99, "skipped_trait": true}"#);
        assert_eq!(block_on(WithSkip::deserialize(&mut de)), Ok(Probe::Miss),);
    }

    #[test]
    fn derive_skip_mixed_with_required() {
        let mut de = JsonDeserializer::new(br#"{"required": 7}"#);
        assert_eq!(
            block_on(WithSkip::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#"{"type": 42, "value": true}"#);
        assert_eq!(
            block_on(RenamedFields::deserialize(&mut de)),
            Ok(Probe::Hit(RenamedFields {
                kind: 42,
                val: true
            })),
        );
    }

    #[test]
    fn derive_rename_original_name_is_miss() {
        let mut de = JsonDeserializer::new(br#"{"kind": 42, "val": true}"#);
        assert_eq!(
            block_on(RenamedFields::deserialize(&mut de)),
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
        let mut de = JsonDeserializer::new(br#""circle""#);
        assert_eq!(
            block_on(RenamedVariants::deserialize(&mut de)),
            Ok(Probe::Hit(RenamedVariants::Circle)),
        );
    }

    #[test]
    fn derive_rename_newtype_variant() {
        let mut de = JsonDeserializer::new(br#"{"rect": 5}"#);
        assert_eq!(
            block_on(RenamedVariants::deserialize(&mut de)),
            Ok(Probe::Hit(RenamedVariants::Rect(5))),
        );
    }

    #[test]
    fn derive_rename_original_variant_name_is_miss() {
        let mut de = JsonDeserializer::new(br#""Circle""#);
        assert_eq!(
            block_on(RenamedVariants::deserialize(&mut de)),
            Ok(Probe::Miss),
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
        let mut de = JsonDeserializer::new(b"42");
        assert_eq!(
            block_on(Untagged::deserialize(&mut de)),
            Ok(Probe::Hit(Untagged::Num(42))),
        );
    }

    #[test]
    fn derive_untagged_newtype_second_match() {
        let mut de = JsonDeserializer::new(b"true");
        assert_eq!(
            block_on(Untagged::deserialize(&mut de)),
            Ok(Probe::Hit(Untagged::Flag(true))),
        );
    }

    #[test]
    fn derive_untagged_tuple_variant() {
        let mut de = JsonDeserializer::new(b"[1, 2]");
        assert_eq!(
            block_on(Untagged::deserialize(&mut de)),
            Ok(Probe::Hit(Untagged::Pt(1, 2))),
        );
    }

    #[test]
    fn derive_untagged_struct_variant() {
        let mut de = JsonDeserializer::new(br#"{"x": 7}"#);
        assert_eq!(
            block_on(Untagged::deserialize(&mut de)),
            Ok(Probe::Hit(Untagged::Named { x: 7 })),
        );
    }

    #[test]
    fn derive_untagged_all_miss() {
        let mut de = JsonDeserializer::new(br#""hello""#);
        assert_eq!(block_on(Untagged::deserialize(&mut de)), Ok(Probe::Miss),);
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
        let mut de = JsonDeserializer::new(br#""Ping""#);
        assert_eq!(
            block_on(MixedTagged::deserialize(&mut de)),
            Ok(Probe::Hit(MixedTagged::Ping)),
        );
    }

    #[test]
    fn derive_mixed_tagged_newtype() {
        let mut de = JsonDeserializer::new(br#"{"Data": 42}"#);
        assert_eq!(
            block_on(MixedTagged::deserialize(&mut de)),
            Ok(Probe::Hit(MixedTagged::Data(42))),
        );
    }

    #[test]
    fn derive_mixed_untagged_fallback() {
        let mut de = JsonDeserializer::new(b"true");
        assert_eq!(
            block_on(MixedTagged::deserialize(&mut de)),
            Ok(Probe::Hit(MixedTagged::Fallback(true))),
        );
    }

    // ---- select_probe! ------------------------------------------------------

    #[test]
    fn select_probe_first_arm_hits() {
        let mut de = JsonDeserializer::new(b"42");
        let result: Result<Probe<i64>, JsonError> = block_on(async {
            de.next(|[e1, e2]| async move {
                strede::select_probe! {
                    (claim, v) = e1.deserialize_i64() => Ok(Probe::Hit((claim, v))),
                    (claim, v) = e2.deserialize_bool() => Ok(Probe::Hit((claim, if v { 1i64 } else { 0 }))),
                }
            }).await
        });
        assert_eq!(result, Ok(Probe::Hit(42)));
    }

    #[test]
    fn select_probe_second_arm_hits() {
        let mut de = JsonDeserializer::new(b"true");
        let result: Result<Probe<i64>, JsonError> = block_on(async {
            de.next(|[e1, e2]| async move {
                strede::select_probe! {
                    (claim, v) = e1.deserialize_i64() => Ok(Probe::Hit((claim, v))),
                    (claim, v) = e2.deserialize_bool() => Ok(Probe::Hit((claim, if v { 1i64 } else { 0 }))),
                }
            }).await
        });
        assert_eq!(result, Ok(Probe::Hit(1)));
    }

    #[test]
    fn select_probe_all_miss_default() {
        let mut de = JsonDeserializer::new(br#""hello""#);
        let result: Result<Probe<i64>, JsonError> = block_on(async {
            de.next(|[e1, e2]| async move {
                strede::select_probe! {
                    (claim, v) = e1.deserialize_i64() => Ok(Probe::Hit((claim, v))),
                    (claim, v) = e2.deserialize_bool() => Ok(Probe::Hit((claim, if v { 1i64 } else { 0 }))),
                }
            }).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn select_probe_all_miss_with_handler() {
        let mut de = JsonDeserializer::new(br#""hello""#);
        let result: Result<Probe<i64>, JsonError> = block_on(async {
            de.next(|[e1, e2]| async move {
                strede::select_probe! {
                    (claim, v) = e1.deserialize_i64() => Ok(Probe::Hit((claim, v))),
                    (claim, v) = e2.deserialize_bool() => Ok(Probe::Hit((claim, if v { 1i64 } else { 0 }))),
                    miss => Ok(Probe::Miss),
                }
            }).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn select_probe_single_arm() {
        let mut de = JsonDeserializer::new(b"false");
        let result: Result<Probe<bool>, JsonError> = block_on(async {
            de.next(|[e]| async move {
                strede::select_probe! {
                    (claim, v) = e.deserialize_bool() => Ok(Probe::Hit((claim, v))),
                }
            })
            .await
        });
        assert_eq!(result, Ok(Probe::Hit(false)));
    }

    #[test]
    fn select_probe_three_arms() {
        let mut de = JsonDeserializer::new(br#""test""#);
        #[derive(Debug, PartialEq)]
        enum Val<'a> {
            Num(i64),
            Bool(bool),
            Str(&'a str),
        }

        let result: Result<Probe<Val>, JsonError> = block_on(async {
            de.next(|[e1, e2, e3]| async move {
                strede::select_probe! {
                    (claim, v) = e1.deserialize_i64()  => Ok(Probe::Hit((claim, Val::Num(v)))),
                    (claim, v) = e2.deserialize_bool() => Ok(Probe::Hit((claim, Val::Bool(v)))),
                    (claim, v) = e3.deserialize_str()  => Ok(Probe::Hit((claim, Val::Str(v)))),
                }
            })
            .await
        });
        assert_eq!(result, Ok(Probe::Hit(Val::Str("test"))));
    }
}
