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
    BytesAccess, Chunk, Deserialize, DeserializeFromMap, DeserializeFromSeq, Deserializer, Entry,
    MapAccess, MapArmStack, MapKeyClaim, MapKeyProbe, MapValueClaim, MapValueProbe, NumberAccess,
    Probe, SeqAccess, SeqEntry, StrAccess, hit, utils::repeat,
};

// ---------------------------------------------------------------------------
// JsonClaim - unified proof-of-consumption type
// ---------------------------------------------------------------------------

/// Carries the post-consumption tokenizer and source state.
/// Returned from every probe method and threaded back to [`Deserializer::entry`]
/// (or [`MapAccess::iterate`] / [`SeqAccess::next`]) to advance the stream.
pub struct JsonClaim<'de> {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) src: &'de [u8],
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
pub struct JsonSubDeserializer<'de> {
    src: &'de [u8],
    /// Buffer position *before* the leading-token bytes were consumed. Used
    /// by [`Self::into_raw_source`] so raw-value types can recover the full
    /// span of the pending value (including any leading punctuation like
    /// `{`, `[`, `"`, or a sign character on a number).
    start_src: &'de [u8],
    pending_tok: Token,
}

impl<'de> JsonSubDeserializer<'de> {
    #[inline(always)]
    pub(crate) fn new(src: &'de [u8], start_src: &'de [u8], tok: Token) -> Self {
        Self {
            src,
            start_src,
            pending_tok: tok,
        }
    }

    /// Hand out the raw-value source state and the pre-loaded token. Returns
    /// `(start_src, src, token)` where `start_src` is the buffer position
    /// before the leading token bytes were consumed (so `&start_src[..n]`
    /// covers the full pending value once `n` bytes are skipped) and `src`
    /// is the post-leading-token position the tokenizer in `token` expects
    /// to be fed.
    #[cfg(feature = "alloc")]
    #[inline(always)]
    pub(crate) fn into_raw_source(self) -> (&'de [u8], &'de [u8], Token) {
        (self.start_src, self.src, self.pending_tok)
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
    let start_src = de.src;
    let token = de.next_dispatch()?;
    let entry = JsonEntry {
        token: token.clone(),
        src: de.src,
        start_src,
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
            start_src: self.start_src,
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
    /// Buffer position before the leading-token bytes were consumed; threaded
    /// into [`JsonSubDeserializer`] when spawning a sub-deserializer so
    /// raw-value capture can recover the full pending-value span.
    start_src: &'de [u8],
}

impl<'de> JsonEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            token: self.token.clone(),
            src: self.src,
            start_src: self.start_src,
        }
    }
}

/// Skip one complete JSON value given its leading token.
/// Returns the tokenizer state positioned after the skipped value.
pub(crate) fn skip_value(src: &mut &[u8], tok: Token) -> Result<Tokenizer, JsonError> {
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
    type SubDeserializer = JsonSubDeserializer<'de>;
    type StrChunks = JsonStrAccess<'de>;
    type BytesChunks = JsonBytesAccess<'de>;
    type NumberChunks = JsonNumberAccess<'de>;
    type Map = JsonMapAccess<'de>;
    type Seq = JsonSeqAccess<'de>;

    fn fork(&mut self) -> Self {
        self.clone()
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

    // ---- Number chunks -----------------------------------------------------

    async fn deserialize_number_chunks(self) -> Result<Probe<Self::NumberChunks>, Self::Error> {
        match self.token {
            Token::Number(access) => Ok(Probe::Hit(JsonNumberAccess {
                access,
                src: self.src,
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
    async fn deserialize_option<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        match self.token {
            Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit((
                JsonClaim {
                    tokenizer: tok,
                    src: self.src,
                },
                None,
            ))),
            other => {
                let sub = JsonSubDeserializer::new(self.src, self.start_src, other);
                let (claim, v) = hit!(T::deserialize(sub, extra).await);
                Ok(Probe::Hit((claim, Some(v))))
            }
        }
    }

    // ---- Value (spawn SubDeserializer and delegate) ------------------------
    #[inline(always)]
    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = JsonSubDeserializer::new(self.src, self.start_src, self.token);
        T::deserialize(sub, extra).await
    }

    // ---- Map / Seq forwarding ----------------------------------------------
    #[inline(always)]
    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>,
    {
        let map = match Entry::deserialize_map(self).await? {
            Probe::Hit(m) => m,
            Probe::Miss => return Ok(Probe::Miss),
        };
        T::deserialize_from_map(map, extra).await
    }

    #[inline(always)]
    async fn deserialize_seq_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeq<'de, Self::Seq>,
    {
        let seq = match Entry::deserialize_seq(self).await? {
            Probe::Hit(s) => s,
            Probe::Miss => return Ok(Probe::Miss),
        };
        T::deserialize_from_seq(seq, extra).await
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
// JsonNumberAccess
// ---------------------------------------------------------------------------

pub struct JsonNumberAccess<'de> {
    access: crate::token::NumberAccess,
    src: &'de [u8],
}

impl<'de> NumberAccess for JsonNumberAccess<'de> {
    type Claim = JsonClaim<'de>;
    type Error = JsonError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            access: self.access,
            src: self.src,
        }
    }

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.access.next_chunk(&mut self.src) {
            Ok(Some(chunk)) => Ok(Chunk::Data((self, f(chunk)))),
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
    start_src: &'de [u8],
    key_tok: Token,
}

impl<'de> JsonMapKeyProbe<'de> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            start_src: self.start_src,
            key_tok: self.key_tok.clone(),
        }
    }
}

impl<'de> MapKeyProbe<'de> for JsonMapKeyProbe<'de> {
    type Error = JsonError;
    type KeyClaim = JsonClaim<'de>;
    type KeySubDeserializer = JsonSubDeserializer<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: Deserialize<'de, Self::KeySubDeserializer>,
    {
        let sub = JsonSubDeserializer::new(self.src, self.start_src, self.key_tok);
        K::deserialize(sub, extra).await
    }
}

// MapKeyClaim is implemented for JsonClaim — see below at the JsonClaim impl block.

/// Value probe: positioned at the start of the value token.
pub struct JsonMapValueProbe<'de> {
    src: &'de [u8],
    start_src: &'de [u8],
    value_tok: Token,
}

impl<'de> JsonMapValueProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            start_src: self.start_src,
            value_tok: self.value_tok.clone(),
        }
    }
}

impl<'de> MapValueProbe<'de> for JsonMapValueProbe<'de> {
    type Error = JsonError;
    type MapClaim = JsonClaim<'de>;
    type ValueClaim = JsonClaim<'de>;
    type ValueSubDeserializer = JsonSubDeserializer<'de>;
    type ValueMap = JsonMapAccess<'de>;
    type ValueSeq = JsonSeqAccess<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    #[inline(always)]
    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: Deserialize<'de, Self::ValueSubDeserializer>,
    {
        let sub = JsonSubDeserializer::new(self.src, self.start_src, self.value_tok);
        V::deserialize(sub, extra).await
    }

    #[inline(always)]
    async fn deserialize_map_into<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeFromMap<'de, Self::ValueMap>,
    {
        match self.value_tok {
            Token::Simple(SimpleToken::ObjectStart, tok) => {
                let map = JsonMapAccess {
                    tokenizer: tok,
                    src: self.src,
                    first: true,
                };
                V::deserialize_from_map(map, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_seq_into<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeFromSeq<'de, Self::ValueSeq>,
    {
        match self.value_tok {
            Token::Simple(SimpleToken::ArrayStart, tok) => {
                let seq = JsonSeqAccess {
                    tokenizer: tok,
                    src: self.src,
                    first: true,
                };
                V::deserialize_from_seq(seq, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let mut src = self.src;
        let tok = skip_value(&mut src, self.value_tok)?;
        Ok(JsonClaim {
            tokenizer: tok,
            src,
        })
    }
}

// MapKeyClaim and MapValueClaim are both implemented on JsonClaim (the unified claim).
impl<'de> MapKeyClaim<'de> for JsonClaim<'de> {
    type Error = JsonError;
    type MapClaim = JsonClaim<'de>;
    type ValueProbe = JsonMapValueProbe<'de>;

    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        let mut src = self.src;
        let colon_new_tok = match self.tokenizer.next_token(&mut src) {
            Ok(Token::Simple(SimpleToken::Colon, new_tok)) => new_tok,
            Ok(_) => return Err(JsonError::UnexpectedByte { byte: 0 }),
            Err(e) => return Err(e),
        };
        let start_src = src;
        let value_tok = colon_new_tok.next_token(&mut src)?;
        Ok(JsonMapValueProbe {
            src,
            start_src,
            value_tok,
        })
    }
}

impl<'de> MapValueClaim<'de> for JsonClaim<'de> {
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
                let start_src = src;
                match new_tok.next_token(&mut src) {
                    Ok(Token::Simple(SimpleToken::ObjectEnd, end_tok)) => {
                        Ok(strede::NextKey::Done(JsonClaim {
                            tokenizer: end_tok,
                            src,
                        }))
                    }
                    Ok(key_tok) => Ok(strede::NextKey::Entry(JsonMapKeyProbe {
                        src,
                        start_src,
                        key_tok,
                    })),
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
            let start_src = src;
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
                Ok(key_tok) => Some(JsonMapKeyProbe {
                    src,
                    start_src,
                    key_tok,
                }),
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
    let start_src = seq.src;
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
        start_src,
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
    start_src: &'de [u8],
    elem_tok: Token,
}

impl<'de> JsonSeqEntry<'de> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            start_src: self.start_src,
            elem_tok: self.elem_tok.clone(),
        }
    }
}

impl<'de> SeqEntry<'de> for JsonSeqEntry<'de> {
    type Error = JsonError;
    type Claim = JsonClaim<'de>;
    type SubDeserializer = JsonSubDeserializer<'de>;
    type Map = JsonMapAccess<'de>;
    type Seq = JsonSeqAccess<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    #[inline(always)]
    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = JsonSubDeserializer::new(self.src, self.start_src, self.elem_tok);
        T::deserialize(sub, extra).await
    }

    #[inline(always)]
    async fn get_map_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>,
    {
        match self.elem_tok {
            Token::Simple(SimpleToken::ObjectStart, tok) => {
                let map = JsonMapAccess {
                    tokenizer: tok,
                    src: self.src,
                    first: true,
                };
                T::deserialize_from_map(map, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn get_seq_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeq<'de, Self::Seq>,
    {
        match self.elem_tok {
            Token::Simple(SimpleToken::ArrayStart, tok) => {
                let seq = JsonSeqAccess {
                    tokenizer: tok,
                    src: self.src,
                    first: true,
                };
                T::deserialize_from_seq(seq, extra).await
            }
            _ => Ok(Probe::Miss),
        }
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

// ---------------------------------------------------------------------------
// Per-format primitive Deserialize impls
//
// Both `JsonDeserializer` and `JsonSubDeserializer` are local types, so both
// are covered here. The orphan rule is satisfied for either.
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_bool {
    ($($de:ty),*) => {
        $(impl<'de> Deserialize<'de, $de> for bool {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
                d.entry(|[e]| async move {
                    match e.token {
                        Token::Simple(SimpleToken::Bool(b), tok) => Ok(Probe::Hit((
                            JsonClaim {
                                tokenizer: tok,
                                src: e.src,
                            },
                            b,
                        ))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        })*
    };
}
impl_deserialize_bool!(JsonDeserializer<'de>, JsonSubDeserializer<'de>);

macro_rules! impl_deserialize_num {
    ($($t:ty),*) => {
        $(impl<'de> Deserialize<'de, JsonDeserializer<'de>> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: JsonDeserializer<'de>,
                _: (),
            ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        }
        impl<'de> Deserialize<'de, JsonSubDeserializer<'de>> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: JsonSubDeserializer<'de>,
                _: (),
            ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })*
    };
}
impl_deserialize_num!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

#[cfg(feature = "arbitrary_precision")]
macro_rules! impl_deserialize_number_borrowed {
    ($($de:ty),*) => {
        $(impl<'de> Deserialize<'de, $de> for crate::number::NumberBorrowed<'de> {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
                d.entry(|[e]| async move {
                    match e.token {
                        Token::Number(mut access) => {
                            let mut src = e.src;
                            while access.next_chunk(&mut src)?.is_some() {}
                            let consumed = e.start_src.len() - src.len();
                            let raw = core::str::from_utf8(&e.start_src[..consumed])
                                .map_err(|_| JsonError::InvalidNumber)?;
                            Ok(Probe::Hit((
                                JsonClaim {
                                    tokenizer: Tokenizer::new(),
                                    src,
                                },
                                crate::number::NumberBorrowed { raw },
                            )))
                        }
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        })*
    };
}

#[cfg(feature = "arbitrary_precision")]
impl_deserialize_number_borrowed!(JsonDeserializer<'de>, JsonSubDeserializer<'de>);

macro_rules! impl_deserialize_char {
    ($($de:ty),*) => {
        $(impl<'de> Deserialize<'de, $de> for char {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
                d.entry(|[e]| async move {
                    let (claim, s) = hit!(e.deserialize_str().await);
                    let mut chars = s.chars();
                    match (chars.next(), chars.next()) {
                        (Some(c), None) => Ok(Probe::Hit((claim, c))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        })*
    };
}
impl_deserialize_char!(JsonDeserializer<'de>, JsonSubDeserializer<'de>);

macro_rules! impl_deserialize_unit {
    ($($de:ty),*) => {
        $(impl<'de> Deserialize<'de, $de> for () {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(JsonClaim<'de>, Self)>, JsonError> {
                d.entry(|[e]| async move {
                    match e.token {
                        Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit((
                            JsonClaim {
                                tokenizer: tok,
                                src: e.src,
                            },
                            (),
                        ))),
                        _ => Ok(Probe::Miss),
                    }
                })
                .await
            }
        })*
    };
}
impl_deserialize_unit!(JsonDeserializer<'de>, JsonSubDeserializer<'de>);
