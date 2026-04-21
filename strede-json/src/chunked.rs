//! Chunked JSON deserializer for async streaming input.
//!
//! Uses [`strede::SharedBuf`]/[`strede::Handle`] to coordinate access to a
//! buffer that is refilled asynchronously by a user-supplied loader closure.
//! The deserializer holds either a `&mut SharedBuf` (top-level) or an
//! `Option<Handle>` (sub-deserializer for option/map values/seq elements).
//!
//! # Capabilities vs. [`crate::JsonDeserializer`]
//!
//! - Implements the **owned** trait family only ([`DeserializerOwned`],
//!   [`EntryOwned`], etc.). Zero-copy `&'de str` / `&'de [u8]` borrowing is
//!   unsupported by design - buffer-chunk lifetimes are shorter than the
//!   caller-facing deserialization session, so there is no `'de` to borrow
//!   for. Use [`EntryOwned::deserialize_str_chunks`] /
//!   [`EntryOwned::deserialize_bytes_chunks`] for string and byte data.
//! - `#[derive(Deserialize)]` today emits a borrow-family impl; types meant
//!   for use with this deserializer must hand-roll a [`DeserializeOwned`]
//!   impl (typically using `deserialize_str_chunks` for keys).
//! - Sub-deserializer Miss recovery is best-effort: if a sub-probe partially
//!   consumed the buffer (advanced past chunk boundaries) and then missed,
//!   the parent's replay state may be inconsistent. Common cases (sync probes
//!   that don't advance) work correctly.
//! - Loader errors must be communicated via the `Buffer` value (e.g. an empty
//!   slice signals EOF). The `F: AsyncFnMut(&mut B)` signature does not
//!   support `Result`.

use crate::JsonError;
use crate::token::{self, SimpleToken, StrChunk, Token, Tokenizer};
use core::future::Future;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeOwned, DeserializerOwned, EntryOwned, Handle,
    MapAccessOwned, MapArmStackOwned, MapKeyClaimOwned, MapKeyProbeOwned, MapValueClaimOwned,
    MapValueProbeOwned, NextKey, Probe, SeqAccessOwned, SeqEntryOwned, SharedBuf, StrAccessOwned,
    hit,
};

// ---------------------------------------------------------------------------
// Claim
// ---------------------------------------------------------------------------

/// Proof-of-consumption returned by every probe and threaded back to
/// [`DeserializerOwned::entry`] / [`MapAccessOwned::next_kv`] / [`SeqAccessOwned::next`].
///
/// For chunked input, this carries the post-token tokenizer, the chunk
/// offset, and (transferred via the [`Handle`] inside) ownership of buffer
/// access. The handle inside is `None` when the probe was synchronous and
/// dropped its handle naturally; in that case the parent re-forks from its
/// `SharedBuf`.
pub struct ChunkedJsonClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    tokenizer: Tokenizer,
    offset: usize,
    handle: Handle<'s, B, F>,
}

// ---------------------------------------------------------------------------
// Helpers - pointer arithmetic to recover offset after tokenizer advances `src`
// ---------------------------------------------------------------------------

#[inline]
fn new_offset(buf: &[u8], src: &[u8]) -> usize {
    let base = buf.as_ptr() as usize;
    let cur = src.as_ptr() as usize;
    debug_assert!(cur >= base);
    let off = cur - base;
    debug_assert!(off <= buf.len());
    off
}

// ---------------------------------------------------------------------------
// Deserializer
// ---------------------------------------------------------------------------

/// Top-level deserializer. Holds a `&mut SharedBuf` and forks fresh handles
/// per `next()`. Claim is `()` - the top-level is consumed and produces
/// nothing the caller needs to thread back.
pub struct ChunkedJsonDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: SharedBuf<'s, B, F>,
    tokenizer: Tokenizer,
    offset: usize,
    pending_tok: Option<Token>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedJsonDeserializer<'s, B, F> {
    pub fn new(shared: SharedBuf<'s, B, F>) -> Self {
        Self {
            shared,
            tokenizer: Tokenizer::new(),
            offset: 0,
            pending_tok: None,
        }
    }
}

/// Sub-deserializer for option / map value / seq element use where the inner
/// type calls `next()` once. Claim is `Self` - the sub-deserializer is
/// returned to the caller so it can extract handle/tokenizer/offset.
pub struct ChunkedJsonSubDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    tokenizer: Tokenizer,
    offset: usize,
    pending_tok: Option<Token>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedJsonSubDeserializer<'s, B, F> {
    fn new(handle: Handle<'s, B, F>, tokenizer: Tokenizer, offset: usize, tok: Token) -> Self {
        Self {
            handle,
            tokenizer,
            offset,
            pending_tok: Some(tok),
        }
    }
}

/// Advance `handle` to the next chunk via `Handle::next`, resetting `offset`
/// and verifying that the new chunk is non-empty (else `UnexpectedEnd`).
async fn refill<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<Handle<'s, B, F>, JsonError> {
    let new_h = handle.next().await;
    let empty = new_h.buf().is_empty();
    *offset = 0;
    if empty {
        return Err(JsonError::UnexpectedEnd);
    }
    Ok(new_h)
}

/// Read the next token from `handle`, advancing chunks as needed.
/// On return, `*tokenizer` and `*offset` reflect the post-token state.
async fn next_dispatch<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    tokenizer: &mut Tokenizer,
    offset: &mut usize,
    pending_tok: &mut Option<Token>,
) -> Result<(Handle<'s, B, F>, Token), JsonError> {
    if let Some(tok) = pending_tok.take() {
        return Ok((handle, tok));
    }
    loop {
        let result = {
            let buf = handle.buf();
            if *offset > buf.len() {
                // Defensive - shouldn't happen.
                *offset = buf.len();
            }
            let mut src: &[u8] = &buf[*offset..];
            let old = core::mem::replace(tokenizer, Tokenizer::new());
            let r = old.next_token(&mut src);
            *offset = new_offset(buf, src);
            r
        };
        match result {
            Ok(Token::Simple(SimpleToken::PartialLiteral, t)) | Ok(Token::NoTokens(t)) => {
                *tokenizer = t;
                handle = refill(handle, offset).await?;
            }
            Ok(t) => return Ok((handle, t)),
            Err(e) => return Err(e),
        }
    }
}

/// Build N entries by forking the main handle N-1 times.
fn build_entries<'s, B: Buffer, F: AsyncFnMut(&mut B), const N: usize>(
    mut main: Handle<'s, B, F>,
    token: Token,
    tokenizer: Tokenizer,
    offset: usize,
) -> [ChunkedJsonEntry<'s, B, F>; N] {
    core::array::from_fn(|_| ChunkedJsonEntry {
        handle: main.fork(),
        token: token.clone(),
        tokenizer: tokenizer.clone(),
        offset,
    })
}

/// Top-level `next`: forks a handle from SharedBuf, runs the closure, does
/// trailing-garbage check, returns `((), R)`.
async fn run_next_top<'s, B, F, const N: usize, Fn_, Fut, R>(
    de: ChunkedJsonDeserializer<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<((), R)>, JsonError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedJsonEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedJsonClaim<'s, B, F>, R)>, JsonError>>,
{
    let main = de.shared.fork();
    let mut tokenizer = de.tokenizer;
    let mut offset = de.offset;
    let mut pending_tok = de.pending_tok;

    let (main, token) = next_dispatch(main, &mut tokenizer, &mut offset, &mut pending_tok).await?;

    let snap_tokenizer = tokenizer.clone();
    let snap_offset = offset;

    let entries = build_entries::<B, F, N>(main, token.clone(), snap_tokenizer, snap_offset);

    let (claim, r) = hit!(f(entries).await);
    // Trailing-garbage check.
    let mut h = claim.handle;
    let mut off = claim.offset;
    loop {
        let buf = h.buf();
        let mut i = off;
        while i < buf.len() && matches!(buf[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        if i < buf.len() {
            return Err(JsonError::ExpectedEnd);
        }
        // Buffer exhausted; refill once and accept EOF.
        let new_h = h.next().await;
        if new_h.buf().is_empty() {
            break;
        }
        h = new_h;
        off = 0;
    }
    Ok(Probe::Hit(((), r)))
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned for ChunkedJsonDeserializer<'s, B, F> {
    type Error = JsonError;
    type Claim = ();
    type EntryClaim = ChunkedJsonClaim<'s, B, F>;
    type Entry = ChunkedJsonEntry<'s, B, F>;

    async fn entry<const N: usize, Fn_, Fut, R>(
        self,
        f: Fn_,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        Fn_: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(<Self::Entry as EntryOwned>::Claim, R)>, Self::Error>>,
    {
        run_next_top(self, f).await
    }
}

/// Sub-deserializer `next`: uses its owned handle, returns
/// `(ChunkedJsonSubDeserializer, R)` as claim.
async fn run_next_sub<'s, B, F, const N: usize, Fn_, Fut, R>(
    de: ChunkedJsonSubDeserializer<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<(ChunkedJsonSubDeserializer<'s, B, F>, R)>, JsonError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedJsonEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedJsonClaim<'s, B, F>, R)>, JsonError>>,
{
    let main = de.handle;
    let mut tokenizer = de.tokenizer;
    let mut offset = de.offset;
    let mut pending_tok = de.pending_tok;

    let (main, token) = next_dispatch(main, &mut tokenizer, &mut offset, &mut pending_tok).await?;

    let snap_tokenizer = tokenizer.clone();
    let snap_offset = offset;

    let entries = build_entries::<B, F, N>(main, token.clone(), snap_tokenizer, snap_offset);

    let (claim, r) = hit!(f(entries).await);
    Ok(Probe::Hit((
        ChunkedJsonSubDeserializer {
            handle: claim.handle,
            tokenizer: claim.tokenizer,
            offset: claim.offset,
            pending_tok: None,
        },
        r,
    )))
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedJsonSubDeserializer<'s, B, F>
{
    type Error = JsonError;
    type Claim = ChunkedJsonSubDeserializer<'s, B, F>;
    type EntryClaim = ChunkedJsonClaim<'s, B, F>;
    type Entry = ChunkedJsonEntry<'s, B, F>;

    async fn entry<const N: usize, Fn_, Fut, R>(
        self,
        f: Fn_,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        Fn_: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(<Self::Entry as EntryOwned>::Claim, R)>, Self::Error>>,
    {
        run_next_sub(self, f).await
    }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

pub struct ChunkedJsonEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    token: Token,
    tokenizer: Tokenizer,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedJsonEntry<'s, B, F> {
    fn into_claim_with_tok(
        handle: Handle<'s, B, F>,
        offset: usize,
        tokenizer: Tokenizer,
    ) -> ChunkedJsonClaim<'s, B, F> {
        ChunkedJsonClaim {
            tokenizer,
            offset,
            handle,
        }
    }

    /// Stack-buffer based number parsing for chunked input.
    /// Accumulates digit chunks across chunk boundaries.
    async fn parse_num<T: ParseNum>(
        mut self,
    ) -> Result<Probe<(ChunkedJsonClaim<'s, B, F>, T)>, JsonError> {
        let Token::Number(mut access) = self.token else {
            return Ok(Probe::Miss);
        };
        let mut scratch = [0u8; 64];
        let mut len = 0usize;
        loop {
            let result = {
                let buf = self.handle.buf();
                let mut src: &[u8] = &buf[self.offset..];
                let r = access.next_chunk(&mut src);
                self.offset = new_offset(buf, src);
                r
            };
            match result {
                Ok(Some(chunk)) => {
                    let bytes = chunk.as_bytes();
                    if len + bytes.len() > scratch.len() {
                        return Err(JsonError::InvalidNumber);
                    }
                    scratch[len..len + bytes.len()].copy_from_slice(bytes);
                    len += bytes.len();
                }
                Ok(None) => break,
                Err(JsonError::UnexpectedEnd) => {
                    self.handle = refill(self.handle, &mut self.offset).await?;
                }
                Err(e) => return Err(e),
            }
        }
        let s = core::str::from_utf8(&scratch[..len]).map_err(|_| JsonError::InvalidNumber)?;
        let value = T::parse(s)?;
        let claim = ChunkedJsonClaim {
            tokenizer: Tokenizer::new(),
            offset: self.offset,
            handle: self.handle,
        };
        Ok(Probe::Hit((claim, value)))
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

/// Drain a chunked string access, discarding all chunks.
async fn drain_str_chunked<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    access: &mut token::StrAccess,
    offset: &mut usize,
) -> Result<Handle<'s, B, F>, JsonError> {
    loop {
        let result = {
            let buf = handle.buf();
            let mut src: &[u8] = &buf[*offset..];
            let r = access.next_chunk(&mut src);
            *offset = new_offset(buf, src);
            r
        };
        match result {
            Ok(Some(_)) => {}
            Ok(None) => return Ok(handle),
            Err(JsonError::UnexpectedEnd) => {
                handle = refill(handle, offset).await?;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Drain a chunked number access, discarding all chunks.
async fn drain_num_chunked<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    access: &mut token::NumberAccess,
    offset: &mut usize,
) -> Result<Handle<'s, B, F>, JsonError> {
    loop {
        let result = {
            let buf = handle.buf();
            let mut src: &[u8] = &buf[*offset..];
            let r = access.next_chunk(&mut src);
            *offset = new_offset(buf, src);
            r
        };
        match result {
            Ok(Some(_)) => {}
            Ok(None) => return Ok(handle),
            Err(JsonError::UnexpectedEnd) => {
                handle = refill(handle, offset).await?;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Skip one complete JSON value (scalar, string, object, or array) in chunked
/// mode. Uses iterative depth tracking instead of recursion to stay no_std.
async fn skip_value_chunked<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    tokenizer: &mut Tokenizer,
    offset: &mut usize,
    tok: Token,
) -> Result<Handle<'s, B, F>, JsonError> {
    match tok {
        Token::Simple(SimpleToken::Null | SimpleToken::Bool(_), t) => {
            *tokenizer = t;
            Ok(handle)
        }
        Token::Number(mut access) => {
            handle = drain_num_chunked(handle, &mut access, offset).await?;
            *tokenizer = Tokenizer::new();
            Ok(handle)
        }
        Token::Str(mut access) => {
            handle = drain_str_chunked(handle, &mut access, offset).await?;
            *tokenizer = Tokenizer::new();
            Ok(handle)
        }
        Token::Simple(SimpleToken::ObjectStart | SimpleToken::ArrayStart, t) => {
            *tokenizer = t;
            let mut depth = 1usize;
            while depth > 0 {
                let mut pending = None;
                let (h, tok) = next_dispatch(handle, tokenizer, offset, &mut pending).await?;
                handle = h;
                match tok {
                    Token::Simple(SimpleToken::ObjectStart | SimpleToken::ArrayStart, t) => {
                        *tokenizer = t;
                        depth += 1;
                    }
                    Token::Simple(SimpleToken::ObjectEnd | SimpleToken::ArrayEnd, t) => {
                        *tokenizer = t;
                        depth -= 1;
                    }
                    Token::Simple(
                        SimpleToken::Null
                        | SimpleToken::Bool(_)
                        | SimpleToken::Comma
                        | SimpleToken::Colon,
                        t,
                    ) => {
                        *tokenizer = t;
                    }
                    Token::Number(mut access) => {
                        handle = drain_num_chunked(handle, &mut access, offset).await?;
                        *tokenizer = Tokenizer::new();
                    }
                    Token::Str(mut access) => {
                        handle = drain_str_chunked(handle, &mut access, offset).await?;
                        *tokenizer = Tokenizer::new();
                    }
                    _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
                }
            }
            Ok(handle)
        }
        _ => Err(JsonError::UnexpectedByte { byte: 0 }),
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EntryOwned for ChunkedJsonEntry<'s, B, F> {
    type Error = JsonError;
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type StrChunks = ChunkedJsonStrAccess<'s, B, F>;
    type BytesChunks = ChunkedJsonBytesAccess<'s, B, F>;
    type Map = ChunkedJsonMapAccessOwned<'s, B, F>;
    type Seq = ChunkedJsonSeqAccess<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
        }
    }

    async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::Bool(b), tok) => Ok(Probe::Hit((
                Self::into_claim_with_tok(self.handle, self.offset, tok),
                b,
            ))),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
        let (c, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((c, n as u8)))
    }
    async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
        let (c, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((c, n as u16)))
    }
    async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
        let (c, n) = hit!(self.parse_num::<u64>().await);
        Ok(Probe::Hit((c, n as u32)))
    }
    async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
        self.parse_num::<u64>().await
    }
    async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
        self.parse_num::<u128>().await
    }
    async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
        let (c, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((c, n as i8)))
    }
    async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
        let (c, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((c, n as i16)))
    }
    async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
        let (c, n) = hit!(self.parse_num::<i64>().await);
        Ok(Probe::Hit((c, n as i32)))
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

    async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
        let mut chunks = hit!(self.deserialize_str_chunks().await);
        // Collect chunks until we have at least one char.
        let mut buf = [0u8; 4];
        let mut len = 0usize;
        let claim = loop {
            match chunks
                .next_str(|s| {
                    let n = s.len().min(buf.len() - len);
                    buf[len..len + n].copy_from_slice(&s.as_bytes()[..n]);
                    len += n;
                    n
                })
                .await?
            {
                Chunk::Data((c, _)) => {
                    // Check if we have a complete char yet.
                    if let Ok(s) = core::str::from_utf8(&buf[..len])
                        && let Some(ch) = s.chars().next()
                        && s.len() == ch.len_utf8()
                    {
                        // Exactly one char and nothing leftover - drain remaining chunks.
                        chunks = c;
                        let claim = loop {
                            match chunks.next_str(|_| ()).await? {
                                Chunk::Data((c, ())) => chunks = c,
                                Chunk::Done(claim) => break claim,
                            }
                        };
                        return Ok(Probe::Hit((claim, ch)));
                    }
                    chunks = c;
                }
                Chunk::Done(claim) => break claim,
            }
        };
        // Ended - should have exactly one char.
        let s = core::str::from_utf8(&buf[..len]).map_err(|_| JsonError::InvalidUtf8)?;
        let mut chars = s.chars();
        match (chars.next(), chars.next()) {
            (Some(ch), None) => Ok(Probe::Hit((claim, ch))),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.token {
            Token::Str(access) => Ok(Probe::Hit(ChunkedJsonStrAccess {
                handle: self.handle,
                access,
                offset: self.offset,
                char_buf: [0; 4],
            })),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.token {
            Token::Str(access) => Ok(Probe::Hit(ChunkedJsonBytesAccess {
                handle: self.handle,
                access,
                offset: self.offset,
                char_buf: [0; 4],
            })),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::ObjectStart, tok) => {
                Ok(Probe::Hit(ChunkedJsonMapAccessOwned {
                    handle: self.handle,
                    tokenizer: tok,
                    offset: self.offset,
                }))
            }
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::ArrayStart, tok) => Ok(Probe::Hit(ChunkedJsonSeqAccess {
                handle: self.handle,
                tokenizer: tok,
                offset: self.offset,
                first: true,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_option<T: DeserializeOwned<Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit((
                Self::into_claim_with_tok(self.handle, self.offset, tok),
                None,
            ))),
            other => {
                let h = self.handle;
                let sub =
                    ChunkedJsonSubDeserializer::new(h, self.tokenizer.clone(), self.offset, other);
                let (sub, v) = hit!(T::deserialize_owned(sub, extra).await);
                let claim = ChunkedJsonClaim {
                    tokenizer: sub.tokenizer,
                    offset: sub.offset,
                    handle: sub.handle,
                };
                Ok(Probe::Hit((claim, Some(v))))
            }
        }
    }

    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
        match self.token {
            Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit(Self::into_claim_with_tok(
                self.handle,
                self.offset,
                tok,
            ))),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_value<T: DeserializeOwned<Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        let h = self.handle;
        let sub =
            ChunkedJsonSubDeserializer::new(h, self.tokenizer.clone(), self.offset, self.token);
        let (sub, v) = hit!(T::deserialize_owned(sub, extra).await);
        let claim = ChunkedJsonClaim {
            tokenizer: sub.tokenizer,
            offset: sub.offset,
            handle: sub.handle,
        };
        Ok(Probe::Hit((claim, v)))
    }

    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut tokenizer = self.tokenizer;
        let mut offset = self.offset;
        let handle =
            skip_value_chunked(self.handle, &mut tokenizer, &mut offset, self.token).await?;
        Ok(ChunkedJsonClaim {
            tokenizer,
            offset,
            handle,
        })
    }
}

// ---------------------------------------------------------------------------
// StrAccess / BytesAccess
// ---------------------------------------------------------------------------

pub struct ChunkedJsonStrAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    access: token::StrAccess,
    offset: usize,
    char_buf: [u8; 4],
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> StrAccessOwned for ChunkedJsonStrAccess<'s, B, F> {
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type Error = JsonError;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            access: self.access,
            offset: self.offset,
            char_buf: self.char_buf,
        }
    }

    async fn next_str<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        enum Step {
            Slice { start: usize, end: usize },
            Char(char),
            Done,
            NeedRefill,
        }
        loop {
            let pre_offset = self.offset;
            let step = {
                let buf = self.handle.buf();
                let mut src: &[u8] = &buf[pre_offset..];
                let r = self.access.next_chunk(&mut src);
                self.offset = new_offset(buf, src);
                match r {
                    Ok(Some(StrChunk::Slice(""))) => Step::NeedRefill,
                    Ok(Some(StrChunk::Slice(_))) => Step::Slice {
                        start: pre_offset,
                        end: self.offset,
                    },
                    Ok(Some(StrChunk::Char(c))) => Step::Char(c),
                    Ok(None) => Step::Done,
                    Err(JsonError::UnexpectedEnd) => Step::NeedRefill,
                    Err(e) => return Err(e),
                }
            };
            match step {
                Step::Slice { start, end } => {
                    let buf = self.handle.buf();
                    let s = core::str::from_utf8(&buf[start..end])
                        .map_err(|_| JsonError::InvalidUtf8)?;
                    let r = f(s);
                    return Ok(Chunk::Data((self, r)));
                }
                Step::Char(c) => {
                    let r = f(c.encode_utf8(&mut self.char_buf));
                    return Ok(Chunk::Data((self, r)));
                }
                Step::Done => {
                    return Ok(Chunk::Done(ChunkedJsonClaim {
                        tokenizer: Tokenizer::new(),
                        offset: self.offset,
                        handle: self.handle,
                    }));
                }
                Step::NeedRefill => {
                    if self.offset > pre_offset {
                        // StrAccess consumed bytes but needs another call
                        // (e.g., high surrogate waiting for low surrogate).
                        continue;
                    }
                    self.handle = refill(self.handle, &mut self.offset).await?;
                }
            }
        }
    }
}

pub struct ChunkedJsonBytesAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    access: token::StrAccess,
    offset: usize,
    char_buf: [u8; 4],
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> BytesAccessOwned for ChunkedJsonBytesAccess<'s, B, F> {
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type Error = JsonError;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            access: self.access,
            offset: self.offset,
            char_buf: self.char_buf,
        }
    }

    async fn next_bytes<R>(
        mut self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        enum Step {
            Slice { start: usize, end: usize },
            Char(char),
            Done,
            NeedRefill,
        }
        loop {
            let pre_offset = self.offset;
            let step = {
                let buf = self.handle.buf();
                let mut src: &[u8] = &buf[pre_offset..];
                let r = self.access.next_chunk(&mut src);
                self.offset = new_offset(buf, src);
                match r {
                    Ok(Some(StrChunk::Slice(""))) => Step::NeedRefill,
                    Ok(Some(StrChunk::Slice(_))) => Step::Slice {
                        start: pre_offset,
                        end: self.offset,
                    },
                    Ok(Some(StrChunk::Char(c))) => Step::Char(c),
                    Ok(None) => Step::Done,
                    Err(JsonError::UnexpectedEnd) => Step::NeedRefill,
                    Err(e) => return Err(e),
                }
            };
            match step {
                Step::Slice { start, end } => {
                    let buf = self.handle.buf();
                    let r = f(&buf[start..end]);
                    return Ok(Chunk::Data((self, r)));
                }
                Step::Char(c) => {
                    let r = f(c.encode_utf8(&mut self.char_buf).as_bytes());
                    return Ok(Chunk::Data((self, r)));
                }
                Step::Done => {
                    return Ok(Chunk::Done(ChunkedJsonClaim {
                        tokenizer: Tokenizer::new(),
                        offset: self.offset,
                        handle: self.handle,
                    }));
                }
                Step::NeedRefill => {
                    if self.offset > pre_offset {
                        continue;
                    }
                    self.handle = refill(self.handle, &mut self.offset).await?;
                }
            }
        }
    }
}

// ===========================================================================
// Map access
// ===========================================================================

pub struct ChunkedJsonMapAccessOwned<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    tokenizer: Tokenizer,
    offset: usize,
}

pub struct ChunkedJsonKeyProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Option<Handle<'s, B, F>>,
    key_tok: Token,
    tokenizer: Tokenizer,
    offset: usize,
}

pub struct ChunkedJsonKeyClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    tokenizer: Tokenizer,
    offset: usize,
}

pub struct ChunkedJsonValueProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Option<Handle<'s, B, F>>,
    value_tok: Token,
    tokenizer: Tokenizer,
    offset: usize,
}

pub struct ChunkedJsonValueClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    tokenizer: Tokenizer,
    offset: usize,
}

// --- KeyProbe ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyProbeOwned for ChunkedJsonKeyProbe<'s, B, F> {
    type Error = JsonError;
    type KeyClaim = ChunkedJsonKeyClaim<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: Some(
                self.handle
                    .as_mut()
                    .expect("fork on consumed key probe")
                    .fork(),
            ),
            key_tok: self.key_tok.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
        }
    }

    async fn deserialize_key<K: DeserializeOwned<Extra>, Extra>(
        mut self,
        extra: Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error> {
        let h = self.handle.take().expect("key probe handle missing");
        let sub = ChunkedJsonSubDeserializer::new(h, self.tokenizer, self.offset, self.key_tok);
        let (sub, k) = hit!(K::deserialize_owned(sub, extra).await);
        Ok(Probe::Hit((
            ChunkedJsonKeyClaim {
                handle: sub.handle,
                tokenizer: sub.tokenizer,
                offset: sub.offset,
            },
            k,
        )))
    }
}

// --- KeyClaim ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyClaimOwned for ChunkedJsonKeyClaim<'s, B, F> {
    type Error = JsonError;
    type MapClaim = ChunkedJsonClaim<'s, B, F>;
    type ValueProbe = ChunkedJsonValueProbe<'s, B, F>;

    async fn into_value_probe(mut self) -> Result<Self::ValueProbe, Self::Error> {
        // Eat the colon.
        let mut pending: Option<Token> = None;
        let (h, colon_tok) = next_dispatch(
            self.handle,
            &mut self.tokenizer,
            &mut self.offset,
            &mut pending,
        )
        .await?;
        match colon_tok {
            Token::Simple(SimpleToken::Colon, t) => self.tokenizer = t,
            _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
        }

        // Read the value start token.
        let (h, value_tok) =
            next_dispatch(h, &mut self.tokenizer, &mut self.offset, &mut pending).await?;

        Ok(ChunkedJsonValueProbe {
            handle: Some(h),
            value_tok,
            tokenizer: self.tokenizer,
            offset: self.offset,
        })
    }
}

// --- ValueProbe ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueProbeOwned for ChunkedJsonValueProbe<'s, B, F> {
    type Error = JsonError;
    type MapClaim = ChunkedJsonClaim<'s, B, F>;
    type ValueClaim = ChunkedJsonValueClaim<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: Some(
                self.handle
                    .as_mut()
                    .expect("fork on consumed value probe")
                    .fork(),
            ),
            value_tok: self.value_tok.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
        }
    }

    async fn deserialize_value<V: DeserializeOwned<Extra>, Extra>(
        mut self,
        extra: Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error> {
        let h = self.handle.take().expect("value probe handle missing");
        let sub = ChunkedJsonSubDeserializer::new(h, self.tokenizer, self.offset, self.value_tok);
        let (sub, v) = hit!(V::deserialize_owned(sub, extra).await);
        Ok(Probe::Hit((
            ChunkedJsonValueClaim {
                handle: sub.handle,
                tokenizer: sub.tokenizer,
                offset: sub.offset,
            },
            v,
        )))
    }

    async fn skip(mut self) -> Result<Self::ValueClaim, Self::Error> {
        let h = self.handle.take().expect("value probe handle missing");
        let mut tokenizer = self.tokenizer;
        let mut offset = self.offset;
        let handle = skip_value_chunked(h, &mut tokenizer, &mut offset, self.value_tok).await?;
        Ok(ChunkedJsonValueClaim {
            handle,
            tokenizer,
            offset,
        })
    }
}

// --- ValueClaim ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueClaimOwned for ChunkedJsonValueClaim<'s, B, F> {
    type Error = JsonError;
    type KeyProbe = ChunkedJsonKeyProbe<'s, B, F>;
    type MapClaim = ChunkedJsonClaim<'s, B, F>;

    async fn next_key(
        mut self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        // Expect comma or closing brace.
        let mut pending: Option<Token> = None;
        let (h, tok) = next_dispatch(
            self.handle,
            &mut self.tokenizer,
            &mut self.offset,
            &mut pending,
        )
        .await?;
        match tok {
            Token::Simple(SimpleToken::Comma, t) => {
                self.tokenizer = t;
            }
            Token::Simple(SimpleToken::ObjectEnd, t) => {
                return Ok(NextKey::Done(ChunkedJsonClaim {
                    tokenizer: t,
                    offset: self.offset,
                    handle: h,
                }));
            }
            _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
        }

        // Read the next key start token.
        let (h, key_tok) =
            next_dispatch(h, &mut self.tokenizer, &mut self.offset, &mut pending).await?;

        Ok(NextKey::Entry(ChunkedJsonKeyProbe {
            handle: Some(h),
            key_tok,
            tokenizer: self.tokenizer,
            offset: self.offset,
        }))
    }
}

// --- MapAccessOwned ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapAccessOwned for ChunkedJsonMapAccessOwned<'s, B, F> {
    type Error = JsonError;
    type MapClaim = ChunkedJsonClaim<'s, B, F>;
    type KeyProbe = ChunkedJsonKeyProbe<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
        }
    }

    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        mut self,
        mut arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        let mut pending: Option<Token> = None;
        let (h, tok) = next_dispatch(
            self.handle,
            &mut self.tokenizer,
            &mut self.offset,
            &mut pending,
        )
        .await?;

        let mut key_probe = match tok {
            Token::Simple(SimpleToken::ObjectEnd, t) => {
                // Empty map.
                let claim = ChunkedJsonClaim {
                    tokenizer: t,
                    offset: self.offset,
                    handle: h,
                };
                return Ok(Probe::Hit((claim, arms.take_outputs())));
            }
            key_tok => ChunkedJsonKeyProbe {
                handle: Some(h),
                key_tok,
                tokenizer: self.tokenizer,
                offset: self.offset,
            },
        };

        loop {
            // Race all arms' key callbacks against this round's key probe.
            let value_claim = match arms.race_keys(key_probe).await? {
                Probe::Hit((arm_index, key_claim)) => {
                    // A known arm matched. Convert key claim → value probe.
                    let vp = key_claim.into_value_probe().await?;
                    match arms.dispatch_value(arm_index, vp).await? {
                        Probe::Hit((vc, ())) => vc,
                        Probe::Miss => return Ok(Probe::Miss),
                    }
                }
                Probe::Miss => {
                    // No arm matched this key - whole map is a Miss.
                    return Ok(Probe::Miss);
                }
            };

            // Advance to next key or end of map.
            match value_claim
                .next_key(arms.unsatisfied_count(), arms.open_count())
                .await?
            {
                strede::NextKey::Entry(next_kp) => key_probe = next_kp,
                strede::NextKey::Done(map_claim) => {
                    return Ok(Probe::Hit((map_claim, arms.take_outputs())));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SeqAccess / SeqEntry
// ---------------------------------------------------------------------------

pub struct ChunkedJsonSeqAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    tokenizer: Tokenizer,
    offset: usize,
    first: bool,
}

pub struct ChunkedJsonElemClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    tokenizer: Tokenizer,
    offset: usize,
}

async fn seq_next<'s, B, F, const N: usize, Fn_, Fut, R>(
    mut seq: ChunkedJsonSeqAccess<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<Chunk<(ChunkedJsonSeqAccess<'s, B, F>, R), ChunkedJsonClaim<'s, B, F>>>, JsonError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedJsonSeqEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedJsonElemClaim<'s, B, F>, R)>, JsonError>>,
{
    if !seq.first {
        let mut pending: Option<Token> = None;
        let h = seq.handle;
        let (h, tok) = next_dispatch(h, &mut seq.tokenizer, &mut seq.offset, &mut pending).await?;
        match tok {
            Token::Simple(SimpleToken::Comma, t) => {
                seq.handle = h;
                seq.tokenizer = t;
            }
            Token::Simple(SimpleToken::ArrayEnd, t) => {
                return Ok(Probe::Hit(Chunk::Done(ChunkedJsonClaim {
                    tokenizer: t,
                    offset: seq.offset,
                    handle: h,
                })));
            }
            _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
        }
    }
    seq.first = false;

    let elem_tok = {
        let mut pending: Option<Token> = None;
        let h = seq.handle;
        let (h, tok) = next_dispatch(h, &mut seq.tokenizer, &mut seq.offset, &mut pending).await?;
        match tok {
            Token::Simple(SimpleToken::ArrayEnd, t) => {
                return Ok(Probe::Hit(Chunk::Done(ChunkedJsonClaim {
                    tokenizer: t,
                    offset: seq.offset,
                    handle: h,
                })));
            }
            t => {
                seq.handle = h;
                t
            }
        }
    };

    let main = seq.handle;
    let mut handles: [Option<Handle<'s, B, F>>; N] = [const { None }; N];
    handles[0] = Some(main);
    for i in 1..N {
        let h0 = handles[0].as_mut().expect("h0 present");
        handles[i] = Some(h0.fork());
    }
    let snap_tok = seq.tokenizer.clone();
    let snap_off = seq.offset;
    let entries: [ChunkedJsonSeqEntry<'s, B, F>; N] =
        core::array::from_fn(|i| ChunkedJsonSeqEntry {
            handle: handles[i].take(),
            elem_tok: elem_tok.clone(),
            tokenizer: snap_tok.clone(),
            offset: snap_off,
        });

    let (elem_claim, r) = hit!(f(entries).await);
    seq.tokenizer = elem_claim.tokenizer;
    seq.offset = elem_claim.offset;
    seq.handle = elem_claim.handle;
    Ok(Probe::Hit(Chunk::Data((seq, r))))
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqAccessOwned for ChunkedJsonSeqAccess<'s, B, F> {
    type Error = JsonError;
    type SeqClaim = ChunkedJsonClaim<'s, B, F>;
    type ElemClaim = ChunkedJsonElemClaim<'s, B, F>;
    type Elem = ChunkedJsonSeqEntry<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            first: self.first,
        }
    }

    async fn next<const N: usize, Fn_, Fut, R>(
        self,
        f: Fn_,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        Fn_: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        seq_next(self, f).await
    }
}

pub struct ChunkedJsonSeqEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Option<Handle<'s, B, F>>,
    elem_tok: Token,
    tokenizer: Tokenizer,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqEntryOwned for ChunkedJsonSeqEntry<'s, B, F> {
    type Error = JsonError;
    type Claim = ChunkedJsonElemClaim<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: Some(self.handle.as_mut().expect("fork on consumed entry").fork()),
            elem_tok: self.elem_tok.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
        }
    }

    async fn get<T: DeserializeOwned<Extra>, Extra>(
        mut self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        let h = self.handle.take().expect("seq entry handle missing");
        let sub =
            ChunkedJsonSubDeserializer::new(h, self.tokenizer.clone(), self.offset, self.elem_tok);
        let (sub, v) = hit!(T::deserialize_owned(sub, extra).await);
        let claim = ChunkedJsonElemClaim {
            tokenizer: sub.tokenizer,
            offset: sub.offset,
            handle: sub.handle,
        };
        Ok(Probe::Hit((claim, v)))
    }

    async fn skip(mut self) -> Result<Self::Claim, Self::Error> {
        let h = self.handle.take().expect("seq entry handle missing");
        let mut tokenizer = self.tokenizer;
        let mut offset = self.offset;
        let handle = skip_value_chunked(h, &mut tokenizer, &mut offset, self.elem_tok).await?;
        Ok(ChunkedJsonElemClaim {
            tokenizer,
            offset,
            handle,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::string::{String, ToString};
    use strede::{
        Chunk, DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapArmBase, Match,
        MatchVals, Probe, SeqAccessOwned, SeqEntryOwned, StrAccessOwned, UnwrapOrElse, owned,
    };

    use strede_test_util::block_on;

    #[derive(Debug, DeserializeOwned, PartialEq, Eq)]
    enum Either<L, R> {
        Left(L),
        Right(R),
    }

    /// Run a chunked deserialization with `input` loaded as a single chunk.
    /// The loader returns an empty slice on the second call (EOF).
    fn with_chunked<L: AsyncFnMut(&mut &[u8]), R>(
        input: &[u8],
        loader: L,
        f: impl AsyncFnOnce(SharedBuf<'_, &[u8], L>) -> R,
    ) -> R {
        block_on(SharedBuf::with_async(input, loader, f))
    }

    fn eof_loader() -> impl AsyncFnMut(&mut &[u8]) {
        async move |buf: &mut &[u8]| {
            *buf = &[];
        }
    }

    // ----- DeserializeOwned newtypes -----------------------------------------

    struct U32(u32);
    impl DeserializeOwned for U32 {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    struct I64(i64);
    impl DeserializeOwned for I64 {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let (c, v) = hit!(e.deserialize_i64().await);
                Ok(Probe::Hit((c, I64(v))))
            })
            .await
        }
    }

    struct F64(f64);
    impl DeserializeOwned for F64 {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let (c, v) = hit!(e.deserialize_f64().await);
                Ok(Probe::Hit((c, F64(v))))
            })
            .await
        }
    }

    struct Bool(bool);
    impl DeserializeOwned for Bool {
        async fn deserialize_owned<D: DeserializerOwned>(
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

    /// Str type that just measures byte length (no allocation needed).
    struct StrLen(usize);
    impl DeserializeOwned for StrLen {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let mut chunks = match e.deserialize_str_chunks().await? {
                    Probe::Hit(c) => c,
                    Probe::Miss => return Ok(Probe::Miss),
                };
                let mut len = 0usize;
                loop {
                    match chunks.next_str(|s| s.len()).await? {
                        Chunk::Data((c, n)) => {
                            len += n;
                            chunks = c;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, StrLen(len)))),
                    }
                }
            })
            .await
        }
    }

    // ---- bool ---------------------------------------------------------------

    #[test]
    fn bool_true() {
        let v = with_chunked(b"true", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = Bool::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert!(v);
    }

    #[test]
    fn bool_false() {
        let v = with_chunked(b"false", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = Bool::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert!(!v);
    }

    // ---- integers -----------------------------------------------------------

    #[test]
    fn u32_positive() {
        let v = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = U32::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert_eq!(v, 42);
    }

    #[test]
    fn i64_negative() {
        let v = with_chunked(b"-7", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = I64::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert_eq!(v, -7);
    }

    // ---- floats -------------------------------------------------------------

    #[test]
    fn f64_decimal() {
        let v = with_chunked(b"3.14", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = F64::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert!((v - 3.14f64).abs() < 1e-10);
    }

    // ---- char ---------------------------------------------------------------

    #[test]
    fn char_single() {
        let (_, v) = with_chunked(b"\"A\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async { e.deserialize_char().await })
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, 'A');
    }

    // ---- str_chunks ---------------------------------------------------------

    #[test]
    fn str_chunks_plain() {
        let len = with_chunked(b"\"hello\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrLen::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert_eq!(len, 5);
    }

    #[test]
    fn str_chunks_empty() {
        let len = with_chunked(b"\"\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrLen::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert_eq!(len, 0);
    }

    #[test]
    fn str_chunks_escape_newline() {
        // "hello\nworld": "hello"(5) + '\n'(1) + "world"(5) = 11 bytes
        let len = with_chunked(b"\"hello\\nworld\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrLen::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert_eq!(len, 11);
    }

    #[test]
    fn str_chunks_unicode_escape() {
        // "\u0041" decodes to 'A' (1 byte in UTF-8)
        let len = with_chunked(b"\"\\u0041\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrLen::deserialize_owned(de, ()).await.unwrap().unwrap();
            v.0
        });
        assert_eq!(len, 1);
    }

    // ---- surrogate pairs (str_chunks) ----------------------------------------

    /// Str type that collects chunks into a String (for content verification).
    struct StrCollect(String);
    impl DeserializeOwned for StrCollect {
        async fn deserialize_owned<D: DeserializerOwned>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let mut chunks = match e.deserialize_str_chunks().await? {
                    Probe::Hit(c) => c,
                    Probe::Miss => return Ok(Probe::Miss),
                };
                let mut out = String::new();
                loop {
                    match chunks.next_str(|s| String::from(s)).await? {
                        Chunk::Data((c, s)) => {
                            out.push_str(&s);
                            chunks = c;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, StrCollect(out)))),
                    }
                }
            })
            .await
        }
    }

    #[test]
    fn str_chunks_surrogate_pair() {
        // \uD834\uDD1E = U+1D11E = 𝄞 (musical symbol G clef)
        let s = with_chunked(b"\"\\uD834\\uDD1E\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrCollect::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap();
            v.0
        });
        assert_eq!(s, "\u{1D11E}");
    }

    #[test]
    fn str_chunks_surrogate_pair_min() {
        let s = with_chunked(b"\"\\uD800\\uDC00\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrCollect::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap();
            v.0
        });
        assert_eq!(s, "\u{10000}");
    }

    #[test]
    fn str_chunks_surrogate_pair_max() {
        let s = with_chunked(b"\"\\uDBFF\\uDFFF\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrCollect::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap();
            v.0
        });
        assert_eq!(s, "\u{10FFFF}");
    }

    #[test]
    fn str_chunks_surrogate_pair_with_text() {
        let s = with_chunked(b"\"abc\\uD834\\uDD1Exyz\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let (_, v) = StrCollect::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap();
            v.0
        });
        assert_eq!(s, "abc\u{1D11E}xyz");
    }

    #[test]
    fn str_chunks_lone_high_surrogate_err() {
        let result = with_chunked(b"\"\\uD834\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            StrCollect::deserialize_owned(de, ()).await
        });
        assert!(result.is_err());
    }

    #[test]
    fn str_chunks_lone_low_surrogate_err() {
        let result = with_chunked(b"\"\\uDD1E\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            StrCollect::deserialize_owned(de, ()).await
        });
        assert!(result.is_err());
    }

    // ---- seq ----------------------------------------------------------------

    #[test]
    fn seq_sum_of_numbers() {
        let (_, sum) = with_chunked(b"[1, 2, 3]", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async {
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
                        Chunk::Data((s, v)) => {
                            sum += v;
                            seq = s;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, sum))),
                    }
                }
            })
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(sum, 6);
    }

    #[test]
    fn seq_empty() {
        let (_, count) = with_chunked(b"[]", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async {
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
                        Chunk::Data((s, _)) => {
                            count += 1;
                            seq = s;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, count))),
                    }
                }
            })
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(count, 0);
    }

    // ---- map ----------------------------------------------------------------

    async fn map_single_pair_impl<D: DeserializerOwned<Error = JsonError>>(
        d: D,
    ) -> Result<Probe<(D::Claim, (usize, u32))>, JsonError> {
        use strede::{MapAccessOwned as _, MapKeyProbeOwned as _, MapValueProbeOwned as _};
        d.entry(|[e]| async {
            let map = hit!(e.deserialize_map().await);
            let arms = strede::map_arms! {
                |kp: owned::KP<D>| kp.deserialize_key::<StrLen, ()>(())
                => |vp: owned::VP2<D>, k: StrLen| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<U32, ()>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
            };
            let (claim, strede::map_outputs!(opt_kv)) = hit!(map.iterate(arms).await);
            let (k, v) = opt_kv.unwrap();
            Ok(Probe::Hit((claim, (k.0, v.0))))
        })
        .await
    }

    #[test]
    fn map_single_pair() {
        let (_, (key_len, val)) = with_chunked(b"{\"x\": 10}", eof_loader(), async |shared| {
            map_single_pair_impl(ChunkedJsonDeserializer::new(shared))
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(key_len, 1); // "x"
        assert_eq!(val, 10);
    }

    async fn map_empty_impl<D: DeserializerOwned<Error = JsonError>>(
        d: D,
    ) -> Result<Probe<(D::Claim, ())>, JsonError> {
        d.entry(|[e]| async {
            let map = hit!(e.deserialize_map().await);
            let (claim, ()) = hit!(iterate_skip_all(map).await);
            Ok(Probe::Hit((claim, ())))
        })
        .await
    }

    #[test]
    fn map_empty() {
        let (_, ()) = with_chunked(b"{}", eof_loader(), async |shared| {
            map_empty_impl(ChunkedJsonDeserializer::new(shared))
                .await
                .unwrap()
                .unwrap()
        });
    }

    async fn map_multiple_pairs_impl<D: DeserializerOwned<Error = JsonError>>(
        d: D,
    ) -> Result<Probe<(D::Claim, u32)>, JsonError> {
        use strede::{MapAccessOwned as _, MapKeyProbeOwned as _, MapValueProbeOwned as _};
        d.entry(|[e]| async {
            let map = hit!(e.deserialize_map().await);
            let arms = strede::map_arms! {
                |kp: owned::KP<D>| kp.deserialize_key::<Match, &str>("a")
                => |vp: owned::VP2<D>, k: Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<U32, ()>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
                |kp: owned::KP<D>| kp.deserialize_key::<Match, &str>("b")
                => |vp: owned::VP2<D>, k: Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<U32, ()>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
                |kp: owned::KP<D>| kp.deserialize_key::<Match, &str>("c")
                => |vp: owned::VP2<D>, k: Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<U32, ()>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
            };
            let (claim, strede::map_outputs!(a, b, c)) = hit!(map.iterate(arms).await);
            let sum = a.map(|(_, v)| v.0).unwrap_or(0)
                + b.map(|(_, v)| v.0).unwrap_or(0)
                + c.map(|(_, v)| v.0).unwrap_or(0);
            Ok(Probe::Hit((claim, sum)))
        })
        .await
    }

    #[test]
    fn map_multiple_pairs() {
        let (_, sum) = with_chunked(
            b"{\"a\": 1, \"b\": 2, \"c\": 3}",
            eof_loader(),
            async |shared| {
                map_multiple_pairs_impl(ChunkedJsonDeserializer::new(shared))
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(sum, 6);
    }

    // ---- option -------------------------------------------------------------

    #[test]
    fn option_null_is_none() {
        let (_, v) = with_chunked(b"null", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async { e.deserialize_option::<Bool, ()>(()).await })
                .await
                .unwrap()
                .unwrap()
        });
        assert!(v.is_none());
    }

    #[test]
    fn option_bool_is_some() {
        let (_, v) = with_chunked(b"true", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async { e.deserialize_option::<Bool, ()>(()).await })
                .await
                .unwrap()
                .unwrap()
        });
        assert!(v.unwrap().0);
    }

    // ---- error handling -----------------------------------------------------

    #[test]
    fn error_truncated_literal() {
        let result = with_chunked(b"tru", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async { e.deserialize_bool().await }).await
        });
        assert!(matches!(result, Err(JsonError::UnexpectedEnd)));
    }

    #[test]
    fn error_invalid_number() {
        let result = with_chunked(b"1.}", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async { e.deserialize_f64().await }).await
        });
        assert!(matches!(result, Err(JsonError::InvalidNumber)));
    }

    // ---- derive(DeserializeOwned) -------------------------------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct Point {
        x: i64,
        y: i64,
    }

    #[test]
    fn derive_owned_basic() {
        let (_, v) = with_chunked(b"{\"x\": 1, \"y\": -2}", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Point::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Point { x: 1, y: -2 });
    }

    #[test]
    fn derive_owned_fields_out_of_order() {
        let (_, v) = with_chunked(b"{\"y\": 7, \"x\": 3}", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Point::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Point { x: 3, y: 7 });
    }

    #[test]
    fn derive_owned_duplicate_field() {
        let result = with_chunked(
            b"{\"x\": 1, \"x\": 2, \"y\": 3}",
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                Point::deserialize_owned(de, ()).await
            },
        );
        assert_eq!(result, Err(JsonError::DuplicateField("x")));
    }

    #[test]
    fn derive_owned_unknown_field_is_miss() {
        let result = with_chunked(
            b"{\"x\": 1, \"z\": 99, \"y\": 2}",
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                Point::deserialize_owned(de, ()).await
            },
        );
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn derive_owned_missing_field_is_miss() {
        let result = with_chunked(b"{\"x\": 5}", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Point::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn derive_owned_non_object_is_miss() {
        let result = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Point::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn derive_owned_array_is_miss() {
        let result = with_chunked(b"[1, 2]", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Point::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct MixedOwned {
        score: f64,
        count: u32,
        active: bool,
    }

    #[test]
    fn derive_owned_mixed_types() {
        let (_, v) = with_chunked(
            br#"{"score": 3.14, "count": 7, "active": true}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                MixedOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            MixedOwned {
                score: 3.14,
                count: 7,
                active: true
            }
        );
    }

    #[test]
    fn derive_owned_mixed_types_reordered() {
        let (_, v) = with_chunked(
            br#"{"active": false, "score": 0.0, "count": 0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                MixedOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            MixedOwned {
                score: 0.0,
                count: 0,
                active: false
            }
        );
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct Wrapper {
        value: i64,
    }

    #[test]
    fn derive_owned_single_field() {
        let (_, v) = with_chunked(br#"{"value": -99}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Wrapper::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Wrapper { value: -99 });
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct Rect {
        origin: Point,
        size: Point,
    }

    #[test]
    fn derive_owned_nested_struct() {
        let (_, v) = with_chunked(
            br#"{"origin": {"x": 1, "y": 2}, "size": {"x": 10, "y": 20}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                Rect::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            Rect {
                origin: Point { x: 1, y: 2 },
                size: Point { x: 10, y: 20 },
            }
        );
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct OptFields {
        required: i64,
        maybe: Option<i64>,
    }

    #[test]
    fn derive_owned_option_present() {
        let (_, v) = with_chunked(
            br#"{"required": 1, "maybe": 42}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OptFields::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            OptFields {
                required: 1,
                maybe: Some(42)
            }
        );
    }

    #[test]
    fn derive_owned_option_null() {
        let (_, v) = with_chunked(
            br#"{"required": 1, "maybe": null}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OptFields::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            OptFields {
                required: 1,
                maybe: None
            }
        );
    }

    // ---- derive(DeserializeOwned): generic type parameters ------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct GenPair<A, B> {
        first: A,
        second: B,
    }

    #[test]
    fn derive_owned_generic_two_type_params() {
        let (_, v) = with_chunked(
            br#"{"first": 10, "second": true}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                <GenPair<i64, bool>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            GenPair {
                first: 10,
                second: true
            }
        );
    }

    #[test]
    fn derive_owned_generic_different_instantiation() {
        let (_, v) = with_chunked(
            br#"{"first": false, "second": 42}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                <GenPair<bool, u32>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            GenPair {
                first: false,
                second: 42
            }
        );
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct GenWrapper<T> {
        inner: T,
    }

    #[test]
    fn derive_owned_generic_single_param() {
        let (_, v) = with_chunked(br#"{"inner": 99}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <GenWrapper<i64>>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, GenWrapper { inner: 99 });
    }

    #[test]
    fn derive_owned_generic_nested() {
        let (_, v) = with_chunked(
            br#"{"inner": {"x": 5, "y": 6}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                <GenWrapper<Point>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            GenWrapper {
                inner: Point { x: 5, y: 6 }
            }
        );
    }

    #[test]
    fn derive_owned_generic_nested_generic() {
        let (_, v) = with_chunked(
            br#"{"inner": {"inner": true}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                <GenWrapper<GenWrapper<bool>>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            GenWrapper {
                inner: GenWrapper { inner: true }
            }
        );
    }

    #[test]
    fn derive_owned_generic_with_option() {
        let (_, v) = with_chunked(br#"{"inner": null}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <GenWrapper<Option<i64>>>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, GenWrapper { inner: None });

        let (_, v) = with_chunked(br#"{"inner": 7}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <GenWrapper<Option<i64>>>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, GenWrapper { inner: Some(7) });
    }

    // ---- derive(DeserializeOwned): enum - unit-only -------------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    #[test]
    fn derive_owned_enum_unit_variant() {
        let (_, v) = with_chunked(br#""Red""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Color::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Color::Red);
    }

    #[test]
    fn derive_owned_enum_unit_variant_other() {
        let (_, v) = with_chunked(br#""Blue""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Color::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Color::Blue);
    }

    #[test]
    fn derive_owned_enum_unit_unknown_is_miss() {
        let result = with_chunked(br#""Yellow""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Color::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn derive_owned_enum_unit_non_string_is_miss() {
        let result = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Color::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ---- derive(DeserializeOwned): enum - mixed -----------------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum Shape {
        Circle,
        Square(Point),
        Rect { origin: Point, size: Point },
    }

    #[test]
    fn derive_owned_enum_mixed_unit_variant() {
        let (_, v) = with_chunked(br#""Circle""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Shape::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Shape::Circle);
    }

    #[test]
    fn derive_owned_enum_newtype_variant() {
        let (_, v) = with_chunked(
            br#"{"Square": {"x": 1, "y": 2}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                Shape::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(v, Shape::Square(Point { x: 1, y: 2 }));
    }

    #[test]
    fn derive_owned_enum_struct_variant() {
        let (_, v) = with_chunked(
            br#"{"Rect": {"origin": {"x": 0, "y": 0}, "size": {"x": 10, "y": 20}}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                Shape::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            Shape::Rect {
                origin: Point { x: 0, y: 0 },
                size: Point { x: 10, y: 20 },
            }
        );
    }

    #[test]
    fn derive_owned_enum_unknown_variant_map_is_miss() {
        let result = with_chunked(br#"{"Triangle": {"x": 1}}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Shape::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn derive_owned_enum_empty_map_is_miss() {
        let result = with_chunked(br#"{}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Shape::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ---- derive(DeserializeOwned): enum - tuple variants ---------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum Geom {
        Point(i64, i64),
        Color(u8, u8, u8),
        Single(bool),
        Unit,
    }

    #[test]
    fn derive_owned_enum_tuple_variant_two_fields() {
        let (_, v) = with_chunked(br#"{"Point": [3, 4]}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Geom::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Geom::Point(3, 4));
    }

    #[test]
    fn derive_owned_enum_tuple_variant_three_fields() {
        let (_, v) = with_chunked(
            br#"{"Color": [255, 128, 0]}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                Geom::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(v, Geom::Color(255, 128, 0));
    }

    #[test]
    fn derive_owned_enum_tuple_variant_mixed_with_unit() {
        let (_, v) = with_chunked(br#""Unit""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Geom::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Geom::Unit);
    }

    #[test]
    fn derive_owned_enum_tuple_variant_mixed_with_newtype() {
        let (_, v) = with_chunked(br#"{"Single": true}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Geom::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Geom::Single(true));
    }

    #[test]
    fn derive_owned_enum_tuple_variant_wrong_length_is_miss() {
        let result = with_chunked(br#"{"Point": [1, 2, 3]}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Geom::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ---- derive(DeserializeOwned): enum - newtype-only ----------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum Value {
        Int(i64),
        Bool(bool),
    }

    #[test]
    fn derive_owned_enum_newtype_only() {
        let (_, v) = with_chunked(br#"{"Int": 42}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Value::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn derive_owned_enum_newtype_only_other() {
        let (_, v) = with_chunked(br#"{"Bool": true}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Value::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn derive_owned_enum_generic_left() {
        let (_, v) = with_chunked(br#"{"Left": 42}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Either<i64, bool>>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, Either::Left(42));
    }

    #[test]
    fn derive_owned_enum_generic_right() {
        let (_, v) = with_chunked(br#"{"Right": true}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Either<i64, bool>>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, Either::Right(true));
    }

    // ---- derive(DeserializeOwned): default attribute -------------------------

    fn default_count() -> i64 {
        99
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct WithDefaults {
        required: i64,
        #[strede(default = "default_count")]
        count: i64,
        #[strede(default)]
        flag: bool,
    }

    #[test]
    fn derive_owned_default_fields_missing() {
        let (_, v) = with_chunked(br#"{"required": 1}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            WithDefaults::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(
            v,
            WithDefaults {
                required: 1,
                count: 99,
                flag: false
            }
        );
    }

    #[test]
    fn derive_owned_default_fields_present_overrides() {
        let (_, v) = with_chunked(
            br#"{"required": 1, "count": 5, "flag": true}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                WithDefaults::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            WithDefaults {
                required: 1,
                count: 5,
                flag: true
            }
        );
    }

    #[test]
    fn derive_owned_default_required_missing_is_miss() {
        let result = with_chunked(br#"{"count": 5}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            WithDefaults::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ---- derive(DeserializeOwned): default expression attribute ---------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct WithDefaultExpr {
        required: i64,
        #[strede(default = "99i64")]
        count: i64,
        #[strede(default = "String::from(\"hello\")")]
        greeting: String,
    }

    #[test]
    fn derive_owned_default_expr_missing() {
        let (_, v) = with_chunked(br#"{"required": 1}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            WithDefaultExpr::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(
            v,
            WithDefaultExpr {
                required: 1,
                count: 99,
                greeting: String::from("hello"),
            }
        );
    }

    #[test]
    fn derive_owned_default_expr_present_overrides() {
        let (_, v) = with_chunked(
            br#"{"required": 1, "count": 5, "greeting": "world"}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                WithDefaultExpr::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            WithDefaultExpr {
                required: 1,
                count: 5,
                greeting: String::from("world"),
            }
        );
    }

    // ---- derive(DeserializeOwned): deserialize_owned_with attribute -----------

    /// Custom owned deserializer: reads an i64 and doubles it.
    async fn double_i64_owned<D: strede::DeserializerOwned>(
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

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct WithCustomDe {
        normal: i64,
        #[strede(deserialize_owned_with = "double_i64_owned")]
        doubled: i64,
    }

    #[test]
    fn derive_owned_deserialize_with() {
        let (_, v) = with_chunked(
            br#"{"normal": 5, "doubled": 3}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                WithCustomDe::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            WithCustomDe {
                normal: 5,
                doubled: 6
            }
        );
    }

    // ---- derive(DeserializeOwned): skip_deserializing attribute ---------------

    fn skipped_default() -> i64 {
        42
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct WithSkip {
        required: i64,
        #[strede(skip_deserializing, default = "skipped_default")]
        skipped: i64,
        #[strede(skip_deserializing, default)]
        skipped_trait: bool,
    }

    #[test]
    fn derive_owned_skip_uses_default_when_absent() {
        let (_, v) = with_chunked(br#"{"required": 1}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            WithSkip::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(
            v,
            WithSkip {
                required: 1,
                skipped: 42,
                skipped_trait: false
            }
        );
    }

    #[test]
    fn derive_owned_skip_ignores_present_value() {
        let result = with_chunked(
            br#"{"required": 1, "skipped": 99, "skipped_trait": true}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                WithSkip::deserialize_owned(de, ()).await
            },
        );
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ---- derive(DeserializeOwned): rename attribute -------------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct RenamedFields {
        #[strede(rename = "type")]
        kind: i64,
        #[strede(rename = "value")]
        val: bool,
    }

    #[test]
    fn derive_owned_rename_struct_fields() {
        let (_, v) = with_chunked(
            br#"{"type": 42, "value": true}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                RenamedFields::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            RenamedFields {
                kind: 42,
                val: true
            }
        );
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum RenamedVariants {
        #[strede(rename = "circle")]
        Circle,
        #[strede(rename = "rect")]
        Rect(i64),
    }

    #[test]
    fn derive_owned_rename_unit_variant() {
        let (_, v) = with_chunked(br#""circle""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            RenamedVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, RenamedVariants::Circle);
    }

    #[test]
    fn derive_owned_rename_newtype_variant() {
        let (_, v) = with_chunked(br#"{"rect": 5}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            RenamedVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, RenamedVariants::Rect(5));
    }

    // ---- derive(DeserializeOwned): alias attribute ----------------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct AliasedFields {
        #[strede(alias = "hostname", alias = "server")]
        host: String,
        port: u16,
    }

    #[test]
    fn derive_owned_alias_primary_name() {
        let (_, v) = with_chunked(
            br#"{"host": "a.com", "port": 80}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AliasedFields::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            AliasedFields {
                host: String::from("a.com"),
                port: 80
            }
        );
    }

    #[test]
    fn derive_owned_alias_first_alias() {
        let (_, v) = with_chunked(
            br#"{"hostname": "b.com", "port": 443}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AliasedFields::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            AliasedFields {
                host: String::from("b.com"),
                port: 443
            }
        );
    }

    #[test]
    fn derive_owned_alias_second_alias() {
        let (_, v) = with_chunked(
            br#"{"server": "c.com", "port": 8080}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AliasedFields::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            AliasedFields {
                host: String::from("c.com"),
                port: 8080
            }
        );
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum AliasedVariants {
        #[strede(alias = "pong")]
        Ping,
        #[strede(alias = "payload")]
        Data(i64),
    }

    #[test]
    fn derive_owned_alias_unit_variant_primary() {
        let (_, v) = with_chunked(br#""Ping""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AliasedVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, AliasedVariants::Ping);
    }

    #[test]
    fn derive_owned_alias_unit_variant_alias() {
        let (_, v) = with_chunked(br#""pong""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AliasedVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, AliasedVariants::Ping);
    }

    #[test]
    fn derive_owned_alias_newtype_variant_primary() {
        let (_, v) = with_chunked(br#"{"Data": 42}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AliasedVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, AliasedVariants::Data(42));
    }

    #[test]
    fn derive_owned_alias_newtype_variant_alias() {
        let (_, v) = with_chunked(br#"{"payload": 42}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AliasedVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, AliasedVariants::Data(42));
    }

    // ---- derive(DeserializeOwned): untagged attribute -----------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(untagged)]
    enum Untagged {
        Num(i64),
        Flag(bool),
        Pt(i64, i64),
        Named { x: i64 },
    }

    #[test]
    fn derive_owned_untagged_newtype_first() {
        let (_, v) = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Untagged::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Untagged::Num(42));
    }

    #[test]
    fn derive_owned_untagged_bool() {
        let (_, v) = with_chunked(b"true", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Untagged::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Untagged::Flag(true));
    }

    #[test]
    fn derive_owned_untagged_tuple() {
        let (_, v) = with_chunked(b"[1, 2]", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Untagged::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Untagged::Pt(1, 2));
    }

    #[test]
    fn derive_owned_untagged_struct() {
        let (_, v) = with_chunked(br#"{"x": 7}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Untagged::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, Untagged::Named { x: 7 });
    }

    #[test]
    fn derive_owned_untagged_all_miss() {
        let result = with_chunked(br#""hello""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            Untagged::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ---- derive(DeserializeOwned): per-variant untagged ---------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum MixedTagged {
        Ping,
        Data(i64),
        #[strede(untagged)]
        Fallback(bool),
    }

    #[test]
    fn derive_owned_mixed_tagged_unit() {
        let (_, v) = with_chunked(br#""Ping""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            MixedTagged::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, MixedTagged::Ping);
    }

    #[test]
    fn derive_owned_mixed_tagged_newtype() {
        let (_, v) = with_chunked(br#"{"Data": 42}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            MixedTagged::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, MixedTagged::Data(42));
    }

    #[test]
    fn derive_owned_mixed_untagged_fallback() {
        let (_, v) = with_chunked(b"true", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            MixedTagged::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, MixedTagged::Fallback(true));
    }

    // ==== allow_unknown_fields (owned family) =================================

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(allow_unknown_fields)]
    struct LenientOwned {
        x: u32,
        y: u32,
    }

    #[test]
    fn derive_owned_allow_unknown_fields_basic() {
        let (_, v) = with_chunked(
            b"{\"x\": 1, \"extra\": 99, \"y\": 2}",
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                LenientOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, LenientOwned { x: 1, y: 2 });
    }

    #[test]
    fn derive_owned_allow_unknown_fields_nested_object() {
        let (_, v) = with_chunked(
            b"{\"x\": 1, \"nested\": {\"a\": [1,2,3]}, \"y\": 2}",
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                LenientOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, LenientOwned { x: 1, y: 2 });
    }

    #[test]
    fn derive_owned_allow_unknown_fields_string_value() {
        let (_, v) = with_chunked(
            b"{\"x\": 1, \"name\": \"hello\", \"y\": 2}",
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                LenientOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, LenientOwned { x: 1, y: 2 });
    }

    #[test]
    fn derive_owned_allow_unknown_fields_null_value() {
        let (_, v) = with_chunked(
            b"{\"x\": 1, \"gone\": null, \"y\": 2}",
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                LenientOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, LenientOwned { x: 1, y: 2 });
    }

    #[test]
    fn derive_owned_allow_unknown_fields_no_extra() {
        let (_, v) = with_chunked(b"{\"x\": 10, \"y\": 20}", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            LenientOwned::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, LenientOwned { x: 10, y: 20 });
    }

    #[test]
    fn derive_owned_allow_unknown_fields_multiple_unknowns() {
        let (_, v) = with_chunked(
            b"{\"a\": 1, \"x\": 5, \"b\": 2, \"y\": 6, \"c\": 3}",
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                LenientOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, LenientOwned { x: 5, y: 6 });
    }

    #[test]
    fn derive_owned_allow_unknown_fields_missing_required_still_misses() {
        let result = with_chunked(b"{\"x\": 1, \"extra\": 99}", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            LenientOwned::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ==== tuple struct derive (owned) =========================================

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct PairOwned(u32, u32);

    #[test]
    fn derive_owned_tuple_struct_basic() {
        let (_, v) = with_chunked(b"[10, 20]", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            PairOwned::deserialize_owned(de, ()).await.unwrap().unwrap()
        });
        assert_eq!(v, PairOwned(10, 20));
    }

    #[test]
    fn derive_owned_tuple_struct_wrong_length_is_miss() {
        let result = with_chunked(b"[1, 2, 3]", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            PairOwned::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[test]
    fn derive_owned_tuple_struct_non_array_is_miss() {
        let result = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            PairOwned::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct TripleOwned(u32, bool, u32);

    #[test]
    fn derive_owned_tuple_struct_mixed_types() {
        let (_, v) = with_chunked(b"[1, true, 3]", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            TripleOwned::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, TripleOwned(1, true, 3));
    }

    // ==== transparent derive (owned) ==========================================

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(transparent)]
    struct WrapperOwned {
        inner: u32,
    }

    #[test]
    fn derive_owned_transparent_named() {
        let (_, v) = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            WrapperOwned::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, WrapperOwned { inner: 42 });
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(transparent)]
    struct NewtypeOwned(u32);

    #[test]
    fn derive_owned_transparent_tuple() {
        let (_, v) = with_chunked(b"99", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            NewtypeOwned::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, NewtypeOwned(99));
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(transparent)]
    struct TransparentBoolOwned(bool);

    #[test]
    fn derive_owned_transparent_bool() {
        let (_, v) = with_chunked(b"true", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            TransparentBoolOwned::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, TransparentBoolOwned(true));
    }

    // ==== unit struct derive (owned) ==========================================

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct UnitStructOwned;

    #[test]
    fn derive_owned_unit_struct_null() {
        let (_, v) = with_chunked(b"null", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            UnitStructOwned::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, UnitStructOwned);
    }

    #[test]
    fn derive_owned_unit_struct_non_null_is_miss() {
        let result = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            UnitStructOwned::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ==== bound attribute (owned) =============================================

    // Supertrait pattern: anything implementing DeserializeOwned also
    // implements MyDeserializeOwned, so the generated body still compiles.
    trait MyDeserializeOwned: strede::DeserializeOwned {}
    impl<T: strede::DeserializeOwned> MyDeserializeOwned for T {}

    // -- container-level bound = "T: Copy + DeserializeOwned" -------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(bound = "T: Copy + strede::DeserializeOwned")]
    struct OwnedBoundedCopy<T> {
        value: T,
    }

    #[test]
    fn derive_owned_container_bound_copy_hit() {
        let (_, v) = with_chunked(br#"{"value": 42}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedBoundedCopy::<u32>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedBoundedCopy { value: 42u32 });
    }

    #[test]
    fn derive_owned_container_bound_copy_miss() {
        let result = with_chunked(br#""oops""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedBoundedCopy::<u32>::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // -- container-level bound replacing T: DeserializeOwned with MyDeserializeOwned

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(bound = "T: MyDeserializeOwned")]
    struct OwnedBoundedMyDeserialize<T> {
        a: T,
        b: u32,
    }

    #[test]
    fn derive_owned_container_bound_custom_trait_hit() {
        let (_, v) = with_chunked(br#"{"a": true, "b": 9}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedBoundedMyDeserialize::<bool>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedBoundedMyDeserialize { a: true, b: 9u32 });
    }

    #[test]
    fn derive_owned_container_bound_custom_trait_miss() {
        let result = with_chunked(br#""nope""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedBoundedMyDeserialize::<bool>::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // -- field-level bound: explicit Copy on one field, auto on the other -----

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct OwnedFieldBoundCopy<T, U> {
        #[strede(bound = "T: Copy + strede::DeserializeOwned")]
        first: T,
        second: U, // auto-bound: U: DeserializeOwned
    }

    #[test]
    fn derive_owned_field_bound_copy_hit() {
        let (_, v) = with_chunked(
            br#"{"first": 3, "second": false}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OwnedFieldBoundCopy::<u32, bool>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            OwnedFieldBoundCopy {
                first: 3u32,
                second: false
            }
        );
    }

    // -- field-level bound replacing T: DeserializeOwned with MyDeserializeOwned

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct OwnedFieldBoundMyDeserialize<T> {
        #[strede(bound = "T: MyDeserializeOwned")]
        inner: T,
        tag: u32,
    }

    #[test]
    fn derive_owned_field_bound_custom_trait_hit() {
        let (_, v) = with_chunked(br#"{"inner": 1, "tag": 2}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedFieldBoundMyDeserialize::<u32>::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(
            v,
            OwnedFieldBoundMyDeserialize {
                inner: 1u32,
                tag: 2
            }
        );
    }

    // ---- derive(DeserializeOwned): rename_all attribute ----------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(rename_all = "camelCase")]
    struct OwnedCamelCaseFields {
        first_name: String,
        last_name: String,
        age_years: u32,
    }

    #[test]
    fn derive_owned_rename_all_camel_case_hit() {
        let (_, v) = with_chunked(
            br#"{"firstName": "Alice", "lastName": "Smith", "ageYears": 30}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OwnedCamelCaseFields::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            OwnedCamelCaseFields {
                first_name: String::from("Alice"),
                last_name: String::from("Smith"),
                age_years: 30,
            }
        );
    }

    #[test]
    fn derive_owned_rename_all_original_name_is_miss() {
        let result = with_chunked(
            br#"{"first_name": "Alice", "last_name": "Smith", "age_years": 30}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OwnedCamelCaseFields::deserialize_owned(de, ()).await
            },
        );
        assert_eq!(result, Ok(Probe::Miss));
    }

    // rename_all with an explicit rename on one field - explicit wins

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(rename_all = "camelCase")]
    struct OwnedRenameAllWithExplicit {
        first_name: String,
        #[strede(rename = "custom_key")]
        last_name: String,
    }

    #[test]
    fn derive_owned_rename_all_explicit_rename_wins() {
        let (_, v) = with_chunked(
            br#"{"firstName": "Bob", "custom_key": "Jones"}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OwnedRenameAllWithExplicit::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            OwnedRenameAllWithExplicit {
                first_name: String::from("Bob"),
                last_name: String::from("Jones"),
            }
        );
    }

    // rename_all on enum variants

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(rename_all = "snake_case")]
    enum OwnedSnakeCaseVariants {
        MyVariant,
        AnotherOne(i64),
        WithStruct { x: u32, y: u32 },
    }

    #[test]
    fn derive_owned_rename_all_unit_variant() {
        let (_, v) = with_chunked(br#""my_variant""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedSnakeCaseVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedSnakeCaseVariants::MyVariant);
    }

    #[test]
    fn derive_owned_rename_all_newtype_variant() {
        let (_, v) = with_chunked(br#"{"another_one": 7}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedSnakeCaseVariants::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedSnakeCaseVariants::AnotherOne(7));
    }

    #[test]
    fn derive_owned_rename_all_struct_variant() {
        let (_, v) = with_chunked(
            br#"{"with_struct": {"x": 1, "y": 2}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OwnedSnakeCaseVariants::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, OwnedSnakeCaseVariants::WithStruct { x: 1, y: 2 });
    }

    #[test]
    fn derive_owned_rename_all_original_variant_name_is_miss() {
        let result = with_chunked(br#""MyVariant""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedSnakeCaseVariants::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(Probe::Miss));
    }

    // ---- derive(DeserializeOwned): other attribute ---------------------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum OwnedWithOther {
        Known,
        #[strede(other)]
        Unknown,
    }

    #[test]
    fn derive_owned_other_known_variant_hits() {
        let (_, v) = with_chunked(br#""Known""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedWithOther::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedWithOther::Known);
    }

    #[test]
    fn derive_owned_other_unknown_string_returns_other() {
        let (_, v) = with_chunked(br#""anything_else""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedWithOther::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedWithOther::Unknown);
    }

    // other in a mixed enum (unit + non-unit variants)

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    enum OwnedMixedWithOther {
        Unit,
        Pair(i64),
        #[strede(other)]
        Unknown,
    }

    #[test]
    fn derive_owned_other_mixed_known_unit() {
        let (_, v) = with_chunked(br#""Unit""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedMixedWithOther::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedMixedWithOther::Unit);
    }

    #[test]
    fn derive_owned_other_mixed_known_nonunit() {
        let (_, v) = with_chunked(br#"{"Pair": 42}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedMixedWithOther::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedMixedWithOther::Pair(42));
    }

    #[test]
    fn derive_owned_other_mixed_unknown_string() {
        let (_, v) = with_chunked(br#""nope""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedMixedWithOther::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedMixedWithOther::Unknown);
    }

    #[test]
    fn derive_owned_other_mixed_unknown_map_key() {
        let (_, v) = with_chunked(br#"{"nope": 99}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedMixedWithOther::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedMixedWithOther::Unknown);
    }

    // ---- derive(DeserializeOwned): from attribute (field level) --------------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct OwnedWithFrom {
        name: String,
        /// Deserializes as u32, then widens to u64 via From.
        #[strede(from = "u32")]
        count: u64,
    }

    #[test]
    fn derive_owned_field_from_converts() {
        let (_, v) = with_chunked(
            br#"{"name": "hi", "count": 7}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OwnedWithFrom::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            OwnedWithFrom {
                name: String::from("hi"),
                count: 7
            }
        );
    }

    // ---- derive(DeserializeOwned): try_from attribute (field level) ----------

    #[derive(Debug, PartialEq)]
    struct OwnedPositive(i64);

    impl TryFrom<i64> for OwnedPositive {
        type Error = ();
        fn try_from(v: i64) -> Result<Self, ()> {
            if v > 0 { Ok(OwnedPositive(v)) } else { Err(()) }
        }
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct OwnedWithTryFrom {
        #[strede(try_from = "i64")]
        value: OwnedPositive,
    }

    #[test]
    fn derive_owned_field_try_from_hit() {
        let (_, v) = with_chunked(br#"{"value": 5}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedWithTryFrom::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(
            v,
            OwnedWithTryFrom {
                value: OwnedPositive(5)
            }
        );
    }

    #[test]
    fn derive_owned_field_try_from_miss_on_conversion_failure() {
        let result = with_chunked(br#"{"value": -3}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedWithTryFrom::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(strede::Probe::Miss));
    }

    // ---- derive(DeserializeOwned): from attribute (container level) ----------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(from = "f64")]
    struct OwnedMetersWrapper(f64);

    impl From<f64> for OwnedMetersWrapper {
        fn from(v: f64) -> Self {
            OwnedMetersWrapper(v)
        }
    }

    #[test]
    fn derive_owned_container_from_converts() {
        let (_, v) = with_chunked(b"3.14", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedMetersWrapper::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedMetersWrapper(3.14));
    }

    // ---- derive(DeserializeOwned): try_from attribute (container level) ------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(try_from = "String")]
    struct OwnedNonEmptyString(String);

    impl TryFrom<String> for OwnedNonEmptyString {
        type Error = ();
        fn try_from(s: String) -> Result<Self, ()> {
            if s.is_empty() {
                Err(())
            } else {
                Ok(OwnedNonEmptyString(s))
            }
        }
    }

    #[test]
    fn derive_owned_container_try_from_hit() {
        let (_, v) = with_chunked(br#""hello""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedNonEmptyString::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedNonEmptyString(String::from("hello")));
    }

    #[test]
    fn derive_owned_container_try_from_miss_on_conversion_failure() {
        let result = with_chunked(br#""""#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedNonEmptyString::deserialize_owned(de, ()).await
        });
        assert_eq!(result, Ok(strede::Probe::Miss));
    }

    // ---- fork ---------------------------------------------------------------

    #[test]
    fn str_access_owned_fork_reads_same_content() {
        // Fork a StrAccessOwned at the start; both forks must collect the full string.
        let (_, (a, b)) = with_chunked(b"\"hello\\nworld\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async {
                let mut chunks = match e.deserialize_str_chunks().await? {
                    Probe::Hit(c) => c,
                    Probe::Miss => panic!("expected str chunks"),
                };
                let mut fork = chunks.fork();

                let mut a = String::new();
                let claim = loop {
                    match chunks.next_str(|s| String::from(s)).await? {
                        Chunk::Data((c, s)) => {
                            a.push_str(&s);
                            chunks = c;
                        }
                        Chunk::Done(claim) => break claim,
                    }
                };

                let mut b = String::new();
                while let Chunk::Data((c, s)) = fork.next_str(|s| String::from(s)).await? {
                    b.push_str(&s);
                    fork = c;
                }

                Ok(Probe::Hit((claim, (a, b))))
            })
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(a, "hello\nworld");
        assert_eq!(b, "hello\nworld");
    }

    /// Helper: create a skip-all arm stack for testing. Uses a closure-taking
    /// pattern so the KP/VP types are inferred from the map.
    async fn iterate_skip_all<M: MapAccessOwned>(
        map: M,
    ) -> Result<Probe<(M::MapClaim, ())>, M::Error> {
        use strede::MapKeyProbeOwned as _;
        use strede::MapValueProbeOwned as _;
        let arms = (
            MapArmBase,
            strede::VirtualArmSlot::new(
                |kp: M::KeyProbe| kp.deserialize_key::<strede::Skip, _>(()),
                |vp: owned::VP<M::KeyProbe>, _k: strede::Skip| async move {
                    let vc = vp.skip().await?;
                    Ok(strede::Probe::Hit((vc, ())))
                },
            ),
        );
        map.iterate(arms).await
    }

    #[test]
    fn map_access_owned_fork_counts_same_pairs() {
        // Fork a MapAccessOwned at the start; both forks must see all pairs.
        let (_, ()) = with_chunked(b"{\"a\": 1, \"b\": 2}", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            de.entry(|[e]| async {
                let mut map = match e.deserialize_map().await? {
                    Probe::Hit(m) => m,
                    Probe::Miss => panic!("expected map"),
                };
                let fork = map.fork();

                let (claim, ()) = hit!(iterate_skip_all(map).await);
                let (_fork_claim, ()) = hit!(iterate_skip_all(fork).await);

                Ok(Probe::Hit((claim, ())))
            })
            .await
            .unwrap()
            .unwrap()
        });
    }

    // ---- Match (owned family) ------------------------------------------------

    #[test]
    fn match_str_owned_hits() {
        let (_, v) = with_chunked(b"\"hello\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Match as DeserializeOwned<&str>>::deserialize_owned(de, "hello")
                .await
                .unwrap()
                .unwrap()
        });
        assert!(matches!(v, Match));
    }

    #[test]
    fn match_str_owned_misses_wrong_content() {
        let probe = with_chunked(b"\"hello\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Match as DeserializeOwned<&str>>::deserialize_owned(de, "world")
                .await
                .unwrap()
        });
        assert!(matches!(probe, Probe::Miss));
    }

    #[test]
    fn match_str_owned_misses_wrong_type() {
        let probe = with_chunked(b"42", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Match as DeserializeOwned<&str>>::deserialize_owned(de, "42")
                .await
                .unwrap()
        });
        assert!(matches!(probe, Probe::Miss));
    }

    #[test]
    fn match_bytes_owned_hits() {
        let (_, v) = with_chunked(b"\"hello\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Match as DeserializeOwned<&[u8]>>::deserialize_owned(de, b"hello")
                .await
                .unwrap()
                .unwrap()
        });
        assert!(matches!(v, Match));
    }

    #[test]
    fn match_bytes_owned_misses_wrong_content() {
        let probe = with_chunked(b"\"hello\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Match as DeserializeOwned<&[u8]>>::deserialize_owned(de, b"world")
                .await
                .unwrap()
        });
        assert!(matches!(probe, Probe::Miss));
    }

    // ---- MatchVals (owned family) --------------------------------------------

    #[test]
    fn match_vals_str_owned_hits_middle() {
        let (_, v) = with_chunked(b"\"warn\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <MatchVals<u8> as DeserializeOwned<[(&str, u8); 3]>>::deserialize_owned(
                de,
                [("ok", 0), ("warn", 1), ("error", 2)],
            )
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(v, MatchVals(1));
    }

    #[test]
    fn match_vals_str_owned_misses_unknown() {
        let probe = with_chunked(b"\"nope\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <MatchVals<u8> as DeserializeOwned<[(&str, u8); 3]>>::deserialize_owned(
                de,
                [("ok", 0), ("warn", 1), ("error", 2)],
            )
            .await
            .unwrap()
        });
        assert!(matches!(probe, Probe::Miss));
    }

    #[test]
    fn match_vals_bytes_owned_hits() {
        let (_, v) = with_chunked(b"\"error\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <MatchVals<u8> as DeserializeOwned<[(&[u8], u8); 3]>>::deserialize_owned(
                de,
                [
                    (b"ok".as_ref(), 0),
                    (b"warn".as_ref(), 1),
                    (b"error".as_ref(), 2),
                ],
            )
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(v, MatchVals(2));
    }

    #[test]
    fn match_str_array_owned_hits_any() {
        let (_, v) = with_chunked(b"\"b\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Match as DeserializeOwned<[&str; 3]>>::deserialize_owned(de, ["a", "b", "c"])
                .await
                .unwrap()
                .unwrap()
        });
        assert!(matches!(v, Match));
    }

    #[test]
    fn match_str_array_owned_misses_none() {
        let probe = with_chunked(b"\"d\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <Match as DeserializeOwned<[&str; 3]>>::deserialize_owned(de, ["a", "b", "c"])
                .await
                .unwrap()
        });
        assert!(matches!(probe, Probe::Miss));
    }

    #[test]
    fn match_vals_usize_str_owned_returns_index() {
        let (_, v) = with_chunked(b"\"b\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <MatchVals<usize> as DeserializeOwned<[&str; 3]>>::deserialize_owned(
                de,
                ["a", "b", "c"],
            )
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(v, MatchVals(1));
    }

    #[test]
    fn match_vals_usize_str_owned_misses() {
        let probe = with_chunked(b"\"z\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <MatchVals<usize> as DeserializeOwned<[&str; 3]>>::deserialize_owned(
                de,
                ["a", "b", "c"],
            )
            .await
            .unwrap()
        });
        assert!(matches!(probe, Probe::Miss));
    }

    // ---- UnwrapOrElse (owned family) -----------------------------------------

    #[test]
    fn unwrap_or_else_owned_hits_inner() {
        let (_, v) = with_chunked(b"\"b\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <UnwrapOrElse<MatchVals<usize>> as DeserializeOwned<(_, [(&str, usize); 3])>>::deserialize_owned(
                de,
                (async || MatchVals(99usize), [("a", 0), ("b", 1), ("c", 2)]),
            )
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(v, UnwrapOrElse(MatchVals(1)));
    }

    #[test]
    fn unwrap_or_else_owned_falls_back_on_miss() {
        let (_, v) = with_chunked(b"\"z\"", eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            <UnwrapOrElse<MatchVals<usize>> as DeserializeOwned<(_, [(&str, usize); 3])>>::deserialize_owned(
                de,
                (async || MatchVals(99usize), [("a", 0), ("b", 1), ("c", 2)]),
            )
            .await
            .unwrap()
            .unwrap()
        });
        assert_eq!(v, UnwrapOrElse(MatchVals(99)));
    }

    // ---- derive: #[strede(tag)] internally tagged enum (owned family) ---------

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(tag = "type")]
    enum OwnedTaggedEvent {
        Ping,
        Pong,
    }

    #[test]
    fn derive_owned_internally_tagged_unit_hits_first() {
        let (_, v) = with_chunked(br#"{"type": "Ping"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedTaggedEvent::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedTaggedEvent::Ping);
    }

    #[test]
    fn derive_owned_internally_tagged_unit_hits_second() {
        let (_, v) = with_chunked(br#"{"type": "Pong"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            OwnedTaggedEvent::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, OwnedTaggedEvent::Pong);
    }

    #[test]
    fn derive_owned_internally_tagged_unit_unknown_variant_misses() {
        let missed = with_chunked(br#"{"type": "Unknown"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let probe = OwnedTaggedEvent::deserialize_owned(de, ()).await.unwrap();
            matches!(probe, Probe::Miss)
        });
        assert!(missed, "unknown variant should return Miss");
    }

    #[test]
    fn derive_owned_internally_tagged_unit_missing_tag_misses() {
        let missed = with_chunked(br#"{"other": "Ping"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let probe = OwnedTaggedEvent::deserialize_owned(de, ()).await.unwrap();
            matches!(probe, Probe::Miss)
        });
        assert!(missed, "missing tag field should return Miss");
    }

    #[test]
    fn derive_owned_internally_tagged_unit_tag_after_other_key() {
        // Tag is not the first key - should still be found.
        let (_, v) = with_chunked(
            br#"{"extra": "ignored", "type": "Ping"}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                OwnedTaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, OwnedTaggedEvent::Ping);
    }

    // ---- derive: #[strede(tag)] internally tagged enum with non-unit variants (owned family) ----

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    struct Vec2 {
        x: f64,
        y: f64,
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(tag = "type")]
    enum TaggedEvent {
        Ping,
        Move { x: f64, y: f64 },
        Teleport(Vec2),
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(tag = "kind")]
    enum SimpleTagged {
        Wrap(Vec2),
    }

    #[test]
    fn derive_owned_internally_tagged_simple_newtype() {
        let (_, v) = with_chunked(
            br#"{"kind": "Wrap", "x": 1.0, "y": 2.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                let result = SimpleTagged::deserialize_owned(de, ()).await;
                result.unwrap().unwrap()
            },
        );
        assert_eq!(v, SimpleTagged::Wrap(Vec2 { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn derive_owned_internally_tagged_struct_variant() {
        let (_, v) = with_chunked(
            br#"{"type": "Move", "x": 1.0, "y": 2.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                TaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, TaggedEvent::Move { x: 1.0, y: 2.0 });
    }

    #[test]
    fn derive_owned_internally_tagged_newtype_variant() {
        let (_, v) = with_chunked(
            br#"{"type": "Teleport", "x": 3.0, "y": 4.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                let result = TaggedEvent::deserialize_owned(de, ()).await;
                result.unwrap().unwrap()
            },
        );
        assert_eq!(v, TaggedEvent::Teleport(Vec2 { x: 3.0, y: 4.0 }));
    }

    #[test]
    fn derive_owned_internally_tagged_unit_variant_mixed() {
        let (_, v) = with_chunked(br#"{"type": "Ping"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            TaggedEvent::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, TaggedEvent::Ping);
    }

    #[test]
    fn derive_owned_internally_tagged_tag_not_first() {
        let (_, v) = with_chunked(
            br#"{"x": 5.0, "type": "Move", "y": 6.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                TaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, TaggedEvent::Move { x: 5.0, y: 6.0 });
    }

    #[test]
    fn derive_owned_internally_tagged_unknown_variant_misses() {
        let missed = with_chunked(
            br#"{"type": "Unknown", "x": 1.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                let probe = TaggedEvent::deserialize_owned(de, ()).await.unwrap();
                probe.is_miss()
            },
        );
        assert!(missed);
    }

    #[test]
    fn derive_owned_internally_tagged_missing_tag_non_unit_misses() {
        let missed = with_chunked(br#"{"x": 1.0, "y": 2.0}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let probe = TaggedEvent::deserialize_owned(de, ()).await.unwrap();
            probe.is_miss()
        });
        assert!(
            missed,
            "missing tag field should return Miss for non-unit enum"
        );
    }

    // ---- derive: #[strede(tag)] with rename/alias on variants ----

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(tag = "t")]
    enum RenamedTagged {
        #[strede(rename = "start")]
        Begin,
        #[strede(alias = "quit", alias = "bye")]
        End,
    }

    #[test]
    fn derive_owned_internally_tagged_rename_variant() {
        let (_, v) = with_chunked(br#"{"t": "start"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            RenamedTagged::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, RenamedTagged::Begin);
    }

    #[test]
    fn derive_owned_internally_tagged_original_name_misses_after_rename() {
        let missed = with_chunked(br#"{"t": "Begin"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            let probe = RenamedTagged::deserialize_owned(de, ()).await.unwrap();
            probe.is_miss()
        });
        assert!(missed, "original name should not match after rename");
    }

    #[test]
    fn derive_owned_internally_tagged_alias_first() {
        let (_, v) = with_chunked(br#"{"t": "quit"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            RenamedTagged::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, RenamedTagged::End);
    }

    #[test]
    fn derive_owned_internally_tagged_alias_second() {
        let (_, v) = with_chunked(br#"{"t": "bye"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            RenamedTagged::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, RenamedTagged::End);
    }

    #[test]
    fn derive_owned_internally_tagged_canonical_name_still_matches_alongside_alias() {
        let (_, v) = with_chunked(br#"{"t": "End"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            RenamedTagged::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, RenamedTagged::End);
    }

    // ---- derive: #[strede(tag)] with rename_all ----

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(tag = "kind", rename_all = "snake_case")]
    enum SnakeCaseTagged {
        UserCreated,
        OrderPlaced { id: u64 },
    }

    #[test]
    fn derive_owned_internally_tagged_rename_all_unit() {
        let (_, v) = with_chunked(
            br#"{"kind": "user_created"}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                SnakeCaseTagged::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, SnakeCaseTagged::UserCreated);
    }

    #[test]
    fn derive_owned_internally_tagged_rename_all_struct_variant() {
        let (_, v) = with_chunked(
            br#"{"kind": "order_placed", "id": 42}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                SnakeCaseTagged::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, SnakeCaseTagged::OrderPlaced { id: 42 });
    }

    // ---- derive: #[strede(tag, content)] adjacently tagged enum (owned family) ----

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(tag = "type", content = "data")]
    enum AdjTaggedUnit {
        Ping,
        Pong,
    }

    #[test]
    fn derive_owned_adjacent_tagged_unit_hits_first() {
        let (_, v) = with_chunked(br#"{"type": "Ping"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AdjTaggedUnit::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, AdjTaggedUnit::Ping);
    }

    #[test]
    fn derive_owned_adjacent_tagged_unit_hits_second() {
        let (_, v) = with_chunked(br#"{"type": "Pong"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AdjTaggedUnit::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, AdjTaggedUnit::Pong);
    }

    #[test]
    fn derive_owned_adjacent_tagged_unit_unknown_misses() {
        let missed = with_chunked(br#"{"type": "Unknown"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AdjTaggedUnit::deserialize_owned(de, ())
                .await
                .unwrap()
                .is_miss()
        });
        assert!(missed);
    }

    #[derive(strede_derive::DeserializeOwned, Debug, PartialEq)]
    #[strede(tag = "t", content = "c")]
    enum AdjTaggedEvent {
        Ping,
        Move { x: f64, y: f64 },
        Teleport(Vec2),
    }

    #[test]
    fn derive_owned_adjacent_tagged_struct_variant() {
        let (_, v) = with_chunked(
            br#"{"t": "Move", "c": {"x": 1.0, "y": 2.0}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AdjTaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, AdjTaggedEvent::Move { x: 1.0, y: 2.0 });
    }

    #[test]
    fn derive_owned_adjacent_tagged_newtype_variant() {
        let (_, v) = with_chunked(
            br#"{"t": "Teleport", "c": {"x": 3.0, "y": 4.0}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AdjTaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, AdjTaggedEvent::Teleport(Vec2 { x: 3.0, y: 4.0 }));
    }

    #[test]
    fn derive_owned_adjacent_tagged_unit_in_mixed_enum() {
        let (_, v) = with_chunked(br#"{"t": "Ping"}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            AdjTaggedEvent::deserialize_owned(de, ())
                .await
                .unwrap()
                .unwrap()
        });
        assert_eq!(v, AdjTaggedEvent::Ping);
    }

    #[test]
    fn derive_owned_adjacent_tagged_content_before_tag() {
        // JSON key order shouldn't matter.
        let (_, v) = with_chunked(
            br#"{"c": {"x": 5.0, "y": 6.0}, "t": "Move"}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AdjTaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, AdjTaggedEvent::Move { x: 5.0, y: 6.0 });
    }

    #[test]
    fn derive_owned_adjacent_tagged_unknown_variant_misses() {
        let missed = with_chunked(
            br#"{"t": "Unknown", "c": {"x": 1.0}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AdjTaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .is_miss()
            },
        );
        assert!(missed);
    }

    #[test]
    fn derive_owned_adjacent_tagged_missing_tag_misses() {
        let missed = with_chunked(
            br#"{"c": {"x": 1.0, "y": 2.0}}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AdjTaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .is_miss()
            },
        );
        assert!(missed);
    }

    #[test]
    fn derive_owned_adjacent_tagged_extra_fields_ignored() {
        let (_, v) = with_chunked(
            br#"{"extra": true, "t": "Move", "c": {"x": 7.0, "y": 8.0}, "also": 1}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                AdjTaggedEvent::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(v, AdjTaggedEvent::Move { x: 7.0, y: 8.0 });
    }

    // -------------------------------------------------------------------------
    // flatten
    // -------------------------------------------------------------------------

    #[derive(Debug, PartialEq, strede_derive::DeserializeOwned)]
    struct FlatInner {
        x: f64,
        y: f64,
    }

    #[derive(Debug, PartialEq, strede_derive::DeserializeOwned)]
    struct FlatOuter {
        name: String,
        #[strede(flatten)]
        pos: FlatInner,
    }

    // A struct where the flatten field comes before a regular field (order stress test).
    #[derive(Debug, PartialEq, strede_derive::DeserializeOwned)]
    struct FlatLeadingInner {
        #[strede(flatten)]
        pos: FlatInner,
        label: String,
    }

    #[test]
    fn derive_owned_flatten_basic() {
        let (_, v) = with_chunked(
            br#"{"name": "origin", "x": 1.0, "y": 2.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                FlatOuter::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            FlatOuter {
                name: "origin".to_string(),
                pos: FlatInner { x: 1.0, y: 2.0 }
            }
        );
    }

    #[test]
    fn derive_owned_flatten_inner_keys_first() {
        // Inner (flatten) keys come before the outer key.
        let (_, v) = with_chunked(
            br#"{"x": 3.0, "y": 4.0, "name": "point"}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                FlatOuter::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            FlatOuter {
                name: "point".to_string(),
                pos: FlatInner { x: 3.0, y: 4.0 }
            }
        );
    }

    #[test]
    fn derive_owned_flatten_extra_fields_ignored() {
        // Extra fields not claimed by outer or inner are skipped.
        let (_, v) = with_chunked(
            br#"{"name": "p", "x": 5.0, "extra": true, "y": 6.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                FlatOuter::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            FlatOuter {
                name: "p".to_string(),
                pos: FlatInner { x: 5.0, y: 6.0 }
            }
        );
    }

    #[test]
    fn derive_owned_flatten_missing_inner_field_misses() {
        // Inner field missing - should return Miss.
        let missed = with_chunked(
            br#"{"name": "p", "x": 1.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                FlatOuter::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .is_miss()
            },
        );
        assert!(missed);
    }

    #[test]
    fn derive_owned_flatten_missing_outer_field_misses() {
        // Outer (non-flatten) field missing - should return Miss.
        let missed = with_chunked(br#"{"x": 1.0, "y": 2.0}"#, eof_loader(), async |shared| {
            let de = ChunkedJsonDeserializer::new(shared);
            FlatOuter::deserialize_owned(de, ())
                .await
                .unwrap()
                .is_miss()
        });
        assert!(missed);
    }

    #[test]
    fn derive_owned_flatten_field_first() {
        // Flatten field declared first in struct - outer field comes after in JSON.
        let (_, v) = with_chunked(
            br#"{"x": 7.0, "y": 8.0, "label": "test"}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                FlatLeadingInner::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            FlatLeadingInner {
                pos: FlatInner { x: 7.0, y: 8.0 },
                label: "test".to_string()
            }
        );
    }

    // -------------------------------------------------------------------------
    // multi-flatten
    // -------------------------------------------------------------------------

    #[derive(Debug, PartialEq, strede_derive::DeserializeOwned)]
    struct FlatColor {
        r: u8,
        g: u8,
        b: u8,
    }

    #[derive(Debug, PartialEq, strede_derive::DeserializeOwned)]
    struct MultiFlat {
        name: String,
        #[strede(flatten(boxed))]
        pos: FlatInner,
        #[strede(flatten(boxed))]
        color: FlatColor,
    }

    #[test]
    fn derive_owned_multi_flatten_basic() {
        let (_, v) = with_chunked(
            br#"{"name": "pt", "x": 1.0, "y": 2.0, "r": 255, "g": 128, "b": 0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                MultiFlat::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            MultiFlat {
                name: "pt".to_string(),
                pos: FlatInner { x: 1.0, y: 2.0 },
                color: FlatColor {
                    r: 255,
                    g: 128,
                    b: 0
                },
            }
        );
    }

    #[test]
    fn derive_owned_multi_flatten_interleaved_keys() {
        // Keys from all three groups interleaved.
        let (_, v) = with_chunked(
            br#"{"r": 10, "x": 3.0, "name": "mix", "g": 20, "y": 4.0, "b": 30}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                MultiFlat::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            MultiFlat {
                name: "mix".to_string(),
                pos: FlatInner { x: 3.0, y: 4.0 },
                color: FlatColor {
                    r: 10,
                    g: 20,
                    b: 30
                },
            }
        );
    }

    #[test]
    fn derive_owned_multi_flatten_missing_second_field_misses() {
        // Missing a field from the second flatten type.
        let missed = with_chunked(
            br#"{"name": "pt", "x": 1.0, "y": 2.0, "r": 255, "g": 128}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                MultiFlat::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .is_miss()
            },
        );
        assert!(missed);
    }

    #[test]
    fn derive_owned_multi_flatten_extra_fields_ignored() {
        let (_, v) = with_chunked(
            br#"{"name": "pt", "x": 1.0, "y": 2.0, "r": 0, "g": 0, "b": 0, "extra": true}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                MultiFlat::deserialize_owned(de, ()).await.unwrap().unwrap()
            },
        );
        assert_eq!(
            v,
            MultiFlat {
                name: "pt".to_string(),
                pos: FlatInner { x: 1.0, y: 2.0 },
                color: FlatColor { r: 0, g: 0, b: 0 },
            }
        );
    }

    #[derive(Debug, PartialEq, strede_derive::DeserializeOwned)]
    struct FlatSize {
        w: f64,
        h: f64,
    }

    #[derive(Debug, PartialEq, strede_derive::DeserializeOwned)]
    struct TripleFlat {
        #[strede(flatten)]
        // #[strede(flatten(boxed))]
        pos: FlatInner,
        #[strede(flatten)]
        // #[strede(flatten(boxed))]
        color: FlatColor,
        #[strede(flatten)]
        // #[strede(flatten(boxed))]
        size: FlatSize,
    }

    #[test]
    fn derive_owned_triple_flatten() {
        let (_, v) = with_chunked(
            br#"{"x": 1.0, "y": 2.0, "r": 10, "g": 20, "b": 30, "w": 100.0, "h": 200.0}"#,
            eof_loader(),
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                TripleFlat::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        );
        assert_eq!(
            v,
            TripleFlat {
                pos: FlatInner { x: 1.0, y: 2.0 },
                color: FlatColor {
                    r: 10,
                    g: 20,
                    b: 30
                },
                size: FlatSize { w: 100.0, h: 200.0 },
            }
        );
    }
}
