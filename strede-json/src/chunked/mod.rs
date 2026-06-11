//! Chunked JSON deserializer for async streaming input.
//!
//! Uses [`strede::SharedBuf`]/[`strede::Handle`] to coordinate access to a
//! buffer that is refilled asynchronously by a user-supplied loader closure.
//! The top-level deserializer holds a `SharedBuf`; sub-deserializers (for
//! option/map values/seq elements) hold a `Handle` forked from it.
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

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::JsonError;
use crate::token::{self, SimpleToken, Token, Tokenizer};
use core::future::Future;
use core::mem;
use strede::utils::repeat;
use strede::{
    Ascii, Buffer, DeserializeFromEnumOwned, DeserializeFromMapOwned, DeserializeFromSeqOwned,
    DeserializeOwned, DeserializerOwned, EntryOwned, EnumAccessOwned, EnumArmStackOwned,
    EnumVariantProbeOwned, Handle, NumberEncoding, Probe, SharedBuf, hit,
};

// ---------------------------------------------------------------------------
// Claim
// ---------------------------------------------------------------------------

/// The concrete [`DeserializerOwned::Claim`] type for the chunked JSON deserializer, threaded back
/// through [`DeserializerOwned::entry`], accessor `next` methods, and `skip` calls.
///
/// Carries the post-token tokenizer state, the chunk offset within the current
/// buffer slice, and the [`Handle`] that holds ownership of buffer access until
/// the parent is ready to advance the stream.
pub struct ChunkedJsonClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) offset: usize,
    pub(crate) handle: Handle<'s, B, F>,
}

// ---------------------------------------------------------------------------
// Helpers - pointer arithmetic to recover offset after tokenizer advances `src`
// ---------------------------------------------------------------------------

#[inline]
pub(super) fn new_offset(buf: &[u8], src: &[u8]) -> usize {
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
    /// Buffer offset before the leading token bytes were consumed. Used by
    /// raw-value capture in [`into_raw_source`].
    start_offset: usize,
    pending_tok: Option<Token>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedJsonSubDeserializer<'s, B, F> {
    #[inline(always)]
    pub(crate) fn new(
        handle: Handle<'s, B, F>,
        tokenizer: Tokenizer,
        offset: usize,
        start_offset: usize,
        tok: Token,
    ) -> Self {
        Self {
            handle,
            tokenizer,
            offset,
            start_offset,
            pending_tok: Some(tok),
        }
    }

    /// Hand out the inner pieces of this sub-deserializer so that raw-value
    /// types (e.g. `RawValueOwned`) can capture the underlying byte stream.
    ///
    /// Returns `(handle, tokenizer, offset, start_offset, pending_tok)`.
    /// Because the chunked side is streaming, this returns the source state
    /// rather than a contiguous byte slice; the caller is responsible for
    /// driving the stream to capture bytes.
    #[cfg(feature = "alloc")]
    #[inline(always)]
    pub(crate) fn into_raw_source(
        self,
    ) -> (Handle<'s, B, F>, Tokenizer, usize, usize, Option<Token>) {
        (
            self.handle,
            self.tokenizer,
            self.offset,
            self.start_offset,
            self.pending_tok,
        )
    }
}

/// Advance `handle` to the next chunk via `Handle::next`, resetting `offset`
/// and verifying that the new chunk is non-empty (else `UnexpectedEnd`).
#[inline(always)]
pub(super) async fn refill<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
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
pub(super) async fn next_dispatch<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
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
            let old = mem::replace(tokenizer, Tokenizer::new());
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
#[inline(always)]
fn build_entries<'s, B: Buffer, F: AsyncFnMut(&mut B), const N: usize>(
    handle: Handle<'s, B, F>,
    token: Token,
    tokenizer: Tokenizer,
    offset: usize,
    start_offset: usize,
) -> [ChunkedJsonEntry<'s, B, F>; N] {
    let entry = ChunkedJsonEntry {
        handle,
        token,
        tokenizer,
        offset,
        start_offset,
    };
    repeat(entry, ChunkedJsonEntry::clone)
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

    let start_offset = offset;
    let (main, token) = next_dispatch(main, &mut tokenizer, &mut offset, &mut pending_tok).await?;

    let snap_tokenizer = tokenizer.clone();
    let snap_offset = offset;

    let entries = build_entries::<B, F, N>(
        main,
        token.clone(),
        snap_tokenizer,
        snap_offset,
        start_offset,
    );

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

    #[inline(always)]
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

/// Sub-deserializer `next`: uses its owned handle, returns `ChunkedJsonClaim`
/// directly (claim = entry claim).
async fn run_next_sub<'s, B, F, const N: usize, Fn_, Fut, R>(
    de: ChunkedJsonSubDeserializer<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<(ChunkedJsonClaim<'s, B, F>, R)>, JsonError>
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

    let start_offset = de.start_offset;
    let (main, token) = next_dispatch(main, &mut tokenizer, &mut offset, &mut pending_tok).await?;

    let snap_tokenizer = tokenizer.clone();
    let snap_offset = offset;

    let entries = build_entries::<B, F, N>(
        main,
        token.clone(),
        snap_tokenizer,
        snap_offset,
        start_offset,
    );

    f(entries).await
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedJsonSubDeserializer<'s, B, F>
{
    type Error = JsonError;
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type EntryClaim = ChunkedJsonClaim<'s, B, F>;
    type Entry = ChunkedJsonEntry<'s, B, F>;

    #[inline(always)]
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
    /// Buffer offset before the leading token bytes were consumed.
    /// Used by `deserialize_value` raw-capture to include the token header.
    start_offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedJsonEntry<'s, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            start_offset: self.start_offset,
        }
    }

    #[inline(always)]
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
            #[inline(always)]
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

/// Capture one complete JSON value as raw bytes, appending to `out`.
/// `start_offset` is the buffer position before the leading token was read.
/// Mirrors `skip_value_chunked` but collects bytes instead of discarding them.
#[cfg(feature = "alloc")]
pub(crate) async fn capture_raw_value_chunked<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    tokenizer: &mut Tokenizer,
    offset: &mut usize,
    start_offset: usize,
    tok: Token,
    out: &mut alloc::vec::Vec<u8>,
) -> Result<Handle<'s, B, F>, JsonError> {
    // `seg_start` tracks the beginning of the not-yet-flushed segment within
    // the current buffer chunk. On refill it resets to 0.
    let mut seg_start = start_offset;

    // Flush `buf[seg_start..*offset]` then refill; resets seg_start to 0.
    macro_rules! flush_and_refill {
        () => {{
            out.extend_from_slice(&handle.buf()[seg_start..*offset]);
            handle = refill(handle, offset).await?;
            seg_start = 0;
        }};
    }

    match tok {
        Token::Simple(SimpleToken::Null | SimpleToken::Bool(_), t) => {
            out.extend_from_slice(&handle.buf()[seg_start..*offset]);
            *tokenizer = t;
        }
        Token::Number(mut access) => {
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
                    Ok(None) => {
                        out.extend_from_slice(&handle.buf()[seg_start..*offset]);
                        break;
                    }
                    Err(JsonError::UnexpectedEnd) => {
                        flush_and_refill!();
                    }
                    Err(e) => return Err(e),
                }
            }
            *tokenizer = Tokenizer::new();
        }
        Token::Str(mut access) => {
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
                    Ok(None) => {
                        out.extend_from_slice(&handle.buf()[seg_start..*offset]);
                        break;
                    }
                    Err(JsonError::UnexpectedEnd) => {
                        flush_and_refill!();
                    }
                    Err(e) => return Err(e),
                }
            }
            *tokenizer = Tokenizer::new();
        }
        Token::Simple(SimpleToken::ObjectStart | SimpleToken::ArrayStart, t) => {
            *tokenizer = t;
            let mut depth = 1usize;
            while depth > 0 {
                let mut pending = None;
                // Flush the accumulated segment up to the current position
                // before calling next_dispatch, which may cross chunk boundaries
                // internally (invalidating seg_start).
                out.extend_from_slice(&handle.buf()[seg_start..*offset]);
                let (h, inner_tok) = next_dispatch(handle, tokenizer, offset, &mut pending).await?;
                handle = h;
                // After next_dispatch, start a new segment from wherever
                // the buffer now is. The token header bytes were emitted
                // implicitly: next_dispatch consumed them advancing *offset.
                // We want to emit them too, so reset seg_start to before
                // the token header. However next_dispatch may have refilled,
                // so we can't go backwards. Instead we must re-emit the
                // structural byte from the token itself.
                //
                // Solution: emit the token's leading byte(s) directly from
                // the token enum, then continue tracking from *offset.
                match inner_tok {
                    Token::Simple(SimpleToken::ObjectStart, t) => {
                        out.push(b'{');
                        *tokenizer = t;
                        seg_start = *offset;
                        depth += 1;
                    }
                    Token::Simple(SimpleToken::ArrayStart, t) => {
                        out.push(b'[');
                        *tokenizer = t;
                        seg_start = *offset;
                        depth += 1;
                    }
                    Token::Simple(SimpleToken::ObjectEnd, t) => {
                        out.push(b'}');
                        *tokenizer = t;
                        seg_start = *offset;
                        depth -= 1;
                    }
                    Token::Simple(SimpleToken::ArrayEnd, t) => {
                        out.push(b']');
                        *tokenizer = t;
                        seg_start = *offset;
                        depth -= 1;
                    }
                    Token::Simple(SimpleToken::Null, t) => {
                        out.extend_from_slice(b"null");
                        *tokenizer = t;
                        seg_start = *offset;
                    }
                    Token::Simple(SimpleToken::Bool(true), t) => {
                        out.extend_from_slice(b"true");
                        *tokenizer = t;
                        seg_start = *offset;
                    }
                    Token::Simple(SimpleToken::Bool(false), t) => {
                        out.extend_from_slice(b"false");
                        *tokenizer = t;
                        seg_start = *offset;
                    }
                    Token::Simple(SimpleToken::Comma, t) => {
                        out.push(b',');
                        *tokenizer = t;
                        seg_start = *offset;
                    }
                    Token::Simple(SimpleToken::Colon, t) => {
                        out.push(b':');
                        *tokenizer = t;
                        seg_start = *offset;
                    }
                    Token::Number(mut access) => {
                        seg_start = *offset;
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
                                Ok(None) => {
                                    out.extend_from_slice(&handle.buf()[seg_start..*offset]);
                                    seg_start = *offset;
                                    break;
                                }
                                Err(JsonError::UnexpectedEnd) => {
                                    out.extend_from_slice(
                                        &handle.buf()[seg_start..handle.buf().len()],
                                    );
                                    handle = refill(handle, offset).await?;
                                    seg_start = 0;
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        *tokenizer = Tokenizer::new();
                    }
                    Token::Str(mut access) => {
                        // The leading `"` was already consumed by next_dispatch;
                        // emit it explicitly. The closing `"` is captured below
                        // because the str-access advances `*offset` past it
                        // before returning `Ok(None)`.
                        out.push(b'"');
                        seg_start = *offset;
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
                                Ok(None) => {
                                    out.extend_from_slice(&handle.buf()[seg_start..*offset]);
                                    seg_start = *offset;
                                    break;
                                }
                                Err(JsonError::UnexpectedEnd) => {
                                    out.extend_from_slice(
                                        &handle.buf()[seg_start..handle.buf().len()],
                                    );
                                    handle = refill(handle, offset).await?;
                                    seg_start = 0;
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        *tokenizer = Tokenizer::new();
                    }
                    _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
                }
            }
            // No trailing flush needed: seg_start == *offset after the last token.
        }
        _ => return Err(JsonError::UnexpectedByte { byte: 0 }),
    }
    Ok(handle)
}

/// Bit-per-depth open-bracket tracker for `skip_value_chunked`. Bit `i` is 1
/// when depth `i+1` was opened by `{`, 0 when opened by `[`. Depths beyond
/// 4096 are not validated.
struct FixedBitSet([u64; 64]);

impl FixedBitSet {
    const fn new() -> Self {
        Self([0u64; 64])
    }

    /// Returns `false` when `depth` exceeds the tracked range (> 4096).
    fn set(&mut self, depth: usize, is_object: bool) -> bool {
        let i = depth - 1;
        if i >= 4096 {
            return false;
        }
        let word = i / 64;
        let bit = i % 64;
        if is_object {
            self.0[word] |= 1u64 << bit;
        } else {
            self.0[word] &= !(1u64 << bit);
        }
        true
    }

    fn is_object(&self, depth: usize) -> Option<bool> {
        let i = depth - 1;
        if i >= 4096 {
            return None;
        }
        let word = i / 64;
        let bit = i % 64;
        Some((self.0[word] >> bit) & 1 == 1)
    }
}

/// Skip one complete JSON value (scalar, string, object, or array) in chunked
/// mode. Uses iterative depth tracking instead of recursion to stay no_std.
pub(super) async fn skip_value_chunked<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
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
        Token::Simple(open @ (SimpleToken::ObjectStart | SimpleToken::ArrayStart), t) => {
            *tokenizer = t;
            let mut depth = 1usize;
            let mut stack = FixedBitSet::new();
            stack.set(depth, open == SimpleToken::ObjectStart);
            while depth > 0 {
                let mut pending = None;
                let (h, tok) = next_dispatch(handle, tokenizer, offset, &mut pending).await?;
                handle = h;
                match tok {
                    Token::Simple(
                        inner @ (SimpleToken::ObjectStart | SimpleToken::ArrayStart),
                        t,
                    ) => {
                        depth += 1;
                        if stack.set(depth, inner == SimpleToken::ObjectStart) {
                            *tokenizer = t;
                        } else {
                            #[cfg(feature = "alloc")]
                            {
                                handle = alloc::boxed::Box::pin(skip_value_chunked(
                                    handle,
                                    tokenizer,
                                    offset,
                                    Token::Simple(inner, t),
                                ))
                                .await?;
                                depth -= 1;
                            }
                            #[cfg(not(feature = "alloc"))]
                            {
                                *tokenizer = t;
                            }
                        }
                    }
                    Token::Simple(close @ (SimpleToken::ObjectEnd | SimpleToken::ArrayEnd), t) => {
                        if let Some(was_object) = stack.is_object(depth) {
                            let is_object_end = close == SimpleToken::ObjectEnd;
                            if was_object != is_object_end {
                                return Err(JsonError::UnexpectedByte { byte: 0 });
                            }
                        }
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
    type SubDeserializer = ChunkedJsonSubDeserializer<'s, B, F>;
    type StrChunks = ChunkedJsonStrAccess<'s, B, F>;
    type BytesChunks = ChunkedJsonBytesAccess<'s, B, F>;
    type NumberChunks<Enc: NumberEncoding> = ChunkedJsonNumberAccess<'s, B, F>;
    type Map = ChunkedJsonMapAccessOwned<'s, B, F>;
    type Seq = ChunkedJsonSeqAccess<'s, B, F>;
    type Enum = ChunkedJsonEnumAccess<'s, B, F>;
    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            start_offset: self.start_offset,
        }
    }

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if Enc::NAME != Ascii::NAME {
            return Ok(Probe::Miss);
        }
        match self.token {
            Token::Number(access) => Ok(Probe::Hit(ChunkedJsonNumberAccess {
                handle: self.handle,
                access,
                offset: self.offset,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
    async fn deserialize_option<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        match self.token {
            Token::Simple(SimpleToken::Null, tok) => Ok(Probe::Hit((
                Self::into_claim_with_tok(self.handle, self.offset, tok),
                None,
            ))),
            other => {
                let sub = ChunkedJsonSubDeserializer::new(
                    self.handle,
                    self.tokenizer,
                    self.offset,
                    self.start_offset,
                    other,
                );
                let (claim, v) = hit!(T::deserialize_owned(sub, extra).await);
                Ok(Probe::Hit((claim, Some(v))))
            }
        }
    }

    #[inline(always)]
    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let sub = ChunkedJsonSubDeserializer::new(
            self.handle,
            self.tokenizer,
            self.offset,
            self.start_offset,
            self.token,
        );
        T::deserialize_owned(sub, extra).await
    }

    #[inline(always)]
    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMapOwned<Self::Map>,
    {
        let map = match EntryOwned::deserialize_map(self).await? {
            Probe::Hit(m) => m,
            Probe::Miss => return Ok(Probe::Miss),
        };
        T::deserialize_from_map_owned(map, extra).await
    }

    #[inline(always)]
    async fn deserialize_seq_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeqOwned<Self::Seq>,
    {
        let seq = match EntryOwned::deserialize_seq(self).await? {
            Probe::Hit(s) => s,
            Probe::Miss => return Ok(Probe::Miss),
        };
        T::deserialize_from_seq_owned(seq, extra).await
    }

    #[inline(always)]
    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        Ok(Probe::Hit(ChunkedJsonEnumAccess {
            handle: self.handle,
            token: self.token,
            tokenizer: self.tokenizer,
            offset: self.offset,
            start_offset: self.start_offset,
        }))
    }

    #[inline(always)]
    async fn deserialize_enum_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnumOwned<Self::Enum>,
    {
        let enum_access = ChunkedJsonEnumAccess {
            handle: self.handle,
            token: self.token,
            tokenizer: self.tokenizer,
            offset: self.offset,
            start_offset: self.start_offset,
        };
        T::deserialize_from_enum_owned(enum_access, extra).await
    }

    #[inline(always)]
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

mod access;
mod primitives;

use access::{
    ChunkedJsonBytesAccess, ChunkedJsonMapAccessOwned, ChunkedJsonNumberAccess,
    ChunkedJsonSeqAccess, ChunkedJsonStrAccess,
};

// ---------------------------------------------------------------------------
// ChunkedJsonEnumAccess / ChunkedJsonEnumVariantProbe
// ---------------------------------------------------------------------------

/// [`EnumAccessOwned`] for the chunked JSON deserializer.
///
/// - Unit variants: bare string token (`"VariantName"`)
/// - Non-unit variants: single-key object (`{"VariantName": <payload>}`)
pub struct ChunkedJsonEnumAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    token: Token,
    tokenizer: Tokenizer,
    offset: usize,
    start_offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumAccessOwned for ChunkedJsonEnumAccess<'s, B, F> {
    type Error = JsonError;
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type VariantProbe = ChunkedJsonEnumVariantProbe<'s, B, F>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStackOwned<Self::VariantProbe>,
    {
        let vp = ChunkedJsonEnumVariantProbe {
            handle: self.handle,
            token: self.token,
            tokenizer: self.tokenizer,
            offset: self.offset,
            start_offset: self.start_offset,
        };
        let (_idx, claim) = hit!(arms.race(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

/// [`EnumVariantProbeOwned`] for the chunked JSON deserializer.
pub struct ChunkedJsonEnumVariantProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    token: Token,
    tokenizer: Tokenizer,
    offset: usize,
    start_offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumVariantProbeOwned
    for ChunkedJsonEnumVariantProbe<'s, B, F>
{
    type Error = JsonError;
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type PayloadDeserializer = ChunkedJsonSubDeserializer<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            start_offset: self.start_offset,
        }
    }

    async fn deserialize_unit_by_name<W>(
        self,
        candidates: W,
    ) -> Result<Probe<(Self::Claim, usize)>, Self::Error>
    where
        W: strede::ConcatableArray<T = (&'static str, usize)> + Copy + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        use access::ChunkedJsonStrAccess;
        use strede::StrAccessOwned as _;

        // Only match string tokens (unit variants are bare strings).
        let mut str_access = match self.token {
            Token::Str(access) => ChunkedJsonStrAccess {
                handle: self.handle,
                access,
                offset: self.offset,
                char_buf: [0; 4],
            },
            _ => return Ok(Probe::Miss),
        };

        let mut viable = candidates.map(|_| true);
        let cands = candidates.as_ref();
        let mut consumed: usize = 0;
        loop {
            let result = str_access
                .next_str(|s: &str| {
                    let new_consumed = consumed + s.len();
                    let v = viable.as_mut();
                    for (i, &(k, _)) in cands.iter().enumerate() {
                        if !v[i] {
                            continue;
                        }
                        if new_consumed > k.len()
                            || &k.as_bytes()[consumed..new_consumed] != s.as_bytes()
                        {
                            v[i] = false;
                        }
                    }
                    consumed = new_consumed;
                })
                .await?;
            if !viable.as_ref().iter().any(|v| *v) {
                return Ok(Probe::Miss);
            }
            match result {
                strede::Chunk::Data((new, ())) => str_access = new,
                strede::Chunk::Done(claim) => {
                    let v = viable.as_ref();
                    for (i, &(k, idx)) in cands.iter().enumerate() {
                        if v[i] && k.len() == consumed {
                            return Ok(Probe::Hit((claim, idx)));
                        }
                    }
                    return Ok(Probe::Miss);
                }
            }
        }
    }

    async fn deserialize_payload_by_name<T, W>(
        self,
        candidates: W,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
        W: strede::ConcatableArray<T = (&'static str, usize)> + Copy + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        use access::ChunkedJsonKeyProbe;

        // Expect a single-key object `{"VariantName": <payload>}`.
        let (handle, mut tokenizer, mut offset) = match self.token {
            Token::Simple(SimpleToken::ObjectStart, tok) => (self.handle, tok, self.offset),
            _ => return Ok(Probe::Miss),
        };

        // Read the key token (or empty-object `}`).
        let mut pending: Option<Token> = None;
        let start_key_offset = offset;
        let (handle, key_tok) =
            next_dispatch(handle, &mut tokenizer, &mut offset, &mut pending).await?;

        if let Token::Simple(SimpleToken::ObjectEnd, _) = key_tok {
            // Empty object — no variant matched.
            return Ok(Probe::Miss);
        }

        // Build a key probe and match against candidates via a sub-deserializer.
        let key_probe = ChunkedJsonKeyProbe {
            handle,
            key_tok,
            tokenizer,
            offset,
            start_offset: start_key_offset,
        };

        // Deserialize the key as MatchVals<usize, [(&'static str, usize); N]> to find the candidate index.
        use strede::{
            MapKeyClaimOwned as _, MapKeyProbeOwned as _, MapValueClaimOwned as _,
            MapValueProbeOwned as _, MatchVals,
        };
        let (key_claim, MatchVals(idx, _)): (ChunkedJsonClaim<'s, B, F>, MatchVals<usize, W>) =
            match key_probe
                .deserialize_key::<MatchVals<usize, W>>(candidates)
                .await?
            {
                Probe::Hit(v) => v,
                Probe::Miss => return Ok(Probe::Miss),
            };

        // Advance past `:` and get the value token.
        let value_probe: access::ChunkedJsonValueProbe<'s, B, F> =
            key_claim.into_value_probe().await?;

        // Deserialize the value as T via MapValueProbeOwned::deserialize_value.
        let (value_claim, t): (ChunkedJsonClaim<'s, B, F>, T) =
            match value_probe.deserialize_value::<T>(extra).await? {
                Probe::Hit(v) => v,
                Probe::Miss => return Ok(Probe::Miss),
            };

        // Expect `}` — externally-tagged enum has exactly one key-value pair.
        let map_claim = match value_claim.next_key(0, 0).await? {
            strede::NextKey::Done(c) => c,
            strede::NextKey::Entry(_) => return Ok(Probe::Miss),
        };

        Ok(Probe::Hit((map_claim, idx, t)))
    }

    async fn deserialize_value_by_shape<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        let sub = ChunkedJsonSubDeserializer::new(
            self.handle,
            self.tokenizer,
            self.offset,
            self.start_offset,
            self.token,
        );
        T::deserialize_owned(sub, extra).await
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::ChunkedJsonDeserializer;
    use std::{boxed::Box, vec::Vec};
    use strede::shared_buf::SharedBuf;
    use strede::{DeserializeOwned, Skip};
    use strede_test_util::block_on;

    fn skip_input(input: &'static [u8]) -> Result<(), crate::JsonError> {
        block_on(SharedBuf::with_async(
            input,
            async move |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                match Skip::deserialize_owned(de, ()).await? {
                    strede::Probe::Hit(_) => Ok(()),
                    strede::Probe::Miss => Err(crate::JsonError::UnexpectedByte { byte: 0 }),
                }
            },
        ))
    }

    #[test]
    fn skip_mismatched_object_close_with_array_end() {
        // {] is not valid JSON — object opened but array-end used to close it.
        assert!(skip_input(b"{]").is_err());
    }

    #[test]
    fn skip_mismatched_array_close_with_object_end() {
        // [} is not valid JSON — array opened but object-end used to close it.
        assert!(skip_input(b"[}").is_err());
    }

    #[test]
    fn skip_mismatched_nested() {
        // {"a": [} — mismatch inside a nested structure.
        assert!(skip_input(b"{\"a\":[}").is_err());
    }

    #[test]
    fn skip_matched_object() {
        assert!(skip_input(b"{}").is_ok());
    }

    #[test]
    fn skip_matched_array() {
        assert!(skip_input(b"[]").is_ok());
    }

    /// Build a `&'static [u8]` from a `Vec` via `Box::leak` for use in tests.
    fn skip_input_owned(input: Vec<u8>) -> Result<(), crate::JsonError> {
        skip_input(Box::leak(input.into_boxed_slice()))
    }

    /// 4096 `[`s + `{}` + 4096 `]`s — valid at any depth, must succeed in both alloc and no-alloc.
    #[test]
    fn skip_deep_nesting_valid() {
        let mut input = Vec::new();
        input.extend(std::iter::repeat_n(b'[', 4096));
        input.extend_from_slice(b"{}");
        input.extend(std::iter::repeat_n(b']', 4096));
        assert!(skip_input_owned(input).is_ok());
    }

    /// 4096 `[`s + `{` + `]` (mismatched close at depth 4097) + 4096 `]`s.
    /// With `alloc`, the recursive call catches the mismatch and returns an error.
    #[cfg(feature = "alloc")]
    #[test]
    fn skip_deep_nesting_mismatch_alloc() {
        let mut input = Vec::new();
        input.extend(std::iter::repeat(b'[').take(4096));
        input.push(b'{');
        input.push(b']'); // closes `{` with `]` — mismatch
        input.extend(std::iter::repeat(b']').take(4096));
        assert!(skip_input_owned(input).is_err());
    }

    /// Same mismatch, but without `alloc`. Validation is skipped beyond depth 4096
    /// so the mismatched close is silently accepted and skip returns `Ok`.
    #[cfg(not(feature = "alloc"))]
    #[test]
    fn skip_deep_nesting_mismatch_no_alloc() {
        let mut input = Vec::new();
        input.extend(std::iter::repeat_n(b'[', 4096));
        input.push(b'{');
        input.push(b']'); // closes `{` with `]` — mismatch silently ignored
        input.extend(std::iter::repeat_n(b']', 4096));
        assert!(skip_input_owned(input).is_ok());
    }
}
