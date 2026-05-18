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
use crate::token::{self, SimpleToken, StrChunk, Token, Tokenizer};
use core::future::Future;
use strede::utils::repeat;
use strede::{
    Buffer, BytesAccessOwned, Chunk, DeserializeFromMapOwned, DeserializeFromSeqOwned,
    DeserializeOwned, Handle, MapAccessOwned, MapArmStackOwned, MapKeyClaimOwned, MapKeyProbeOwned,
    MapValueClaimOwned, MapValueProbeOwned, NextKey, NumberAccessOwned, Probe, SeqAccessOwned,
    SeqEntryOwned, StrAccessOwned, hit,
};

use super::{
    ChunkedJsonClaim, ChunkedJsonSubDeserializer, new_offset, next_dispatch, refill,
    skip_value_chunked,
};

// ---------------------------------------------------------------------------
// StrAccess / BytesAccess
// ---------------------------------------------------------------------------

pub struct ChunkedJsonStrAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) access: token::StrAccess,
    pub(super) offset: usize,
    pub(super) char_buf: [u8; 4],
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> StrAccessOwned for ChunkedJsonStrAccess<'s, B, F> {
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type Error = JsonError;

    #[inline(always)]
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
    pub(super) handle: Handle<'s, B, F>,
    pub(super) access: token::StrAccess,
    pub(super) offset: usize,
    pub(super) char_buf: [u8; 4],
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> BytesAccessOwned for ChunkedJsonBytesAccess<'s, B, F> {
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type Error = JsonError;

    #[inline(always)]
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

// ---------------------------------------------------------------------------
// NumberAccessOwned
// ---------------------------------------------------------------------------

pub struct ChunkedJsonNumberAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) access: token::NumberAccess,
    pub(super) offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> NumberAccessOwned for ChunkedJsonNumberAccess<'s, B, F> {
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type Error = JsonError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            access: self.access,
            offset: self.offset,
        }
    }

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        loop {
            let pre_offset = self.offset;
            let result = {
                let buf = self.handle.buf();
                let mut src: &[u8] = &buf[pre_offset..];
                let r = self.access.next_chunk(&mut src);
                self.offset = new_offset(buf, src);
                r
            };
            match result {
                Ok(Some(chunk)) => {
                    let v = f(chunk);
                    return Ok(Chunk::Data((self, v)));
                }
                Ok(None) => {
                    return Ok(Chunk::Done(ChunkedJsonClaim {
                        tokenizer: Tokenizer::new(),
                        offset: self.offset,
                        handle: self.handle,
                    }));
                }
                Err(JsonError::UnexpectedEnd) => {
                    self.handle = refill(self.handle, &mut self.offset).await?;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

// ===========================================================================
// Map access
// ===========================================================================

pub struct ChunkedJsonMapAccessOwned<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(super) handle: Handle<'s, B, F>,
    pub(super) tokenizer: Tokenizer,
    pub(super) offset: usize,
}

pub struct ChunkedJsonKeyProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    key_tok: Token,
    tokenizer: Tokenizer,
    offset: usize,
    /// Buffer offset before the leading-key-token bytes were consumed; threaded
    /// into the spawned [`ChunkedJsonSubDeserializer`] so raw-value capture
    /// covers the full key span.
    start_offset: usize,
}

pub struct ChunkedJsonValueProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    value_tok: Token,
    tokenizer: Tokenizer,
    offset: usize,
    /// Same role as on [`ChunkedJsonKeyProbe`], but for the value's leading token.
    start_offset: usize,
}

// --- KeyProbe ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyProbeOwned for ChunkedJsonKeyProbe<'s, B, F> {
    type Error = JsonError;
    type KeyClaim = ChunkedJsonClaim<'s, B, F>;
    type KeySubDeserializer = ChunkedJsonSubDeserializer<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            key_tok: self.key_tok.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            start_offset: self.start_offset,
        }
    }

    #[inline(always)]
    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: DeserializeOwned<Self::KeySubDeserializer>,
    {
        let sub = ChunkedJsonSubDeserializer::new(
            self.handle,
            self.tokenizer,
            self.offset,
            self.start_offset,
            self.key_tok,
        );
        K::deserialize_owned(sub, extra).await
    }
}

// --- ValueProbe ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueProbeOwned for ChunkedJsonValueProbe<'s, B, F> {
    type Error = JsonError;
    type MapClaim = ChunkedJsonClaim<'s, B, F>;
    type ValueClaim = ChunkedJsonClaim<'s, B, F>;
    type ValueSubDeserializer = ChunkedJsonSubDeserializer<'s, B, F>;
    type ValueMap = ChunkedJsonMapAccessOwned<'s, B, F>;
    type ValueSeq = ChunkedJsonSeqAccess<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            value_tok: self.value_tok.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            start_offset: self.start_offset,
        }
    }

    #[inline(always)]
    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeOwned<Self::ValueSubDeserializer>,
    {
        let sub = ChunkedJsonSubDeserializer::new(
            self.handle,
            self.tokenizer,
            self.offset,
            self.start_offset,
            self.value_tok,
        );
        V::deserialize_owned(sub, extra).await
    }

    #[inline(always)]
    async fn deserialize_map_into<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeFromMapOwned<Self::ValueMap>,
    {
        match self.value_tok {
            Token::Simple(SimpleToken::ObjectStart, tok) => {
                let map = ChunkedJsonMapAccessOwned {
                    handle: self.handle,
                    tokenizer: tok,
                    offset: self.offset,
                };
                V::deserialize_from_map_owned(map, extra).await
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
        V: DeserializeFromSeqOwned<Self::ValueSeq>,
    {
        match self.value_tok {
            Token::Simple(SimpleToken::ArrayStart, tok) => {
                let seq = ChunkedJsonSeqAccess {
                    handle: self.handle,
                    tokenizer: tok,
                    offset: self.offset,
                    first: true,
                };
                V::deserialize_from_seq_owned(seq, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let mut tokenizer = self.tokenizer;
        let mut offset = self.offset;
        let handle =
            skip_value_chunked(self.handle, &mut tokenizer, &mut offset, self.value_tok).await?;
        Ok(ChunkedJsonClaim {
            handle,
            tokenizer,
            offset,
        })
    }
}

// --- MapKeyClaim and MapValueClaim are both implemented on ChunkedJsonClaim ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapKeyClaimOwned for ChunkedJsonClaim<'s, B, F> {
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
        let start_offset = self.offset;
        let (handle, value_tok) =
            next_dispatch(h, &mut self.tokenizer, &mut self.offset, &mut pending).await?;

        Ok(ChunkedJsonValueProbe {
            handle,
            value_tok,
            tokenizer: self.tokenizer,
            offset: self.offset,
            start_offset,
        })
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapValueClaimOwned for ChunkedJsonClaim<'s, B, F> {
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
        let start_offset = self.offset;
        let (handle, key_tok) =
            next_dispatch(h, &mut self.tokenizer, &mut self.offset, &mut pending).await?;

        Ok(NextKey::Entry(ChunkedJsonKeyProbe {
            handle,
            key_tok,
            tokenizer: self.tokenizer,
            offset: self.offset,
            start_offset,
        }))
    }
}

// --- MapAccessOwned ---

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> MapAccessOwned for ChunkedJsonMapAccessOwned<'s, B, F> {
    type Error = JsonError;
    type MapClaim = ChunkedJsonClaim<'s, B, F>;
    type KeyProbe = ChunkedJsonKeyProbe<'s, B, F>;

    #[inline(always)]
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
        let start_offset = self.offset;
        let (handle, tok) = next_dispatch(
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
                    handle,
                };
                return Ok(Probe::Hit((claim, arms.take_outputs())));
            }
            key_tok => ChunkedJsonKeyProbe {
                handle,
                key_tok,
                tokenizer: self.tokenizer,
                offset: self.offset,
                start_offset,
            },
        };

        loop {
            // Race all arms' key callbacks against this round's key probe.
            let value_claim: ChunkedJsonClaim<'s, B, F> = match arms.race_keys(key_probe).await? {
                Probe::Hit((arm_index, key_claim)) => {
                    // A known arm matched. Convert key claim → value probe.
                    let vp: ChunkedJsonValueProbe<'s, B, F> = key_claim.into_value_probe().await?;
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
    pub(super) handle: Handle<'s, B, F>,
    pub(super) tokenizer: Tokenizer,
    pub(super) offset: usize,
    pub(super) first: bool,
}

async fn seq_next<'s, B, F, const N: usize, Fn_, Fut, R>(
    mut seq: ChunkedJsonSeqAccess<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<Chunk<(ChunkedJsonSeqAccess<'s, B, F>, R), ChunkedJsonClaim<'s, B, F>>>, JsonError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedJsonSeqEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedJsonClaim<'s, B, F>, R)>, JsonError>>,
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

    let start_offset = seq.offset;
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

    let entries: [ChunkedJsonSeqEntry<'s, B, F>; N] =
        repeat(seq.handle, Handle::fork).map(|handle| ChunkedJsonSeqEntry {
            handle,
            elem_tok: elem_tok.clone(),
            tokenizer: seq.tokenizer.clone(),
            offset: seq.offset,
            start_offset,
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
    type ElemClaim = ChunkedJsonClaim<'s, B, F>;
    type Elem = ChunkedJsonSeqEntry<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            first: self.first,
        }
    }

    #[inline(always)]
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
    handle: Handle<'s, B, F>,
    elem_tok: Token,
    tokenizer: Tokenizer,
    offset: usize,
    /// Buffer offset before the leading-element-token bytes were consumed.
    start_offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SeqEntryOwned for ChunkedJsonSeqEntry<'s, B, F> {
    type Error = JsonError;
    type Claim = ChunkedJsonClaim<'s, B, F>;
    type SubDeserializer = ChunkedJsonSubDeserializer<'s, B, F>;
    type Map = ChunkedJsonMapAccessOwned<'s, B, F>;
    type Seq = ChunkedJsonSeqAccess<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            elem_tok: self.elem_tok.clone(),
            tokenizer: self.tokenizer.clone(),
            offset: self.offset,
            start_offset: self.start_offset,
        }
    }

    #[inline(always)]
    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeOwned<Self::SubDeserializer>,
    {
        let sub = ChunkedJsonSubDeserializer::new(
            self.handle,
            self.tokenizer,
            self.offset,
            self.start_offset,
            self.elem_tok,
        );
        T::deserialize_owned(sub, extra).await
    }

    #[inline(always)]
    async fn get_map_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMapOwned<Self::Map>,
    {
        match self.elem_tok {
            Token::Simple(SimpleToken::ObjectStart, tok) => {
                let map = ChunkedJsonMapAccessOwned {
                    handle: self.handle,
                    tokenizer: tok,
                    offset: self.offset,
                };
                T::deserialize_from_map_owned(map, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn get_seq_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeqOwned<Self::Seq>,
    {
        match self.elem_tok {
            Token::Simple(SimpleToken::ArrayStart, tok) => {
                let seq = ChunkedJsonSeqAccess {
                    handle: self.handle,
                    tokenizer: tok,
                    offset: self.offset,
                    first: true,
                };
                T::deserialize_from_seq_owned(seq, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut tokenizer = self.tokenizer;
        let mut offset = self.offset;
        let handle =
            skip_value_chunked(self.handle, &mut tokenizer, &mut offset, self.elem_tok).await?;
        Ok(ChunkedJsonClaim {
            tokenizer,
            offset,
            handle,
        })
    }
}
