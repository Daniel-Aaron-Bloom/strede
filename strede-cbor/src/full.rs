//! In-memory borrow-family CBOR deserializer.
//!
//! Implements [`strede::Deserializer`] over a borrowed `&[u8]` source.
//!
//! # Tag stripping
//!
//! All `CborToken::Tag` tokens are consumed transparently before dispatching
//! to probe methods.  Normal `Deserialize` impls never see tags.  User code
//! that needs the tag number uses `CborTag<T, H>` from the `tag` module.
//!
//! # Indefinite-length strings
//!
//! `deserialize_str` / `deserialize_bytes` return `Probe::Hit` for definite-
//! length strings and for indefinite-length strings whose entire payload fits
//! in a single chunk (i.e. exactly one inner chunk + break byte all present in
//! the current buffer).  Otherwise they return `Probe::Miss`, and
//! `deserialize_str_chunks` / `deserialize_bytes_chunks` handle the general
//! case.

use crate::{
    CborError,
    tag::{CborTag, TagHandler},
    token::{CborToken, next_token, skip_value},
};
use strede::{
    BigEndian, BytesAccess, Chunk, Deserialize, DeserializeFromEnum, DeserializeFromMap,
    DeserializeFromSeq, Deserializer, Entry, EnumAccess, EnumArmStack, EnumVariantProbe, MapAccess,
    MapArmStack, MapKeyClaim, MapKeyProbe, MapValueClaim, MapValueProbe, MatchVals, NextKey,
    NumberAccess, NumberEncoding, Probe, SeqAccess, SeqEntry, StrAccess, hit,
    match_entry_str_against, utils::repeat,
};

// ---------------------------------------------------------------------------
// CborClaim
// ---------------------------------------------------------------------------

pub struct CborClaim<'de> {
    pub(crate) src: &'de [u8],
    /// Remaining map key-value pairs after this claim (map context only); 0 otherwise.
    pub(crate) remaining_after: usize,
}

// ---------------------------------------------------------------------------
// CborDeserializer
// ---------------------------------------------------------------------------

pub struct CborDeserializer<'de> {
    pub(crate) src: &'de [u8],
}

impl<'de> CborDeserializer<'de> {
    pub fn new(src: &'de [u8]) -> Self {
        Self { src }
    }
}

// ---------------------------------------------------------------------------
// CborSubDeserializer
// ---------------------------------------------------------------------------

pub struct CborSubDeserializer<'de> {
    src: &'de [u8],
    pending_tok: CborToken,
    bignum_tag: Option<u64>,
    /// See `CborEntry::raw_start`. Defaults to `src` (i.e. "no leading tags
    /// known to exist beyond `pending_tok`") for call sites that construct a
    /// sub-deserializer without threading the true origin through; call sites
    /// that dispatch to an arbitrary nested type (`deserialize_value`,
    /// `deserialize_option`, `SeqEntry::get`, `MapValueProbe::deserialize_value`)
    /// set this explicitly via `new_with_raw` so `CborTag`/`CborValue` nested
    /// at that position can still see the real tag chain.
    pub(crate) raw_start: &'de [u8],
}

impl<'de> CborSubDeserializer<'de> {
    #[inline(always)]
    pub(crate) fn new(src: &'de [u8], tok: CborToken) -> Self {
        Self {
            src,
            pending_tok: tok,
            bignum_tag: None,
            raw_start: src,
        }
    }

    #[inline(always)]
    pub(crate) fn new_with_raw(
        src: &'de [u8],
        tok: CborToken,
        bignum_tag: Option<u64>,
        raw_start: &'de [u8],
    ) -> Self {
        Self {
            src,
            pending_tok: tok,
            bignum_tag,
            raw_start,
        }
    }
}

// ---------------------------------------------------------------------------
// Deserializer impls
// ---------------------------------------------------------------------------

/// Strip leading Tag tokens from `src`, calling `handler.handle(n)` for each.
/// Returns the first non-Tag token and updated `src`.
/// Returns `None` if the handler vetoes any tag.
fn strip_tags(src: &mut &[u8], tok: CborToken) -> Result<Option<CborToken>, CborError> {
    let mut cur = tok;
    loop {
        match cur {
            CborToken::Tag(n) => {
                let _ = n;
                cur = next_token(src)?;
            }
            other => return Ok(Some(other)),
        }
    }
}

/// Like `strip_tags` but also records the innermost bignum tag (2 or 3) seen.
/// Returns `(bignum_tag, inner_token)` where `bignum_tag` is `Some(2)` or
/// `Some(3)` if the outermost meaningful tag was a CBOR bignum tag, else `None`.
fn strip_tags_bignum(
    src: &mut &[u8],
    tok: CborToken,
) -> Result<Option<(Option<u64>, CborToken)>, CborError> {
    let mut cur = tok;
    let mut bignum_tag: Option<u64> = None;
    loop {
        match cur {
            CborToken::Tag(n @ (2 | 3)) => {
                bignum_tag = Some(n);
                cur = next_token(src)?;
            }
            CborToken::Tag(_) => {
                bignum_tag = None;
                cur = next_token(src)?;
            }
            other => return Ok(Some((bignum_tag, other))),
        }
    }
}

/// `(handler, final_token, raw_start, rest)` — `raw_start` is the position
/// the final (non-tag) token began at, `rest` is the position right after
/// that token's header.
pub(crate) type StripTagsWithRaw<'de, H> = (H, CborToken, &'de [u8], &'de [u8]);

/// Like `strip_tags` but uses a user-supplied `TagHandler`, and reads from a
/// truly untouched position (i.e. before any generic entry-construction has
/// already stripped tags). `raw_start` in the result lets a nested
/// `CborTag`/`CborValue` re-read from there to see further tags of its own,
/// exactly matching `strip_tags`/`strip_tags_bignum`'s `rest` semantics
/// otherwise.
pub(crate) fn strip_tags_with_raw<H: TagHandler>(
    start: &[u8],
    mut handler: H,
) -> Result<Option<StripTagsWithRaw<'_, H>>, CborError> {
    let mut cur = start;
    loop {
        let raw_start = cur;
        let tok = next_token(&mut cur)?;
        match tok {
            CborToken::Tag(n) => {
                handler = match handler.handle(n) {
                    Some(h) => h,
                    None => return Ok(None),
                };
            }
            other => return Ok(Some((handler, other, raw_start, cur))),
        }
    }
}

#[inline(always)]
async fn cbor_root_next<'de, const N: usize, F, Fut, R>(
    mut de: CborDeserializer<'de>,
    mut f: F,
) -> Result<Probe<(CborClaim<'de>, R)>, CborError>
where
    F: FnMut([CborEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(CborClaim<'de>, R)>, CborError>>,
{
    let raw_start = de.src;
    let raw_tok = next_token(&mut de.src)?;
    let (bignum_tag, tok) = match strip_tags_bignum(&mut de.src, raw_tok)? {
        Some(t) => t,
        None => return Ok(Probe::Miss),
    };
    let entry = CborEntry {
        token: tok,
        src: de.src,
        bignum_tag,
        raw_start,
    };
    match f(repeat(entry, |e| e.clone())).await? {
        Probe::Hit((claim, r)) => {
            if !claim.src.is_empty() {
                return Err(CborError::ExpectedEnd);
            }
            Ok(Probe::Hit((claim, r)))
        }
        Probe::Miss => Ok(Probe::Miss),
    }
}

impl<'de> Deserializer<'de> for CborDeserializer<'de> {
    type Error = CborError;
    type Claim = CborClaim<'de>;
    type EntryClaim = CborClaim<'de>;
    type Entry = CborEntry<'de>;

    #[inline(always)]
    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        cbor_root_next(self, f).await
    }
}

impl<'de> Deserializer<'de> for CborSubDeserializer<'de> {
    type Error = CborError;
    type Claim = CborClaim<'de>;
    type EntryClaim = CborClaim<'de>;
    type Entry = CborEntry<'de>;

    #[inline(always)]
    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        // pending_tok is already tag-stripped by the parent
        let entry = CborEntry {
            token: self.pending_tok,
            src: self.src,
            bignum_tag: self.bignum_tag,
            raw_start: self.raw_start,
        };
        f(repeat(entry, |e| e.clone())).await
    }
}

// ---------------------------------------------------------------------------
// CborEntry
// ---------------------------------------------------------------------------

pub struct CborEntry<'de> {
    pub(crate) token: CborToken,
    pub(crate) src: &'de [u8],
    /// Set to `Some(2)` or `Some(3)` when this entry is the payload of a CBOR
    /// bignum tag (unsigned or negative). Used by `deserialize_number_chunks`.
    pub(crate) bignum_tag: Option<u64>,
    /// The position of this value before any leading tags were stripped.
    /// Re-reading from here (via `next_token`) recovers the true, untouched
    /// token chain. Used by `CborTag`/`CborValue`, which need to observe
    /// tags that ordinary probe methods strip transparently.
    pub(crate) raw_start: &'de [u8],
}

impl<'de> CborEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            token: self.token,
            src: self.src,
            bignum_tag: self.bignum_tag,
            raw_start: self.raw_start,
        }
    }

    fn into_claim(self) -> CborClaim<'de> {
        CborClaim {
            src: self.src,
            remaining_after: 0,
        }
    }
}

/// Try to get a zero-copy &'de [u8] slice from an indefinite-length
/// bstr/tstr when there is exactly one chunk + break byte in the buffer.
/// Returns `Some((payload, rest))` on success, `None` if multi-chunk or truncated.
fn try_single_chunk_indef(src: &[u8]) -> Option<(&[u8], &[u8])> {
    let mut s = src;
    let first = *s.first()?;
    let major = first >> 5;
    let info = first & 0x1f;
    if info == 31 {
        return None; // nested indefinite
    }
    s = &s[1..];
    // Read the length argument
    let len: usize = match info {
        0..=23 => info as usize,
        24 => {
            if s.is_empty() {
                return None;
            }
            let v = s[0] as usize;
            s = &s[1..];
            v
        }
        25 => {
            if s.len() < 2 {
                return None;
            }
            let v = u16::from_be_bytes([s[0], s[1]]) as usize;
            s = &s[2..];
            v
        }
        26 => {
            if s.len() < 4 {
                return None;
            }
            let v = u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize;
            s = &s[4..];
            v
        }
        27 => {
            if s.len() < 8 {
                return None;
            }
            let v = u64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) as usize;
            s = &s[8..];
            v
        }
        _ => return None,
    };
    // Validate major type matches (2 for bstr, 3 for tstr)
    if major != 2 && major != 3 {
        return None;
    }
    if s.len() < len + 1 {
        return None;
    } // need payload + break byte
    let payload = &s[..len];
    let rest = &s[len..];
    if rest.first() != Some(&0xff) {
        return None;
    } // no break
    if rest.is_empty() {
        return None;
    }
    Some((payload, &rest[1..]))
}

impl<'de> Entry<'de> for CborEntry<'de> {
    type Error = CborError;
    type Claim = CborClaim<'de>;
    type SubDeserializer = CborSubDeserializer<'de>;
    type StrChunks = CborStrAccess<'de>;
    type BytesChunks = CborBytesAccess<'de>;
    type NumberChunks<Enc: NumberEncoding> = CborNumberAccess<'de>;
    type Map = CborMapAccess<'de>;
    type Seq = CborSeqAccess<'de>;
    type Enum = CborEnumAccess<'de>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    // ---- Strings ------------------------------------------------------------

    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
        match self.token {
            CborToken::Tstr(len) => {
                if self.src.len() < len {
                    return Err(CborError::UnexpectedEnd);
                }
                let (payload, rest) = self.src.split_at(len);
                let s = core::str::from_utf8(payload).map_err(|_| CborError::InvalidUtf8)?;
                Ok(Probe::Hit((
                    CborClaim {
                        src: rest,
                        remaining_after: 0,
                    },
                    s,
                )))
            }
            CborToken::TstrIndef => {
                // Single-chunk fast path
                if let Some((payload, rest)) = try_single_chunk_indef(self.src) {
                    let s = core::str::from_utf8(payload).map_err(|_| CborError::InvalidUtf8)?;
                    return Ok(Probe::Hit((
                        CborClaim {
                            src: rest,
                            remaining_after: 0,
                        },
                        s,
                    )));
                }
                Ok(Probe::Miss)
            }
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.token {
            CborToken::Tstr(len) => {
                if self.src.len() < len {
                    return Err(CborError::UnexpectedEnd);
                }
                Ok(Probe::Hit(CborStrAccess {
                    src: self.src,
                    state: StrState::Definite { remaining: len },
                }))
            }
            CborToken::TstrIndef => Ok(Probe::Hit(CborStrAccess {
                src: self.src,
                state: StrState::Indefinite,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Bytes --------------------------------------------------------------

    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
        match self.token {
            CborToken::Bstr(len) => {
                if self.src.len() < len {
                    return Err(CborError::UnexpectedEnd);
                }
                let (payload, rest) = self.src.split_at(len);
                Ok(Probe::Hit((
                    CborClaim {
                        src: rest,
                        remaining_after: 0,
                    },
                    payload,
                )))
            }
            CborToken::BstrIndef => {
                if let Some((payload, rest)) = try_single_chunk_indef(self.src) {
                    return Ok(Probe::Hit((
                        CborClaim {
                            src: rest,
                            remaining_after: 0,
                        },
                        payload,
                    )));
                }
                Ok(Probe::Miss)
            }
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.token {
            CborToken::Bstr(len) => {
                if self.src.len() < len {
                    return Err(CborError::UnexpectedEnd);
                }
                Ok(Probe::Hit(CborBytesAccess {
                    src: self.src,
                    state: BytesState::Definite { remaining: len },
                }))
            }
            CborToken::BstrIndef => Ok(Probe::Hit(CborBytesAccess {
                src: self.src,
                state: BytesState::Indefinite,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Numbers ------------------------------------------------------------

    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if Enc::NAME != BigEndian::NAME {
            return Ok(Probe::Miss);
        }
        match (self.bignum_tag, self.token) {
            (Some(2 | 3), CborToken::Bstr(len)) => Ok(Probe::Hit(CborNumberAccess {
                src: self.src,
                state: BytesState::Definite { remaining: len },
            })),
            (Some(2 | 3), CborToken::BstrIndef) => Ok(Probe::Hit(CborNumberAccess {
                src: self.src,
                state: BytesState::Indefinite,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Map / Seq ----------------------------------------------------------

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.token {
            CborToken::Map(count) => Ok(Probe::Hit(CborMapAccess {
                src: self.src,
                remaining: count,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.token {
            CborToken::Array(count) => Ok(Probe::Hit(CborSeqAccess {
                src: self.src,
                remaining: count,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Option -------------------------------------------------------------

    #[inline(always)]
    async fn deserialize_option<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        match self.token {
            CborToken::Null | CborToken::Undefined => Ok(Probe::Hit((self.into_claim(), None))),
            other => {
                let sub = CborSubDeserializer::new_with_raw(self.src, other, None, self.raw_start);
                let (claim, v) = hit!(T::deserialize(sub, extra).await);
                Ok(Probe::Hit((claim, Some(v))))
            }
        }
    }

    // ---- Value / Map / Seq forwarding ---------------------------------------

    #[inline(always)]
    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        // bignum_tag intentionally not forwarded here, matching prior behavior:
        // only the map-value path (`MapValueProbe::deserialize_value`) threads it.
        let sub = CborSubDeserializer::new_with_raw(self.src, self.token, None, self.raw_start);
        T::deserialize(sub, extra).await
    }

    #[inline(always)]
    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>,
    {
        let map = hit!(Entry::deserialize_map(self).await);
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
        let seq = hit!(Entry::deserialize_seq(self).await);
        T::deserialize_from_seq(seq, extra).await
    }

    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        match self.token {
            CborToken::Tstr(_) | CborToken::TstrIndef | CborToken::Map(_) => {
                Ok(Probe::Hit(CborEnumAccess {
                    token: self.token,
                    src: self.src,
                }))
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_enum_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromEnum<'de, Self::Enum>,
    {
        let e = match Entry::deserialize_enum(self).await? {
            Probe::Hit(e) => e,
            Probe::Miss => return Ok(Probe::Miss),
        };
        T::deserialize_from_enum(e, extra).await
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut src = self.src;
        skip_value(&mut src, self.token)?;
        Ok(CborClaim {
            src,
            remaining_after: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// parse_num helper — used by format-specific Deserialize impls
// ---------------------------------------------------------------------------

impl<'de> CborEntry<'de> {
    pub(crate) async fn parse_num<T: ParseNum>(
        self,
    ) -> Result<Probe<(CborClaim<'de>, T)>, CborError> {
        let v = match self.token {
            CborToken::UInt(n) => T::from_uint(n),
            CborToken::NegInt(n) => T::from_negint(n),
            CborToken::Float16(f) => T::from_f32(f),
            CborToken::Float32(f) => T::from_f32(f),
            CborToken::Float64(f) => T::from_f64(f),
            _ => return Ok(Probe::Miss),
        };
        match v {
            Some(value) => Ok(Probe::Hit((self.into_claim(), value))),
            None => Ok(Probe::Miss),
        }
    }
}

pub(crate) trait ParseNum: Sized {
    fn from_uint(v: u64) -> Option<Self>;
    /// `n` is the raw CBOR additional value; actual = -1 - n.
    fn from_negint(n: u64) -> Option<Self>;
    fn from_f32(v: f32) -> Option<Self>;
    fn from_f64(v: f64) -> Option<Self>;
}

// ---------------------------------------------------------------------------
// CborTag impl for borrow family (concrete types)
// ---------------------------------------------------------------------------

impl<'de, T, H> Deserialize<'de, CborDeserializer<'de>> for CborTag<T, H>
where
    T: Deserialize<'de, CborSubDeserializer<'de>>,
    H: TagHandler,
{
    type Extra = (H, T::Extra);

    /// Reads directly from `d.src` — nothing has stripped tags yet at the
    /// document root — instead of going through `Deserializer::entry`, which
    /// would strip them first and leave nothing for this impl to observe.
    async fn deserialize(
        d: CborDeserializer<'de>,
        (handler, extra): (H, T::Extra),
    ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
        let (h, tok, raw_start, src) = match strip_tags_with_raw(d.src, handler)? {
            Some(x) => x,
            None => return Ok(Probe::Miss),
        };
        let sub = CborSubDeserializer::new_with_raw(src, tok, None, raw_start);
        match T::deserialize(sub, extra).await? {
            Probe::Hit((claim, v)) => {
                if h.finish() {
                    if !claim.src.is_empty() {
                        return Err(CborError::ExpectedEnd);
                    }
                    Ok(Probe::Hit((
                        claim,
                        CborTag {
                            handler: h,
                            value: v,
                        },
                    )))
                } else {
                    Ok(Probe::Miss)
                }
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

impl<'de, T, H> Deserialize<'de, CborSubDeserializer<'de>> for CborTag<T, H>
where
    T: Deserialize<'de, CborSubDeserializer<'de>>,
    H: TagHandler,
{
    type Extra = (H, T::Extra);

    /// Reads from `d.raw_start` (the true origin of this value, threaded
    /// through by whichever entry/probe constructed `d`) rather than `d.src`/
    /// `d.pending_tok`, which have already been stripped of any leading tags.
    async fn deserialize(
        d: CborSubDeserializer<'de>,
        (handler, extra): (H, T::Extra),
    ) -> Result<Probe<(CborClaim<'de>, Self)>, CborError> {
        let (h, tok, raw_start, src) = match strip_tags_with_raw(d.raw_start, handler)? {
            Some(x) => x,
            None => return Ok(Probe::Miss),
        };
        let sub = CborSubDeserializer::new_with_raw(src, tok, None, raw_start);
        match T::deserialize(sub, extra).await? {
            Probe::Hit((claim, v)) => {
                if h.finish() {
                    Ok(Probe::Hit((
                        claim,
                        CborTag {
                            handler: h,
                            value: v,
                        },
                    )))
                } else {
                    Ok(Probe::Miss)
                }
            }
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

// ---------------------------------------------------------------------------
// CborStrAccess — borrow family string chunk accessor
// ---------------------------------------------------------------------------

enum StrState {
    Definite { remaining: usize },
    Indefinite,
    Done,
}

pub struct CborStrAccess<'de> {
    src: &'de [u8],
    state: StrState,
}

impl<'de> StrAccess for CborStrAccess<'de> {
    type Claim = CborClaim<'de>;
    type Error = CborError;

    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.state {
            StrState::Done => Ok(Chunk::Done(CborClaim {
                src: self.src,
                remaining_after: 0,
            })),
            StrState::Definite { remaining: 0 } => Ok(Chunk::Done(CborClaim {
                src: self.src,
                remaining_after: 0,
            })),
            StrState::Definite { remaining } => {
                if self.src.len() < remaining {
                    return Err(CborError::UnexpectedEnd);
                }
                let (payload, rest) = self.src.split_at(remaining);
                let s = core::str::from_utf8(payload).map_err(|_| CborError::InvalidUtf8)?;
                let r = f(s);
                Ok(Chunk::Data((
                    Self {
                        src: rest,
                        state: StrState::Done,
                    },
                    r,
                )))
            }
            StrState::Indefinite => {
                // Read the next chunk header
                let mut src = self.src;
                let tok = next_token(&mut src)?;
                match tok {
                    CborToken::Break => Ok(Chunk::Done(CborClaim {
                        src,
                        remaining_after: 0,
                    })),
                    CborToken::Tstr(len) => {
                        if src.len() < len {
                            return Err(CborError::UnexpectedEnd);
                        }
                        let (payload, rest) = src.split_at(len);
                        let s =
                            core::str::from_utf8(payload).map_err(|_| CborError::InvalidUtf8)?;
                        let r = f(s);
                        Ok(Chunk::Data((
                            Self {
                                src: rest,
                                state: StrState::Indefinite,
                            },
                            r,
                        )))
                    }
                    _ => Err(CborError::UnexpectedByte { byte: 0 }),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CborBytesAccess — borrow family bytes chunk accessor
// ---------------------------------------------------------------------------

enum BytesState {
    Definite { remaining: usize },
    Indefinite,
    Done,
}

pub struct CborBytesAccess<'de> {
    src: &'de [u8],
    state: BytesState,
}

impl<'de> BytesAccess for CborBytesAccess<'de> {
    type Claim = CborClaim<'de>;
    type Error = CborError;

    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.state {
            BytesState::Done => Ok(Chunk::Done(CborClaim {
                src: self.src,
                remaining_after: 0,
            })),
            BytesState::Definite { remaining: 0 } => Ok(Chunk::Done(CborClaim {
                src: self.src,
                remaining_after: 0,
            })),
            BytesState::Definite { remaining } => {
                if self.src.len() < remaining {
                    return Err(CborError::UnexpectedEnd);
                }
                let (payload, rest) = self.src.split_at(remaining);
                let r = f(payload);
                Ok(Chunk::Data((
                    Self {
                        src: rest,
                        state: BytesState::Done,
                    },
                    r,
                )))
            }
            BytesState::Indefinite => {
                let mut src = self.src;
                let tok = next_token(&mut src)?;
                match tok {
                    CborToken::Break => Ok(Chunk::Done(CborClaim {
                        src,
                        remaining_after: 0,
                    })),
                    CborToken::Bstr(len) => {
                        if src.len() < len {
                            return Err(CborError::UnexpectedEnd);
                        }
                        let (payload, rest) = src.split_at(len);
                        let r = f(payload);
                        Ok(Chunk::Data((
                            Self {
                                src: rest,
                                state: BytesState::Indefinite,
                            },
                            r,
                        )))
                    }
                    _ => Err(CborError::UnexpectedByte { byte: 0 }),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CborNumberAccess — borrow family number chunk accessor (BigEndian bignums)
// ---------------------------------------------------------------------------

pub struct CborNumberAccess<'de> {
    src: &'de [u8],
    state: BytesState,
}

impl<'de, Enc: NumberEncoding> NumberAccess<Enc> for CborNumberAccess<'de> {
    type Claim = CborClaim<'de>;
    type Error = CborError;

    async fn next_number_chunk<R>(
        self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        // Only reachable when Enc == BigEndian; other encodings never produce this accessor.
        match self.state {
            BytesState::Done | BytesState::Definite { remaining: 0 } => {
                Ok(Chunk::Done(CborClaim {
                    src: self.src,
                    remaining_after: 0,
                }))
            }
            BytesState::Definite { remaining } => {
                if self.src.len() < remaining {
                    return Err(CborError::UnexpectedEnd);
                }
                let (payload, rest) = self.src.split_at(remaining);
                let r = f(Enc::from_bytes(payload));
                Ok(Chunk::Data((
                    Self {
                        src: rest,
                        state: BytesState::Done,
                    },
                    r,
                )))
            }
            BytesState::Indefinite => {
                let mut src = self.src;
                let tok = next_token(&mut src)?;
                match tok {
                    CborToken::Break => Ok(Chunk::Done(CborClaim {
                        src,
                        remaining_after: 0,
                    })),
                    CborToken::Bstr(len) => {
                        if src.len() < len {
                            return Err(CborError::UnexpectedEnd);
                        }
                        let (payload, rest) = src.split_at(len);
                        let r = f(Enc::from_bytes(payload));
                        Ok(Chunk::Data((
                            Self {
                                src: rest,
                                state: BytesState::Indefinite,
                            },
                            r,
                        )))
                    }
                    _ => Err(CborError::UnexpectedByte { byte: 0 }),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Map access type chain
// ---------------------------------------------------------------------------

pub struct CborMapAccess<'de> {
    src: &'de [u8],
    /// `Some(n)` = definite-length with n pairs remaining.
    /// `None` = indefinite-length (terminated by Break).
    remaining: Option<usize>,
}

impl<'de> MapKeyClaim<'de> for CborClaim<'de> {
    type Error = CborError;
    type MapClaim = CborClaim<'de>;
    type ValueProbe = CborMapValueProbe<'de>;

    async fn into_value_probe(mut self) -> Result<Self::ValueProbe, Self::Error> {
        let raw_start = self.src;
        let raw = next_token(&mut self.src)?;
        let (bignum_tag, value_tok) = match strip_tags_bignum(&mut self.src, raw)? {
            Some(t) => t,
            None => return Err(CborError::UnexpectedByte { byte: 0 }),
        };
        Ok(CborMapValueProbe {
            src: self.src,
            value_tok,
            remaining_after: self.remaining_after,
            bignum_tag,
            raw_start,
        })
    }
}

impl<'de> MapValueClaim<'de> for CborClaim<'de> {
    type Error = CborError;
    type KeyProbe = CborMapKeyProbe<'de>;
    type MapClaim = CborClaim<'de>;

    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        match self.remaining_after {
            0 => {
                // Definite map with no remaining pairs, or indefinite map needing break check.
                // remaining_after == 0 can mean "definite, done" OR "indefinite, check for break".
                // We encode indefinite-done as usize::MAX in the key probe; here 0 = definite done.
                Ok(NextKey::Done(CborClaim {
                    src: self.src,
                    remaining_after: 0,
                }))
            }
            usize::MAX => {
                // Indefinite: check for break
                let mut src = self.src;
                let tok = next_token(&mut src)?;
                if matches!(tok, CborToken::Break) {
                    return Ok(NextKey::Done(CborClaim {
                        src,
                        remaining_after: 0,
                    }));
                }
                let key_tok = match strip_tags(&mut src, tok)? {
                    Some(t) => t,
                    None => return Err(CborError::UnexpectedByte { byte: 0 }),
                };
                Ok(NextKey::Entry(CborMapKeyProbe {
                    src,
                    key_tok,
                    remaining_after: usize::MAX,
                }))
            }
            n => {
                let mut src = self.src;
                let raw = next_token(&mut src)?;
                let key_tok = match strip_tags(&mut src, raw)? {
                    Some(t) => t,
                    None => return Err(CborError::UnexpectedByte { byte: 0 }),
                };
                Ok(NextKey::Entry(CborMapKeyProbe {
                    src,
                    key_tok,
                    remaining_after: n - 1,
                }))
            }
        }
    }
}

pub struct CborMapKeyProbe<'de> {
    src: &'de [u8],
    key_tok: CborToken,
    remaining_after: usize,
}

impl<'de> CborMapKeyProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            key_tok: self.key_tok,
            remaining_after: self.remaining_after,
        }
    }
}

impl<'de> MapKeyProbe<'de> for CborMapKeyProbe<'de> {
    type Error = CborError;
    type KeyClaim = CborClaim<'de>;
    type KeySubDeserializer = CborSubDeserializer<'de>;

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
        let sub = CborSubDeserializer::new(self.src, self.key_tok);
        match K::deserialize(sub, extra).await? {
            Probe::Hit((claim, k)) => Ok(Probe::Hit((
                CborClaim {
                    src: claim.src,
                    remaining_after: self.remaining_after,
                },
                k,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

pub struct CborMapValueProbe<'de> {
    src: &'de [u8],
    value_tok: CborToken,
    remaining_after: usize,
    bignum_tag: Option<u64>,
    /// See `CborEntry::raw_start`.
    raw_start: &'de [u8],
}

impl<'de> CborMapValueProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            value_tok: self.value_tok,
            remaining_after: self.remaining_after,
            bignum_tag: self.bignum_tag,
            raw_start: self.raw_start,
        }
    }
}

impl<'de> MapValueProbe<'de> for CborMapValueProbe<'de> {
    type Error = CborError;
    type MapClaim = CborClaim<'de>;
    type ValueClaim = CborClaim<'de>;
    type ValueSubDeserializer = CborSubDeserializer<'de>;
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
        let remaining_after = self.remaining_after;
        let sub = CborSubDeserializer::new_with_raw(
            self.src,
            self.value_tok,
            self.bignum_tag,
            self.raw_start,
        );
        match V::deserialize(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((
                CborClaim {
                    src: claim.src,
                    remaining_after,
                },
                v,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let remaining_after = self.remaining_after;
        let mut src = self.src;
        skip_value(&mut src, self.value_tok)?;
        Ok(CborClaim {
            src,
            remaining_after,
        })
    }
}

impl<'de> MapAccess<'de> for CborMapAccess<'de> {
    type Error = CborError;
    type MapClaim = CborClaim<'de>;
    type KeyProbe = CborMapKeyProbe<'de>;

    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        cbor_map_iterate(self, arms).await
    }

    async fn iterate_dyn<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        cbor_map_iterate(self, arms).await
    }
}

/// Shared body for [`MapAccess::iterate`] / [`MapAccess::iterate_dyn`] — CBOR
/// maps (definite or `Break`-terminated indefinite) are wire-identical for
/// structs and dynamic collections, so both trait methods delegate here
/// rather than one calling the other.
async fn cbor_map_iterate<'de, S: MapArmStack<'de, CborMapKeyProbe<'de>>>(
    map: CborMapAccess<'de>,
    mut arms: S,
) -> Result<Probe<(CborClaim<'de>, S::Outputs)>, CborError> {
    let mut src = map.src;

    // Build the first key probe (or return immediately for definite empty map).
    let mut key_probe_opt: Option<CborMapKeyProbe<'de>> = match map.remaining {
        Some(0) => {
            return Ok(Probe::Hit((
                CborClaim {
                    src,
                    remaining_after: 0,
                },
                arms.take_outputs(),
            )));
        }
        Some(n) => {
            let raw = next_token(&mut src)?;
            let key_tok = match strip_tags(&mut src, raw)? {
                Some(t) => t,
                None => return Err(CborError::UnexpectedByte { byte: 0 }),
            };
            Some(CborMapKeyProbe {
                src,
                key_tok,
                remaining_after: n - 1,
            })
        }
        None => {
            // Indefinite: check for immediate break
            let raw = next_token(&mut src)?;
            if matches!(raw, CborToken::Break) {
                return Ok(Probe::Hit((
                    CborClaim {
                        src,
                        remaining_after: 0,
                    },
                    arms.take_outputs(),
                )));
            }
            let key_tok = match strip_tags(&mut src, raw)? {
                Some(t) => t,
                None => return Err(CborError::UnexpectedByte { byte: 0 }),
            };
            Some(CborMapKeyProbe {
                src,
                key_tok,
                remaining_after: usize::MAX,
            })
        }
    };

    loop {
        let key_probe = key_probe_opt.take().unwrap();

        let (arm_index, key_claim) = match arms.race_keys(key_probe).await? {
            Probe::Miss => return Ok(Probe::Miss),
            Probe::Hit(x) => x,
        };

        let value_probe = key_claim.into_value_probe().await?;

        let (value_claim, ()) = match arms.dispatch_value(arm_index, value_probe).await? {
            Probe::Miss => return Ok(Probe::Miss),
            Probe::Hit(x) => x,
        };

        match value_claim
            .next_key(arms.unsatisfied_count(), arms.open_count())
            .await?
        {
            NextKey::Done(map_claim) => {
                return Ok(Probe::Hit((map_claim, arms.take_outputs())));
            }
            NextKey::Entry(next_kp) => {
                key_probe_opt = Some(next_kp);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CborSeqAccess / CborSeqEntry
// ---------------------------------------------------------------------------

pub struct CborSeqAccess<'de> {
    pub(crate) src: &'de [u8],
    /// `Some(n)` = definite, `None` = indefinite
    pub(crate) remaining: Option<usize>,
}

pub struct CborSeqEntry<'de> {
    pub(crate) src: &'de [u8],
    pub(crate) elem_tok: CborToken,
    pub(crate) bignum_tag: Option<u64>,
    /// See `CborEntry::raw_start`.
    pub(crate) raw_start: &'de [u8],
}

impl<'de> CborSeqEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            elem_tok: self.elem_tok,
            bignum_tag: self.bignum_tag,
            raw_start: self.raw_start,
        }
    }
}

#[inline(always)]
async fn cbor_seq_next<'de, const N: usize, F, Fut, R>(
    seq: CborSeqAccess<'de>,
    mut f: F,
) -> Result<Probe<Chunk<(CborSeqAccess<'de>, R), CborClaim<'de>>>, CborError>
where
    F: FnMut([CborSeqEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(CborClaim<'de>, R)>, CborError>>,
{
    let mut src = seq.src;

    match seq.remaining {
        Some(0) => Ok(Probe::Hit(Chunk::Done(CborClaim {
            src,
            remaining_after: 0,
        }))),
        Some(n) => {
            let raw_start = src;
            let raw = next_token(&mut src)?;
            let (bignum_tag, elem_tok) = match strip_tags_bignum(&mut src, raw)? {
                Some(t) => t,
                None => return Err(CborError::UnexpectedByte { byte: 0 }),
            };
            let new_seq = CborSeqAccess {
                src,
                remaining: Some(n - 1),
            };
            let entry = CborSeqEntry {
                src,
                elem_tok,
                bignum_tag,
                raw_start,
            };
            let (claim, r) = hit!(f(repeat(entry, |e| e.clone())).await);
            let updated_seq = CborSeqAccess {
                src: claim.src,
                remaining: new_seq.remaining,
            };
            Ok(Probe::Hit(Chunk::Data((updated_seq, r))))
        }
        None => {
            // Indefinite: check for break
            let raw_start = src;
            let raw = next_token(&mut src)?;
            if matches!(raw, CborToken::Break) {
                return Ok(Probe::Hit(Chunk::Done(CborClaim {
                    src,
                    remaining_after: 0,
                })));
            }
            let (bignum_tag, elem_tok) = match strip_tags_bignum(&mut src, raw)? {
                Some(t) => t,
                None => return Err(CborError::UnexpectedByte { byte: 0 }),
            };
            let entry = CborSeqEntry {
                src,
                elem_tok,
                bignum_tag,
                raw_start,
            };
            let (claim, r) = hit!(f(repeat(entry, |e| e.clone())).await);
            let updated_seq = CborSeqAccess {
                src: claim.src,
                remaining: None,
            };
            Ok(Probe::Hit(Chunk::Data((updated_seq, r))))
        }
    }
}

impl<'de> SeqAccess<'de> for CborSeqAccess<'de> {
    type Error = CborError;
    type SeqClaim = CborClaim<'de>;
    type ElemClaim = CborClaim<'de>;
    type Elem = CborSeqEntry<'de>;

    #[inline(always)]
    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        cbor_seq_next(self, f).await
    }
}

impl<'de> SeqEntry<'de> for CborSeqEntry<'de> {
    type Error = CborError;
    type Claim = CborClaim<'de>;
    type SubDeserializer = CborSubDeserializer<'de>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    #[inline(always)]
    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = CborSubDeserializer::new_with_raw(
            self.src,
            self.elem_tok,
            self.bignum_tag,
            self.raw_start,
        );
        T::deserialize(sub, extra).await
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut src = self.src;
        skip_value(&mut src, self.elem_tok)?;
        Ok(CborClaim {
            src,
            remaining_after: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// CborEnumAccess / CborEnumVariantProbe
// ---------------------------------------------------------------------------

/// [`EnumAccess`] for CBOR externally-tagged enums.
///
/// - Unit variants: bare text string (`"VariantName"`)
/// - Non-unit variants: single-key map (`{"VariantName": <payload>}`)
pub struct CborEnumAccess<'de> {
    token: CborToken,
    src: &'de [u8],
}

impl<'de> EnumAccess<'de> for CborEnumAccess<'de> {
    type Error = CborError;
    type Claim = CborClaim<'de>;
    type VariantProbe = CborEnumVariantProbe<'de>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStack<'de, Self::VariantProbe>,
    {
        let vp = CborEnumVariantProbe {
            token: self.token,
            src: self.src,
        };
        let (_idx, claim) = hit!(arms.race(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

/// [`EnumVariantProbe`] for CBOR externally-tagged enums.
pub struct CborEnumVariantProbe<'de> {
    token: CborToken,
    src: &'de [u8],
}

impl<'de> EnumVariantProbe<'de> for CborEnumVariantProbe<'de> {
    type Error = CborError;
    type Claim = CborClaim<'de>;
    type PayloadDeserializer = CborSubDeserializer<'de>;

    fn fork(&mut self) -> Self {
        Self {
            token: self.token,
            src: self.src,
        }
    }

    async fn deserialize_unit_by_name<W>(
        self,
        candidates: W,
    ) -> Result<Probe<(Self::Claim, usize)>, Self::Error>
    where
        W: strede::ConcatableArray<T = (&'static str, usize)>
            + Copy
            + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        let entry = CborEntry {
            token: self.token,
            src: self.src,
            bignum_tag: None,
            raw_start: self.src,
        };
        match_entry_str_against(entry, candidates.as_ref()).await
    }

    async fn deserialize_payload_by_name<T, W>(
        self,
        candidates: W,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, usize, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
        W: strede::ConcatableArray<T = (&'static str, usize)>
            + Copy
            + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        // Expect a single-key map `{"VariantName": <payload>}`.
        let (mut src, remaining) = match self.token {
            CborToken::Map(Some(n)) if n >= 1 => (self.src, Some(n - 1)),
            CborToken::Map(None) => (self.src, None),
            _ => return Ok(Probe::Miss),
        };

        // Read the key token.
        let raw_key = next_token(&mut src)?;
        let key_tok = match strip_tags(&mut src, raw_key)? {
            Some(t) => t,
            None => return Ok(Probe::Miss),
        };

        // Match key against candidates using MatchVals.
        let key_entry = CborEntry {
            token: key_tok,
            src,
            bignum_tag: None,
            raw_start: src,
        };
        let (key_claim, MatchVals(idx, _)) = hit!(
            key_entry
                .deserialize_value::<MatchVals<usize, W>>(candidates)
                .await
        );

        // Read value token.
        let val_raw_start = key_claim.src;
        let mut val_src = key_claim.src;
        let raw_val = next_token(&mut val_src)?;
        let val_tok = match strip_tags(&mut val_src, raw_val)? {
            Some(t) => t,
            None => return Ok(Probe::Miss),
        };

        let sub = CborSubDeserializer::new_with_raw(val_src, val_tok, None, val_raw_start);
        let (val_claim, t) = hit!(T::deserialize(sub, extra).await);

        // Verify no additional keys follow (externally-tagged = exactly one KV pair).
        let map_done = match remaining {
            Some(0) => CborClaim {
                src: val_claim.src,
                remaining_after: 0,
            },
            None => {
                // Indefinite map: expect break byte.
                let mut s = val_claim.src;
                let tok = next_token(&mut s)?;
                match tok {
                    CborToken::Break => CborClaim {
                        src: s,
                        remaining_after: 0,
                    },
                    _ => return Ok(Probe::Miss),
                }
            }
            _ => return Ok(Probe::Miss),
        };

        Ok(Probe::Hit((map_done, idx, t)))
    }

    async fn deserialize_value_by_shape<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::PayloadDeserializer>,
    {
        let sub = CborSubDeserializer::new(self.src, self.token);
        T::deserialize(sub, extra).await
    }
}
