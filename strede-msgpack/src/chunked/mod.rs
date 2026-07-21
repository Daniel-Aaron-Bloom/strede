//! Chunked MessagePack deserializer for async streaming input.
//!
//! Uses [`strede::SharedBuf`]/[`strede::Handle`] to coordinate access to a
//! buffer that is refilled asynchronously by a user-supplied loader closure.
//!
//! # Key differences from the borrow family
//!
//! - Implements the **owned** trait family only.  Zero-copy string/byte
//!   borrowing is unsupported (chunk lifetimes are shorter than the session).
//! - The format byte and any multi-byte length prefix may straddle buffer
//!   boundaries; `next_dispatch` refills the buffer as needed.
//! - Numbers are dispatched from the token value directly (no text round-trip).

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::MsgpackError;
use crate::token::MsgpackToken;
use core::future::Future;
use strede::utils::repeat;
use strede::{
    BigEndian, Buffer, DeserializeFromEnumOwned, DeserializeFromMapOwned, DeserializeFromSeqOwned,
    DeserializeOwned, DeserializerOwned, EntryOwned, EnumAccessOwned, EnumArmStackOwned,
    EnumVariantProbeOwned, Handle, MatchVals, NextKey, NumberAccessOwned, NumberEncoding, Probe,
    RawSlot, SharedBuf, hit,
};

// ---------------------------------------------------------------------------
// Claim
// ---------------------------------------------------------------------------

pub struct ChunkedMsgpackClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) offset: usize,
    pub(crate) handle: Handle<'s, B, F>,
    /// Remaining map key-value pairs after this claim (map context only); 0 otherwise.
    pub(crate) remaining_after: usize,
}

// ---------------------------------------------------------------------------
// RawSlot — drives `strede::PairSeqMapAccess`. Msgpack maps are wire-identical
// to "N pairs in a flat, count-prefixed stream", so the key/value probe
// quintet is generic infrastructure rather than hand-rolled here; see
// `access::ChunkedMsgpackMapAccess` (removed) for the shape this replaces.
// ---------------------------------------------------------------------------

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> RawSlot for ChunkedMsgpackClaim<'s, B, F> {
    type Error = MsgpackError;
    type Token = MsgpackToken;
    type SubDeserializer = ChunkedMsgpackSubDeserializer<'s, B, F>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            offset: self.offset,
            handle: self.handle.fork(),
            remaining_after: self.remaining_after,
        }
    }

    #[inline(always)]
    async fn next_token(mut self) -> Result<(Self, Self::Token), Self::Error> {
        let (handle, tok) = next_dispatch(self.handle, &mut self.offset).await?;
        Ok((
            Self {
                handle,
                offset: self.offset,
                remaining_after: self.remaining_after,
            },
            tok,
        ))
    }

    #[inline(always)]
    fn into_sub_deserializer(self, token: Self::Token) -> Self::SubDeserializer {
        ChunkedMsgpackSubDeserializer::new(self.handle, self.offset, token)
    }

    #[inline(always)]
    async fn skip_token(mut self, token: Self::Token) -> Result<Self, Self::Error> {
        let handle = skip_value_chunked(self.handle, &mut self.offset, token).await?;
        Ok(Self {
            handle,
            offset: self.offset,
            remaining_after: self.remaining_after,
        })
    }

    #[inline(always)]
    fn remaining_after(&self) -> Option<usize> {
        // Msgpack maps are always definite-length (fixmap/map16/map32 all
        // carry an explicit pair count) — `None` never occurs.
        Some(self.remaining_after)
    }

    #[inline(always)]
    fn with_remaining_after(mut self, remaining: Option<usize>) -> Self {
        self.remaining_after = remaining.expect("msgpack maps are always definite-length");
        self
    }
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Advance handle to the next chunk, resetting offset.
#[inline(always)]
pub(super) async fn refill<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<Handle<'s, B, F>, MsgpackError> {
    let h = handle.next().await;
    *offset = 0;
    if h.buf().is_empty() {
        return Err(MsgpackError::UnexpectedEnd);
    }
    Ok(h)
}

/// Read exactly one byte from the buffer, refilling if needed.
async fn read_byte<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, u8), MsgpackError> {
    loop {
        let buf = handle.buf();
        if *offset < buf.len() {
            let b = buf[*offset];
            *offset += 1;
            return Ok((handle, b));
        }
        handle = refill(handle, offset).await?;
    }
}

/// Read `N` big-endian bytes into a `[u8; N]` array, refilling across chunk
/// boundaries.
async fn read_bytes_exact<'s, B: Buffer, F: AsyncFnMut(&mut B), const N: usize>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, [u8; N]), MsgpackError> {
    let mut out = [0u8; N];
    let mut filled = 0;
    while filled < N {
        let buf = handle.buf();
        let avail = buf.len() - *offset;
        if avail == 0 {
            handle = refill(handle, offset).await?;
            continue;
        }
        let take = (N - filled).min(avail);
        out[filled..filled + take].copy_from_slice(&buf[*offset..*offset + take]);
        *offset += take;
        filled += take;
    }
    Ok((handle, out))
}

/// Read the next msgpack token from the streaming buffer.
///
/// May cross chunk boundaries for multi-byte headers (uint16, str32, etc.).
/// On return, `handle` and `*offset` are positioned immediately after the
/// header bytes (at the start of any payload, or the next token for scalars).
pub(crate) async fn next_dispatch<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, MsgpackToken), MsgpackError> {
    let (handle, byte) = read_byte(handle, offset).await?;
    decode_header(handle, offset, byte).await
}

async fn decode_header<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
    byte: u8,
) -> Result<(Handle<'s, B, F>, MsgpackToken), MsgpackError> {
    match byte {
        0x00..=0x7f => Ok((handle, MsgpackToken::UFixInt([byte]))),
        0x80..=0x8f => Ok((handle, MsgpackToken::Map((byte & 0x0f) as usize))),
        0x90..=0x9f => Ok((handle, MsgpackToken::Array((byte & 0x0f) as usize))),
        0xa0..=0xbf => Ok((handle, MsgpackToken::Str((byte & 0x1f) as usize))),
        0xc0 => Ok((handle, MsgpackToken::Nil)),
        0xc1 => Err(MsgpackError::UnexpectedByte { byte }),
        0xc2 => Ok((handle, MsgpackToken::Bool(false))),
        0xc3 => Ok((handle, MsgpackToken::Bool(true))),
        0xc4 => {
            let (h, [len]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
            Ok((h, MsgpackToken::Bin(len as usize)))
        }
        0xc5 => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((h, MsgpackToken::Bin(u16::from_be_bytes(b) as usize)))
        }
        0xc6 => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((h, MsgpackToken::Bin(u32::from_be_bytes(b) as usize)))
        }
        // ext8/16/32: read type byte, leave data bytes in stream
        0xc7 => {
            let (h, [len]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
            let (h, [type_byte]) = read_bytes_exact::<_, _, 1>(h, offset).await?;
            Ok((
                h,
                MsgpackToken::Ext {
                    type_id: type_byte as i8,
                    len: len as usize,
                },
            ))
        }
        0xc8 => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            let len = u16::from_be_bytes(b) as usize;
            let (h, [type_byte]) = read_bytes_exact::<_, _, 1>(h, offset).await?;
            Ok((
                h,
                MsgpackToken::Ext {
                    type_id: type_byte as i8,
                    len,
                },
            ))
        }
        0xc9 => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            let len = u32::from_be_bytes(b) as usize;
            let (h, [type_byte]) = read_bytes_exact::<_, _, 1>(h, offset).await?;
            Ok((
                h,
                MsgpackToken::Ext {
                    type_id: type_byte as i8,
                    len,
                },
            ))
        }
        0xca => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((
                h,
                MsgpackToken::Float32(f32::from_bits(u32::from_be_bytes(b))),
            ))
        }
        0xcb => {
            let (h, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
            Ok((
                h,
                MsgpackToken::Float64(f64::from_bits(u64::from_be_bytes(b))),
            ))
        }
        0xcc => {
            let (h, b) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
            Ok((h, MsgpackToken::UInt8(b)))
        }
        0xcd => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((h, MsgpackToken::UInt16(b)))
        }
        0xce => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((h, MsgpackToken::UInt32(b)))
        }
        0xcf => {
            let (h, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
            Ok((h, MsgpackToken::UInt64(b)))
        }
        0xd0 => {
            let (h, b) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
            Ok((h, MsgpackToken::Int8(b)))
        }
        0xd1 => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((h, MsgpackToken::Int16(b)))
        }
        0xd2 => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((h, MsgpackToken::Int32(b)))
        }
        0xd3 => {
            let (h, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
            Ok((h, MsgpackToken::Int64(b)))
        }
        // fixext1/2/4/8/16: type byte + N data bytes, all embedded in token
        0xd4 => {
            let (h, bytes) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            let type_id = bytes[0] as i8;
            let mut data = [0u8; 16];
            data[0] = bytes[1];
            Ok((
                h,
                MsgpackToken::FixExt {
                    type_id,
                    data,
                    len: 1,
                },
            ))
        }
        0xd5 => {
            let (h, bytes) = read_bytes_exact::<_, _, 3>(handle, offset).await?;
            let type_id = bytes[0] as i8;
            let mut data = [0u8; 16];
            data[..2].copy_from_slice(&bytes[1..3]);
            Ok((
                h,
                MsgpackToken::FixExt {
                    type_id,
                    data,
                    len: 2,
                },
            ))
        }
        0xd6 => {
            let (h, bytes) = read_bytes_exact::<_, _, 5>(handle, offset).await?;
            let type_id = bytes[0] as i8;
            let mut data = [0u8; 16];
            data[..4].copy_from_slice(&bytes[1..5]);
            Ok((
                h,
                MsgpackToken::FixExt {
                    type_id,
                    data,
                    len: 4,
                },
            ))
        }
        0xd7 => {
            let (h, bytes) = read_bytes_exact::<_, _, 9>(handle, offset).await?;
            let type_id = bytes[0] as i8;
            let mut data = [0u8; 16];
            data[..8].copy_from_slice(&bytes[1..9]);
            Ok((
                h,
                MsgpackToken::FixExt {
                    type_id,
                    data,
                    len: 8,
                },
            ))
        }
        0xd8 => {
            let (h, bytes) = read_bytes_exact::<_, _, 17>(handle, offset).await?;
            let type_id = bytes[0] as i8;
            let mut data = [0u8; 16];
            data.copy_from_slice(&bytes[1..17]);
            Ok((
                h,
                MsgpackToken::FixExt {
                    type_id,
                    data,
                    len: 16,
                },
            ))
        }
        0xd9 => {
            let (h, [len]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
            Ok((h, MsgpackToken::Str(len as usize)))
        }
        0xda => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((h, MsgpackToken::Str(u16::from_be_bytes(b) as usize)))
        }
        0xdb => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((h, MsgpackToken::Str(u32::from_be_bytes(b) as usize)))
        }
        0xdc => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((h, MsgpackToken::Array(u16::from_be_bytes(b) as usize)))
        }
        0xdd => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((h, MsgpackToken::Array(u32::from_be_bytes(b) as usize)))
        }
        0xde => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((h, MsgpackToken::Map(u16::from_be_bytes(b) as usize)))
        }
        0xdf => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((h, MsgpackToken::Map(u32::from_be_bytes(b) as usize)))
        }
        0xe0..=0xff => Ok((handle, MsgpackToken::IFixInt([byte]))),
    }
}

/// Skip `n` bytes from the buffer, refilling as needed.
async fn skip_n_bytes<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
    mut n: usize,
) -> Result<Handle<'s, B, F>, MsgpackError> {
    while n > 0 {
        let avail = handle.buf().len() - *offset;
        if avail == 0 {
            handle = refill(handle, offset).await?;
            continue;
        }
        let skip = n.min(avail);
        *offset += skip;
        n -= skip;
    }
    Ok(handle)
}

// ---------------------------------------------------------------------------
// Compact stack for iterative skip (avoids recursive async fn layout cycle)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Counter(u8);

impl Counter {
    fn new() -> Self {
        Self(0)
    }
    fn count(self) -> u8 {
        self.0 + 1
    }
    fn bits(self) -> u32 {
        64 / self.count() as u32
    }
    fn try_inc(self) -> Result<Self, Self> {
        if self.0 == 63 {
            Err(self)
        } else {
            Ok(Self(self.0 + 1))
        }
    }
    fn try_dec(self) -> Result<Self, Self> {
        if self.0 == 0 {
            Err(self)
        } else {
            Ok(Self(self.0 - 1))
        }
    }
}

#[derive(Clone, Copy)]
struct Accumulator {
    storage: u64,
    counter: Counter,
}

impl Accumulator {
    const fn empty() -> Self {
        Self {
            storage: 0,
            counter: Counter(0),
        }
    }

    fn new(v: u64) -> Self {
        Self {
            storage: v,
            counter: Counter::new(),
        }
    }

    fn slot_mask(bits: u32) -> u64 {
        if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        }
    }

    fn get_slot(&self, i: u8, bits: u32) -> u64 {
        (self.storage >> (i as u32 * bits)) & Self::slot_mask(bits)
    }

    fn try_push(&mut self, v: u64) -> Result<(), u64> {
        let new_counter = self.counter.try_inc().map_err(|_| v)?;
        let new_bits = new_counter.bits();
        let mask = Self::slot_mask(new_bits);
        if v & !mask != 0 {
            return Err(v);
        }
        let old_count = self.counter.count();
        let old_bits = self.counter.bits();
        for i in 0..old_count {
            if self.get_slot(i, old_bits) & !mask != 0 {
                return Err(v);
            }
        }
        let mut new_storage = 0u64;
        for i in 0..old_count {
            new_storage |= self.get_slot(i, old_bits) << (i as u32 * new_bits);
        }
        new_storage |= v << (old_count as u32 * new_bits);
        self.storage = new_storage;
        self.counter = new_counter;
        Ok(())
    }

    fn try_pop(&mut self) -> Result<u64, u64> {
        let count = self.counter.count();
        let bits = self.counter.bits();
        let top = self.get_slot(count - 1, bits);
        match self.counter.try_dec() {
            Err(_) => Err(top),
            Ok(new_counter) => {
                let new_bits = new_counter.bits();
                let mut new_storage = 0u64;
                for i in 0..new_counter.count() {
                    new_storage |= self.get_slot(i, bits) << (i as u32 * new_bits);
                }
                self.storage = new_storage;
                self.counter = new_counter;
                Ok(top)
            }
        }
    }
}

struct Stack<const N: usize> {
    data: [Accumulator; N],
    len: usize,
}

impl<const N: usize> Stack<N> {
    fn new() -> Self {
        Self {
            data: [Accumulator::empty(); N],
            len: 0,
        }
    }

    fn push(&mut self, val: u64) -> bool {
        if self.len > 0 && self.data[self.len - 1].try_push(val).is_ok() {
            return true;
        }
        if self.len == N {
            return false;
        }
        self.data[self.len] = Accumulator::new(val);
        self.len += 1;
        true
    }

    fn pop(&mut self) -> Option<u64> {
        if self.len == 0 {
            return None;
        }
        match self.data[self.len - 1].try_pop() {
            Ok(v) => Some(v),
            Err(v) => {
                self.len -= 1;
                Some(v)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Iterative skip — no recursive async fn, so no layout cycle
// ---------------------------------------------------------------------------

/// Skip one complete msgpack value (including its payload) in chunked mode.
pub(super) async fn skip_value_chunked<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
    tok: MsgpackToken,
) -> Result<Handle<'s, B, F>, MsgpackError> {
    let mut stack = Stack::<32>::new();

    // Seed the stack from the root token.
    match tok {
        MsgpackToken::Nil
        | MsgpackToken::Bool(_)
        | MsgpackToken::UFixInt(_)
        | MsgpackToken::UInt8(_)
        | MsgpackToken::UInt16(_)
        | MsgpackToken::UInt32(_)
        | MsgpackToken::UInt64(_)
        | MsgpackToken::IFixInt(_)
        | MsgpackToken::Int8(_)
        | MsgpackToken::Int16(_)
        | MsgpackToken::Int32(_)
        | MsgpackToken::Int64(_)
        | MsgpackToken::Float32(_)
        | MsgpackToken::Float64(_)
        | MsgpackToken::FixExt { .. } => return Ok(handle),
        MsgpackToken::Ext { len, .. } => {
            return skip_n_bytes(handle, offset, len).await;
        }
        MsgpackToken::Str(len) | MsgpackToken::Bin(len) => {
            return skip_n_bytes(handle, offset, len).await;
        }
        MsgpackToken::Array(count) => {
            if count == 0 {
                return Ok(handle);
            }
            if !stack.push(count as u64) {
                return Err(MsgpackError::SkipDepthExceeded);
            }
        }
        MsgpackToken::Map(count) => {
            if count == 0 {
                return Ok(handle);
            }
            // Map(n) = 2n values (alternating key and value).
            let total = (count as u64)
                .checked_mul(2)
                .ok_or(MsgpackError::SkipDepthExceeded)?;
            if !stack.push(total) {
                return Err(MsgpackError::SkipDepthExceeded);
            }
        }
    }

    loop {
        let remaining = match stack.pop() {
            None => break,
            Some(r) => r,
        };
        // Push back the count of remaining siblings (we are about to consume one).
        if remaining > 1 && !stack.push(remaining - 1) {
            return Err(MsgpackError::SkipDepthExceeded);
        }

        let (h, next_tok) = next_dispatch(handle, offset).await?;
        handle = h;

        match next_tok {
            MsgpackToken::Nil
            | MsgpackToken::Bool(_)
            | MsgpackToken::UFixInt(_)
            | MsgpackToken::UInt8(_)
            | MsgpackToken::UInt16(_)
            | MsgpackToken::UInt32(_)
            | MsgpackToken::UInt64(_)
            | MsgpackToken::IFixInt(_)
            | MsgpackToken::Int8(_)
            | MsgpackToken::Int16(_)
            | MsgpackToken::Int32(_)
            | MsgpackToken::Int64(_)
            | MsgpackToken::Float32(_)
            | MsgpackToken::Float64(_)
            | MsgpackToken::FixExt { .. } => {}
            MsgpackToken::Ext { len, .. } => {
                handle = skip_n_bytes(handle, offset, len).await?;
            }
            MsgpackToken::Str(len) | MsgpackToken::Bin(len) => {
                handle = skip_n_bytes(handle, offset, len).await?;
            }
            MsgpackToken::Array(count) => {
                if count > 0 && !stack.push(count as u64) {
                    #[cfg(feature = "alloc")]
                    {
                        handle = alloc::boxed::Box::pin(skip_value_chunked(
                            handle,
                            offset,
                            MsgpackToken::Array(count),
                        ))
                        .await?;
                    }
                    #[cfg(not(feature = "alloc"))]
                    return Err(MsgpackError::SkipDepthExceeded);
                }
            }
            MsgpackToken::Map(count) => {
                if count > 0 {
                    let total = (count as u64)
                        .checked_mul(2)
                        .ok_or(MsgpackError::SkipDepthExceeded)?;
                    if !stack.push(total) {
                        #[cfg(feature = "alloc")]
                        {
                            handle = alloc::boxed::Box::pin(skip_value_chunked(
                                handle,
                                offset,
                                MsgpackToken::Map(count),
                            ))
                            .await?;
                        }
                        #[cfg(not(feature = "alloc"))]
                        return Err(MsgpackError::SkipDepthExceeded);
                    }
                }
            }
        }
    }

    Ok(handle)
}

// ---------------------------------------------------------------------------
// Deserializer
// ---------------------------------------------------------------------------

pub struct ChunkedMsgpackDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: SharedBuf<'s, B, F>,
    offset: usize,
    pending_tok: Option<MsgpackToken>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedMsgpackDeserializer<'s, B, F> {
    pub fn new(shared: SharedBuf<'s, B, F>) -> Self {
        Self {
            shared,
            offset: 0,
            pending_tok: None,
        }
    }
}

pub struct ChunkedMsgpackSubDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    offset: usize,
    pending_tok: Option<MsgpackToken>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedMsgpackSubDeserializer<'s, B, F> {
    #[inline(always)]
    pub(crate) fn new(handle: Handle<'s, B, F>, offset: usize, tok: MsgpackToken) -> Self {
        Self {
            handle,
            offset,
            pending_tok: Some(tok),
        }
    }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

pub struct ChunkedMsgpackEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) token: MsgpackToken,
    pub(crate) offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedMsgpackEntry<'s, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token,
            offset: self.offset,
        }
    }

    pub(crate) fn into_claim(self) -> ChunkedMsgpackClaim<'s, B, F> {
        ChunkedMsgpackClaim {
            offset: self.offset,
            handle: self.handle,
            remaining_after: 0,
        }
    }

    /// Direct numeric dispatch for typed msgpack numbers.
    pub(crate) async fn parse_num<T: crate::full::ParseNum>(
        self,
    ) -> Result<Probe<(ChunkedMsgpackClaim<'s, B, F>, T)>, MsgpackError> {
        let v = match self.token {
            MsgpackToken::UFixInt(b) => T::from_ufixint(b),
            MsgpackToken::UInt8(b) => T::from_uint8(b),
            MsgpackToken::UInt16(b) => T::from_uint16(b),
            MsgpackToken::UInt32(b) => T::from_uint32(b),
            MsgpackToken::UInt64(b) => T::from_uint64(b),
            MsgpackToken::IFixInt(b) => T::from_ifixint(b),
            MsgpackToken::Int8(b) => T::from_int8(b),
            MsgpackToken::Int16(b) => T::from_int16(b),
            MsgpackToken::Int32(b) => T::from_int32(b),
            MsgpackToken::Int64(b) => T::from_int64(b),
            MsgpackToken::Float32(f) => T::from_f32(f),
            MsgpackToken::Float64(f) => T::from_f64(f),
            _ => return Ok(Probe::Miss),
        };
        match v {
            Some(value) => Ok(Probe::Hit((self.into_claim(), value))),
            None => Ok(Probe::Miss),
        }
    }
}

// ---------------------------------------------------------------------------
// Deserializer entry helpers
// ---------------------------------------------------------------------------

async fn run_next_top<'s, B, F, const N: usize, Fn_, Fut, R>(
    de: ChunkedMsgpackDeserializer<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<((), R)>, MsgpackError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedMsgpackEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedMsgpackClaim<'s, B, F>, R)>, MsgpackError>>,
{
    let main = de.shared.fork();
    let mut offset = de.offset;

    let (main, token) = if let Some(tok) = de.pending_tok {
        (main, tok)
    } else {
        next_dispatch(main, &mut offset).await?
    };

    let snap_offset = offset;
    let entry = ChunkedMsgpackEntry {
        handle: main,
        token,
        offset: snap_offset,
    };
    let (claim, r) = hit!(f(repeat(entry, ChunkedMsgpackEntry::clone)).await);

    // Trailing-garbage check: drain remaining buffer, then verify EOF.
    let mut h = claim.handle;
    let mut off = claim.offset;
    loop {
        let buf = h.buf();
        if off < buf.len() {
            return Err(MsgpackError::ExpectedEnd);
        }
        let new_h = h.next().await;
        if new_h.buf().is_empty() {
            break;
        }
        h = new_h;
        off = 0;
    }
    Ok(Probe::Hit(((), r)))
}

async fn run_next_sub<'s, B, F, const N: usize, Fn_, Fut, R>(
    de: ChunkedMsgpackSubDeserializer<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<(ChunkedMsgpackClaim<'s, B, F>, R)>, MsgpackError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedMsgpackEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedMsgpackClaim<'s, B, F>, R)>, MsgpackError>>,
{
    let main = de.handle;
    let mut offset = de.offset;

    let (main, token) = if let Some(tok) = de.pending_tok {
        (main, tok)
    } else {
        next_dispatch(main, &mut offset).await?
    };

    let snap_offset = offset;
    let entry = ChunkedMsgpackEntry {
        handle: main,
        token,
        offset: snap_offset,
    };
    f(repeat(entry, ChunkedMsgpackEntry::clone)).await
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedMsgpackDeserializer<'s, B, F>
{
    type Error = MsgpackError;
    type Claim = ();
    type EntryClaim = ChunkedMsgpackClaim<'s, B, F>;
    type Entry = ChunkedMsgpackEntry<'s, B, F>;

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

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned
    for ChunkedMsgpackSubDeserializer<'s, B, F>
{
    type Error = MsgpackError;
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type EntryClaim = ChunkedMsgpackClaim<'s, B, F>;
    type Entry = ChunkedMsgpackEntry<'s, B, F>;

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
// EntryOwned
// ---------------------------------------------------------------------------

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EntryOwned for ChunkedMsgpackEntry<'s, B, F> {
    type Error = MsgpackError;
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type SubDeserializer = ChunkedMsgpackSubDeserializer<'s, B, F>;
    type StrChunks = access::ChunkedMsgpackStrAccess<'s, B, F>;
    type BytesChunks = access::ChunkedMsgpackBytesAccess<'s, B, F>;
    type NumberChunks<Enc: NumberEncoding> = ChunkedMsgpackNumberAccess<'s, B, F>;
    type Map = strede::PairSeqMapAccess<ChunkedMsgpackClaim<'s, B, F>>;
    type Seq = access::ChunkedMsgpackSeqAccess<'s, B, F>;
    type Enum = ChunkedMsgpackEnumAccess<'s, B, F>;
    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token,
            offset: self.offset,
        }
    }

    #[inline(always)]
    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.token {
            MsgpackToken::Str(len) => Ok(Probe::Hit(access::ChunkedMsgpackStrAccess {
                handle: self.handle,
                offset: self.offset,
                remaining: len,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.token {
            MsgpackToken::Bin(len) | MsgpackToken::Str(len) => {
                Ok(Probe::Hit(access::ChunkedMsgpackBytesAccess {
                    handle: self.handle,
                    offset: self.offset,
                    remaining: len,
                }))
            }
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if Enc::NAME != BigEndian::NAME {
            return Ok(Probe::Miss);
        }
        match self.token {
            MsgpackToken::UFixInt(_)
            | MsgpackToken::UInt8(_)
            | MsgpackToken::UInt16(_)
            | MsgpackToken::UInt32(_)
            | MsgpackToken::UInt64(_)
            | MsgpackToken::IFixInt(_)
            | MsgpackToken::Int8(_)
            | MsgpackToken::Int16(_)
            | MsgpackToken::Int32(_)
            | MsgpackToken::Int64(_) => Ok(Probe::Hit(ChunkedMsgpackNumberAccess {
                token: self.token,
                handle: self.handle,
                offset: self.offset,
                done: false,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.token {
            MsgpackToken::Map(count) => Ok(Probe::Hit(strede::PairSeqMapAccess::new(
                ChunkedMsgpackClaim {
                    handle: self.handle,
                    offset: self.offset,
                    remaining_after: 0,
                },
                Some(count),
            ))),
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.token {
            MsgpackToken::Array(count) => Ok(Probe::Hit(access::ChunkedMsgpackSeqAccess {
                handle: self.handle,
                offset: self.offset,
                remaining: count,
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
            MsgpackToken::Nil => Ok(Probe::Hit((self.into_claim(), None))),
            other => {
                let sub = ChunkedMsgpackSubDeserializer::new(self.handle, self.offset, other);
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
        let sub = ChunkedMsgpackSubDeserializer::new(self.handle, self.offset, self.token);
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
        let map = hit!(EntryOwned::deserialize_map(self).await);
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
        let seq = hit!(EntryOwned::deserialize_seq(self).await);
        T::deserialize_from_seq_owned(seq, extra).await
    }

    #[inline(always)]
    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        match self.token {
            MsgpackToken::Str(_) | MsgpackToken::Map(_) => {
                Ok(Probe::Hit(ChunkedMsgpackEnumAccess {
                    handle: self.handle,
                    token: self.token,
                    offset: self.offset,
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
        T: DeserializeFromEnumOwned<Self::Enum>,
    {
        let e = match EntryOwned::deserialize_enum(self).await? {
            Probe::Hit(e) => e,
            Probe::Miss => return Ok(Probe::Miss),
        };
        T::deserialize_from_enum_owned(e, extra).await
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut offset = self.offset;
        let handle = skip_value_chunked(self.handle, &mut offset, self.token).await?;
        Ok(ChunkedMsgpackClaim {
            offset,
            handle,
            remaining_after: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// ChunkedMsgpackNumberAccess
// ---------------------------------------------------------------------------

pub struct ChunkedMsgpackNumberAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    token: MsgpackToken,
    handle: Handle<'s, B, F>,
    offset: usize,
    done: bool,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B), Enc: NumberEncoding> NumberAccessOwned<Enc>
    for ChunkedMsgpackNumberAccess<'s, B, F>
{
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type Error = MsgpackError;

    async fn next_number_chunk<R>(
        mut self,
        f: impl FnOnce(&Enc::Data) -> R,
    ) -> Result<strede::Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.done {
            return Ok(strede::Chunk::Done(ChunkedMsgpackClaim {
                offset: self.offset,
                handle: self.handle,
                remaining_after: 0,
            }));
        }
        let bytes: &[u8] = match &self.token {
            MsgpackToken::UFixInt(b)
            | MsgpackToken::UInt8(b)
            | MsgpackToken::IFixInt(b)
            | MsgpackToken::Int8(b) => b.as_slice(),
            MsgpackToken::UInt16(b) | MsgpackToken::Int16(b) => b.as_slice(),
            MsgpackToken::UInt32(b) | MsgpackToken::Int32(b) => b.as_slice(),
            MsgpackToken::UInt64(b) | MsgpackToken::Int64(b) => b.as_slice(),
            _ => unreachable!(),
        };
        let r = f(Enc::from_bytes(bytes));
        self.done = true;
        Ok(strede::Chunk::Data((self, r)))
    }
}

// ---------------------------------------------------------------------------
// ChunkedMsgpackEnumAccess / ChunkedMsgpackEnumVariantProbe
// ---------------------------------------------------------------------------
//
// Externally-tagged enums:
//   - Unit variants:     bare string token  ("VariantName")
//   - Non-unit variants: single-key map     ({"VariantName": <payload>})

pub struct ChunkedMsgpackEnumAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    token: MsgpackToken,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumAccessOwned for ChunkedMsgpackEnumAccess<'s, B, F> {
    type Error = MsgpackError;
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type VariantProbe = ChunkedMsgpackEnumVariantProbe<'s, B, F>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStackOwned<Self::VariantProbe>,
    {
        let vp = ChunkedMsgpackEnumVariantProbe {
            handle: self.handle,
            token: self.token,
            offset: self.offset,
        };
        let (_idx, claim) = hit!(arms.race(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

pub struct ChunkedMsgpackEnumVariantProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    token: MsgpackToken,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumVariantProbeOwned
    for ChunkedMsgpackEnumVariantProbe<'s, B, F>
{
    type Error = MsgpackError;
    type Claim = ChunkedMsgpackClaim<'s, B, F>;
    type PayloadDeserializer = ChunkedMsgpackSubDeserializer<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token,
            offset: self.offset,
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
        use access::ChunkedMsgpackStrAccess;
        use strede::StrAccessOwned as _;

        let mut str_access = match self.token {
            MsgpackToken::Str(len) => ChunkedMsgpackStrAccess {
                handle: self.handle,
                offset: self.offset,
                remaining: len,
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
        W: strede::ConcatableArray<T = (&'static str, usize)>
            + Copy
            + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        use strede::{
            MapKeyClaimOwned as _, MapKeyProbeOwned as _, MapValueClaimOwned as _,
            MapValueProbeOwned as _, PairSeqKeyProbe,
        };

        // Expect a single-key map {"VariantName": <payload>}.
        let count = match self.token {
            MsgpackToken::Map(n) if n >= 1 => n,
            _ => return Ok(Probe::Miss),
        };

        let mut offset = self.offset;
        let (handle, key_tok) = next_dispatch(self.handle, &mut offset).await?;
        let cursor = ChunkedMsgpackClaim {
            handle,
            offset,
            remaining_after: 0,
        };

        let key_probe = PairSeqKeyProbe::new(cursor, key_tok, Some(count - 1));

        let (key_claim, MatchVals(idx, _)) = match key_probe
            .deserialize_key::<MatchVals<usize, W>>(candidates)
            .await?
        {
            Probe::Hit(v) => v,
            Probe::Miss => return Ok(Probe::Miss),
        };

        let value_probe = key_claim.into_value_probe().await?;
        let (value_claim, t) = match value_probe.deserialize_value::<T>(extra).await? {
            Probe::Hit(v) => v,
            Probe::Miss => return Ok(Probe::Miss),
        };

        // Externally-tagged enum has exactly one key-value pair.
        let map_claim = match value_claim.next_key(0, 0).await? {
            NextKey::Done(c) => c,
            NextKey::Entry(_) => return Ok(Probe::Miss),
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
        let sub = ChunkedMsgpackSubDeserializer::new(self.handle, self.offset, self.token);
        T::deserialize_owned(sub, extra).await
    }
}

pub(crate) mod access;

#[cfg(test)]
mod tests {
    extern crate std;
    use super::ChunkedMsgpackDeserializer;
    use std::{boxed::Box, vec::Vec};
    use strede::shared_buf::SharedBuf;
    use strede::{DeserializeOwned, Skip};
    use strede_test_util::block_on_loop;

    // Stack<32> overflow analysis: each level of Array(2) nesting adds ~1 item
    // to the top accumulator (capacity 64). data[k] fills at level 64 + 63*k,
    // so data[31] fills at 64 + 63*31 = 2017. Level 2018 causes push() → false.
    const OVERFLOW_DEPTH: usize = 2100;

    /// Build `Array(2)[Array(2)[..., nil], nil]` at depth `d`.
    /// Encoding: [0x92; d] ++ [0xc0; d + 1]
    fn skip_deep_nested(depth: usize) -> Result<(), crate::MsgpackError> {
        let mut input = Vec::<u8>::with_capacity(depth * 2 + 1);
        input.extend(core::iter::repeat_n(0x92u8, depth));
        input.extend(core::iter::repeat_n(0xc0u8, depth + 1));
        let input: &'static [u8] = Box::leak(input.into_boxed_slice());
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedMsgpackDeserializer::new(shared);
                match Skip::deserialize_owned(de, ()).await? {
                    strede::Probe::Hit(_) => Ok(()),
                    strede::Probe::Miss => Err(crate::MsgpackError::UnexpectedEnd),
                }
            },
        ))
    }

    /// Shallow nesting — always succeeds in both alloc and no-alloc.
    #[test]
    fn skip_deep_nesting_valid() {
        assert!(skip_deep_nested(100).is_ok());
    }

    /// Beyond Stack<32> capacity without alloc: must return SkipDepthExceeded.
    #[cfg(not(feature = "alloc"))]
    #[test]
    fn skip_deep_nesting_no_alloc() {
        assert!(matches!(
            skip_deep_nested(OVERFLOW_DEPTH),
            Err(crate::MsgpackError::SkipDepthExceeded)
        ));
    }

    /// Beyond Stack<32> capacity with alloc: Box::pin recursion must succeed.
    #[cfg(feature = "alloc")]
    #[test]
    fn skip_deep_nesting_alloc() {
        assert!(skip_deep_nested(OVERFLOW_DEPTH).is_ok());
    }
}
