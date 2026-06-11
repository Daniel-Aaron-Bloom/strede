//! Chunked CBOR deserializer for async streaming input.
//!
//! Uses [`strede::SharedBuf`]/[`strede::Handle`] to coordinate access to a
//! buffer refilled asynchronously by a user-supplied loader closure.

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::CborError;
use crate::tag::{CborTag, TagHandler};
use crate::token::CborToken;
use core::future::Future;
use strede::utils::repeat;
use strede::{
    BigEndian, Buffer, DeserializeFromEnumOwned, DeserializeFromMapOwned,
    DeserializeFromSeqOwned, DeserializeOwned, DeserializerOwned, EntryOwned, EnumAccessOwned,
    EnumArmStackOwned, EnumVariantProbeOwned, Handle, NumberEncoding, Probe,
    SharedBuf, hit, match_str_chunks_against_owned,
};

// ---------------------------------------------------------------------------
// Claim
// ---------------------------------------------------------------------------

pub struct ChunkedCborClaim<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) offset: usize,
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) remaining_after: usize,
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

#[inline(always)]
pub(super) async fn refill<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<Handle<'s, B, F>, CborError> {
    let h = handle.next().await;
    *offset = 0;
    if h.buf().is_empty() {
        return Err(CborError::UnexpectedEnd);
    }
    Ok(h)
}

async fn read_byte<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, u8), CborError> {
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

pub(crate) async fn read_bytes_exact<'s, B: Buffer, F: AsyncFnMut(&mut B), const N: usize>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, [u8; N]), CborError> {
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

/// Read the CBOR additional argument (0..=27) given the `info` bits.
async fn read_argument<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
    info: u8,
) -> Result<(Handle<'s, B, F>, u64), CborError> {
    match info {
        0..=23 => Ok((handle, info as u64)),
        24 => {
            let (h, [v]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
            Ok((h, v as u64))
        }
        25 => {
            let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
            Ok((h, u16::from_be_bytes(b) as u64))
        }
        26 => {
            let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
            Ok((h, u32::from_be_bytes(b) as u64))
        }
        27 => {
            let (h, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
            Ok((h, u64::from_be_bytes(b)))
        }
        28..=30 => Err(CborError::UnexpectedByte { byte: info }),
        31 => Err(CborError::UnexpectedByte { byte: 31 }), // indefinite — callers check before calling
        _ => unreachable!(),
    }
}

/// Read the next CBOR token from the streaming buffer.
/// May cross chunk boundaries for multi-byte headers.
pub(crate) async fn next_dispatch<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
) -> Result<(Handle<'s, B, F>, CborToken), CborError> {
    let (handle, byte) = read_byte(handle, offset).await?;
    decode_header(handle, offset, byte).await
}

async fn decode_header<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    handle: Handle<'s, B, F>,
    offset: &mut usize,
    byte: u8,
) -> Result<(Handle<'s, B, F>, CborToken), CborError> {
    let major = byte >> 5;
    let info = byte & 0x1f;

    match major {
        0 => {
            if info == 31 {
                return Err(CborError::UnexpectedByte { byte });
            }
            let (h, v) = read_argument(handle, offset, info).await?;
            Ok((h, CborToken::UInt(v)))
        }
        1 => {
            if info == 31 {
                return Err(CborError::UnexpectedByte { byte });
            }
            let (h, v) = read_argument(handle, offset, info).await?;
            Ok((h, CborToken::NegInt(v)))
        }
        2 => {
            if info == 31 {
                return Ok((handle, CborToken::BstrIndef));
            }
            let (h, v) = read_argument(handle, offset, info).await?;
            Ok((h, CborToken::Bstr(v as usize)))
        }
        3 => {
            if info == 31 {
                return Ok((handle, CborToken::TstrIndef));
            }
            let (h, v) = read_argument(handle, offset, info).await?;
            Ok((h, CborToken::Tstr(v as usize)))
        }
        4 => {
            if info == 31 {
                return Ok((handle, CborToken::Array(None)));
            }
            let (h, v) = read_argument(handle, offset, info).await?;
            Ok((h, CborToken::Array(Some(v as usize))))
        }
        5 => {
            if info == 31 {
                return Ok((handle, CborToken::Map(None)));
            }
            let (h, v) = read_argument(handle, offset, info).await?;
            Ok((h, CborToken::Map(Some(v as usize))))
        }
        6 => {
            let (h, v) = read_argument(handle, offset, info).await?;
            Ok((h, CborToken::Tag(v)))
        }
        7 => match info {
            20 => Ok((handle, CborToken::Bool(false))),
            21 => Ok((handle, CborToken::Bool(true))),
            22 => Ok((handle, CborToken::Null)),
            23 => Ok((handle, CborToken::Undefined)),
            24 => {
                let (_h, [_v]) = read_bytes_exact::<_, _, 1>(handle, offset).await?;
                Err(CborError::UnexpectedByte { byte })
            }
            25 => {
                let (h, b) = read_bytes_exact::<_, _, 2>(handle, offset).await?;
                Ok((
                    h,
                    CborToken::Float16(crate::token::decode_f16(u16::from_be_bytes(b))),
                ))
            }
            26 => {
                let (h, b) = read_bytes_exact::<_, _, 4>(handle, offset).await?;
                Ok((h, CborToken::Float32(f32::from_bits(u32::from_be_bytes(b)))))
            }
            27 => {
                let (h, b) = read_bytes_exact::<_, _, 8>(handle, offset).await?;
                Ok((h, CborToken::Float64(f64::from_bits(u64::from_be_bytes(b)))))
            }
            31 => Ok((handle, CborToken::Break)),
            _ => Err(CborError::UnexpectedByte { byte }),
        },
        _ => unreachable!(),
    }
}

/// Strip leading `Tag` tokens using `Ignored` handler.
async fn strip_tags_dispatch<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
    mut tok: CborToken,
) -> Result<(Handle<'s, B, F>, CborToken), CborError> {
    loop {
        match tok {
            CborToken::Tag(_) => {
                let (h, next) = next_dispatch(handle, offset).await?;
                handle = h;
                tok = next;
            }
            other => return Ok((handle, other)),
        }
    }
}

async fn strip_tags_bignum_dispatch<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
    mut tok: CborToken,
) -> Result<(Handle<'s, B, F>, Option<u64>, CborToken), CborError> {
    let mut bignum_tag: Option<u64> = None;
    loop {
        match tok {
            CborToken::Tag(n @ (2 | 3)) => {
                bignum_tag = Some(n);
                let (h, next) = next_dispatch(handle, offset).await?;
                handle = h;
                tok = next;
            }
            CborToken::Tag(_) => {
                bignum_tag = None;
                let (h, next) = next_dispatch(handle, offset).await?;
                handle = h;
                tok = next;
            }
            other => return Ok((handle, bignum_tag, other)),
        }
    }
}

/// Strip leading `Tag` tokens using a `TagHandler`.
/// Returns `None` if the handler vetoes any tag.
pub(crate) async fn strip_tags_with_dispatch<
    's,
    B: Buffer,
    F: AsyncFnMut(&mut B),
    H: TagHandler,
>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
    mut tok: CborToken,
    mut handler: H,
) -> Result<Option<(Handle<'s, B, F>, CborToken, H)>, CborError> {
    loop {
        match tok {
            CborToken::Tag(n) => {
                handler = match handler.handle(n) {
                    Some(h) => h,
                    None => return Ok(None),
                };
                let (h, next) = next_dispatch(handle, offset).await?;
                handle = h;
                tok = next;
            }
            other => return Ok(Some((handle, other, handler))),
        }
    }
}

/// Skip `n` bytes from the buffer, refilling as needed.
async fn skip_n_bytes<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
    mut n: usize,
) -> Result<Handle<'s, B, F>, CborError> {
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
// Compact stack for iterative skip
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

// Special sentinel for indefinite-length containers in the skip stack.
// We use u64::MAX since actual counts won't reach that value.
const INDEF_SENTINEL: u64 = u64::MAX;

/// Skip one complete CBOR value (including payload) in chunked mode.
///
/// Fully iterative — no recursive async calls — so the compiler can compute a
/// finite layout for the future.  A `Stack` tracks pending array/map item counts
/// and indefinite-length container sentinels; `skip_one_token` (a sync macro)
/// handles the per-token dispatch inline.
pub(super) async fn skip_value_chunked<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
    mut handle: Handle<'s, B, F>,
    offset: &mut usize,
    mut tok: CborToken,
) -> Result<Handle<'s, B, F>, CborError> {
    // Strip leading tag wrappers without any recursive call.
    while let CborToken::Tag(_) = tok {
        let (h, next) = next_dispatch(handle, offset).await?;
        handle = h;
        tok = next;
    }

    // Process `tok`, then drain `stack`.  Every async operation is a direct
    // `.await` on a non-recursive helper (`skip_n_bytes`, `next_dispatch`).
    let mut stack = Stack::<32>::new();

    // `process_tok` inlines the per-token logic for both the initial token and
    // each token popped while draining the stack.
    //
    // Returns `true` if the caller should `break` out of the drain loop (i.e.
    // the token was self-contained and pushed nothing onto `stack`).  For tokens
    // that extend the stack this is a no-op that lets the loop continue.
    //
    // Because this is a macro (not an async fn), it has no separate future and
    // therefore introduces no layout cycle.
    macro_rules! process_tok {
        ($tok:expr) => {{
            match $tok {
                CborToken::UInt(_)
                | CborToken::NegInt(_)
                | CborToken::Bool(_)
                | CborToken::Null
                | CborToken::Undefined
                | CborToken::Float16(_)
                | CborToken::Float32(_)
                | CborToken::Float64(_) => { /* leaf, nothing to push */ }
                CborToken::Break => return Err(CborError::InvalidBreak),
                CborToken::Bstr(len) | CborToken::Tstr(len) => {
                    handle = skip_n_bytes(handle, offset, len).await?;
                }
                CborToken::BstrIndef | CborToken::TstrIndef => {
                    loop {
                        let (h, chunk_tok) = next_dispatch(handle, offset).await?;
                        handle = h;
                        match chunk_tok {
                            CborToken::Break => break,
                            CborToken::Bstr(len) | CborToken::Tstr(len) => {
                                handle = skip_n_bytes(handle, offset, len).await?;
                            }
                            _ => return Err(CborError::UnexpectedByte { byte: 0 }),
                        }
                    }
                }
                CborToken::Tag(_) => {
                    // A tag wraps one value: push 1 so the drain loop reads it.
                    if !stack.push(1) {
                        return Err(CborError::SkipDepthExceeded);
                    }
                }
                CborToken::Array(Some(0)) | CborToken::Map(Some(0)) => { /* empty */ }
                CborToken::Array(Some(n)) => {
                    if !stack.push(n as u64) {
                        return Err(CborError::SkipDepthExceeded);
                    }
                }
                CborToken::Map(Some(n)) => {
                    let total = (n as u64)
                        .checked_mul(2)
                        .ok_or(CborError::SkipDepthExceeded)?;
                    if !stack.push(total) {
                        return Err(CborError::SkipDepthExceeded);
                    }
                }
                CborToken::Array(None) | CborToken::Map(None) => {
                    if !stack.push(INDEF_SENTINEL) {
                        return Err(CborError::SkipDepthExceeded);
                    }
                }
            }
        }};
    }

    process_tok!(tok);

    loop {
        let remaining = match stack.pop() {
            None => break,
            Some(r) => r,
        };

        if remaining == INDEF_SENTINEL {
            let (h, next_tok) = next_dispatch(handle, offset).await?;
            handle = h;
            if matches!(next_tok, CborToken::Break) {
                continue;
            }
            // Push sentinel back so we keep reading items until the break.
            if !stack.push(INDEF_SENTINEL) {
                return Err(CborError::SkipDepthExceeded);
            }
            process_tok!(next_tok);
        } else {
            if remaining > 1 && !stack.push(remaining - 1) {
                return Err(CborError::SkipDepthExceeded);
            }
            let (h, next_tok) = next_dispatch(handle, offset).await?;
            handle = h;
            process_tok!(next_tok);
        }
    }

    Ok(handle)
}

// ---------------------------------------------------------------------------
// Deserializer
// ---------------------------------------------------------------------------

pub struct ChunkedCborDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: SharedBuf<'s, B, F>,
    offset: usize,
    pending_tok: Option<CborToken>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedCborDeserializer<'s, B, F> {
    pub fn new(shared: SharedBuf<'s, B, F>) -> Self {
        Self {
            shared,
            offset: 0,
            pending_tok: None,
        }
    }
}

pub struct ChunkedCborSubDeserializer<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) offset: usize,
    pub(crate) pending_tok: Option<CborToken>,
    pub(crate) bignum_tag: Option<u64>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedCborSubDeserializer<'s, B, F> {
    #[inline(always)]
    pub(crate) fn new(handle: Handle<'s, B, F>, offset: usize, tok: CborToken) -> Self {
        Self {
            handle,
            offset,
            pending_tok: Some(tok),
            bignum_tag: None,
        }
    }

    #[inline(always)]
    pub(crate) fn new_bignum(
        handle: Handle<'s, B, F>,
        offset: usize,
        tok: CborToken,
        bignum_tag: Option<u64>,
    ) -> Self {
        Self {
            handle,
            offset,
            pending_tok: Some(tok),
            bignum_tag,
        }
    }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

pub struct ChunkedCborEntry<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    pub(crate) handle: Handle<'s, B, F>,
    pub(crate) token: CborToken,
    pub(crate) offset: usize,
    pub(crate) bignum_tag: Option<u64>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> ChunkedCborEntry<'s, B, F> {
    #[inline(always)]
    fn clone(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token,
            offset: self.offset,
            bignum_tag: self.bignum_tag,
        }
    }

    pub(crate) fn into_claim(self) -> ChunkedCborClaim<'s, B, F> {
        ChunkedCborClaim {
            offset: self.offset,
            handle: self.handle,
            remaining_after: 0,
        }
    }

    pub(crate) async fn parse_num<T: crate::full::ParseNum>(
        self,
    ) -> Result<Probe<(ChunkedCborClaim<'s, B, F>, T)>, CborError> {
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

// ---------------------------------------------------------------------------
// Deserializer entry helpers
// ---------------------------------------------------------------------------

async fn run_next_top<'s, B, F, const N: usize, Fn_, Fut, R>(
    de: ChunkedCborDeserializer<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<((), R)>, CborError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedCborEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedCborClaim<'s, B, F>, R)>, CborError>>,
{
    let main = de.shared.fork();
    let mut offset = de.offset;

    let (main, raw_token) = if let Some(tok) = de.pending_tok {
        (main, tok)
    } else {
        next_dispatch(main, &mut offset).await?
    };

    let (main, bignum_tag, token) =
        strip_tags_bignum_dispatch(main, &mut offset, raw_token).await?;

    let snap_offset = offset;
    let entry = ChunkedCborEntry {
        handle: main,
        token,
        offset: snap_offset,
        bignum_tag,
    };
    let (claim, r) = hit!(f(repeat(entry, ChunkedCborEntry::clone)).await);

    // Trailing-garbage check
    let mut h = claim.handle;
    let mut off = claim.offset;
    loop {
        let buf = h.buf();
        if off < buf.len() {
            return Err(CborError::ExpectedEnd);
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
    de: ChunkedCborSubDeserializer<'s, B, F>,
    mut f: Fn_,
) -> Result<Probe<(ChunkedCborClaim<'s, B, F>, R)>, CborError>
where
    B: Buffer,
    F: AsyncFnMut(&mut B),
    Fn_: FnMut([ChunkedCborEntry<'s, B, F>; N]) -> Fut,
    Fut: Future<Output = Result<Probe<(ChunkedCborClaim<'s, B, F>, R)>, CborError>>,
{
    let main = de.handle;
    let mut offset = de.offset;

    // pending_tok is already tag-stripped by the parent; bignum_tag carried through
    let (main, bignum_tag, token) = if let Some(tok) = de.pending_tok {
        (main, de.bignum_tag, tok)
    } else {
        let (h, raw) = next_dispatch(main, &mut offset).await?;
        strip_tags_bignum_dispatch(h, &mut offset, raw).await?
    };

    let snap_offset = offset;
    let entry = ChunkedCborEntry {
        handle: main,
        token,
        offset: snap_offset,
        bignum_tag,
    };
    f(repeat(entry, ChunkedCborEntry::clone)).await
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializerOwned for ChunkedCborDeserializer<'s, B, F> {
    type Error = CborError;
    type Claim = ();
    type EntryClaim = ChunkedCborClaim<'s, B, F>;
    type Entry = ChunkedCborEntry<'s, B, F>;

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
    for ChunkedCborSubDeserializer<'s, B, F>
{
    type Error = CborError;
    type Claim = ChunkedCborClaim<'s, B, F>;
    type EntryClaim = ChunkedCborClaim<'s, B, F>;
    type Entry = ChunkedCborEntry<'s, B, F>;

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

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EntryOwned for ChunkedCborEntry<'s, B, F> {
    type Error = CborError;
    type Claim = ChunkedCborClaim<'s, B, F>;
    type SubDeserializer = ChunkedCborSubDeserializer<'s, B, F>;
    type StrChunks = access::ChunkedCborStrAccess<'s, B, F>;
    type BytesChunks = access::ChunkedCborBytesAccess<'s, B, F>;
    type NumberChunks<Enc: NumberEncoding> = access::ChunkedCborNumberAccess<'s, B, F>;
    type Map = access::ChunkedCborMapAccess<'s, B, F>;
    type Seq = access::ChunkedCborSeqAccess<'s, B, F>;
    type Enum = ChunkedCborEnumAccess<'s, B, F>;

    fn fork(&mut self) -> Self {
        Self {
            handle: self.handle.fork(),
            token: self.token,
            offset: self.offset,
            bignum_tag: self.bignum_tag,
        }
    }

    #[inline(always)]
    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.token {
            CborToken::Tstr(len) => Ok(Probe::Hit(access::ChunkedCborStrAccess {
                handle: self.handle,
                offset: self.offset,
                state: access::ChunkedStrState::Definite { remaining: len },
            })),
            CborToken::TstrIndef => Ok(Probe::Hit(access::ChunkedCborStrAccess {
                handle: self.handle,
                offset: self.offset,
                state: access::ChunkedStrState::Indefinite,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.token {
            CborToken::Bstr(len) => Ok(Probe::Hit(access::ChunkedCborBytesAccess {
                handle: self.handle,
                offset: self.offset,
                state: access::ChunkedBytesState::Definite { remaining: len },
            })),
            CborToken::BstrIndef => Ok(Probe::Hit(access::ChunkedCborBytesAccess {
                handle: self.handle,
                offset: self.offset,
                state: access::ChunkedBytesState::Indefinite,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_number_chunks<Enc: NumberEncoding>(
        self,
    ) -> Result<Probe<Self::NumberChunks<Enc>>, Self::Error> {
        if Enc::NAME != BigEndian::NAME {
            return Ok(Probe::Miss);
        }
        match (self.bignum_tag, self.token) {
            (Some(2 | 3), CborToken::Bstr(len)) => {
                Ok(Probe::Hit(access::ChunkedCborNumberAccess {
                    handle: self.handle,
                    offset: self.offset,
                    state: access::ChunkedBytesState::Definite { remaining: len },
                }))
            }
            (Some(2 | 3), CborToken::BstrIndef) => {
                Ok(Probe::Hit(access::ChunkedCborNumberAccess {
                    handle: self.handle,
                    offset: self.offset,
                    state: access::ChunkedBytesState::Indefinite,
                }))
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.token {
            CborToken::Map(count) => Ok(Probe::Hit(access::ChunkedCborMapAccess {
                handle: self.handle,
                offset: self.offset,
                remaining: count,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.token {
            CborToken::Array(count) => Ok(Probe::Hit(access::ChunkedCborSeqAccess {
                handle: self.handle,
                offset: self.offset,
                remaining: count,
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
            CborToken::Null | CborToken::Undefined => Ok(Probe::Hit((self.into_claim(), None))),
            other => {
                let sub = ChunkedCborSubDeserializer::new(self.handle, self.offset, other);
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
        let sub = ChunkedCborSubDeserializer::new(self.handle, self.offset, self.token);
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

    async fn deserialize_enum(self) -> Result<Probe<Self::Enum>, Self::Error> {
        match self.token {
            CborToken::Tstr(_) | CborToken::TstrIndef | CborToken::Map(_) => {
                Ok(Probe::Hit(ChunkedCborEnumAccess {
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
        Ok(ChunkedCborClaim {
            offset,
            handle,
            remaining_after: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// ChunkedCborEnumAccess / ChunkedCborEnumVariantProbe
// ---------------------------------------------------------------------------

/// [`EnumAccessOwned`] for CBOR externally-tagged enums (chunked/streaming).
///
/// - Unit variants: bare text string (`"VariantName"`)
/// - Non-unit variants: single-key map (`{"VariantName": <payload>}`)
pub struct ChunkedCborEnumAccess<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    token: CborToken,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumAccessOwned for ChunkedCborEnumAccess<'s, B, F> {
    type Error = CborError;
    type Claim = ChunkedCborClaim<'s, B, F>;
    type VariantProbe = ChunkedCborEnumVariantProbe<'s, B, F>;

    async fn iterate<S>(self, mut arms: S) -> Result<Probe<(Self::Claim, S::Outputs)>, Self::Error>
    where
        S: EnumArmStackOwned<Self::VariantProbe>,
    {
        let vp = ChunkedCborEnumVariantProbe {
            handle: self.handle,
            token: self.token,
            offset: self.offset,
        };
        let (_idx, claim) = hit!(arms.race(vp).await);
        let outputs = arms.take_outputs();
        Ok(Probe::Hit((claim, outputs)))
    }
}

/// [`EnumVariantProbeOwned`] for CBOR externally-tagged enums (chunked/streaming).
pub struct ChunkedCborEnumVariantProbe<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    handle: Handle<'s, B, F>,
    token: CborToken,
    offset: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> EnumVariantProbeOwned
    for ChunkedCborEnumVariantProbe<'s, B, F>
{
    type Error = CborError;
    type Claim = ChunkedCborClaim<'s, B, F>;
    type PayloadDeserializer = ChunkedCborSubDeserializer<'s, B, F>;

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
        W: strede::ConcatableArray<T = (&'static str, usize)> + Copy + AsRef<[(&'static str, usize)]>,
        W::OtherArray<bool>: AsRef<[bool]> + AsMut<[bool]>,
    {
        let str_access = match self.token {
            CborToken::Tstr(len) => access::ChunkedCborStrAccess {
                handle: self.handle,
                offset: self.offset,
                state: access::ChunkedStrState::Definite { remaining: len },
            },
            CborToken::TstrIndef => access::ChunkedCborStrAccess {
                handle: self.handle,
                offset: self.offset,
                state: access::ChunkedStrState::Indefinite,
            },
            _ => return Ok(Probe::Miss),
        };
        match_str_chunks_against_owned(str_access, candidates.as_ref()).await
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
        // Expect a single-key map `{"VariantName": <payload>}`.
        let (handle, mut offset, remaining) = match self.token {
            CborToken::Map(Some(n)) if n >= 1 => (self.handle, self.offset, Some(n - 1)),
            CborToken::Map(None) => (self.handle, self.offset, None),
            _ => return Ok(Probe::Miss),
        };

        // Read key token.
        let (handle, raw_key) = next_dispatch(handle, &mut offset).await?;
        let (handle, key_tok) = strip_tags_dispatch(handle, &mut offset, raw_key).await?;

        // Match key as a text string against candidates.
        let key_str_access = match key_tok {
            CborToken::Tstr(len) => access::ChunkedCborStrAccess {
                handle,
                offset,
                state: access::ChunkedStrState::Definite { remaining: len },
            },
            CborToken::TstrIndef => access::ChunkedCborStrAccess {
                handle,
                offset,
                state: access::ChunkedStrState::Indefinite,
            },
            _ => return Ok(Probe::Miss),
        };

        let (key_claim, idx) =
            match match_str_chunks_against_owned(key_str_access, candidates.as_ref()).await? {
                Probe::Hit(v) => v,
                Probe::Miss => return Ok(Probe::Miss),
            };

        // Read value token (already stripped of tags by sub-deserializer entry).
        let mut val_offset = key_claim.offset;
        let (val_handle, raw_val) = next_dispatch(key_claim.handle, &mut val_offset).await?;
        let (val_handle, val_tok) = strip_tags_dispatch(val_handle, &mut val_offset, raw_val).await?;

        let sub = ChunkedCborSubDeserializer::new(val_handle, val_offset, val_tok);
        let (val_claim, t) = hit!(T::deserialize_owned(sub, extra).await);

        // Verify no additional keys (externally-tagged = exactly one KV pair).
        let map_done = match remaining {
            Some(0) => ChunkedCborClaim {
                offset: val_claim.offset,
                handle: val_claim.handle,
                remaining_after: 0,
            },
            None => {
                // Indefinite map: expect break.
                let mut off = val_claim.offset;
                let (h, tok) = next_dispatch(val_claim.handle, &mut off).await?;
                match tok {
                    CborToken::Break => ChunkedCborClaim {
                        offset: off,
                        handle: h,
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
        T: DeserializeOwned<Self::PayloadDeserializer>,
    {
        let sub = ChunkedCborSubDeserializer::new(self.handle, self.offset, self.token);
        T::deserialize_owned(sub, extra).await
    }
}

// ---------------------------------------------------------------------------
// CborTag impl for owned family
// ---------------------------------------------------------------------------

macro_rules! impl_cbor_tag_owned {
    ($de:ty) => {
        impl<'s, B: Buffer, F: AsyncFnMut(&mut B), T, H> DeserializeOwned<$de> for CborTag<T, H>
        where
            T: DeserializeOwned<ChunkedCborSubDeserializer<'s, B, F>>,
            H: TagHandler,
        {
            type Extra = (H, T::Extra);

            async fn deserialize_owned(
                d: $de,
                (handler, extra): (H, T::Extra),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, CborError> {
                let mut handler_slot = Some(handler);
                let mut extra_slot = Some(extra);
                d.entry(|[e]| {
                    let handler = handler_slot.take().unwrap();
                    let extra = extra_slot.take().unwrap();
                    async move {
                        let mut offset = e.offset;
                        let (handle, tok, h) =
                            match strip_tags_with_dispatch(e.handle, &mut offset, e.token, handler)
                                .await?
                            {
                                Some(x) => x,
                                None => return Ok(Probe::Miss),
                            };
                        let sub = ChunkedCborSubDeserializer::new(handle, offset, tok);
                        match T::deserialize_owned(sub, extra).await? {
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
                })
                .await
            }
        }
    };
}

impl_cbor_tag_owned!(ChunkedCborDeserializer<'s, B, F>);
impl_cbor_tag_owned!(ChunkedCborSubDeserializer<'s, B, F>);

pub(crate) mod access;
