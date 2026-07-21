#![allow(dead_code)]

/// Encode a ULEB128 varint (postcard unsigned integer encoding).
pub fn varint(mut v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            break;
        } else {
            out.push(byte | 0x80);
        }
    }
    out
}

/// Encode a zigzag + ULEB128 varint (postcard signed integer encoding).
pub fn zigzag(v: i64) -> Vec<u8> {
    let encoded = ((v << 1) ^ (v >> 63)) as u64;
    varint(encoded)
}

/// Encode a postcard string: varint length + raw UTF-8 bytes.
pub fn pstr(s: &str) -> Vec<u8> {
    let mut out = varint(s.len() as u64);
    out.extend_from_slice(s.as_bytes());
    out
}

/// Encode postcard bytes: varint length + raw bytes.
pub fn pbytes(data: &[u8]) -> Vec<u8> {
    let mut out = varint(data.len() as u64);
    out.extend_from_slice(data);
    out
}

/// Encode a postcard f32: 4 bytes little-endian.
pub fn pf32(v: f32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Encode a postcard f64: 8 bytes little-endian.
pub fn pf64(v: f64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// Encode a postcard bool: 0x00 = false, 0x01 = true.
pub fn pbool(v: bool) -> Vec<u8> {
    vec![v as u8]
}

/// Encode a postcard Option::None.
pub fn pnone() -> Vec<u8> {
    vec![0x00]
}

/// Encode a postcard Option::Some(inner).
pub fn psome(inner: &[u8]) -> Vec<u8> {
    let mut out = vec![0x01];
    out.extend_from_slice(inner);
    out
}

/// Encode a postcard char: varint of the Unicode codepoint.
pub fn pchar(c: char) -> Vec<u8> {
    varint(c as u64)
}

/// Encode a postcard sequence: varint count + elements concatenated.
pub fn pseq(elements: &[&[u8]]) -> Vec<u8> {
    let mut out = varint(elements.len() as u64);
    for e in elements {
        out.extend_from_slice(e);
    }
    out
}

/// Encode a postcard map (HashMap/BTreeMap): varint count + key/value byte
/// pairs concatenated. Same shape as `pseq`, just alternating key then value.
pub fn pmap(pairs: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut out = varint(pairs.len() as u64);
    for (k, v) in pairs {
        out.extend_from_slice(k);
        out.extend_from_slice(v);
    }
    out
}

/// Encode a postcard u128: two consecutive varints (lo, hi).
pub fn pu128(v: u128) -> Vec<u8> {
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    let mut out = varint(lo);
    out.extend_from_slice(&varint(hi));
    out
}

/// Encode a postcard i128: zigzag on two consecutive varints.
pub fn pi128(v: i128) -> Vec<u8> {
    let encoded = ((v << 1) ^ (v >> 127)) as u128;
    let lo = encoded as u64;
    let hi = (encoded >> 64) as u64;
    let mut out = varint(lo);
    out.extend_from_slice(&varint(hi));
    out
}

/// Deserialize `$ty` from `$input: &[u8]` through the owned/chunked family,
/// feeding the whole input upfront via a trivial "empty the buffer to signal
/// EOF" loader (matches the pattern `strede-msgpack`/`strede-cbor`'s owned
/// tests use). Expands to a `Result<Option<$ty>, PostcardError>`, mirroring
/// the borrow-family `parse::<T>(input)` helper pattern used elsewhere in
/// this crate's tests.
///
/// A macro rather than a generic function: the loader closure's type is
/// anonymous per call site, and threading it through a `for<'s> T:
/// DeserializeOwned<ChunkedPostcardDeserializer<'s, ..., F>>` bound on a
/// shared function runs into HRTB/generic-closure ergonomics that
/// `strede-msgpack`/`strede-cbor`'s own owned-family tests sidestep by
/// inlining `SharedBuf::with_async` per test - this macro gets the same
/// inlining without the per-test boilerplate.
#[allow(unused_macros)]
macro_rules! parse_owned {
    ($ty:ty, $input:expr) => {{
        use strede::{Probe, SharedBuf};
        use strede_postcard::chunked::ChunkedPostcardDeserializer;
        use strede_test_util::block_on_loop;

        let input: &[u8] = $input;
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedPostcardDeserializer::new(shared);
                match <$ty as strede::DeserializeOwned<_>>::deserialize_owned(de, ()).await {
                    Ok(Probe::Hit((_, v))) => Ok(Some(v)),
                    Ok(Probe::Miss) => Ok(None),
                    Err(e) => Err(e),
                }
            },
        ))
    }};
}

/// Like [`parse_owned!`], but feeds `$input` through the loader
/// `$chunk_size` bytes at a time (via a `Cell<usize>`-tracked position)
/// instead of handing over the whole input upfront. Forces every
/// varint/byte read in the deserialization path to refill mid-value at
/// every possible split point - this is the harness `varint_chunking.rs`
/// uses to prove the resumable varint decoder (postcard's one genuinely new
/// piece of chunked-specific logic, with no msgpack/CBOR precedent) is
/// correct across real chunk boundaries, not just against a
/// whole-input-upfront loader that never actually forces a mid-value
/// refill.
#[allow(unused_macros)]
macro_rules! parse_owned_chunked {
    ($ty:ty, $input:expr, $chunk_size:expr) => {{
        use strede::{Probe, SharedBuf};
        use strede_postcard::chunked::ChunkedPostcardDeserializer;
        use strede_test_util::block_on_loop;

        let input: &[u8] = $input;
        let chunk_size: usize = $chunk_size;
        let pos = ::core::cell::Cell::new(chunk_size.min(input.len()));
        block_on_loop(SharedBuf::with_async(
            &input[..chunk_size.min(input.len())],
            async |buf: &mut &[u8]| {
                let start = pos.get();
                let end = (start + chunk_size).min(input.len());
                pos.set(end);
                *buf = &input[start..end];
            },
            async |shared| {
                let de = ChunkedPostcardDeserializer::new(shared);
                match <$ty as strede::DeserializeOwned<_>>::deserialize_owned(de, ()).await {
                    Ok(Probe::Hit((_, v))) => Ok(Some(v)),
                    Ok(Probe::Miss) => Ok(None),
                    Err(e) => Err(e),
                }
            },
        ))
    }};
}
