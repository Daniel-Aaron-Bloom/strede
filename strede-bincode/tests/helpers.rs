#![allow(dead_code)]

/// Byte order for the test encoder — mirrors `strede_bincode::config::ByteOrder`
/// but as a plain runtime enum (tests don't need zero-cost dispatch).
#[derive(Clone, Copy)]
pub enum Order {
    Le,
    Be,
}

/// Int-width strategy for the test encoder — mirrors
/// `strede_bincode::config::IntEncoding`.
#[derive(Clone, Copy)]
pub enum Mode {
    Fix,
    Var,
}

/// Encodes values the way `strede-bincode` under a given `(Order, Mode)`
/// config expects to decode them. Used to build test fixtures independent
/// of the crate's own decode logic (so tests don't just check the decoder
/// against itself).
#[derive(Clone, Copy)]
pub struct Enc {
    pub order: Order,
    pub mode: Mode,
}

impl Enc {
    /// Little-endian, varint — matches `strede_bincode::Standard` (bincode2's
    /// `standard()` default).
    pub const STANDARD: Enc = Enc {
        order: Order::Le,
        mode: Mode::Var,
    };
    /// Little-endian, fixed-width — matches `strede_bincode::Legacy`
    /// (bincode 1.x / bincode2's `legacy()`).
    pub const LEGACY: Enc = Enc {
        order: Order::Le,
        mode: Mode::Fix,
    };
    /// Big-endian, varint — matches `strede_bincode::BigStandard`.
    pub const BIG_STANDARD: Enc = Enc {
        order: Order::Be,
        mode: Mode::Var,
    };
    /// Big-endian, fixed-width — matches `strede_bincode::BigLegacy`.
    pub const BIG_LEGACY: Enc = Enc {
        order: Order::Be,
        mode: Mode::Fix,
    };

    /// Reorder a little-endian byte array into this encoder's configured order.
    fn tail(&self, bytes_le: &[u8]) -> Vec<u8> {
        match self.order {
            Order::Le => bytes_le.to_vec(),
            Order::Be => {
                let mut v = bytes_le.to_vec();
                v.reverse();
                v
            }
        }
    }

    /// Bincode's own unsigned varint scheme: `0..=250` single byte;
    /// `251`/`252`/`253`/`254` prefix a 2/4/8/16-byte tail.
    fn varint_from_u128(&self, v: u128) -> Vec<u8> {
        if v <= 250 {
            return vec![v as u8];
        }
        if v <= u16::MAX as u128 {
            let mut out = vec![251u8];
            out.extend(self.tail(&(v as u16).to_le_bytes()));
            return out;
        }
        if v <= u32::MAX as u128 {
            let mut out = vec![252u8];
            out.extend(self.tail(&(v as u32).to_le_bytes()));
            return out;
        }
        if v <= u64::MAX as u128 {
            let mut out = vec![253u8];
            out.extend(self.tail(&(v as u64).to_le_bytes()));
            return out;
        }
        let mut out = vec![254u8];
        out.extend(self.tail(&v.to_le_bytes()));
        out
    }

    /// Builds a deliberately *non-canonical* varint: forces the tail width
    /// implied by `prefix` regardless of whether `v` would actually need
    /// that much room — e.g. `varint_with_prefix(254, 100)` encodes the
    /// value `100` (which fits in a single byte) via the 16-byte
    /// `u128`-tail prefix. Used to test rejection of non-canonical wire
    /// forms; real `varint_from_u128`/per-width encoders above always pick
    /// the minimal canonical prefix, so they can't produce this on their own.
    pub fn varint_with_prefix(&self, prefix: u8, v: u128) -> Vec<u8> {
        let mut out = vec![prefix];
        match prefix {
            251 => out.extend(self.tail(&(v as u16).to_le_bytes())),
            252 => out.extend(self.tail(&(v as u32).to_le_bytes())),
            253 => out.extend(self.tail(&(v as u64).to_le_bytes())),
            254 => out.extend(self.tail(&v.to_le_bytes())),
            _ => panic!("prefix must be one of 251..=254"),
        }
        out
    }

    pub fn u8(&self, v: u8) -> Vec<u8> {
        vec![v]
    }
    pub fn i8(&self, v: i8) -> Vec<u8> {
        vec![v as u8]
    }

    pub fn u16(&self, v: u16) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(v as u128),
        }
    }
    pub fn u32(&self, v: u32) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(v as u128),
        }
    }
    pub fn u64(&self, v: u64) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(v as u128),
        }
    }
    pub fn u128(&self, v: u128) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(v),
        }
    }

    /// Zigzag: same formula `strede-postcard`'s own test helpers use
    /// (`(n << 1) ^ (n >> (bits - 1))`), widened to i128.
    fn zigzag128(&self, v: i128) -> u128 {
        ((v << 1) ^ (v >> 127)) as u128
    }

    pub fn i16(&self, v: i16) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(self.zigzag128(v as i128)),
        }
    }
    pub fn i32(&self, v: i32) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(self.zigzag128(v as i128)),
        }
    }
    pub fn i64(&self, v: i64) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(self.zigzag128(v as i128)),
        }
    }
    pub fn i128(&self, v: i128) -> Vec<u8> {
        match self.mode {
            Mode::Fix => self.tail(&v.to_le_bytes()),
            Mode::Var => self.varint_from_u128(self.zigzag128(v)),
        }
    }

    /// Floats are always fixed-width IEEE754 — order-dependent only, never
    /// varies by `Mode`.
    pub fn f32(&self, v: f32) -> Vec<u8> {
        self.tail(&v.to_le_bytes())
    }
    pub fn f64(&self, v: f64) -> Vec<u8> {
        self.tail(&v.to_le_bytes())
    }

    /// Always exactly 1 raw byte, regardless of config.
    pub fn bool(&self, v: bool) -> Vec<u8> {
        vec![v as u8]
    }

    /// Bincode encodes `char` as its own literal UTF-8 byte sequence,
    /// independent of both config axes.
    pub fn char(&self, c: char) -> Vec<u8> {
        let mut buf = [0u8; 4];
        c.encode_utf8(&mut buf).as_bytes().to_vec()
    }

    /// Length prefix: `u64`, subject to `Mode`.
    pub fn len(&self, n: usize) -> Vec<u8> {
        self.u64(n as u64)
    }

    /// Enum discriminant: `u32`, subject to `Mode`.
    pub fn discriminant(&self, idx: u32) -> Vec<u8> {
        self.u32(idx)
    }

    pub fn str(&self, s: &str) -> Vec<u8> {
        let mut out = self.len(s.len());
        out.extend_from_slice(s.as_bytes());
        out
    }

    pub fn bytes(&self, data: &[u8]) -> Vec<u8> {
        let mut out = self.len(data.len());
        out.extend_from_slice(data);
        out
    }

    /// Option tag: always exactly 1 raw byte regardless of config, ignoring
    /// `Mode` entirely (bincode's `Option` encoding delegates to plain `u8`).
    pub fn none(&self) -> Vec<u8> {
        vec![0]
    }
    pub fn some(&self, inner: &[u8]) -> Vec<u8> {
        let mut out = vec![1];
        out.extend_from_slice(inner);
        out
    }

    pub fn seq(&self, elements: &[&[u8]]) -> Vec<u8> {
        let mut out = self.len(elements.len());
        for e in elements {
            out.extend_from_slice(e);
        }
        out
    }
}

/// Deserialize `T` under config `C` from `input` through the borrow family.
///
/// A plain generic function (not a macro, unlike the owned-family helpers
/// below): callers pass an inline expression like `&E.u8(0)` as `input`,
/// and a function call's argument temporaries live for the whole enclosing
/// statement — exactly what's needed here. A macro expanding to multiple
/// `let` statements would instead drop that temporary at the end of its own
/// statement, before the later statement that borrows from it runs.
#[allow(dead_code)]
pub fn parse<'de, T, C>(input: &'de [u8]) -> Result<Option<T>, strede_bincode::BincodeError>
where
    C: strede_bincode::BincodeConfig,
    T: strede::Deserialize<'de, strede_bincode::BincodeDeserializer<'de, C>, Extra = ()>,
{
    use strede::Probe;
    use strede_bincode::BincodeDeserializer;
    use strede_test_util::block_on;

    let de = BincodeDeserializer::<C>::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Ok(Some(v)),
        Probe::Miss => Ok(None),
    }
}

/// Like [`parse`], but propagates the deserialization error instead of
/// unwrapping it, for tests that assert on a specific error variant.
#[allow(dead_code)]
pub fn parse_err<'de, T, C>(input: &'de [u8]) -> strede_bincode::BincodeError
where
    C: strede_bincode::BincodeConfig,
    T: strede::Deserialize<'de, strede_bincode::BincodeDeserializer<'de, C>, Extra = ()>,
{
    use strede_bincode::BincodeDeserializer;
    use strede_test_util::block_on;

    let de = BincodeDeserializer::<C>::new(input);
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())) {
        Err(e) => e,
        Ok(_) => panic!("expected error"),
    }
}

/// Deserialize `$ty` under config `$cfg` through the owned/chunked family,
/// feeding the whole input upfront — mirrors `strede-postcard`'s
/// `parse_owned!` macro exactly (a macro rather than a generic function for
/// the same HRTB/generic-closure ergonomics reason documented there).
#[allow(unused_macros)]
macro_rules! parse_owned {
    ($ty:ty, $cfg:ty, $input:expr) => {{
        use strede::{Probe, SharedBuf};
        use strede_bincode::{BincodeError, chunked::ChunkedBincodeDeserializer};
        use strede_test_util::block_on_loop;

        let input: &[u8] = $input;
        let result: Result<Option<$ty>, BincodeError> = block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedBincodeDeserializer::<$cfg, _, _>::new(shared);
                match <$ty as strede::DeserializeOwned<_>>::deserialize_owned(de, ()).await {
                    Ok(Probe::Hit((_, v))) => Ok(Some(v)),
                    Ok(Probe::Miss) => Ok(None),
                    Err(e) => Err(e),
                }
            },
        ));
        result
    }};
}

/// Like [`parse_owned!`], but feeds `$input` through the loader
/// `$chunk_size` bytes at a time — forces every read in the deserialization
/// path to refill mid-value at every possible split point. Mirrors
/// `strede-postcard`'s `parse_owned_chunked!`.
#[allow(unused_macros)]
macro_rules! parse_owned_chunked {
    ($ty:ty, $cfg:ty, $input:expr, $chunk_size:expr) => {{
        use strede::{Probe, SharedBuf};
        use strede_bincode::{BincodeError, chunked::ChunkedBincodeDeserializer};
        use strede_test_util::block_on_loop;

        let input: &[u8] = $input;
        let chunk_size: usize = $chunk_size;
        let pos = ::core::cell::Cell::new(chunk_size.min(input.len()));
        let result: Result<Option<$ty>, BincodeError> = block_on_loop(SharedBuf::with_async(
            &input[..chunk_size.min(input.len())],
            async |buf: &mut &[u8]| {
                let start = pos.get();
                let end = (start + chunk_size).min(input.len());
                pos.set(end);
                *buf = &input[start..end];
            },
            async |shared| {
                let de = ChunkedBincodeDeserializer::<$cfg, _, _>::new(shared);
                match <$ty as strede::DeserializeOwned<_>>::deserialize_owned(de, ()).await {
                    Ok(Probe::Hit((_, v))) => Ok(Some(v)),
                    Ok(Probe::Miss) => Ok(None),
                    Err(e) => Err(e),
                }
            },
        ));
        result
    }};
}
