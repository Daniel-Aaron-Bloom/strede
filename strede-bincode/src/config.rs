//! Compile-time wire-format configuration.
//!
//! Real bincode's wire format is not fixed the way postcard's is: the actual
//! `bincode` crate exposes a `Configuration<Endianness, IntEncoding, Limit>`.
//! `Limit` is a decode-time safety cap with no effect on wire bytes and is
//! out of scope here. The two wire-affecting axes become compile-time
//! generic parameters, bundled behind one [`BincodeConfig`] trait so every
//! deserializer/entry/accessor type only needs a single generic parameter
//! rather than two.

use core::marker::PhantomData;

/// Byte order used to interpret fixed-width integer/float payloads (and the
/// multi-byte tail of a varint-encoded value).
///
/// Implemented for strede's own [`strede::LittleEndian`]/[`strede::BigEndian`]
/// marker types (foreign types, local trait — orphan-rule legal) rather than
/// inventing parallel marker types. Requires `NumberEncoding<Data = [u8]>`
/// (which both marker types already implement in strede core) so
/// `deserialize_number_chunks` can compare a caller's requested `Enc::NAME`
/// against `C::Order`'s own encoding name directly.
pub trait ByteOrder: strede::NumberEncoding<Data = [u8]> {
    fn read_u16(b: [u8; 2]) -> u16;
    fn read_u32(b: [u8; 4]) -> u32;
    fn read_u64(b: [u8; 8]) -> u64;
    fn read_u128(b: [u8; 16]) -> u128;
}

impl ByteOrder for strede::LittleEndian {
    #[inline(always)]
    fn read_u16(b: [u8; 2]) -> u16 {
        u16::from_le_bytes(b)
    }
    #[inline(always)]
    fn read_u32(b: [u8; 4]) -> u32 {
        u32::from_le_bytes(b)
    }
    #[inline(always)]
    fn read_u64(b: [u8; 8]) -> u64 {
        u64::from_le_bytes(b)
    }
    #[inline(always)]
    fn read_u128(b: [u8; 16]) -> u128 {
        u128::from_le_bytes(b)
    }
}

impl ByteOrder for strede::BigEndian {
    #[inline(always)]
    fn read_u16(b: [u8; 2]) -> u16 {
        u16::from_be_bytes(b)
    }
    #[inline(always)]
    fn read_u32(b: [u8; 4]) -> u32 {
        u32::from_be_bytes(b)
    }
    #[inline(always)]
    fn read_u64(b: [u8; 8]) -> u64 {
        u64::from_be_bytes(b)
    }
    #[inline(always)]
    fn read_u128(b: [u8; 16]) -> u128 {
        u128::from_be_bytes(b)
    }
}

/// Integer width strategy: fixed-width or bincode's own compact varint.
pub trait IntEncoding {
    const NAME: &'static str;
}

/// Fixed-width integers: 1/2/4/8/16 raw bytes per width. Matches bincode
/// 1.x's only mode, and bincode2's `legacy()` config.
#[derive(Debug)]
pub struct Fixint;

/// Bincode's own compact varint scheme (byte-prefix 0..=250 / 251 / 252 /
/// 253 / 254, see [`crate::num`] for the decode algorithm). Matches
/// bincode2's `standard()` default config.
#[derive(Debug)]
pub struct Varint;

impl IntEncoding for Fixint {
    const NAME: &'static str = "fixint";
}
impl IntEncoding for Varint {
    const NAME: &'static str = "varint";
}

/// Umbrella config trait bundling both wire-affecting axes. The single
/// generic parameter threaded through every deserializer/entry/accessor
/// type in both families.
///
/// `'static`: every `BincodeConfig` impl is a zero-sized marker type with no
/// lifetime parameters (`Configuration<O, I>` where `O`/`I` are themselves
/// lifetime-free marker types), so this is always trivially satisfied — it
/// exists so that `BincodeClaim<'de, C>` (which must satisfy `Claim: 'de`
/// per the core `Entry`/`Deserializer` traits) doesn't need `C: 'de` spelled
/// out at every one of its many impl sites throughout this crate.
pub trait BincodeConfig: 'static {
    type Order: ByteOrder;
    type Int: IntEncoding;
}

/// A concrete `(ByteOrder, IntEncoding)` pair. Pure marker type — never
/// constructed, only named as a type parameter.
#[derive(Debug)]
pub struct Configuration<O, I>(PhantomData<(O, I)>);

// Manual Clone/Copy (not `#[derive]`): the struct is a zero-sized phantom
// marker, so it should be `Copy` regardless of whether `O`/`I` themselves
// are `Copy` — `#[derive(Copy)]` would incorrectly add `O: Copy, I: Copy`
// bounds.
impl<O, I> Clone for Configuration<O, I> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<O, I> Copy for Configuration<O, I> {}

impl<O: ByteOrder + 'static, I: IntEncoding + 'static> BincodeConfig for Configuration<O, I> {
    type Order = O;
    type Int = I;
}

/// bincode2's `standard()` default: little-endian, varint.
pub type Standard = Configuration<strede::LittleEndian, Varint>;
/// bincode 1.x's only mode / bincode2's `legacy()`: little-endian, fixed-width.
pub type Legacy = Configuration<strede::LittleEndian, Fixint>;
/// Big-endian, varint.
pub type BigStandard = Configuration<strede::BigEndian, Varint>;
/// Big-endian, fixed-width.
pub type BigLegacy = Configuration<strede::BigEndian, Fixint>;
