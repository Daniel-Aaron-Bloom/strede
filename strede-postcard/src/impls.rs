//! Format-specific primitive `Deserialize`/`DeserializeOwned` impls for postcard.
//!
//! Postcard numbers are encoded as varints (unsigned) or zigzag varints (signed);
//! floats are fixed 4/8 LE bytes. We decode directly without going through
//! `deserialize_number_chunks`.

use crate::{
    PostcardError,
    chunked::{
        ChunkedPostcardClaim, ChunkedPostcardDeserializer, ChunkedPostcardSubDeserializer,
        ParseNumOwned, read_bytes_exact,
        varint::{read_varint, read_zigzag},
    },
    full::{ParseNum, PostcardClaim, PostcardDeserializer, PostcardSubDeserializer},
    varint::{decode_varint, decode_zigzag},
};
use strede::{
    Buffer, Deserialize, DeserializeOwned, Deserializer, DeserializerOwned, Handle, Probe,
};

// ---------------------------------------------------------------------------
// ParseNum implementations
// ---------------------------------------------------------------------------

macro_rules! impl_parse_uint {
    ($($t:ty),*) => {
        $(impl ParseNum for $t {
            #[inline(always)]
            fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
                let (v, consumed) = decode_varint(src)?;
                match <$t>::try_from(v) {
                    Ok(n) => Ok(Probe::Hit((PostcardClaim { src: &src[consumed..] }, n))),
                    Err(_) => Ok(Probe::Miss),
                }
            }
        })*
    };
}

macro_rules! impl_parse_sint {
    ($($t:ty),*) => {
        $(impl ParseNum for $t {
            #[inline(always)]
            fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
                let (v, consumed) = decode_zigzag(src)?;
                match <$t>::try_from(v) {
                    Ok(n) => Ok(Probe::Hit((PostcardClaim { src: &src[consumed..] }, n))),
                    Err(_) => Ok(Probe::Miss),
                }
            }
        })*
    };
}

impl_parse_uint!(u8, u16, u32, u64);
impl_parse_sint!(i8, i16, i32, i64);

impl ParseNum for u128 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
        // Postcard encodes u128 as two consecutive varints (lo, hi).
        let (lo, c1) = decode_varint(src)?;
        let (hi, c2) = decode_varint(&src[c1..])?;
        let v = (lo as u128) | ((hi as u128) << 64);
        Ok(Probe::Hit((
            PostcardClaim {
                src: &src[c1 + c2..],
            },
            v,
        )))
    }
}

impl ParseNum for i128 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
        // Postcard encodes i128 as zigzag on two consecutive varints.
        let (lo, c1) = decode_varint(src)?;
        let (hi, c2) = decode_varint(&src[c1..])?;
        let raw = (lo as u128) | ((hi as u128) << 64);
        let v = ((raw >> 1) as i128) ^ (-((raw & 1) as i128));
        Ok(Probe::Hit((
            PostcardClaim {
                src: &src[c1 + c2..],
            },
            v,
        )))
    }
}

impl ParseNum for f32 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
        if src.len() < 4 {
            return Err(PostcardError::UnexpectedEnd);
        }
        let bytes: [u8; 4] = src[..4].try_into().unwrap();
        Ok(Probe::Hit((
            PostcardClaim { src: &src[4..] },
            f32::from_le_bytes(bytes),
        )))
    }
}

impl ParseNum for f64 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
        if src.len() < 8 {
            return Err(PostcardError::UnexpectedEnd);
        }
        let bytes: [u8; 8] = src[..8].try_into().unwrap();
        Ok(Probe::Hit((
            PostcardClaim { src: &src[8..] },
            f64::from_le_bytes(bytes),
        )))
    }
}

impl ParseNum for bool {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
        match src.first() {
            None => Err(PostcardError::UnexpectedEnd),
            Some(&0x00) => Ok(Probe::Hit((PostcardClaim { src: &src[1..] }, false))),
            Some(&0x01) => Ok(Probe::Hit((PostcardClaim { src: &src[1..] }, true))),
            Some(_) => Ok(Probe::Miss),
        }
    }
}

impl ParseNum for () {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
        Ok(Probe::Hit((PostcardClaim { src }, ())))
    }
}

impl ParseNum for char {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(PostcardClaim<'_>, Self)>, PostcardError> {
        let (cp, consumed) = decode_varint(src)?;
        match char::from_u32(cp as u32) {
            Some(c) => Ok(Probe::Hit((
                PostcardClaim {
                    src: &src[consumed..],
                },
                c,
            ))),
            None => Ok(Probe::Miss),
        }
    }
}

// ---------------------------------------------------------------------------
// Borrow-family Deserialize impls
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_borrow_one {
    ($de:ty; $($t:ty),+) => {
        $(impl<'de> Deserialize<'de, $de> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(PostcardClaim<'de>, Self)>, PostcardError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_borrow_both {
    ($($t:ty),+) => {
        $(
            impl_deserialize_borrow_one!(PostcardDeserializer<'de>; $t);
            impl_deserialize_borrow_one!(PostcardSubDeserializer<'de>; $t);
        )+
    };
}

impl_deserialize_borrow_both!(
    bool,
    (),
    u8,
    u16,
    u32,
    u64,
    u128,
    i8,
    i16,
    i32,
    i64,
    i128,
    f32,
    f64,
    char
);

// ---------------------------------------------------------------------------
// ParseNumOwned implementations — async counterparts to `ParseNum` above,
// built on the resumable `chunked::varint` readers.
// ---------------------------------------------------------------------------

macro_rules! impl_parse_num_owned_uint {
    ($($t:ty),*) => {
        $(impl ParseNumOwned for $t {
            #[inline(always)]
            async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
                handle: Handle<'s, B, F>,
                mut offset: usize,
            ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
                let (handle, v) = read_varint(handle, &mut offset).await?;
                match <$t>::try_from(v) {
                    Ok(n) => Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, n))),
                    Err(_) => Ok(Probe::Miss),
                }
            }
        })*
    };
}

macro_rules! impl_parse_num_owned_sint {
    ($($t:ty),*) => {
        $(impl ParseNumOwned for $t {
            #[inline(always)]
            async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
                handle: Handle<'s, B, F>,
                mut offset: usize,
            ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
                let (handle, v) = read_zigzag(handle, &mut offset).await?;
                match <$t>::try_from(v) {
                    Ok(n) => Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, n))),
                    Err(_) => Ok(Probe::Miss),
                }
            }
        })*
    };
}

impl_parse_num_owned_uint!(u8, u16, u32, u64);
impl_parse_num_owned_sint!(i8, i16, i32, i64);

impl ParseNumOwned for u128 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
        // Postcard encodes u128 as two consecutive varints (lo, hi).
        let (handle, lo) = read_varint(handle, &mut offset).await?;
        let (handle, hi) = read_varint(handle, &mut offset).await?;
        let v = (lo as u128) | ((hi as u128) << 64);
        Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, v)))
    }
}

impl ParseNumOwned for i128 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
        // Postcard encodes i128 as zigzag on two consecutive varints.
        let (handle, lo) = read_varint(handle, &mut offset).await?;
        let (handle, hi) = read_varint(handle, &mut offset).await?;
        let raw = (lo as u128) | ((hi as u128) << 64);
        let v = ((raw >> 1) as i128) ^ (-((raw & 1) as i128));
        Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, v)))
    }
}

impl ParseNumOwned for f32 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
        let (handle, bytes) = read_bytes_exact::<_, _, 4>(handle, &mut offset).await?;
        Ok(Probe::Hit((
            ChunkedPostcardClaim { handle, offset },
            f32::from_le_bytes(bytes),
        )))
    }
}

impl ParseNumOwned for f64 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
        let (handle, bytes) = read_bytes_exact::<_, _, 8>(handle, &mut offset).await?;
        Ok(Probe::Hit((
            ChunkedPostcardClaim { handle, offset },
            f64::from_le_bytes(bytes),
        )))
    }
}

impl ParseNumOwned for bool {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
        let (handle, [tag]) = read_bytes_exact::<_, _, 1>(handle, &mut offset).await?;
        match tag {
            0x00 => Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, false))),
            0x01 => Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, true))),
            _ => Ok(Probe::Miss),
        }
    }
}

impl ParseNumOwned for () {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
        Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, ())))
    }
}

impl ParseNumOwned for char {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedPostcardClaim<'s, B, F>, Self)>, PostcardError> {
        let (handle, cp) = read_varint(handle, &mut offset).await?;
        match char::from_u32(cp as u32) {
            Some(c) => Ok(Probe::Hit((ChunkedPostcardClaim { handle, offset }, c))),
            None => Ok(Probe::Miss),
        }
    }
}

// ---------------------------------------------------------------------------
// Owned-family DeserializeOwned impls
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_owned_one {
    ($de:ty; $($t:ty),+) => {
        $(impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, PostcardError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_owned_both {
    ($($t:ty),+) => {
        $(
            impl_deserialize_owned_one!(ChunkedPostcardDeserializer<'s, B, F>; $t);
            impl_deserialize_owned_one!(ChunkedPostcardSubDeserializer<'s, B, F>; $t);
        )+
    };
}

impl_deserialize_owned_both!(
    bool,
    (),
    u8,
    u16,
    u32,
    u64,
    u128,
    i8,
    i16,
    i32,
    i64,
    i128,
    f32,
    f64,
    char
);
