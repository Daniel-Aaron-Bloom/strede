//! Format-specific primitive `Deserialize`/`DeserializeOwned` impls for
//! bincode.
//!
//! Each `ParseNum<C>`/`ParseNumOwned<C>` impl is written once, generic over
//! `C: BincodeConfig`, delegating to `crate::num`/`crate::chunked::num`'s
//! config-generic decode functions — the compiler monomorphizes one body
//! per concrete `C` rather than four hand-duplicated code paths.

use crate::{
    BincodeError,
    chunked::{
        ChunkedBincodeClaim, ChunkedBincodeDeserializer, ChunkedBincodeSubDeserializer,
        ParseNumOwned, num as num_owned,
    },
    config::BincodeConfig,
    full::{BincodeClaim, BincodeDeserializer, BincodeSubDeserializer, ParseNum},
    num,
};
use strede::{Buffer, Deserialize, DeserializeOwned, Deserializer, DeserializerOwned, Probe};

// ---------------------------------------------------------------------------
// ParseNum implementations (borrow family)
// ---------------------------------------------------------------------------

// Not written as a `path`-fragment macro: a macro-captured `path` fragment
// can't be directly followed by `::<...>` turbofish at expansion (a known
// `macro_rules!` restriction), so each width gets its own small impl instead.

impl<C: BincodeConfig> ParseNum<C> for u16 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let Some((v, consumed)) = num::decode_u16::<C>(src)? else {
            return Ok(Probe::Miss);
        };
        match u16::try_from(v) {
            Ok(n) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNum<C> for u32 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let Some((v, consumed)) = num::decode_u32::<C>(src)? else {
            return Ok(Probe::Miss);
        };
        match u32::try_from(v) {
            Ok(n) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNum<C> for u64 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let Some((v, consumed)) = num::decode_u64::<C>(src)? else {
            return Ok(Probe::Miss);
        };
        match u64::try_from(v) {
            Ok(n) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNum<C> for i16 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let Some((v, consumed)) = num::decode_i16::<C>(src)? else {
            return Ok(Probe::Miss);
        };
        match i16::try_from(v) {
            Ok(n) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNum<C> for i32 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let Some((v, consumed)) = num::decode_i32::<C>(src)? else {
            return Ok(Probe::Miss);
        };
        match i32::try_from(v) {
            Ok(n) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNum<C> for i64 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let Some((v, consumed)) = num::decode_i64::<C>(src)? else {
            return Ok(Probe::Miss);
        };
        match i64::try_from(v) {
            Ok(n) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNum<C> for u8 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let (v, consumed) = num::decode_u8(src)?;
        Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), v)))
    }
}

impl<C: BincodeConfig> ParseNum<C> for i8 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let (v, consumed) = num::decode_i8(src)?;
        Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), v)))
    }
}

impl<C: BincodeConfig> ParseNum<C> for u128 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let (v, consumed) = num::decode_u128::<C>(src)?;
        Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), v)))
    }
}

impl<C: BincodeConfig> ParseNum<C> for i128 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let (v, consumed) = num::decode_i128::<C>(src)?;
        Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), v)))
    }
}

impl<C: BincodeConfig> ParseNum<C> for f32 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let (v, consumed) = num::decode_f32::<C>(src)?;
        Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), v)))
    }
}

impl<C: BincodeConfig> ParseNum<C> for f64 {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        let (v, consumed) = num::decode_f64::<C>(src)?;
        Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), v)))
    }
}

impl<C: BincodeConfig> ParseNum<C> for bool {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        match num::decode_bool(src)? {
            Some((v, consumed)) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), v))),
            None => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNum<C> for () {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        Ok(Probe::Hit((BincodeClaim::new(src), ())))
    }
}

impl<C: BincodeConfig> ParseNum<C> for char {
    #[inline(always)]
    fn parse(src: &[u8]) -> Result<Probe<(BincodeClaim<'_, C>, Self)>, BincodeError> {
        match num::decode_char(src)? {
            Some((c, consumed)) => Ok(Probe::Hit((BincodeClaim::new(&src[consumed..]), c))),
            None => Ok(Probe::Miss),
        }
    }
}

// ---------------------------------------------------------------------------
// Borrow-family Deserialize impls
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_borrow_one {
    ($de:ty; $($t:ty),+) => {
        $(impl<'de, C: BincodeConfig> Deserialize<'de, $de> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize(
                d: $de,
                _: (),
            ) -> Result<Probe<(BincodeClaim<'de, C>, Self)>, BincodeError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_borrow_both {
    ($($t:ty),+) => {
        $(
            impl_deserialize_borrow_one!(BincodeDeserializer<'de, C>; $t);
            impl_deserialize_borrow_one!(BincodeSubDeserializer<'de, C>; $t);
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
// built on the resumable `chunked::num` readers.
// ---------------------------------------------------------------------------

// See the borrow-family note above on why these aren't a `path`-fragment macro.

impl<C: BincodeConfig> ParseNumOwned<C> for u16 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_u16::<_, _, C>(handle, &mut offset).await?;
        let Some(v) = v else {
            return Ok(Probe::Miss);
        };
        match u16::try_from(v) {
            Ok(n) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for u32 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_u32::<_, _, C>(handle, &mut offset).await?;
        let Some(v) = v else {
            return Ok(Probe::Miss);
        };
        match u32::try_from(v) {
            Ok(n) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for u64 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_u64::<_, _, C>(handle, &mut offset).await?;
        let Some(v) = v else {
            return Ok(Probe::Miss);
        };
        match u64::try_from(v) {
            Ok(n) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for i16 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_i16::<_, _, C>(handle, &mut offset).await?;
        let Some(v) = v else {
            return Ok(Probe::Miss);
        };
        match i16::try_from(v) {
            Ok(n) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for i32 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_i32::<_, _, C>(handle, &mut offset).await?;
        let Some(v) = v else {
            return Ok(Probe::Miss);
        };
        match i32::try_from(v) {
            Ok(n) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for i64 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_i64::<_, _, C>(handle, &mut offset).await?;
        let Some(v) = v else {
            return Ok(Probe::Miss);
        };
        match i64::try_from(v) {
            Ok(n) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), n))),
            Err(_) => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for u8 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_u8(handle, &mut offset).await?;
        Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), v)))
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for i8 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_i8(handle, &mut offset).await?;
        Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), v)))
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for u128 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_u128::<_, _, C>(handle, &mut offset).await?;
        Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), v)))
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for i128 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_i128::<_, _, C>(handle, &mut offset).await?;
        Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), v)))
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for f32 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_f32::<_, _, C>(handle, &mut offset).await?;
        Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), v)))
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for f64 {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_f64::<_, _, C>(handle, &mut offset).await?;
        Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), v)))
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for bool {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_bool(handle, &mut offset).await?;
        match v {
            Some(v) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), v))),
            None => Ok(Probe::Miss),
        }
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for () {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), ())))
    }
}

impl<C: BincodeConfig> ParseNumOwned<C> for char {
    #[inline(always)]
    async fn parse_owned<'s, B: Buffer, F: AsyncFnMut(&mut B)>(
        handle: strede::Handle<'s, B, F>,
        mut offset: usize,
    ) -> Result<Probe<(ChunkedBincodeClaim<'s, C, B, F>, Self)>, BincodeError> {
        let (handle, v) = num_owned::decode_char(handle, &mut offset).await?;
        match v {
            Some(c) => Ok(Probe::Hit((ChunkedBincodeClaim::new(handle, offset), c))),
            None => Ok(Probe::Miss),
        }
    }
}

// ---------------------------------------------------------------------------
// Owned-family DeserializeOwned impls
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_owned_one {
    ($de:ty; $($t:ty),+) => {
        $(impl<'s, C: BincodeConfig, B: Buffer, F: AsyncFnMut(&mut B)> DeserializeOwned<$de> for $t {
            type Extra = ();
            #[inline(always)]
            async fn deserialize_owned(
                d: $de,
                _: (),
            ) -> Result<Probe<(<$de as DeserializerOwned>::Claim, Self)>, BincodeError> {
                d.entry(|[e]| async move { e.parse_num::<$t>().await }).await
            }
        })+
    };
}

macro_rules! impl_deserialize_owned_both {
    ($($t:ty),+) => {
        $(
            impl_deserialize_owned_one!(ChunkedBincodeDeserializer<'s, C, B, F>; $t);
            impl_deserialize_owned_one!(ChunkedBincodeSubDeserializer<'s, C, B, F>; $t);
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
