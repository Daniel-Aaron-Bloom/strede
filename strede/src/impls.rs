//! Blanket `Deserialize` / `DeserializeOwned` implementations for standard
//! library types beyond the primitives defined in `borrow.rs` and `owned.rs`.

use core::cell::{Cell, RefCell};
use core::cmp::Reverse;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::num::{
    NonZeroI8, NonZeroI16, NonZeroI32, NonZeroI64, NonZeroI128, NonZeroIsize, NonZeroU8,
    NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU128, NonZeroUsize, Saturating, Wrapping,
};

use strede_derive::{Deserialize, DeserializeOwned};

use crate::borrow::{BytesAccess, Deserialize, Deserializer, Entry, SeqAccess, SeqEntry};
use crate::owned::{
    BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, SeqAccessOwned,
    SeqEntryOwned, StrAccessOwned,
};
use crate::{Chunk, DeserializeError, Probe, StrAccess, hit, or_miss, select_probe};

// ===========================================================================
// Helper: ArrayGuard — drop-safe partial array initialization
// ===========================================================================

struct ArrayGuard<T, const N: usize> {
    arr: [MaybeUninit<T>; N],
    len: usize,
}

impl<T, const N: usize> ArrayGuard<T, N> {
    fn new() -> Self {
        Self {
            arr: [const { MaybeUninit::uninit() }; N],
            len: 0,
        }
    }

    fn push(&mut self, val: T) {
        debug_assert!(self.len < N);
        self.arr[self.len].write(val);
        self.len += 1;
    }

    /// # Safety
    /// All N elements must have been pushed.
    unsafe fn into_array(self) -> [T; N] {
        debug_assert_eq!(self.len, N);
        let me = core::mem::ManuallyDrop::new(self);
        // Safety: all N elements initialized, ManuallyDrop prevents double-drop
        unsafe { (&me.arr as *const [MaybeUninit<T>; N] as *const [T; N]).read() }
    }
}

impl<T, const N: usize> Drop for ArrayGuard<T, N> {
    fn drop(&mut self) {
        for i in 0..self.len {
            // Safety: elements 0..len have been initialized via push()
            unsafe {
                self.arr[i].assume_init_drop();
            }
        }
    }
}

// ===========================================================================
// Helper: StackStr — fixed-capacity stack-allocated string buffer
// ===========================================================================

struct StackStr<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> StackStr<N> {
    fn new() -> Self {
        Self {
            buf: [0; N],
            len: 0,
        }
    }

    /// Returns `false` if the buffer would overflow.
    fn push_str(&mut self, s: &str) -> bool {
        let bytes = s.as_bytes();
        let end = self.len + bytes.len();
        if end > N {
            return false;
        }
        self.buf[self.len..end].copy_from_slice(bytes);
        self.len = end;
        true
    }

    fn as_str(&self) -> &str {
        // Safety: only valid UTF-8 is ever pushed via push_str
        unsafe { core::str::from_utf8_unchecked(&self.buf[..self.len]) }
    }
}

// ===========================================================================
// Core impls — no features required
// ===========================================================================

// --- Unit ---

impl<'de> Deserialize<'de> for () {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, ())))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for () {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, ())))
        })
        .await
    }
}

// --- usize / isize ---

impl<'de> Deserialize<'de> for usize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_u32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_u64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as usize)))
        })
        .await
    }
}

impl<'de> Deserialize<'de> for isize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_i32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_i64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as isize)))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for usize {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_u32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_u64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as usize)))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for isize {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            #[cfg(target_pointer_width = "32")]
            let res = e.deserialize_i32().await;
            #[cfg(target_pointer_width = "64")]
            let res = e.deserialize_i64().await;
            let (claim, v) = hit!(res);
            Ok(Probe::Hit((claim, v as isize)))
        })
        .await
    }
}

// --- PhantomData ---

impl<'de, T: ?Sized> Deserialize<'de> for PhantomData<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, PhantomData)))
        })
        .await
    }
}

impl<'s, T: ?Sized> DeserializeOwned<'s> for PhantomData<T> {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = hit!(e.deserialize_null().await);
            Ok(Probe::Hit((claim, PhantomData)))
        })
        .await
    }
}

// --- Skip ---

/// Deserializes by discarding the current token unconditionally.
///
/// Always returns `Probe::Hit(Skip)` on well-formed input — it never misses,
/// since [`Entry::skip`](crate::Entry::skip) accepts any token type.
pub struct Skip;

impl<'de> Deserialize<'de> for Skip {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = e.skip().await?;
            Ok(Probe::Hit((claim, Skip)))
        })
        .await
    }
}

impl<'s> DeserializeOwned<'s> for Skip {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let claim = e.skip().await?;
            Ok(Probe::Hit((claim, Skip)))
        })
        .await
    }
}

// --- Match ---

/// Deserializes a string or byte token and checks it for an exact match
/// against a caller-supplied value passed as `Extra`.
///
/// Returns `Probe::Hit(Match)` if the token's content equals `extra`,
/// `Probe::Miss` if the token is the wrong type or the content differs.
///
/// Useful for discriminated dispatch in `select_probe!`:
/// ```rust,ignore
/// d.entry(|[e1, e2]| select_probe! {
///     e1.deserialize_value::<Match, &str>("ok"),
///     e2.deserialize_value::<Match, &str>("err"),
///     miss => Ok(Probe::Miss),
/// })
/// ```
pub struct Match;

impl<'de, 'a> Deserialize<'de, &'a str> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: &'a str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        // Fast path: zero-copy. Compare inside closure so a mismatch returns
        // Miss without advancing the stream.
        let string_miss = Cell::new(false);
        let string_miss = &string_miss;
        d.entry(async |[e1, e2]| {
            select_probe!(
                async move {
                    let (claim, s) = hit!(e1.deserialize_str().await);
                    if s == extra {
                        return Ok(Probe::Hit((claim, Match)));
                    }
                    string_miss.set(true);
                    Ok(Probe::Miss)
                },
                async move {
                    let mut acc = hit!(e2.deserialize_str_chunks().await);
                    let mut remaining = extra;
                    let mut matched = true;
                    let claim = loop {
                        match acc
                            .next_str(|chunk| {
                                if matched {
                                    if remaining.starts_with(chunk) {
                                        remaining = &remaining[chunk.len()..];
                                    } else {
                                        matched = false;
                                    }
                                }
                            })
                            .await?
                        {
                            Chunk::Data((new_acc, ())) => {
                                if string_miss.get() {
                                    matched = false;
                                }
                                acc = new_acc;
                            }
                            Chunk::Done(claim) => break claim,
                        }
                    };
                    Ok(if matched && remaining.is_empty() {
                        Probe::Hit((claim, Match))
                    } else {
                        Probe::Miss
                    })
                }
            )
        })
        .await
    }
}

impl<'de, 'a> Deserialize<'de, &'a [u8]> for Match {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: &'a [u8],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let bytes_miss = Cell::new(false);
        let bytes_miss = &bytes_miss;
        d.entry(async |[e1, e2]| {
            select_probe!(
                async move {
                    let (claim, b) = hit!(e1.deserialize_bytes().await);
                    if b == extra {
                        return Ok(Probe::Hit((claim, Match)));
                    }
                    bytes_miss.set(true);
                    Ok(Probe::Miss)
                },
                async move {
                    let mut acc = hit!(e2.deserialize_bytes_chunks().await);
                    let mut remaining = extra;
                    let mut matched = true;
                    let claim = loop {
                        match acc
                            .next_bytes(|chunk| {
                                if matched {
                                    if remaining.starts_with(chunk) {
                                        remaining = &remaining[chunk.len()..];
                                    } else {
                                        matched = false;
                                    }
                                }
                            })
                            .await?
                        {
                            Chunk::Data((new_acc, ())) => {
                                if bytes_miss.get() {
                                    matched = false;
                                }
                                acc = new_acc;
                            }
                            Chunk::Done(claim) => break claim,
                        }
                    };
                    Ok(if matched && remaining.is_empty() {
                        Probe::Hit((claim, Match))
                    } else {
                        Probe::Miss
                    })
                }
            )
        })
        .await
    }
}

impl<'s, 'a> DeserializeOwned<'s, &'a str> for Match {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        extra: &'a str,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let mut acc = hit!(e.deserialize_str_chunks().await);
            let mut remaining = extra;
            let mut matched = true;
            let claim = loop {
                match acc
                    .next_str(|chunk| {
                        if matched {
                            if remaining.starts_with(chunk) {
                                remaining = &remaining[chunk.len()..];
                            } else {
                                matched = false;
                            }
                        }
                    })
                    .await?
                {
                    Chunk::Data((new_acc, ())) => acc = new_acc,
                    Chunk::Done(claim) => break claim,
                }
            };
            Ok(if matched && remaining.is_empty() {
                Probe::Hit((claim, Match))
            } else {
                Probe::Miss
            })
        })
        .await
    }
}

impl<'s, 'a> DeserializeOwned<'s, &'a [u8]> for Match {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        extra: &'a [u8],
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let mut acc = hit!(e.deserialize_bytes_chunks().await);
            let mut remaining = extra;
            let mut matched = true;
            let claim = loop {
                match acc
                    .next_bytes(|chunk| {
                        if matched {
                            if remaining.starts_with(chunk) {
                                remaining = &remaining[chunk.len()..];
                            } else {
                                matched = false;
                            }
                        }
                    })
                    .await?
                {
                    Chunk::Data((new_acc, ())) => acc = new_acc,
                    Chunk::Done(claim) => break claim,
                }
            };
            Ok(if matched && remaining.is_empty() {
                Probe::Hit((claim, Match))
            } else {
                Probe::Miss
            })
        })
        .await
    }
}

// --- Newtype wrappers (delegate to inner) ---

macro_rules! impl_newtype_wrapper {
    ($wrapper:ident) => {
        impl<'de, T: Deserialize<'de>> Deserialize<'de> for $wrapper<T> {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, $wrapper(v))))
            }
        }

        impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for $wrapper<T> {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, $wrapper(v))))
            }
        }
    };
}

impl_newtype_wrapper!(Wrapping);
impl_newtype_wrapper!(Saturating);
impl_newtype_wrapper!(Reverse);

// --- Cell<T> ---

impl<'de, T: Deserialize<'de> + Copy> Deserialize<'de> for Cell<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, Cell::new(v))))
    }
}

impl<'s, T: DeserializeOwned<'s> + Copy> DeserializeOwned<'s> for Cell<T> {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, Cell::new(v))))
    }
}

// --- RefCell<T> ---

impl<'de, T: Deserialize<'de>> Deserialize<'de> for RefCell<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize(d, ())
            .await?
            .map(|(c, v)| (c, RefCell::new(v))))
    }
}

impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for RefCell<T> {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        Ok(T::deserialize(d, ())
            .await?
            .map(|(c, v)| (c, RefCell::new(v))))
    }
}

// --- NonZero types ---

macro_rules! impl_nonzero {
    ($nonzero:ty, $method:ident) => {
        impl<'de> Deserialize<'de> for $nonzero {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let (claim, v) = hit!(e.$method().await);
                    let nz = or_miss!(<$nonzero>::new(v));
                    Ok(Probe::Hit((claim, nz)))
                })
                .await
            }
        }

        impl<'s> DeserializeOwned<'s> for $nonzero {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let (claim, v) = hit!(e.$method().await);
                    let nz = or_miss!(<$nonzero>::new(v));
                    Ok(Probe::Hit((claim, nz)))
                })
                .await
            }
        }
    };
}

impl_nonzero!(NonZeroU8, deserialize_u8);
impl_nonzero!(NonZeroU16, deserialize_u16);
impl_nonzero!(NonZeroU32, deserialize_u32);
impl_nonzero!(NonZeroU64, deserialize_u64);
impl_nonzero!(NonZeroU128, deserialize_u128);
impl_nonzero!(NonZeroI8, deserialize_i8);
impl_nonzero!(NonZeroI16, deserialize_i16);
impl_nonzero!(NonZeroI32, deserialize_i32);
impl_nonzero!(NonZeroI64, deserialize_i64);
impl_nonzero!(NonZeroI128, deserialize_i128);

// NonZeroUsize / NonZeroIsize — delegate to usize/isize

impl<'de> Deserialize<'de> for NonZeroUsize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<usize as Deserialize>::deserialize(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

impl<'s> DeserializeOwned<'s> for NonZeroUsize {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<usize as DeserializeOwned>::deserialize(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

impl<'de> Deserialize<'de> for NonZeroIsize {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<isize as Deserialize>::deserialize(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

impl<'s> DeserializeOwned<'s> for NonZeroIsize {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, v) = hit!(<isize as DeserializeOwned>::deserialize(d, ()).await);
        let nz = or_miss!(Self::new(v));
        Ok(Probe::Hit((claim, nz)))
    }
}

// --- Arrays [T; N] (const generic) ---

impl<'de, T: Deserialize<'de>, const N: usize> Deserialize<'de> for [T; N] {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            let mut seq = hit!(e.deserialize_seq().await);
            let mut guard = ArrayGuard::<T, N>::new();
            for _ in 0..N {
                let v = hit!(seq.next(|[elem]| elem.get::<T, _>(())).await);
                let (n_seq, v) = or_miss!(v.data());
                seq = n_seq;
                guard.push(v);
            }
            let v = hit!(seq.next(|[elem]| { elem.get::<T, _>(()) }).await);
            let claim = or_miss!(v.done());
            Ok(Probe::Hit((claim, unsafe { guard.into_array() })))
        })
        .await
    }
}

impl<'s, T: DeserializeOwned<'s>, const N: usize> DeserializeOwned<'s> for [T; N] {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        d.entry(|[e]| async {
            let mut seq = hit!(e.deserialize_seq().await);
            let mut guard = ArrayGuard::<T, N>::new();
            for _ in 0..N {
                let v = hit!(seq.next(|[elem]| { elem.get::<T, _>(()) }).await);
                let (s, v) = or_miss!(v.data());
                guard.push(v);
                seq = s;
            }
            let v = hit!(
                seq.next::<1, _, _, T>(|[elem]| { elem.get::<T, _>(()) })
                    .await
            );
            let claim = or_miss!(v.done());
            Ok(Probe::Hit((claim, unsafe { guard.into_array() })))
        })
        .await
    }
}

// --- Tuples up to 16 ---

macro_rules! impl_tuple {
    ($($T:ident $v:ident),+ $(,)?) => {
        impl<'de, $($T: Deserialize<'de>),+> Deserialize<'de> for ($($T,)+) {
            async fn deserialize<D: Deserializer<'de>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let seq = hit!(e.deserialize_seq().await);
                    $(
                        let v = hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await);
                        let (seq, $v) = or_miss!(v.data());
                    )+
                    let v = hit!(seq.next(|[elem]| async {
                        elem.get::<impl_tuple!(@first $($T),+), _>(()).await
                    }).await);
                    let claim = or_miss!(v.done());
                    Ok(Probe::Hit((claim, ($($v,)+))))
                }).await
            }
        }

        impl<'s, $($T: DeserializeOwned<'s>),+> DeserializeOwned<'s> for ($($T,)+) {
            async fn deserialize<D: DeserializerOwned<'s>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let seq = hit!(e.deserialize_seq().await);
                    $(
                        let v = hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await);
                        let (seq, $v) = or_miss!(v.data());
                    )+
                    let v = hit!(seq.next::<1, _, _, impl_tuple!(@first $($T),+)>(|[elem]| async {
                        elem.get::<impl_tuple!(@first $($T),+), _>(()).await
                    }).await);
                    let claim = or_miss!(v.done());
                    Ok(Probe::Hit((claim, ($($v,)+))))
                }).await
            }
        }
    };

    (@first $first:ident $(, $rest:ident)*) => { $first };
}

impl_tuple!(T0 v0);
impl_tuple!(T0 v0, T1 v1);
impl_tuple!(T0 v0, T1 v1, T2 v2);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14);
impl_tuple!(T0 v0, T1 v1, T2 v2, T3 v3, T4 v4, T5 v5, T6 v6, T7 v7, T8 v8, T9 v9, T10 v10, T11 v11, T12 v12, T13 v13, T14 v14, T15 v15);

// ===========================================================================
// IP address and socket address types — string-parsed
// ===========================================================================

macro_rules! impl_from_str {
    ($ty:ty, $max_len:expr) => {
        impl<'de> Deserialize<'de> for $ty {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e1, e2]| async {
                    select_probe! {
                        async move {
                            let (claim, s) = hit!(e1.deserialize_str().await);
                            let v = or_miss!(s.parse::<$ty>().ok());
                            Ok(Probe::Hit((claim, v)))
                        },
                        async move {
                            let mut chunks = hit!(e2.deserialize_str_chunks().await);
                            let mut buf = StackStr::<$max_len>::new();
                            let mut overflow = false;
                            let claim = loop {
                                match chunks.next_str(|s| {
                                    if !overflow && !buf.push_str(s) {
                                        overflow = true;
                                    }
                                }).await? {
                                    Chunk::Data((new, ())) => chunks = new,
                                    Chunk::Done(claim) => break claim,
                                }
                            };
                            if overflow {
                                return Ok(Probe::Miss);
                            }
                            let v = or_miss!(buf.as_str().parse::<$ty>().ok());
                            Ok(Probe::Hit((claim, v)))
                        },
                        miss => Ok(Probe::Miss),
                    }
                })
                .await
            }
        }

        impl<'s> DeserializeOwned<'s> for $ty {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error>
            where
                D::Error: DeserializeError,
            {
                d.entry(|[e]| async {
                    let mut chunks = hit!(e.deserialize_str_chunks().await);
                    let mut buf = StackStr::<$max_len>::new();
                    let mut overflow = false;
                    let claim = loop {
                        match chunks
                            .next_str(|s| {
                                if !overflow {
                                    if !buf.push_str(s) {
                                        overflow = true;
                                    }
                                }
                            })
                            .await?
                        {
                            Chunk::Data((c, ())) => chunks = c,
                            Chunk::Done(claim) => break claim,
                        }
                    };
                    if overflow {
                        return Ok(Probe::Miss);
                    }
                    let v = or_miss!(buf.as_str().parse::<$ty>().ok());
                    Ok(Probe::Hit((claim, v)))
                })
                .await
            }
        }
    };
}

impl_from_str!(core::net::IpAddr, 45);
impl_from_str!(core::net::Ipv4Addr, 15);
impl_from_str!(core::net::Ipv6Addr, 45);
impl_from_str!(core::net::SocketAddr, 51);
impl_from_str!(core::net::SocketAddrV4, 21);
impl_from_str!(core::net::SocketAddrV6, 51);

// --- Duration ---

#[derive(DeserializeOwned, Deserialize)]
#[strede(crate = "crate")]
struct Duration {
    secs: u64,
    nanos: u32,
}

impl<'de> Deserialize<'de> for core::time::Duration {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, Duration { secs, nanos }) =
            hit!(<Duration as Deserialize>::deserialize(d, ()).await);
        Ok(Probe::Hit((claim, Self::new(secs, nanos))))
    }
}

impl<'s> DeserializeOwned<'s> for core::time::Duration {
    async fn deserialize<D: DeserializerOwned<'s>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>
    where
        D::Error: DeserializeError,
    {
        let (claim, Duration { secs, nanos }) =
            hit!(<Duration as DeserializeOwned>::deserialize(d, ()).await);
        Ok(Probe::Hit((claim, Self::new(secs, nanos))))
    }
}

// ===========================================================================
// alloc impls
// ===========================================================================

#[cfg(feature = "alloc")]
mod alloc_impls {
    use crate::borrow::{
        Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry, SeqAccess,
        SeqEntry,
    };
    use crate::owned::{
        BytesAccessOwned, DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned,
        MapKeyEntryOwned, MapValueEntryOwned, SeqAccessOwned, SeqEntryOwned, StrAccessOwned,
    };
    use crate::{BytesAccess, Chunk, DeserializeError, Probe, StrAccess, hit, or_miss};

    extern crate alloc;
    use alloc::{
        borrow::{Cow, ToOwned},
        boxed::Box,
        collections::{BTreeMap, BTreeSet, BinaryHeap, LinkedList, VecDeque},
        ffi::CString,
        string::String,
        vec::Vec,
    };

    // --- String ---

    impl<'de> Deserialize<'de> for String {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_str_chunks().await);
                let mut out = String::new();
                let claim = loop {
                    match chunks.next_str(|s| out.push_str(s)).await? {
                        Chunk::Data((new, ())) => chunks = new,
                        Chunk::Done(claim) => break claim,
                    }
                };
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    impl<'s> DeserializeOwned<'s> for String {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_str_chunks().await);
                let mut out = String::new();
                let claim = loop {
                    match chunks
                        .next_str(|s| {
                            out.push_str(s);
                        })
                        .await?
                    {
                        Chunk::Data((c, ())) => chunks = c,
                        Chunk::Done(claim) => break claim,
                    }
                };
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    // --- Vec<T> ---

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for Vec<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = Vec::new();
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((n_seq, v)) => {
                            out.push(v);
                            seq = n_seq;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for Vec<T> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = Vec::new();
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((s, v)) => {
                            out.push(v);
                            seq = s;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- Box<T>, Box<str>, Box<[T]> ---

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for Box<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, Box::new(v))))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for Box<T> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ()).await?.map(|(c, v)| (c, Box::new(v))))
        }
    }

    impl<'de> Deserialize<'de> for Box<str> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_str())))
        }
    }

    impl<'s> DeserializeOwned<'s> for Box<str> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as DeserializeOwned>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_str())))
        }
    }

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for Box<[T]> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            Ok(Vec::<T>::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, v.into_boxed_slice())))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for Box<[T]> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            Ok(Vec::<T>::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, v.into_boxed_slice())))
        }
    }

    // --- Sequence-collected types (BTreeSet, BinaryHeap, LinkedList, VecDeque) ---

    macro_rules! impl_seq_collect {
        ($ty:ident < $T:ident $(: $($bound:path),+)? >, $new:expr, $add:ident) => {
            impl<'de, $T: Deserialize<'de> $($(+ $bound)+)?> Deserialize<'de> for $ty<$T> {
                async fn deserialize<D: Deserializer<'de>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
                where
                    D::Error: DeserializeError,
                {
                    d.entry(|[e]| async {
                        let mut seq = hit!(e.deserialize_seq().await);
                        let mut out: $ty<$T> = $new;
                        loop {
                            match hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await) {
                                Chunk::Data((n_seq, v)) => {
                                    out.$add(v);
                                    seq = n_seq;
                                }
                                Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                            }
                        }
                    }).await
                }
            }

            impl<'s, $T: DeserializeOwned<'s> $($(+ $bound)+)?> DeserializeOwned<'s> for $ty<$T> {
                async fn deserialize<D: DeserializerOwned<'s>>(d: D, _extra: ()) -> Result<Probe<(D::Claim, Self)>, D::Error>
                where
                    D::Error: DeserializeError,
                {
                    d.entry(|[e]| async {
                        let mut seq = hit!(e.deserialize_seq().await);
                        let mut out: $ty<$T> = $new;
                        loop {
                            match hit!(seq.next(|[elem]| async { elem.get::<$T, _>(()).await }).await) {
                                Chunk::Data((s, v)) => { out.$add(v); seq = s; }
                                Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                            }
                        }
                    }).await
                }
            }
        };
    }

    impl_seq_collect!(BTreeSet<T: Eq, Ord>, BTreeSet::new(), insert);
    impl_seq_collect!(BinaryHeap<T: Ord>, BinaryHeap::new(), push);
    impl_seq_collect!(LinkedList<T>, LinkedList::new(), push_back);
    impl_seq_collect!(VecDeque<T>, VecDeque::new(), push_back);

    // --- BTreeMap<K, V> ---

    impl<'de, K: Deserialize<'de> + Ord, V: Deserialize<'de>> Deserialize<'de> for BTreeMap<K, V> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = BTreeMap::new();
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) =
                                hit!(ke.key((), |_k, [ve]| ve.value::<V, _>(())).await);
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, K: DeserializeOwned<'s> + Ord, V: DeserializeOwned<'s>> DeserializeOwned<'s>
        for BTreeMap<K, V>
    {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = BTreeMap::new();
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) =
                                hit!(ke.key((), |_k, [ve]| ve.value::<V, _>(())).await);
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- Cow<'a, T> (borrow: try zero-copy first, then Owned) ---

    impl<'de: 'a, 'a, T: ?Sized + ToOwned> Deserialize<'de> for Cow<'a, T>
    where
        &'a T: Deserialize<'de>,
        T::Owned: Deserialize<'de>,
    {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e1, e2]| async {
                if let Probe::Hit((claim, t)) = e1.deserialize_value::<&'a T, ()>(()).await? {
                    return Ok(Probe::Hit((claim, Cow::Borrowed(t))));
                }
                match e2.deserialize_value::<T::Owned, ()>(()).await? {
                    Probe::Hit((claim, owned)) => Ok(Probe::Hit((claim, Cow::Owned(owned)))),
                    Probe::Miss => Ok(Probe::Miss),
                }
            })
            .await
        }
    }

    // --- Cow<'a, T> (owned family: always Cow::Owned) ---

    impl<'s, 'a, T: ?Sized + ToOwned> DeserializeOwned<'s> for Cow<'a, T>
    where
        T::Owned: DeserializeOwned<'s>,
    {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::Owned::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, Cow::Owned(v))))
        }
    }

    // --- CString ---

    impl<'de> Deserialize<'de> for CString {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_bytes_chunks().await);
                let mut out = Vec::new();
                let claim = loop {
                    match chunks.next_bytes(|b| out.extend_from_slice(b)).await? {
                        Chunk::Data((new, ())) => chunks = new,
                        Chunk::Done(claim) => break claim,
                    }
                };
                let cs = or_miss!(CString::new(out).ok());
                Ok(Probe::Hit((claim, cs)))
            })
            .await
        }
    }

    impl<'s> DeserializeOwned<'s> for CString {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut chunks = hit!(e.deserialize_bytes_chunks().await);
                let mut out = Vec::new();
                let claim = loop {
                    match chunks
                        .next_bytes(|b| {
                            out.extend_from_slice(b);
                        })
                        .await?
                    {
                        Chunk::Data((c, ())) => chunks = c,
                        Chunk::Done(claim) => break claim,
                    }
                };
                let cs = or_miss!(CString::new(out).ok());
                Ok(Probe::Hit((claim, cs)))
            })
            .await
        }
    }

    // --- Rc<T>, Arc<T> (behind rc feature) ---

    #[cfg(feature = "rc")]
    mod rc_impls {
        use crate::Probe;
        use crate::borrow::{Deserialize, Deserializer};
        use crate::owned::{DeserializeOwned, DeserializerOwned};
        extern crate alloc;
        use alloc::{boxed::Box, rc::Rc, sync::Arc};

        impl<'de, T: ?Sized> Deserialize<'de> for Rc<T>
        where
            Box<T>: Deserialize<'de>,
        {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Rc::from(b))))
            }
        }

        impl<'s, T: ?Sized> DeserializeOwned<'s> for Rc<T>
        where
            Box<T>: DeserializeOwned<'s>,
        {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Rc::from(b))))
            }
        }

        impl<'de, T: ?Sized> Deserialize<'de> for Arc<T>
        where
            Box<T>: Deserialize<'de>,
        {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Arc::from(b))))
            }
        }

        impl<'s, T: ?Sized> DeserializeOwned<'s> for Arc<T>
        where
            Box<T>: DeserializeOwned<'s>,
        {
            async fn deserialize<D: DeserializerOwned<'s>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(<Box<T>>::deserialize(d, ())
                    .await?
                    .map(|(c, b)| (c, Arc::from(b))))
            }
        }
    }
}

// ===========================================================================
// std impls
// ===========================================================================

#[cfg(feature = "std")]
mod std_impls {
    extern crate std;

    use crate::borrow::{
        Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry, SeqAccess,
        SeqEntry,
    };
    use crate::owned::{
        DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapKeyEntryOwned,
        MapValueEntryOwned, SeqAccessOwned, SeqEntryOwned,
    };
    use crate::{Chunk, DeserializeError, Probe, hit, or_miss};
    use core::hash::{BuildHasher, Hash};
    use std::collections::{HashMap, HashSet};

    extern crate alloc;
    use alloc::{boxed::Box, string::String};

    // --- HashMap<K, V, S> ---

    impl<'de, K, V, S> Deserialize<'de> for HashMap<K, V, S>
    where
        K: Deserialize<'de> + Eq + Hash,
        V: Deserialize<'de>,
        S: BuildHasher + Default,
    {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = HashMap::with_hasher(S::default());
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) = hit!(
                                ke.key((), |_k, [ve]| async { ve.value::<V, _>(()).await })
                                    .await
                            );
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, K, V, S> DeserializeOwned<'s> for HashMap<K, V, S>
    where
        K: DeserializeOwned<'s> + Eq + Hash,
        V: DeserializeOwned<'s>,
        S: BuildHasher + Default,
    {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut out = HashMap::with_hasher(S::default());
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, k, v) = hit!(
                                ke.key((), |_k, [ve]| async { ve.value::<V, _>(()).await })
                                    .await
                            );
                            Ok(Probe::Hit((claim, (k, v))))
                        })
                        .await
                    ) {
                        Chunk::Data((m, (k, v))) => {
                            out.insert(k, v);
                            map = m;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- HashSet<T, S> ---

    impl<'de, T, S> Deserialize<'de> for HashSet<T, S>
    where
        T: Deserialize<'de> + Eq + Hash,
        S: BuildHasher + Default,
    {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = HashSet::with_hasher(S::default());
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((n_seq, v)) => {
                            out.insert(v);
                            seq = n_seq;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    impl<'s, T, S> DeserializeOwned<'s> for HashSet<T, S>
    where
        T: DeserializeOwned<'s> + Eq + Hash,
        S: BuildHasher + Default,
    {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut seq = hit!(e.deserialize_seq().await);
                let mut out = HashSet::with_hasher(S::default());
                loop {
                    match hit!(
                        seq.next(|[elem]| async { elem.get::<T, _>(()).await })
                            .await
                    ) {
                        Chunk::Data((s, v)) => {
                            out.insert(v);
                            seq = s;
                        }
                        Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
                    }
                }
            })
            .await
        }
    }

    // --- Mutex<T>, RwLock<T> ---

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for std::sync::Mutex<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::Mutex::new(v))))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for std::sync::Mutex<T> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::Mutex::new(v))))
        }
    }

    impl<'de, T: Deserialize<'de>> Deserialize<'de> for std::sync::RwLock<T> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::RwLock::new(v))))
        }
    }

    impl<'s, T: DeserializeOwned<'s>> DeserializeOwned<'s> for std::sync::RwLock<T> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(T::deserialize(d, ())
                .await?
                .map(|(c, v)| (c, std::sync::RwLock::new(v))))
        }
    }

    // --- PathBuf, OsString ---

    impl<'de> Deserialize<'de> for std::path::PathBuf {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::path::PathBuf::from(s))))
        }
    }

    impl<'s> DeserializeOwned<'s> for std::path::PathBuf {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as DeserializeOwned>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::path::PathBuf::from(s))))
        }
    }

    impl<'de> Deserialize<'de> for std::ffi::OsString {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::ffi::OsString::from(s))))
        }
    }

    impl<'s> DeserializeOwned<'s> for std::ffi::OsString {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<String as DeserializeOwned>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::ffi::OsString::from(s))))
        }
    }

    // --- Box<Path>, Box<OsStr> ---

    impl<'de> Deserialize<'de> for Box<std::path::Path> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::path::PathBuf as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, p)| (c, p.into_boxed_path())))
        }
    }

    impl<'s> DeserializeOwned<'s> for Box<std::path::Path> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::path::PathBuf as DeserializeOwned>::deserialize(d, ())
                .await?
                .map(|(c, p)| (c, p.into_boxed_path())))
        }
    }

    impl<'de> Deserialize<'de> for Box<std::ffi::OsStr> {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::ffi::OsString as Deserialize>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_os_str())))
        }
    }

    impl<'s> DeserializeOwned<'s> for Box<std::ffi::OsStr> {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<std::ffi::OsString as DeserializeOwned>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, s.into_boxed_os_str())))
        }
    }

    // --- &'a Path (borrow only) ---

    impl<'de: 'a, 'a> Deserialize<'de> for &'a std::path::Path {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            Ok(<&'de str>::deserialize(d, ())
                .await?
                .map(|(c, s)| (c, std::path::Path::new(s))))
        }
    }

    // --- SystemTime ---

    enum SysTimeField {
        Secs(u64),
        Nanos(u32),
    }

    impl<'de> Deserialize<'de> for std::time::SystemTime {
        async fn deserialize<D: Deserializer<'de>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut secs: Option<u64> = None;
                let mut nanos: Option<u32> = None;
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, _k, r) = hit!(
                                ke.key((), |k, [ve]| {
                                    let field = match *k {
                                        "secs_since_epoch" => 0u8,
                                        "nanos_since_epoch" => 1u8,
                                        _ => 2u8,
                                    };
                                    async move {
                                        match field {
                                            0 => {
                                                let (claim, v) = hit!(ve.value::<u64, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Secs(v))))
                                            }
                                            1 => {
                                                let (claim, v) = hit!(ve.value::<u32, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Nanos(v))))
                                            }
                                            _ => Ok(Probe::Miss),
                                        }
                                    }
                                })
                                .await
                            );
                            Ok(Probe::Hit((claim, r)))
                        })
                        .await
                    ) {
                        Chunk::Data((m, SysTimeField::Secs(v))) => {
                            secs = Some(v);
                            map = m;
                        }
                        Chunk::Data((m, SysTimeField::Nanos(v))) => {
                            nanos = Some(v);
                            map = m;
                        }
                        Chunk::Done(claim) => {
                            let s = or_miss!(secs);
                            let n = or_miss!(nanos);
                            let dur = core::time::Duration::new(s, n);
                            return Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + dur)));
                        }
                    }
                }
            })
            .await
        }
    }

    impl<'s> DeserializeOwned<'s> for std::time::SystemTime {
        async fn deserialize<D: DeserializerOwned<'s>>(
            d: D,
            _extra: (),
        ) -> Result<Probe<(D::Claim, Self)>, D::Error>
        where
            D::Error: DeserializeError,
        {
            d.entry(|[e]| async {
                let mut map = hit!(e.deserialize_map().await);
                let mut secs: Option<u64> = None;
                let mut nanos: Option<u32> = None;
                loop {
                    match hit!(
                        map.next_kv(|[ke]| async {
                            let (claim, _k, r) = hit!(
                                ke.key((), |k: &String, [ve]| {
                                    let field = match k.as_str() {
                                        "secs_since_epoch" => 0u8,
                                        "nanos_since_epoch" => 1u8,
                                        _ => 2u8,
                                    };
                                    async move {
                                        match field {
                                            0 => {
                                                let (claim, v) = hit!(ve.value::<u64, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Secs(v))))
                                            }
                                            1 => {
                                                let (claim, v) = hit!(ve.value::<u32, _>(()).await);
                                                Ok(Probe::Hit((claim, SysTimeField::Nanos(v))))
                                            }
                                            _ => Ok(Probe::Miss),
                                        }
                                    }
                                })
                                .await
                            );
                            Ok(Probe::Hit((claim, r)))
                        })
                        .await
                    ) {
                        Chunk::Data((m, SysTimeField::Secs(v))) => {
                            secs = Some(v);
                            map = m;
                        }
                        Chunk::Data((m, SysTimeField::Nanos(v))) => {
                            nanos = Some(v);
                            map = m;
                        }
                        Chunk::Done(claim) => {
                            let s = or_miss!(secs);
                            let n = or_miss!(nanos);
                            let dur = core::time::Duration::new(s, n);
                            return Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + dur)));
                        }
                    }
                }
            })
            .await
        }
    }

    // --- Atomic types ---

    macro_rules! impl_atomic {
        ($atomic:path, $inner:ty) => {
            impl<'de> Deserialize<'de> for $atomic {
                async fn deserialize<D: Deserializer<'de>>(
                    d: D,
                    _extra: (),
                ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                    Ok(<$inner as Deserialize>::deserialize(d, ())
                        .await?
                        .map(|(c, v)| (c, <$atomic>::new(v))))
                }
            }

            impl<'s> DeserializeOwned<'s> for $atomic {
                async fn deserialize<D: DeserializerOwned<'s>>(
                    d: D,
                    _extra: (),
                ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                    Ok(<$inner as DeserializeOwned>::deserialize(d, ())
                        .await?
                        .map(|(c, v)| (c, <$atomic>::new(v))))
                }
            }
        };
    }

    #[cfg(target_has_atomic = "8")]
    impl_atomic!(core::sync::atomic::AtomicBool, bool);
    #[cfg(target_has_atomic = "8")]
    impl_atomic!(core::sync::atomic::AtomicI8, i8);
    #[cfg(target_has_atomic = "8")]
    impl_atomic!(core::sync::atomic::AtomicU8, u8);
    #[cfg(target_has_atomic = "16")]
    impl_atomic!(core::sync::atomic::AtomicI16, i16);
    #[cfg(target_has_atomic = "16")]
    impl_atomic!(core::sync::atomic::AtomicU16, u16);
    #[cfg(target_has_atomic = "32")]
    impl_atomic!(core::sync::atomic::AtomicI32, i32);
    #[cfg(target_has_atomic = "32")]
    impl_atomic!(core::sync::atomic::AtomicU32, u32);
    #[cfg(target_has_atomic = "64")]
    impl_atomic!(core::sync::atomic::AtomicI64, i64);
    #[cfg(target_has_atomic = "64")]
    impl_atomic!(core::sync::atomic::AtomicU64, u64);
    #[cfg(target_has_atomic = "ptr")]
    impl_atomic!(core::sync::atomic::AtomicIsize, isize);
    #[cfg(target_has_atomic = "ptr")]
    impl_atomic!(core::sync::atomic::AtomicUsize, usize);
}

// ===========================================================================
// Tag-filtering map facades (used by #[strede(tag)] derive output)
// ===========================================================================

pub mod tag_facade {
    use super::Match;
    use crate::{
        Chunk, Probe, Skip, borrow::{Deserialize, Deserializer, Entry, MapAccess, MapKeyEntry, MapValueEntry}, hit, or_miss, owned::{
            DeserializeOwned, DeserializerOwned, EntryOwned, MapAccessOwned, MapKeyEntryOwned,
            MapValueEntryOwned,
        }
    };
    use core::{array, cell::RefCell, future::Future, marker::PhantomData};

    // -----------------------------------------------------------------------
    // Borrow family
    // -----------------------------------------------------------------------

    /// A `Deserializer` that presents a single already-opened `MapAccess` as
    /// if it were a fresh entry delivering a map.  Used by internally-tagged
    /// enum derives to hand the in-progress map to a variant's `Deserialize`
    /// impl after the tag key has been consumed.
    ///
    /// When the inner type calls `d.entry(|[e]| e.deserialize_map())` the
    /// facade returns the wrapped map directly.  Any other probe returns Miss.
    pub struct MapDeserializer<'de, M: MapAccess<'de>> {
        pub map: Option<M>,
        pub _marker: core::marker::PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> MapDeserializer<'de, M> {
        pub fn new(map: M) -> Self {
            Self {
                map: Some(map),
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'de, M: MapAccess<'de>> Deserializer<'de> for MapDeserializer<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type Entry = MapDeserializerEntry<'de, M>;

        async fn entry<const N: usize, F, Fut, R>(
            mut self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let entries: [_; N] = array::from_fn(|i| {
                let map = if i == N - 1 {
                    self.map.as_mut().map(|m| m.fork())
                } else {
                    self.map.take()
                };
                MapDeserializerEntry {
                    map,
                    _marker: PhantomData,
                }
            });
            f(entries).await
        }
    }

    pub struct MapDeserializerEntry<'de, M: MapAccess<'de>> {
        map: Option<M>,
        _marker: PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> Entry<'de> for MapDeserializerEntry<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type StrChunks = crate::Never<'de, M::Claim, M::Error>;
        type BytesChunks = crate::Never<'de, M::Claim, M::Error>;
        type Map = M;
        type Seq = crate::Never<'de, M::Claim, M::Error>;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            match self.map {
                Some(m) => Ok(Probe::Hit(m)),
                None => Ok(Probe::Miss),
            }
        }

        async fn deserialize_option<T: Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_value<T: Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        fn fork(&mut self) -> Self {
            Self {
                map: self.map.as_mut().map(|v| v.fork()),
                _marker: PhantomData,
            }
        }

        async fn skip(self) -> Result<Self::Claim, Self::Error> {
            match self.map {
                Some(mut m) => loop {
                    match m
                        .next_kv::<1, _, _, ()>(|[ke]| async {
                            let (c, _, ()) = hit!(
                                ke.key((), |_k: &&'de str, [ve]| async {
                                    let c = ve.skip().await?;
                                    Ok(Probe::Hit((c, ())))
                                })
                                .await
                            );
                            Ok(Probe::Hit((c, ())))
                        })
                        .await?
                    {
                        Probe::Hit(Chunk::Done(c)) => return Ok(c),
                        Probe::Hit(Chunk::Data((new_m, ()))) => {
                            m = new_m;
                        }
                        Probe::Miss => panic!("unexpected Miss draining map"),
                    }
                },
                None => panic!("MapDeserializerEntry::skip called with no map"),
            }
        }
    }

    /// A `MapAccess` wrapper that skips one specific key (the tag field) and
    /// passes everything else through unchanged.
    pub struct TagFilteredMap<'de, M: MapAccess<'de>> {
        pub inner: M,
        pub tag_key: &'static str,
        pub _marker: core::marker::PhantomData<&'de ()>,
    }

    impl<'de, M: MapAccess<'de>> TagFilteredMap<'de, M> {
        pub fn new(inner: M, tag_key: &'static str) -> Self {
            Self {
                inner,
                tag_key,
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'de, M: MapAccess<'de>> MapAccess<'de> for TagFilteredMap<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type KeyEntry = M::KeyEntry;

        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_key: self.tag_key,
                _marker: core::marker::PhantomData,
            }
        }

        async fn next_kv<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
        where
            F: FnMut([Self::KeyEntry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let tag_key = self.tag_key;
            let mut inner = self.inner;
            loop {
                let tag_check = inner
                    .fork()
                    .next_kv::<1, _, _, bool>(|[ke]| async move {
                        let result = ke
                            .key((), |k: &&'de str, [ve]| {
                                let is_tag = *k == tag_key;
                                async move {
                                    let c = ve.skip().await?;
                                    Ok(Probe::Hit((c, is_tag)))
                                }
                            })
                            .await?;
                        match result {
                            Probe::Hit((c, _, is_tag)) => Ok(Probe::Hit((c, is_tag))),
                            Probe::Miss => Ok(Probe::Miss),
                        }
                    })
                    .await?;

                match tag_check {
                    Probe::Hit(Chunk::Done(_)) => {
                        // Map exhausted via fork — drain real inner to get the claim.
                        return inner
                            .next_kv::<1, _, _, ()>(|[_ke]| async move { Ok(Probe::Miss) })
                            .await
                            .map(|p| match p {
                                Probe::Hit(Chunk::Done(c)) => Probe::Hit(Chunk::Done(c)),
                                Probe::Hit(Chunk::Data(_)) => Probe::Miss,
                                Probe::Miss => Probe::Miss,
                            });
                    }
                    Probe::Hit(Chunk::Data((_, true))) => {
                        // This key IS the tag — skip it on the real inner and loop.
                        let skipped = inner
                            .next_kv::<1, _, _, ()>(|[ke]| async {
                                let (c, _, ()) = hit!(
                                    ke.key((), |_k: &&'de str, [ve]| async {
                                        let c = ve.skip().await?;
                                        Ok(Probe::Hit((c, ())))
                                    })
                                    .await
                                );
                                Ok(Probe::Hit((c, ())))
                            })
                            .await?;
                        match skipped {
                            Probe::Hit(Chunk::Data((new_inner, ()))) => {
                                inner = new_inner;
                            }
                            Probe::Hit(Chunk::Done(c)) => return Ok(Probe::Hit(Chunk::Done(c))),
                            Probe::Miss => return Ok(Probe::Miss),
                        }
                    }
                    Probe::Hit(Chunk::Data((_, false))) | Probe::Miss => {
                        // Not the tag — delegate to f on real inner.
                        return inner.next_kv(|keys| f(keys)).await.map(|p| match p {
                            Probe::Hit(Chunk::Data((new_inner, r))) => Probe::Hit(Chunk::Data((
                                TagFilteredMap {
                                    inner: new_inner,
                                    tag_key,
                                    _marker: core::marker::PhantomData,
                                },
                                r,
                            ))),
                            Probe::Hit(Chunk::Done(c)) => Probe::Hit(Chunk::Done(c)),
                            Probe::Miss => Probe::Miss,
                        });
                    }
                }
            }
        }
    }

    /// Wraps a `TagFilteredMap` as a `Deserializer` so that variant struct
    /// impls can call `T::deserialize` against an already-opened, tag-filtered map.
    pub struct TagFilteredMapDeserializer<'de, M: MapAccess<'de>> {
        pub map: TagFilteredMap<'de, M>,
    }

    impl<'de, M: MapAccess<'de>> TagFilteredMapDeserializer<'de, M> {
        pub fn new(inner: M, tag_key: &'static str) -> Self {
            Self {
                map: TagFilteredMap::new(inner, tag_key),
            }
        }
    }

    impl<'de, M: MapAccess<'de>> Deserializer<'de> for TagFilteredMapDeserializer<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type Entry = TagFilteredMapEntry<'de, M>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let mut map = Some(self.map);
            let entries: [_; N] = array::from_fn(|i| {
                let map = if i == N - 1 {
                    map.as_mut().unwrap().fork()
                } else {
                    map.take().unwrap()
                };
                TagFilteredMapEntry { map }
            });
            f(entries).await
        }
    }

    pub struct TagFilteredMapEntry<'de, M: MapAccess<'de>> {
        map: TagFilteredMap<'de, M>,
    }

    impl<'de, M: MapAccess<'de>> Entry<'de> for TagFilteredMapEntry<'de, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type StrChunks = crate::Never<'de, M::Claim, M::Error>;
        type BytesChunks = crate::Never<'de, M::Claim, M::Error>;
        type Map = TagFilteredMap<'de, M>;
        type Seq = crate::Never<'de, M::Claim, M::Error>;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_option<T: Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_value<T: Deserialize<'de, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        fn fork(&mut self) -> Self {
            Self {
                map: self.map.fork(),
            }
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            Ok(Probe::Hit(self.map))
        }

        async fn skip(mut self) -> Result<Self::Claim, Self::Error> {
            loop {
                match self
                    .map
                    .next_kv::<1, _, _, ()>(|[ke]| async {
                        let (c, _, ()) = hit!(
                            ke.key((), |_k: &&'de str, [ve]| async {
                                let c = ve.skip().await?;
                                Ok(Probe::Hit((c, ())))
                            })
                            .await
                        );
                        Ok(Probe::Hit((c, ())))
                    })
                    .await?
                {
                    Probe::Hit(Chunk::Done(c)) => return Ok(c),
                    Probe::Hit(Chunk::Data((new_m, ()))) => {
                        self.map = new_m;
                    }
                    Probe::Miss => panic!("unexpected Miss draining map"),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Owned family
    // -----------------------------------------------------------------------

    /// Owned counterpart to `MapDeserializer`.
    pub struct MapDeserializerOwned<'s, M: MapAccessOwned<'s>> {
        pub map: M,
        pub _marker: core::marker::PhantomData<&'s ()>,
    }

    impl<'s, M: MapAccessOwned<'s>> MapDeserializerOwned<'s, M> {
        pub fn new(map: M) -> Self {
            Self {
                map,
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'s, M: MapAccessOwned<'s>> DeserializerOwned<'s> for MapDeserializerOwned<'s, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type Entry = MapDeserializerEntryOwned<'s, M>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<
                Output = Result<Probe<(<Self::Entry as EntryOwned<'s>>::Claim, R)>, Self::Error>,
            >,
        {
            let mut map = Some(self.map);
            let entries: [_; N] = array::from_fn(|i| {
                let map = if i == N - 1 {
                    map.as_mut().unwrap().fork()
                } else {
                    map.take().unwrap()
                };
                MapDeserializerEntryOwned {
                    map,
                    _marker: core::marker::PhantomData,
                }
            });
            f(entries).await
        }
    }

    pub struct MapDeserializerEntryOwned<'s, M: MapAccessOwned<'s>> {
        map: M,
        _marker: core::marker::PhantomData<&'s ()>,
    }

    impl<'s, M: MapAccessOwned<'s>> EntryOwned<'s> for MapDeserializerEntryOwned<'s, M> {
        type Error = M::Error;
        type Claim = M::Claim;
        type StrChunks = crate::Never<'s, M::Claim, M::Error>;
        type BytesChunks = crate::Never<'s, M::Claim, M::Error>;
        type Map = M;
        type Seq = crate::Never<'s, M::Claim, M::Error>;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_option<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
            Ok(Probe::Miss)
        }
        async fn deserialize_value<T: DeserializeOwned<'s, Extra>, Extra>(
            self,
            _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
            Ok(Probe::Miss)
        }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            Ok(Probe::Hit(self.map))
        }

        fn fork(&mut self) -> Self {
            Self {
                map: self.map.fork(),
                _marker: PhantomData,
            }
        }

        async fn skip(mut self) -> Result<Self::Claim, Self::Error> {
            loop {
                match self
                    .map
                    .next_kv::<1, _, _, ()>(|[ke]| async {
                        let (c, _, ()) = hit!(
                            ke.key((), |_k: &super::Skip, [ve]| async {
                                let c = ve.skip().await?;
                                Ok(Probe::Hit((c, ())))
                            })
                            .await
                        );
                        Ok(Probe::Hit((c, ())))
                    })
                    .await?
                {
                    Probe::Hit(Chunk::Done(c)) => return Ok(c),
                    Probe::Hit(Chunk::Data((new_m, ()))) => {
                        self.map = new_m;
                    }
                    Probe::Miss => panic!("unexpected Miss"),
                }
            }
        }
    }

    /// Owned counterpart to `TagFilteredMap`.
    pub struct TagFilteredMapOwned<'s, M: MapAccessOwned<'s>, V = Skip> {
        inner: M,
        tag_key: &'static str,
        _marker: core::marker::PhantomData<fn() -> (&'s (), V)>,
    }

    impl<'s, M: MapAccessOwned<'s>, V> TagFilteredMapOwned<'s, M, V> {
        pub fn new(inner: M, tag_key: &'static str) -> Self {
            Self {
                inner,
                tag_key,
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<'s, M: MapAccessOwned<'s>> TagFilteredMapOwned<'s, M, Skip> {
        pub fn new_skip(inner: M, tag_key: &'static str) -> Self {
            Self::new(inner, tag_key)
        }
    }

    impl<'s, M: MapAccessOwned<'s>, V: DeserializeOwned<'s>> MapAccessOwned<'s> for TagFilteredMapOwned<'s, M, V> {
        type Error = M::Error;
        type Claim = M::Claim;
        type KeyEntry = M::KeyEntry;

        fn fork(&mut self) -> Self {
            Self {
                inner: self.inner.fork(),
                tag_key: self.tag_key,
                _marker: core::marker::PhantomData,
            }
        }

        async fn next_kv<const N: usize, F, Fut, R>(
            mut self,
            f: F,
        ) -> Result<Probe<Chunk<(Self, R), Self::Claim>>, Self::Error>
        where
            F: FnMut([Self::KeyEntry; N]) -> Fut,
            Fut: Future<Output = Result<Probe<(Self::Claim, R)>, Self::Error>>,
        {
            let f = RefCell::new(f);
            let f = &f;
            // If we're only looking for the tag we have to do a special case
            if N == 0 {
                let tag_key = self.tag_key;
                let result = hit!(self.inner.next_kv(async |[ke]| {
                    let (c, _, ()) = hit!(
                        ke.key(tag_key, |_: &Match, [ve]| async move {
                            let (c, _v) = hit!(ve.value::<V, ()>(()).await);
                            Ok(Probe::Hit((c, ())))
                        })
                        .await
                    );
                    Ok(Probe::Hit((c, ())))
                }).await);
                self.tag_key = "";
                let (next_inner, ()) = or_miss!(result.data());
                self.inner = next_inner;
            }

            // Try to read the tag until we do
            while !self.tag_key.is_empty() {
                let tag_key = self.tag_key;
                let result = hit!(self.inner.next_kv(|mut keys| async move {
                    let ke = keys[0].fork();
                    let f_fut = f.borrow_mut()(keys);
                    let (c, v) = hit!(crate::select_probe! {
                        // Arm 0: check if this key is the tag on a fork.
                        async move {
                            let (c, _, ()) = hit!(ke.key(tag_key, |_: &Match, [ve]| async move {
                                kill!(1);
                                let (c, _v) = hit!(ve.value::<V, ()>(()).await);
                                Ok(Probe::Hit((c, ())))
                            }).await);
                            Ok(Probe::Hit((c, None)))
                        },
                        // Arm 1: delegate to f.
                        async move {
                            let (c, r) = hit!(f_fut.await);
                            Ok(Probe::Hit((c, Some(r))))
                        },
                    });
                    Ok(Probe::Hit((c, v)))
                }).await);
                match or_miss!(result.data()) { // Miss if no tag found
                    (next_inner, None) => {
                        self.tag_key = "";
                        self.inner = next_inner;
                    }
                    (next_inner, Some(r)) => {
                        self.inner = next_inner;
                        return Ok(Probe::Hit(Chunk::Data((self, r))));
                    }
                }
                continue;
            }

            // Read everything else normally
            let tag_key = self.tag_key;
            return self
                .inner
                .next_kv(|keys| f.borrow_mut()(keys))
                .await
                .map(|p| match p {
                    Probe::Hit(Chunk::Data((new_inner, r))) => Probe::Hit(Chunk::Data((
                        TagFilteredMapOwned {
                            inner: new_inner,
                            tag_key,
                            _marker: PhantomData,
                        },
                        r,
                    ))),
                    Probe::Hit(Chunk::Done(c)) => Probe::Hit(Chunk::Done(c)),
                    Probe::Miss => Probe::Miss,
                });
        }
    }

    /// Wraps a `TagFilteredMapOwned` as a `DeserializerOwned` so that variant
    /// struct impls can call `T::deserialize_owned` against an already-opened,
    /// tag-filtered map.
    pub struct TagFilteredMapDeserializerOwned<'s, M: MapAccessOwned<'s>, V = Skip> {
        pub map: TagFilteredMapOwned<'s, M, V>,
    }

    impl<'s, M: MapAccessOwned<'s>, V> TagFilteredMapDeserializerOwned<'s, M, V> {
        pub fn new(inner: M, tag_key: &'static str) -> Self {
            Self { map: TagFilteredMapOwned::new(inner, tag_key) }
        }
    }

    impl<'s, M: MapAccessOwned<'s>> TagFilteredMapDeserializerOwned<'s, M, Skip> {
        pub fn new_skip(inner: M, tag_key: &'static str) -> Self {
            Self { map: TagFilteredMapOwned::new_skip(inner, tag_key) }
        }
    }

    impl<'s, M: MapAccessOwned<'s>, V: DeserializeOwned<'s>> DeserializerOwned<'s>
        for TagFilteredMapDeserializerOwned<'s, M, V>
    {
        type Error = M::Error;
        type Claim = M::Claim;
        type Entry = TagFilteredMapEntryOwned<'s, M, V>;

        async fn entry<const N: usize, F, Fut, R>(
            self,
            mut f: F,
        ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
        where
            F: FnMut([Self::Entry; N]) -> Fut,
            Fut: Future<
                Output = Result<Probe<(<Self::Entry as EntryOwned<'s>>::Claim, R)>, Self::Error>,
            >,
        {
            let mut map = Some(self.map);
            let entries: [_; N] = array::from_fn(|i| {
                let map = if i == N - 1 {
                    map.as_mut().unwrap().fork()
                } else {
                    map.take().unwrap()
                };
                TagFilteredMapEntryOwned { map }
            });
            f(entries).await
        }
    }

    pub struct TagFilteredMapEntryOwned<'s, M: MapAccessOwned<'s>, V = Skip> {
        map: TagFilteredMapOwned<'s, M, V>,
    }

    impl<'s, M: MapAccessOwned<'s>, V: DeserializeOwned<'s>> EntryOwned<'s>
        for TagFilteredMapEntryOwned<'s, M, V>
    {
        type Error = M::Error;
        type Claim = M::Claim;
        type StrChunks = crate::Never<'s, M::Claim, M::Error>;
        type BytesChunks = crate::Never<'s, M::Claim, M::Error>;
        type Map = TagFilteredMapOwned<'s, M, V>;
        type Seq = crate::Never<'s, M::Claim, M::Error>;

        async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_option<T: DeserializeOwned<'s, Extra>, Extra>(
            self, _extra: Extra,
        ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> { Ok(Probe::Miss) }
        async fn deserialize_value<T: DeserializeOwned<'s, Extra>, Extra>(
            self, _extra: Extra,
        ) -> Result<Probe<(Self::Claim, T)>, Self::Error> { Ok(Probe::Miss) }

        async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
            Ok(Probe::Hit(self.map))
        }

        fn fork(&mut self) -> Self {
            Self { map: self.map.fork() }
        }

        async fn skip(mut self) -> Result<Self::Claim, Self::Error> {
            loop {
                match self
                    .map
                    .next_kv::<1, _, _, ()>(|[ke]| async {
                        let (c, _, ()) = hit!(
                            ke.key((), |_k: &super::Skip, [ve]| async {
                                let c = ve.skip().await?;
                                Ok(Probe::Hit((c, ())))
                            })
                            .await
                        );
                        Ok(Probe::Hit((c, ())))
                    })
                    .await?
                {
                    Probe::Hit(Chunk::Done(c)) => return Ok(c),
                    Probe::Hit(Chunk::Data((new_m, ()))) => { self.map = new_m; }
                    Probe::Miss => panic!("unexpected Miss draining map"),
                }
            }
        }
    }
}
