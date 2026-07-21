use super::*;
use crate::borrow::{DeserializeFromMap, MapAccess, MapKeyProbe, MapValueProbe};
use crate::owned::{
    DeserializeFromMapOwned, MapAccessOwned, MapKeyProbeOwned, MapValueProbeOwned,
};
use crate::{hit, or_miss, select_probe};

// -------------------------------------------------------------------------
// Vec<T> — shape-specific DeserializeFromSeq only (iterates an already-opened
// seq; format-agnostic, no ambiguity). There is deliberately no universal
// `Deserialize`/`DeserializeOwned` impl here: whether `Vec<u8>` can safely
// race a "raw bytes" reading against "a sequence of u8 elements" depends on
// whether the format's wire representation can tell the two apart, which
// only the format itself knows (see `strede::utils` and the `strede-*`
// format crates, each of which provides its own `Vec<T>` impl).
// -------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<'de, S, T> DeserializeFromSeq<'de, S> for alloc::vec::Vec<T>
where
    S: SeqAccess<'de>,
    T: Deserialize<'de, <S::Elem as SeqEntry<'de>>::SubDeserializer, Extra = ()>,
{
    type Extra = ();
    async fn deserialize_from_seq(
        mut seq: S,
        _: (),
    ) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
        let mut out: alloc::vec::Vec<T> = alloc::vec::Vec::new();
        loop {
            match seq
                .next::<1, _, _, _>(|[elem]| async move {
                    let (claim, v) = hit!(elem.get::<T>(()).await);
                    Ok(Probe::Hit((claim, v)))
                })
                .await?
            {
                Probe::Hit(Chunk::Data((next, v))) => {
                    seq = next;
                    out.push(v);
                }
                Probe::Hit(Chunk::Done(claim)) => return Ok(Probe::Hit((claim, out))),
                Probe::Miss => return Ok(Probe::Miss),
            }
        }
    }
}

#[cfg(feature = "alloc")]
impl<S, T> DeserializeFromSeqOwned<S> for alloc::vec::Vec<T>
where
    S: SeqAccessOwned,
    T: DeserializeOwned<<S::Elem as SeqEntryOwned>::SubDeserializer, Extra = ()>,
{
    type Extra = ();
    async fn deserialize_from_seq_owned(
        mut seq: S,
        _: (),
    ) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
        let mut out: alloc::vec::Vec<T> = alloc::vec::Vec::new();
        loop {
            match seq
                .next::<1, _, _, _>(|[elem]| async move {
                    let (claim, v) = hit!(elem.get::<T>(()).await);
                    Ok(Probe::Hit((claim, v)))
                })
                .await?
            {
                Probe::Hit(Chunk::Data((next, v))) => {
                    seq = next;
                    out.push(v);
                }
                Probe::Hit(Chunk::Done(claim)) => return Ok(Probe::Hit((claim, out))),
                Probe::Miss => return Ok(Probe::Miss),
            }
        }
    }
}

// -------------------------------------------------------------------------
// Cow<'de, [u8]> — borrow-family: zero-copy fast path + chunked alloc fallback.
// -------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<'de, D> Deserialize<'de, D> for alloc::borrow::Cow<'de, [u8]>
where
    D: Deserializer<'de>,
{
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        use alloc::borrow::Cow;
        use alloc::vec::Vec;
        d.entry(|[e1, e2]| async move {
            select_probe! {
                async move {
                    let (claim, b) = hit!(e1.deserialize_bytes().await);
                    Ok(Probe::Hit((claim, Cow::Borrowed(b))))
                },
                async move {
                    let mut chunks = hit!(e2.deserialize_bytes_chunks().await);
                    let mut out: Vec<u8> = Vec::new();
                    loop {
                        match chunks.next_bytes(|b| out.extend_from_slice(b)).await? {
                            Chunk::Data((new, ())) => chunks = new,
                            Chunk::Done(claim) => return Ok(Probe::Hit((claim, Cow::Owned(out)))),
                        }
                    }
                }
            }
        })
        .await
    }
}

// -------------------------------------------------------------------------
// Box<[T]> — both families. Delegates to Vec<T>.
// -------------------------------------------------------------------------

#[cfg(feature = "alloc")]
impl<'de, D, T> Deserialize<'de, D> for alloc::boxed::Box<[T]>
where
    D: Deserializer<'de>,
    T: Deserialize<
        'de,
        <<<D::Entry as Entry<'de>>::Seq as SeqAccess<'de>>::Elem as SeqEntry<'de>>::SubDeserializer,
        Extra = (),
    >,
    alloc::vec::Vec<T>: Deserialize<'de, D, Extra = ()>,
{
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let (claim, v) = hit!(<alloc::vec::Vec<T> as Deserialize<'de, D>>::deserialize(d, ()).await);
        Ok(Probe::Hit((claim, v.into_boxed_slice())))
    }
}

#[cfg(feature = "alloc")]
impl<D, T> DeserializeOwned<D> for alloc::boxed::Box<[T]>
where
    D: DeserializerOwned,
    T: DeserializeOwned<
        <<<D::Entry as EntryOwned>::Seq as SeqAccessOwned>::Elem as SeqEntryOwned>::SubDeserializer,
        Extra = (),
    >,
    alloc::vec::Vec<T>: DeserializeOwned<D, Extra = ()>,
{
    type Extra = ();
    async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        let (claim, v) = hit!(<alloc::vec::Vec<T> as DeserializeOwned<D>>::deserialize_owned(d, ()).await);
        Ok(Probe::Hit((claim, v.into_boxed_slice())))
    }
}

// -------------------------------------------------------------------------
// Tuples — both families.  Each emits universal Deserialize + shape-specific
// DeserializeFromSeq, delegating element extraction to seq.next.
// -------------------------------------------------------------------------

macro_rules! impl_tuple {
    ($($T:ident $v:ident),+ $(,)?) => {
        impl<'de, M, $($T),+> DeserializeFromMap<'de, M> for ($($T,)+)
        where
            M: MapAccess<'de>,
            $(
                $T: Deserialize<
                    'de,
                    <crate::borrow::VP<'de, M::KeyProbe> as MapValueProbe<'de>>::ValueSubDeserializer,
                    Extra = (),
                >,
            )+
        {
            type Extra = ();
            async fn deserialize_from_map(
                map: M,
                _: (),
            ) -> Result<Probe<(M::MapClaim, Self)>, M::Error> {
                let arms = crate::MapArmBase
                    $(+ crate::MapArm(crate::MapArmSlot::new(
                        |mut __kp: M::KeyProbe, __i: usize| async move {
                            let (__kc, ()) = hit!(__kp.deserialize_key_by_index(__i).await);
                            Ok(Probe::Hit((__kc, ())))
                        },
                        |__vp: crate::borrow::VP<'de, M::KeyProbe>, _k: ()| async move {
                            let (__vc, __v) = hit!(__vp.deserialize_value::<$T>(()).await);
                            Ok(Probe::Hit((__vc, ((), __v))))
                        }
                    )))+;
                match map.iterate(arms).await? {
                    Probe::Hit((claim, crate::map_outputs!($($v,)+))) => {
                        Ok(Probe::Hit((claim, ($(or_miss!($v.map(|((), v)| v)),)+))))
                    }
                    Probe::Miss => Ok(Probe::Miss),
                }
            }
        }

        impl<M, $($T),+> DeserializeFromMapOwned<M> for ($($T,)+)
        where
            M: MapAccessOwned,
            $(
                $T: DeserializeOwned<
                    <crate::owned::VP<M::KeyProbe> as MapValueProbeOwned>::ValueSubDeserializer,
                    Extra = (),
                >,
            )+
        {
            type Extra = ();
            async fn deserialize_from_map_owned(
                map: M,
                _: (),
            ) -> Result<Probe<(M::MapClaim, Self)>, M::Error> {
                let arms = crate::MapArmBase
                    $(+ crate::MapArm(crate::MapArmSlot::new(
                        |mut __kp: M::KeyProbe, __i: usize| async move {
                            let (__kc, ()) = hit!(__kp.deserialize_key_by_index(__i).await);
                            Ok(Probe::Hit((__kc, ())))
                        },
                        |__vp: crate::owned::VP<M::KeyProbe>, _k: ()| async move {
                            let (__vc, __v) = hit!(__vp.deserialize_value::<$T>(()).await);
                            Ok(Probe::Hit((__vc, ((), __v))))
                        }
                    )))+;
                match map.iterate(arms).await? {
                    Probe::Hit((claim, crate::map_outputs!($($v,)+))) => {
                        Ok(Probe::Hit((claim, ($(or_miss!($v.map(|((), v)| v)),)+))))
                    }
                    Probe::Miss => Ok(Probe::Miss),
                }
            }
        }

        impl<'de, D, $($T),+> Deserialize<'de, D> for ($($T,)+)
        where
            D: Deserializer<'de>,
            ($($T,)+): DeserializeFromMap<'de, <D::Entry as Entry<'de>>::Map, Extra = ()>,
            ($($T,)+): DeserializeFromSeq<'de, <D::Entry as Entry<'de>>::Seq, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e1, e2]| async move {
                    select_probe!(biased;
                        e1.deserialize_map_into::<Self>(()),
                        e2.deserialize_seq_into::<Self>(()),
                    )
                }).await
            }
        }

        impl<'de, S, $($T),+> DeserializeFromSeq<'de, S> for ($($T,)+)
        where
            S: SeqAccess<'de>,
            $(
                $T: Deserialize<'de, <S::Elem as SeqEntry<'de>>::SubDeserializer, Extra = ()>,
            )+
        {
            type Extra = ();
            async fn deserialize_from_seq(seq: S, _: ()) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
                let seq = seq;
                $(
                    let r = hit!(seq.next::<1, _, _, _>(|[elem]| async move {
                        let (c, v) = hit!(elem.get::<$T>(()).await);
                        Ok(Probe::Hit((c, v)))
                    }).await);
                    let (seq, $v) = or_miss!(r.data());
                )+
                let r = hit!(seq.next::<1, _, _, _>(|[elem]| async move {
                    Ok::<_, S::Error>(Probe::Hit((elem.skip().await?, ())))
                }).await);
                let claim = or_miss!(r.done());
                Ok(Probe::Hit((claim, ($($v,)+))))
            }
        }

        impl<D, $($T),+> DeserializeOwned<D> for ($($T,)+)
        where
            D: DeserializerOwned,
            ($($T,)+): DeserializeFromMapOwned<<D::Entry as EntryOwned>::Map, Extra = ()>,
            ($($T,)+): DeserializeFromSeqOwned<<D::Entry as EntryOwned>::Seq, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e1, e2]| async move {
                    select_probe!(biased;
                        e1.deserialize_map_into::<Self>(()),
                        e2.deserialize_seq_into::<Self>(()),
                    )
                }).await
            }
        }

        impl<S, $($T),+> DeserializeFromSeqOwned<S> for ($($T,)+)
        where
            S: SeqAccessOwned,
            $(
                $T: DeserializeOwned<<S::Elem as SeqEntryOwned>::SubDeserializer, Extra = ()>,
            )+
        {
            type Extra = ();
            async fn deserialize_from_seq_owned(seq: S, _: ()) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
                let seq = seq;
                $(
                    let r = hit!(seq.next::<1, _, _, _>(|[elem]| async move {
                        let (c, v) = hit!(elem.get::<$T>(()).await);
                        Ok(Probe::Hit((c, v)))
                    }).await);
                    let (seq, $v) = or_miss!(r.data());
                )+
                let r = hit!(seq.next::<1, _, _, _>(|[elem]| async move {
                    Ok::<_, S::Error>(Probe::Hit((elem.skip().await?, ())))
                }).await);
                let claim = or_miss!(r.done());
                Ok(Probe::Hit((claim, ($($v,)+))))
            }
        }
    };
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

// -------------------------------------------------------------------------
// Newtype wrappers around T: Wrapping, Saturating, Reverse, Cell, RefCell.
// All delegate fully to T's Deserialize impl.
// -------------------------------------------------------------------------

macro_rules! impl_newtype_wrapper {
    ($wrapper:ident, $ctor:path $(, $bound:ident)?) => {
        impl<'de, D, T> Deserialize<'de, D> for $wrapper<T>
        where
            D: Deserializer<'de>,
            T: Deserialize<'de, D> $(+ $bound)?,
        {
            type Extra = T::Extra;
            async fn deserialize(d: D, extra: Self::Extra) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(T::deserialize(d, extra).await?.map(|(c, v)| (c, $ctor(v))))
            }
        }

        impl<D, T> DeserializeOwned<D> for $wrapper<T>
        where
            D: DeserializerOwned,
            T: DeserializeOwned<D> $(+ $bound)?,
        {
            type Extra = T::Extra;
            async fn deserialize_owned(d: D, extra: Self::Extra) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                Ok(T::deserialize_owned(d, extra).await?.map(|(c, v)| (c, $ctor(v))))
            }
        }
    };
}

impl_newtype_wrapper!(Wrapping, core::num::Wrapping);
impl_newtype_wrapper!(Saturating, core::num::Saturating);
impl_newtype_wrapper!(Reverse, core::cmp::Reverse);
impl_newtype_wrapper!(Cell, core::cell::Cell::new, Copy);
impl_newtype_wrapper!(RefCell, core::cell::RefCell::new);

use core::cell::{Cell, RefCell};
use core::cmp::Reverse;
use core::num::{Saturating, Wrapping};

// -------------------------------------------------------------------------
// NonZero* — both families. Delegate to the underlying integer Deserialize,
// then call NonZero::new and miss on zero.
// -------------------------------------------------------------------------

macro_rules! impl_nonzero {
    ($nonzero:ty, $prim:ty) => {
        impl<'de, D> Deserialize<'de, D> for $nonzero
        where
            D: Deserializer<'de>,
            $prim: Deserialize<'de, <D::Entry as Entry<'de>>::SubDeserializer, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async move {
                    let (claim, v) = hit!(e.deserialize_value::<$prim>(()).await);
                    let nz = or_miss!(Self::new(v));
                    Ok(Probe::Hit((claim, nz)))
                })
                .await
            }
        }

        impl<D> DeserializeOwned<D> for $nonzero
        where
            D: DeserializerOwned,
            $prim: DeserializeOwned<<D::Entry as EntryOwned>::SubDeserializer, Extra = ()>,
        {
            type Extra = ();
            async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async move {
                    let (claim, v) = hit!(e.deserialize_value::<$prim>(()).await);
                    let nz = or_miss!(Self::new(v));
                    Ok(Probe::Hit((claim, nz)))
                })
                .await
            }
        }
    };
}

impl_nonzero!(core::num::NonZeroU8, u8);
impl_nonzero!(core::num::NonZeroU16, u16);
impl_nonzero!(core::num::NonZeroU32, u32);
impl_nonzero!(core::num::NonZeroU64, u64);
impl_nonzero!(core::num::NonZeroU128, u128);
impl_nonzero!(core::num::NonZeroI8, i8);
impl_nonzero!(core::num::NonZeroI16, i16);
impl_nonzero!(core::num::NonZeroI32, i32);
impl_nonzero!(core::num::NonZeroI64, i64);
impl_nonzero!(core::num::NonZeroI128, i128);
impl_nonzero!(core::num::NonZeroUsize, usize);
impl_nonzero!(core::num::NonZeroIsize, isize);

// -------------------------------------------------------------------------
// [T; N] — both families.  Reads N elements from a seq.  Uses MaybeUninit
// for partial-init safety on Miss/Err.
// -------------------------------------------------------------------------

struct ArrayGuard<T, const N: usize> {
    buf: [core::mem::MaybeUninit<T>; N],
    filled: usize,
}

impl<T, const N: usize> ArrayGuard<T, N> {
    fn new() -> Self {
        Self {
            buf: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
            filled: 0,
        }
    }
    fn push(&mut self, i: usize, v: T) {
        self.buf[i].write(v);
        self.filled = i + 1;
    }
    fn into_array(mut self) -> [T; N] {
        let arr = unsafe {
            let p = &self.buf as *const _ as *const [T; N];
            p.read()
        };
        self.filled = 0;
        core::mem::forget(self);
        arr
    }
}

impl<T, const N: usize> Drop for ArrayGuard<T, N> {
    fn drop(&mut self) {
        for i in 0..self.filled {
            unsafe { self.buf[i].assume_init_drop() };
        }
    }
}

impl<'de, D, T, const N: usize> Deserialize<'de, D> for [T; N]
where
    D: Deserializer<'de>,
    T: Deserialize<
        'de,
        <<<D::Entry as Entry<'de>>::Seq as SeqAccess<'de>>::Elem as SeqEntry<'de>>::SubDeserializer,
        Extra = (),
    >,
{
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move { e.deserialize_seq_into::<Self>(()).await }).await
    }
}

impl<'de, S, T, const N: usize> DeserializeFromSeq<'de, S> for [T; N]
where
    S: SeqAccess<'de>,
    T: Deserialize<'de, <S::Elem as SeqEntry<'de>>::SubDeserializer, Extra = ()>,
{
    type Extra = ();
    async fn deserialize_from_seq(seq: S, _: ()) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
        let mut guard = ArrayGuard::<T, N>::new();
        let mut seq = seq;
        for i in 0..N {
            let r = match seq
                .next::<1, _, _, _>(|[elem]| async move {
                    let (c, v) = hit!(elem.get::<T>(()).await);
                    Ok(Probe::Hit((c, v)))
                })
                .await?
            {
                Probe::Hit(c) => c,
                Probe::Miss => return Ok(Probe::Miss),
            };
            match r {
                Chunk::Data((next, v)) => {
                    seq = next;
                    guard.push(i, v);
                }
                Chunk::Done(_) => return Ok(Probe::Miss),
            }
        }
        let r = match seq
            .next::<1, _, _, _>(|[elem]| async move {
                Ok::<_, S::Error>(Probe::Hit((elem.skip().await?, ())))
            })
            .await?
        {
            Probe::Hit(c) => c,
            Probe::Miss => return Ok(Probe::Miss),
        };
        let claim = match r {
            Chunk::Done(c) => c,
            Chunk::Data(_) => return Ok(Probe::Miss),
        };
        Ok(Probe::Hit((claim, guard.into_array())))
    }
}

impl<D, T, const N: usize> DeserializeOwned<D> for [T; N]
where
    D: DeserializerOwned,
    T: DeserializeOwned<
        <<<D::Entry as EntryOwned>::Seq as SeqAccessOwned>::Elem as SeqEntryOwned>::SubDeserializer,
        Extra = (),
    >,
{
    type Extra = ();
    async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move { e.deserialize_seq_into::<Self>(()).await }).await
    }
}

impl<S, T, const N: usize> DeserializeFromSeqOwned<S> for [T; N]
where
    S: SeqAccessOwned,
    T: DeserializeOwned<<S::Elem as SeqEntryOwned>::SubDeserializer, Extra = ()>,
{
    type Extra = ();
    async fn deserialize_from_seq_owned(
        seq: S,
        _: (),
    ) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
        let mut guard = ArrayGuard::<T, N>::new();
        let mut seq = seq;
        for i in 0..N {
            let r = match seq
                .next::<1, _, _, _>(|[elem]| async move {
                    let (c, v) = hit!(elem.get::<T>(()).await);
                    Ok(Probe::Hit((c, v)))
                })
                .await?
            {
                Probe::Hit(c) => c,
                Probe::Miss => return Ok(Probe::Miss),
            };
            match r {
                Chunk::Data((next, v)) => {
                    seq = next;
                    guard.push(i, v);
                }
                Chunk::Done(_) => return Ok(Probe::Miss),
            }
        }
        let r = match seq
            .next::<1, _, _, _>(|[elem]| async move {
                Ok::<_, S::Error>(Probe::Hit((elem.skip().await?, ())))
            })
            .await?
        {
            Probe::Hit(c) => c,
            Probe::Miss => return Ok(Probe::Miss),
        };
        let claim = match r {
            Chunk::Done(c) => c,
            Chunk::Data(_) => return Ok(Probe::Miss),
        };
        Ok(Probe::Hit((claim, guard.into_array())))
    }
}

// -------------------------------------------------------------------------
// Seq-collected types: BTreeSet, BinaryHeap, LinkedList, VecDeque, HashSet.
// Universal Deserialize + shape-specific DeserializeFromSeq.
// -------------------------------------------------------------------------

macro_rules! impl_seq_collect {
    ($(#[$attr:meta])* $ty:ty, $new:expr, $add:ident $(, $bound:ident)*) => {
        $(#[$attr])*
        impl<'de, D, T> Deserialize<'de, D> for $ty
        where
            D: Deserializer<'de>,
            T: Deserialize<
                'de,
                <<<D::Entry as Entry<'de>>::Seq as SeqAccess<'de>>::Elem as SeqEntry<'de>>::SubDeserializer,
                Extra = (),
            > $(+ $bound)*,
        {
            type Extra = ();
            async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async move { e.deserialize_seq_into::<Self>(()).await }).await
            }
        }

        $(#[$attr])*
        impl<'de, S, T> DeserializeFromSeq<'de, S> for $ty
        where
            S: SeqAccess<'de>,
            T: Deserialize<'de, <S::Elem as SeqEntry<'de>>::SubDeserializer, Extra = ()> $(+ $bound)*,
        {
            type Extra = ();
            async fn deserialize_from_seq(mut seq: S, _: ()) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
                let mut out: $ty = $new;
                loop {
                    match seq.next::<1, _, _, _>(|[elem]| async move {
                        let (c, v) = hit!(elem.get::<T>(()).await);
                        Ok(Probe::Hit((c, v)))
                    }).await? {
                        Probe::Hit(Chunk::Data((next, v))) => { seq = next; out.$add(v); }
                        Probe::Hit(Chunk::Done(claim)) => return Ok(Probe::Hit((claim, out))),
                        Probe::Miss => return Ok(Probe::Miss),
                    }
                }
            }
        }

        $(#[$attr])*
        impl<D, T> DeserializeOwned<D> for $ty
        where
            D: DeserializerOwned,
            T: DeserializeOwned<
                <<<D::Entry as EntryOwned>::Seq as SeqAccessOwned>::Elem as SeqEntryOwned>::SubDeserializer,
                Extra = (),
            > $(+ $bound)*,
        {
            type Extra = ();
            async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async move { e.deserialize_seq_into::<Self>(()).await }).await
            }
        }

        $(#[$attr])*
        impl<S, T> DeserializeFromSeqOwned<S> for $ty
        where
            S: SeqAccessOwned,
            T: DeserializeOwned<<S::Elem as SeqEntryOwned>::SubDeserializer, Extra = ()> $(+ $bound)*,
        {
            type Extra = ();
            async fn deserialize_from_seq_owned(mut seq: S, _: ()) -> Result<Probe<(S::SeqClaim, Self)>, S::Error> {
                let mut out: $ty = $new;
                loop {
                    match seq.next::<1, _, _, _>(|[elem]| async move {
                        let (c, v) = hit!(elem.get::<T>(()).await);
                        Ok(Probe::Hit((c, v)))
                    }).await? {
                        Probe::Hit(Chunk::Data((next, v))) => { seq = next; out.$add(v); }
                        Probe::Hit(Chunk::Done(claim)) => return Ok(Probe::Hit((claim, out))),
                        Probe::Miss => return Ok(Probe::Miss),
                    }
                }
            }
        }
    };
}

impl_seq_collect!(
    #[cfg(feature = "alloc")]
    alloc::collections::BTreeSet<T>, alloc::collections::BTreeSet::new(), insert, Ord
);
impl_seq_collect!(
    #[cfg(feature = "alloc")]
    alloc::collections::BinaryHeap<T>, alloc::collections::BinaryHeap::new(), push, Ord
);
impl_seq_collect!(
    #[cfg(feature = "alloc")]
    alloc::collections::LinkedList<T>, alloc::collections::LinkedList::new(), push_back
);
impl_seq_collect!(
    #[cfg(feature = "alloc")]
    alloc::collections::VecDeque<T>, alloc::collections::VecDeque::new(), push_back
);

#[cfg(feature = "std")]
mod hashset_impls {
    extern crate std;
    use super::*;
    use core::hash::Hash;
    impl_seq_collect!(
        std::collections::HashSet<T>,
        std::collections::HashSet::new(),
        insert,
        Eq,
        Hash
    );
}
