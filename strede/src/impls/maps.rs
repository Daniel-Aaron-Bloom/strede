use super::*;
use crate::{hit, or_miss, select_probe};

// -------------------------------------------------------------------------
// Duration / SystemTime — map-shaped {secs, nanos} / {secs_since_epoch, nanos_since_epoch}.
// Reuses the derive-style map_arms! infrastructure.
// -------------------------------------------------------------------------

use crate::borrow::{MapAccess, MapKeyProbe, MapValueProbe};
use crate::owned::{MapAccessOwned, MapKeyProbeOwned, MapValueProbeOwned};

impl<'de, D> Deserialize<'de, D> for core::time::Duration
where
    D: Deserializer<'de>,
    u64: Deserialize<
            'de,
            <crate::borrow::VP2<'de, D> as MapValueProbe<'de>>::ValueSubDeserializer,
            Extra = (),
        >,
    u32: Deserialize<
            'de,
            <crate::borrow::VP2<'de, D> as MapValueProbe<'de>>::ValueSubDeserializer,
            Extra = (),
        >,
{
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move {
            let map = hit!(e.deserialize_map().await);
            let arms = crate::map_arms! {
                |kp: crate::borrow::KP<'de, D>, _i: usize| kp.deserialize_key::<crate::Match>("secs")
                => |vp: crate::borrow::VP2<'de, D>, k: crate::Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<u64>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
                |kp: crate::borrow::KP<'de, D>, _i: usize| kp.deserialize_key::<crate::Match>("nanos")
                => |vp: crate::borrow::VP2<'de, D>, k: crate::Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<u32>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
            };
            let (claim, crate::map_outputs!(opt_secs, opt_nanos)) = hit!(map.iterate(arms).await);
            let s = or_miss!(opt_secs.map(|(_, v)| v));
            let n = or_miss!(opt_nanos.map(|(_, v)| v));
            Ok(Probe::Hit((claim, core::time::Duration::new(s, n))))
        })
        .await
    }
}

impl<D> DeserializeOwned<D> for core::time::Duration
where
    D: DeserializerOwned,
    u64: DeserializeOwned<
            <crate::owned::VP2<D> as MapValueProbeOwned>::ValueSubDeserializer,
            Extra = (),
        >,
    u32: DeserializeOwned<
            <crate::owned::VP2<D> as MapValueProbeOwned>::ValueSubDeserializer,
            Extra = (),
        >,
{
    type Extra = ();
    async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async move {
            let map = hit!(e.deserialize_map().await);
            let arms = crate::map_arms! {
                |kp: crate::owned::KP<D>, _i: usize| kp.deserialize_key::<crate::Match>("secs")
                => |vp: crate::owned::VP2<D>, k: crate::Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<u64>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
                |kp: crate::owned::KP<D>, _i: usize| kp.deserialize_key::<crate::Match>("nanos")
                => |vp: crate::owned::VP2<D>, k: crate::Match| async move {
                    let (vc, v) = hit!(vp.deserialize_value::<u32>(()).await);
                    Ok(Probe::Hit((vc, (k, v))))
                },
            };
            let (claim, crate::map_outputs!(opt_secs, opt_nanos)) = hit!(map.iterate(arms).await);
            let s = or_miss!(opt_secs.map(|(_, v)| v));
            let n = or_miss!(opt_nanos.map(|(_, v)| v));
            Ok(Probe::Hit((claim, core::time::Duration::new(s, n))))
        })
        .await
    }
}

#[cfg(feature = "std")]
mod systemtime_impls {
    extern crate std;
    use super::*;
    use crate::borrow::MapValueProbe;
    use crate::owned::MapValueProbeOwned;

    impl<'de, D> Deserialize<'de, D> for std::time::SystemTime
    where
        D: Deserializer<'de>,
        u64: Deserialize<
                'de,
                <crate::borrow::VP2<'de, D> as MapValueProbe<'de>>::ValueSubDeserializer,
                Extra = (),
            >,
        u32: Deserialize<
                'de,
                <crate::borrow::VP2<'de, D> as MapValueProbe<'de>>::ValueSubDeserializer,
                Extra = (),
            >,
    {
        type Extra = ();
        async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async move {
                let map = hit!(e.deserialize_map().await);
                let arms = crate::map_arms! {
                    |kp: crate::borrow::KP<'de, D>, _i: usize| kp.deserialize_key::<crate::Match>("secs_since_epoch")
                    => |vp: crate::borrow::VP2<'de, D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u64>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                    |kp: crate::borrow::KP<'de, D>, _i: usize| kp.deserialize_key::<crate::Match>("nanos_since_epoch")
                    => |vp: crate::borrow::VP2<'de, D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u32>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                };
                let (claim, crate::map_outputs!(opt_secs, opt_nanos)) = hit!(map.iterate(arms).await);
                let s = or_miss!(opt_secs.map(|(_, v)| v));
                let n = or_miss!(opt_nanos.map(|(_, v)| v));
                Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + core::time::Duration::new(s, n))))
            }).await
        }
    }

    impl<D> DeserializeOwned<D> for std::time::SystemTime
    where
        D: DeserializerOwned,
        u64: DeserializeOwned<
                <crate::owned::VP2<D> as MapValueProbeOwned>::ValueSubDeserializer,
                Extra = (),
            >,
        u32: DeserializeOwned<
                <crate::owned::VP2<D> as MapValueProbeOwned>::ValueSubDeserializer,
                Extra = (),
            >,
    {
        type Extra = ();
        async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async move {
                let map = hit!(e.deserialize_map().await);
                let arms = crate::map_arms! {
                    |kp: crate::owned::KP<D>, _i: usize| kp.deserialize_key::<crate::Match>("secs_since_epoch")
                    => |vp: crate::owned::VP2<D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u64>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                    |kp: crate::owned::KP<D>, _i: usize| kp.deserialize_key::<crate::Match>("nanos_since_epoch")
                    => |vp: crate::owned::VP2<D>, k: crate::Match| async move {
                        let (vc, v) = hit!(vp.deserialize_value::<u32>(()).await);
                        Ok(Probe::Hit((vc, (k, v))))
                    },
                };
                let (claim, crate::map_outputs!(opt_secs, opt_nanos)) = hit!(map.iterate(arms).await);
                let s = or_miss!(opt_secs.map(|(_, v)| v));
                let n = or_miss!(opt_nanos.map(|(_, v)| v));
                Ok(Probe::Hit((claim, std::time::UNIX_EPOCH + core::time::Duration::new(s, n))))
            }).await
        }
    }
}

// -------------------------------------------------------------------------
// Map-collected types: BTreeMap, HashMap.
// CollectMap<K, V, C, KeyFn, ValFn> is a single MapArmStack/MapArmStackOwned
// implementation parameterized over a MapCollect<K, V> collection type.
// -------------------------------------------------------------------------

mod collect_map {
    use super::*;
    use crate::borrow::{MapKeyProbe, VC, VP};
    use crate::owned::{self as owned_, MapKeyProbeOwned};
    use core::future::Future;
    use core::marker::PhantomData;
    use core::pin::Pin;
    use core::task::{Context, Poll};

    pub(super) trait MapCollect<K, V>: Sized {
        fn new_empty() -> Self;
        fn insert_entry(&mut self, k: K, v: V);
    }

    pub(super) struct CollectMap<K, V, C, KeyFn, ValFn> {
        key_fn: KeyFn,
        val_fn: ValFn,
        pending_key: Option<K>,
        out: C,
        _phantom: PhantomData<fn() -> V>,
    }

    impl<K, V, C: MapCollect<K, V>, KeyFn, ValFn> CollectMap<K, V, C, KeyFn, ValFn> {
        pub(super) fn new(key_fn: KeyFn, val_fn: ValFn) -> Self {
            Self {
                key_fn,
                val_fn,
                pending_key: None,
                out: C::new_empty(),
                _phantom: PhantomData,
            }
        }
    }

    impl<'de, KP, K, V, C, KeyFn, KeyFut, ValFn, ValFut> crate::MapArmStack<'de, KP>
        for CollectMap<K, V, C, KeyFn, ValFn>
    where
        KP: MapKeyProbe<'de>,
        C: MapCollect<K, V>,
        KeyFn: FnMut(KP) -> KeyFut,
        KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
        ValFn: FnMut(VP<'de, KP>, K) -> ValFut,
        ValFut: Future<Output = Result<Probe<(VC<'de, KP>, (K, V))>, KP::Error>>,
    {
        const SIZE: usize = 1;
        const FIELD_COUNT: usize = 1;
        type Dynamic = crate::True;
        type Outputs = C;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        }

        type RaceState = KeyFut;
        fn init_race(&mut self, kp: KP, _: usize, _: usize) -> KeyFut {
            (self.key_fn)(kp)
        }
        #[allow(clippy::type_complexity)]
        fn poll_race_one(
            &mut self,
            state: Pin<&mut KeyFut>,
            _: usize,
            cx: &mut Context<'_>,
        ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            match state.poll(cx) {
                Poll::Ready(Ok(Probe::Hit((kc, k)))) => {
                    self.pending_key = Some(k);
                    Poll::Ready(Ok(Probe::Hit((0, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        }

        type DispatchState = ValFut;
        fn init_dispatch(&mut self, _: usize, vp: VP<'de, KP>) -> ValFut {
            let k = self
                .pending_key
                .take()
                .expect("dispatch without pending key");
            (self.val_fn)(vp, k)
        }
        #[allow(clippy::type_complexity)]
        fn poll_dispatch(
            &mut self,
            state: Pin<&mut ValFut>,
            cx: &mut Context<'_>,
        ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
            match state.poll(cx) {
                Poll::Ready(Ok(Probe::Hit((vc, (k, v))))) => {
                    self.out.insert_entry(k, v);
                    Poll::Ready(Ok(Probe::Hit((vc, ()))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        }

        fn take_outputs(&mut self) -> C {
            core::mem::replace(&mut self.out, C::new_empty())
        }
    }

    impl<KP, K, V, C, KeyFn, KeyFut, ValFn, ValFut> crate::MapArmStackOwned<KP>
        for CollectMap<K, V, C, KeyFn, ValFn>
    where
        KP: MapKeyProbeOwned,
        C: MapCollect<K, V>,
        KeyFn: FnMut(KP) -> KeyFut,
        KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
        ValFn: FnMut(owned_::VP<KP>, K) -> ValFut,
        ValFut: Future<Output = Result<Probe<(owned_::VC<KP>, (K, V))>, KP::Error>>,
    {
        const SIZE: usize = 1;
        const FIELD_COUNT: usize = 1;
        type Dynamic = crate::True;
        type Outputs = C;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        }

        type RaceState = KeyFut;
        fn init_race(&mut self, kp: KP, _: usize, _: usize) -> KeyFut {
            (self.key_fn)(kp)
        }
        #[allow(clippy::type_complexity)]
        fn poll_race_one(
            &mut self,
            state: Pin<&mut KeyFut>,
            _: usize,
            cx: &mut Context<'_>,
        ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            match state.poll(cx) {
                Poll::Ready(Ok(Probe::Hit((kc, k)))) => {
                    self.pending_key = Some(k);
                    Poll::Ready(Ok(Probe::Hit((0, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        }

        type DispatchState = ValFut;
        fn init_dispatch(&mut self, _: usize, vp: owned_::VP<KP>) -> ValFut {
            let k = self
                .pending_key
                .take()
                .expect("dispatch without pending key");
            (self.val_fn)(vp, k)
        }
        #[allow(clippy::type_complexity)]
        fn poll_dispatch(
            &mut self,
            state: Pin<&mut ValFut>,
            cx: &mut Context<'_>,
        ) -> Poll<Result<Probe<(owned_::VC<KP>, ())>, KP::Error>> {
            match state.poll(cx) {
                Poll::Ready(Ok(Probe::Hit((vc, (k, v))))) => {
                    self.out.insert_entry(k, v);
                    Poll::Ready(Ok(Probe::Hit((vc, ()))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        }

        fn take_outputs(&mut self) -> C {
            core::mem::replace(&mut self.out, C::new_empty())
        }
    }
}

#[cfg(feature = "alloc")]
mod btreemap_impls {
    use super::collect_map::{CollectMap, MapCollect};
    use super::*;
    use crate::borrow::{KP, MapKeyProbe, VP};
    use crate::owned::{self as owned_, MapKeyProbeOwned, MapValueProbeOwned};
    use alloc::collections::BTreeMap;

    impl<K: Ord, V> MapCollect<K, V> for BTreeMap<K, V> {
        fn new_empty() -> Self {
            BTreeMap::new()
        }
        fn insert_entry(&mut self, k: K, v: V) {
            self.insert(k, v);
        }
    }

    impl<'de, D, K, V> Deserialize<'de, D> for BTreeMap<K, V>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de, <KP<'de, D> as MapKeyProbe<'de>>::KeySubDeserializer, Extra = ()> + Ord,
        V: Deserialize<
                'de,
                <crate::borrow::VP2<'de, D> as crate::MapValueProbe<'de>>::ValueSubDeserializer,
                Extra = (),
            >,
    {
        type Extra = ();
        async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async move {
                let map = hit!(e.deserialize_map().await);
                let (claim, out) = hit!(
                    map.iterate_dyn(CollectMap::<K, V, BTreeMap<K, V>, _, _>::new(
                        |kp: KP<'de, D>| kp.deserialize_key::<K>(()),
                        |vp: VP<'de, KP<'de, D>>, k| async move {
                            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
                            Ok(Probe::Hit((vc, (k, v))))
                        },
                    ))
                    .await
                );
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    impl<D, K, V> DeserializeOwned<D> for BTreeMap<K, V>
    where
        D: DeserializerOwned,
        K: DeserializeOwned<<owned_::KP<D> as MapKeyProbeOwned>::KeySubDeserializer, Extra = ()>
            + Ord,
        V: DeserializeOwned<
                <owned_::VP2<D> as MapValueProbeOwned>::ValueSubDeserializer,
                Extra = (),
            >,
    {
        type Extra = ();
        async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async move {
                let map = hit!(e.deserialize_map().await);
                let (claim, out) = hit!(
                    map.iterate_dyn(CollectMap::<K, V, BTreeMap<K, V>, _, _>::new(
                        |kp: owned_::KP<D>| kp.deserialize_key::<K>(()),
                        |vp: owned_::VP<owned_::KP<D>>, k| async move {
                            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
                            Ok(Probe::Hit((vc, (k, v))))
                        },
                    ))
                    .await
                );
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }
}

#[cfg(feature = "std")]
mod hashmap_impls {
    extern crate std;
    use super::collect_map::{CollectMap, MapCollect};
    use super::*;
    use crate::borrow::{KP, MapKeyProbe, VP};
    use crate::owned::{self as owned_, MapKeyProbeOwned, MapValueProbeOwned};
    use core::hash::{BuildHasher, Hash};
    use std::collections::HashMap;

    impl<K: Eq + Hash, V, S: BuildHasher + Default> MapCollect<K, V> for HashMap<K, V, S> {
        fn new_empty() -> Self {
            HashMap::with_hasher(S::default())
        }
        fn insert_entry(&mut self, k: K, v: V) {
            self.insert(k, v);
        }
    }

    impl<'de, D, K, V, S> Deserialize<'de, D> for HashMap<K, V, S>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de, <KP<'de, D> as MapKeyProbe<'de>>::KeySubDeserializer, Extra = ()>
            + Eq
            + Hash,
        V: Deserialize<
                'de,
                <crate::borrow::VP2<'de, D> as crate::MapValueProbe<'de>>::ValueSubDeserializer,
                Extra = (),
            >,
        S: BuildHasher + Default,
    {
        type Extra = ();
        async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async move {
                let map = hit!(e.deserialize_map().await);
                let (claim, out) = hit!(
                    map.iterate_dyn(CollectMap::<K, V, HashMap<K, V, S>, _, _>::new(
                        |kp: KP<'de, D>| kp.deserialize_key::<K>(()),
                        |vp: VP<'de, KP<'de, D>>, k| async move {
                            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
                            Ok(Probe::Hit((vc, (k, v))))
                        },
                    ))
                    .await
                );
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    impl<D, K, V, S> DeserializeOwned<D> for HashMap<K, V, S>
    where
        D: DeserializerOwned,
        K: DeserializeOwned<<owned_::KP<D> as MapKeyProbeOwned>::KeySubDeserializer, Extra = ()>
            + Eq
            + Hash,
        V: DeserializeOwned<
                <owned_::VP2<D> as MapValueProbeOwned>::ValueSubDeserializer,
                Extra = (),
            >,
        S: BuildHasher + Default,
    {
        type Extra = ();
        async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
            d.entry(|[e]| async move {
                let map = hit!(e.deserialize_map().await);
                let (claim, out) = hit!(
                    map.iterate_dyn(CollectMap::<K, V, HashMap<K, V, S>, _, _>::new(
                        |kp: owned_::KP<D>| kp.deserialize_key::<K>(()),
                        |vp: owned_::VP<owned_::KP<D>>, k| async move {
                            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
                            Ok(Probe::Hit((vc, (k, v))))
                        },
                    ))
                    .await
                );
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }
}

// -------------------------------------------------------------------------
// IP / Socket addresses — parsed from a string via FromStr.
// Borrow path tries zero-copy str first, falls back to chunked + fixed-cap buf.
// Owned path uses chunks only.
// -------------------------------------------------------------------------

macro_rules! impl_from_str {
    ($ty:ty, $max_len:expr) => {
        impl<'de, D> Deserialize<'de, D> for $ty
        where
            D: Deserializer<'de>,
        {
            type Extra = ();
            async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e1, e2]| async move {
                    select_probe! {
                        async move {
                            let (claim, s) = hit!(e1.deserialize_str().await);
                            let v = or_miss!(s.parse::<$ty>().ok());
                            Ok(Probe::Hit((claim, v)))
                        },
                        async move {
                            let mut chunks = hit!(e2.deserialize_str_chunks().await);
                            let mut buf = arrayvec::ArrayString::<$max_len>::new();
                            let mut overflow = false;
                            let claim = loop {
                                match chunks.next_str(|s| {
                                    if !overflow && buf.try_push_str(s).is_err() {
                                        overflow = true;
                                    }
                                }).await? {
                                    Chunk::Data((c, ())) => chunks = c,
                                    Chunk::Done(claim) => break claim,
                                }
                            };
                            if overflow { return Ok(Probe::Miss); }
                            let v = or_miss!(buf.as_str().parse::<$ty>().ok());
                            Ok(Probe::Hit((claim, v)))
                        }
                    }
                })
                .await
            }
        }

        impl<D> DeserializeOwned<D> for $ty
        where
            D: DeserializerOwned,
        {
            type Extra = ();
            async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async move {
                    let mut chunks = hit!(e.deserialize_str_chunks().await);
                    let mut buf = arrayvec::ArrayString::<$max_len>::new();
                    let mut overflow = false;
                    let claim = loop {
                        match chunks
                            .next_str(|s| {
                                if !overflow && buf.try_push_str(s).is_err() {
                                    overflow = true;
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
