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
// Custom MapArmStack collector with SIZE = 1 that overrides race_keys /
// dispatch_value directly; the lower-level state-machine methods are
// unreachable since the format only ever calls race_keys/dispatch_value.
// -------------------------------------------------------------------------

#[cfg(feature = "alloc")]
mod btreemap_impls {
    use super::*;
    use crate::borrow::{KP, MapKeyProbe, VC, VP};
    use crate::owned::{self as owned_, MapKeyProbeOwned, MapValueProbeOwned};
    use alloc::collections::BTreeMap;
    use core::pin::Pin;
    use core::task::{Context, Poll};

    pub struct CollectBTreeMap<K, V> {
        pending_key: Option<K>,
        out: BTreeMap<K, V>,
    }

    impl<K: Ord, V> CollectBTreeMap<K, V> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: BTreeMap::new(),
            }
        }
    }

    impl<'de, KP, K, V> crate::MapArmStack<'de, KP> for CollectBTreeMap<K, V>
    where
        KP: MapKeyProbe<'de>,
        K: Deserialize<'de, KP::KeySubDeserializer, Extra = ()> + Ord,
        V: Deserialize<
                'de,
                <VP<'de, KP> as crate::MapValueProbe<'de>>::ValueSubDeserializer,
                Extra = (),
            >,
    {
        const SIZE: usize = 1;
        type Outputs = BTreeMap<K, V>;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        }

        type RaceState = ();
        fn init_race(&mut self, _: KP, _: usize) -> () {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _: Pin<&mut ()>,
            _: usize,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _: usize, _: VP<'de, KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _: Pin<&mut ()>,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }

        async fn dispatch_value(
            &mut self,
            _: usize,
            vp: VP<'de, KP>,
        ) -> Result<Probe<(VC<'de, KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch without pending key");
            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
            self.out.insert(k, v);
            Ok(Probe::Hit((vc, ())))
        }

        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::take(&mut self.out)
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
                let (claim, out) = hit!(map.iterate(CollectBTreeMap::<K, V>::new()).await);
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    // Owned family ---

    pub struct CollectBTreeMapOwned<K, V> {
        pending_key: Option<K>,
        out: BTreeMap<K, V>,
    }

    impl<K: Ord, V> CollectBTreeMapOwned<K, V> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: BTreeMap::new(),
            }
        }
    }

    impl<KP, K, V> crate::MapArmStackOwned<KP> for CollectBTreeMapOwned<K, V>
    where
        KP: MapKeyProbeOwned,
        K: DeserializeOwned<KP::KeySubDeserializer, Extra = ()> + Ord,
        V: DeserializeOwned<
                <owned_::VP<KP> as MapValueProbeOwned>::ValueSubDeserializer,
                Extra = (),
            >,
    {
        const SIZE: usize = 1;
        type Outputs = BTreeMap<K, V>;

        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        }

        type RaceState = ();
        fn init_race(&mut self, _: KP, _: usize) -> () {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _: Pin<&mut ()>,
            _: usize,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _: usize, _: owned_::VP<KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _: Pin<&mut ()>,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(owned_::VC<KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }

        async fn dispatch_value(
            &mut self,
            _: usize,
            vp: owned_::VP<KP>,
        ) -> Result<Probe<(owned_::VC<KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch without pending key");
            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
            self.out.insert(k, v);
            Ok(Probe::Hit((vc, ())))
        }

        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::take(&mut self.out)
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
                let (claim, out) = hit!(map.iterate(CollectBTreeMapOwned::<K, V>::new()).await);
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }
}

#[cfg(feature = "std")]
mod hashmap_impls {
    extern crate std;
    use super::*;
    use crate::borrow::{KP, MapKeyProbe, VC, VP};
    use crate::owned::{self as owned_, MapKeyProbeOwned, MapValueProbeOwned};
    use core::hash::{BuildHasher, Hash};
    use core::pin::Pin;
    use core::task::{Context, Poll};
    use std::collections::HashMap;

    pub struct CollectHashMap<K, V, S> {
        pending_key: Option<K>,
        out: HashMap<K, V, S>,
    }

    impl<K, V, S: BuildHasher + Default> CollectHashMap<K, V, S> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: HashMap::with_hasher(S::default()),
            }
        }
    }

    impl<'de, KP, K, V, S> crate::MapArmStack<'de, KP> for CollectHashMap<K, V, S>
    where
        KP: MapKeyProbe<'de>,
        K: Deserialize<'de, KP::KeySubDeserializer, Extra = ()> + Eq + Hash,
        V: Deserialize<
                'de,
                <VP<'de, KP> as crate::MapValueProbe<'de>>::ValueSubDeserializer,
                Extra = (),
            >,
        S: BuildHasher + Default,
    {
        const SIZE: usize = 1;
        type Outputs = HashMap<K, V, S>;
        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        }

        type RaceState = ();
        fn init_race(&mut self, _: KP, _: usize) -> () {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _: Pin<&mut ()>,
            _: usize,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _: usize, _: VP<'de, KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _: Pin<&mut ()>,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }
        async fn dispatch_value(
            &mut self,
            _: usize,
            vp: VP<'de, KP>,
        ) -> Result<Probe<(VC<'de, KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch without pending key");
            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
            self.out.insert(k, v);
            Ok(Probe::Hit((vc, ())))
        }
        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::replace(&mut self.out, HashMap::with_hasher(S::default()))
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
                let (claim, out) = hit!(map.iterate(CollectHashMap::<K, V, S>::new()).await);
                Ok(Probe::Hit((claim, out)))
            })
            .await
        }
    }

    // Owned ---

    pub struct CollectHashMapOwned<K, V, S> {
        pending_key: Option<K>,
        out: HashMap<K, V, S>,
    }

    impl<K, V, S: BuildHasher + Default> CollectHashMapOwned<K, V, S> {
        fn new() -> Self {
            Self {
                pending_key: None,
                out: HashMap::with_hasher(S::default()),
            }
        }
    }

    impl<KP, K, V, S> crate::MapArmStackOwned<KP> for CollectHashMapOwned<K, V, S>
    where
        KP: MapKeyProbeOwned,
        K: DeserializeOwned<KP::KeySubDeserializer, Extra = ()> + Eq + Hash,
        V: DeserializeOwned<
                <owned_::VP<KP> as MapValueProbeOwned>::ValueSubDeserializer,
                Extra = (),
            >,
        S: BuildHasher + Default,
    {
        const SIZE: usize = 1;
        type Outputs = HashMap<K, V, S>;
        fn unsatisfied_count(&self) -> usize {
            0
        }
        fn open_count(&self) -> usize {
            1
        }

        type RaceState = ();
        fn init_race(&mut self, _: KP, _: usize) -> () {
            unreachable!()
        }
        fn poll_race_one(
            &mut self,
            _: Pin<&mut ()>,
            _: usize,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
            unreachable!()
        }

        type DispatchState = ();
        fn init_dispatch(&mut self, _: usize, _: owned_::VP<KP>) -> () {
            unreachable!()
        }
        fn poll_dispatch(
            &mut self,
            _: Pin<&mut ()>,
            _: &mut Context<'_>,
        ) -> Poll<Result<Probe<(owned_::VC<KP>, ())>, KP::Error>> {
            unreachable!()
        }

        async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
            let (kc, k) = hit!(kp.deserialize_key::<K>(()).await);
            self.pending_key = Some(k);
            Ok(Probe::Hit((0, kc)))
        }
        async fn dispatch_value(
            &mut self,
            _: usize,
            vp: owned_::VP<KP>,
        ) -> Result<Probe<(owned_::VC<KP>, ())>, KP::Error> {
            let k = self
                .pending_key
                .take()
                .expect("dispatch without pending key");
            let (vc, v) = hit!(vp.deserialize_value::<V>(()).await);
            self.out.insert(k, v);
            Ok(Probe::Hit((vc, ())))
        }
        fn take_outputs(&mut self) -> Self::Outputs {
            core::mem::replace(&mut self.out, HashMap::with_hasher(S::default()))
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
                let (claim, out) = hit!(map.iterate(CollectHashMapOwned::<K, V, S>::new()).await);
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
