use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};

pub use crate::map_arm::{
    ArmState, ConcatDispatchState, ConcatRaceState, DetectDuplicatesOwned, MapArm, MapArmBase,
    MapArmSlot, NextKey, SlotDispatchState, SlotRaceState, StackConcat, TagDispatchState,
    TagInjectingStackOwned, TagRaceState, VirtualArmSlot, WrapperDispatchState, WrapperRaceState,
};
use crate::map_arm::{
    ConcatDispatchProj, SlotDispatchProj, TagDispatchProj, WrapperDispatchProj, poll_key_slot,
};
use crate::{Chunk, DeserializeError, Probe, hit};

// ===========================================================================
// Owned family - parallel trait hierarchy for formats that cannot deliver
// borrowed-for-`'de` slices (e.g. chunked/streaming input).
//
// Mirrors the borrow family, minus `deserialize_str` / `deserialize_bytes`.
// The `'s` lifetime is the deserializer's *session* - the region over which
// the deserializer and its `Claim`s remain valid. Analogous to the borrow
// family's `'de`, but with no zero-copy borrow methods.
//
// `Probe`, `Chunk`, `StrAccess`, `BytesAccess`, `hit!`, `DeserializeError`,
// and `select_probe!` are shared between families.
//
// The two families are independent: no supertrait relationship, no blanket
// impls. A format implements whichever family (or both) it can support.
//
// # Parallel scanning - deadlock hazard
//
// The owned family reads from a streaming source where data arrives
// incrementally.  Every reader (entry handle, map/seq accessor, or
// str/bytes chunk accessor) shares the same underlying buffer and
// advances through it cooperatively via `fork`.
//
// **For callers:**  You must not read one forked handle to completion
// and then decide what to do with another - that will deadlock.  The
// first handle may block waiting for more data to arrive, but the
// buffer cannot advance until *all* sibling handles have consumed the
// current chunk.  Instead, race all handles concurrently (e.g. via
// `select_probe!`).  This is safe: forked handles never interfere with
// each other, and every reader is polled and paused as new data becomes
// available, as long as all of them are making forward progress together.
//
// **For implementers:**  Your `fork` implementation must ensure that
// the underlying buffer does not advance past data that any live handle
// still needs to read.  Every forked reader must be independently
// resumable: when new data arrives, all suspended readers must be woken
// and given the opportunity to process it.  You must never require one
// reader to finish before another can make progress - doing so creates
// a circular dependency that deadlocks the single-threaded executor.
// The `shared_buf` module provides a reference implementation of this
// contract.
// ===========================================================================

/// Owned counterpart to [`Deserialize`](crate::Deserialize).
///
/// The `Extra` type parameter is side-channel context passed into
/// [`EntryOwned::deserialize_value`], [`SeqEntryOwned::get`],
/// site.  Defaults to `()` for types that need no extra context.
pub trait DeserializeOwned<Extra = ()>: Sized {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>;
}

/// Owned counterpart to [`Deserializer`](crate::Deserializer).
///
/// # Streaming and parallel scanning
///
/// Unlike the borrow family, the owned family reads from a streaming source
/// where data arrives incrementally.  When `entry` passes `N` handles to
/// the closure (or when you `fork` an accessor), the resulting readers share
/// the same underlying buffer and advance through it cooperatively.
///
/// **You must drive all forked readers concurrently** - typically via
/// [`select_probe!`](crate::select_probe).  Sequentially awaiting one reader
/// to completion before polling another will deadlock: the first reader may
/// block waiting for buffer data that cannot arrive until all sibling readers
/// have consumed the current chunk.
///
/// This is safe to do: forked readers are guaranteed not to interfere with
/// each other, and every reader is automatically suspended and resumed as
/// new data becomes available, provided all readers are being polled.
pub trait DeserializerOwned: Sized {
    type Error: DeserializeError;

    type Claim;
    type EntryClaim;
    type Entry: EntryOwned<Claim = Self::EntryClaim, Error = Self::Error>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>;
}

/// Owned counterpart to [`Entry`](crate::Entry). Drops `deserialize_str` / `deserialize_bytes`
/// (the borrow-only methods); strings and bytes must be read via
/// [`StrAccessOwned`] / [`BytesAccessOwned`].
///
/// # For implementors - `fork` contract
///
/// [`EntryOwned::fork`] creates a sibling handle for the same token slot.
/// Your implementation must guarantee that:
///
/// 1. The underlying buffer does not advance past data any live handle still
///    needs.  All forked handles read the same data independently.
/// 2. When new data arrives, **every** suspended reader is woken and given the
///    chance to process it.
/// 3. No reader is required to finish before another can make progress.
///    Violating this creates a circular wait that deadlocks the
///    single-threaded executor.
pub trait EntryOwned: Sized {
    type Error: DeserializeError;
    type Claim;

    type StrChunks: StrAccessOwned<Claim = Self::Claim, Error = Self::Error>;
    type BytesChunks: BytesAccessOwned<Claim = Self::Claim, Error = Self::Error>;
    type Map: MapAccessOwned<MapClaim = Self::Claim, Error = Self::Error>;
    type Seq: SeqAccessOwned<SeqClaim = Self::Claim, Error = Self::Error>;

    async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error>;
    async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error>;
    async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error>;
    async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error>;
    async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error>;
    async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error>;
    async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error>;
    async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error>;
    async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error>;
    async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error>;
    async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error>;
    async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error>;
    async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error>;
    async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error>;

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error>;
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error>;
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error>;
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error>;

    async fn deserialize_option<T: DeserializeOwned<Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>;

    /// Probe for a null token.
    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error>;

    /// Delegate to `T::deserialize` from this entry handle, forwarding `extra`.
    async fn deserialize_value<T: DeserializeOwned<Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

    /// Fork a sibling entry handle for the same item slot.
    ///
    /// Both `self` and the returned handle refer to the same slot.
    /// Whichever resolves a probe first claims the slot; the other
    /// becomes inert and may be dropped without advancing the stream.
    fn fork(&mut self) -> Self;

    /// Consume and discard the current token regardless of its type.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

/// Owned counterpart to [`StrAccess`]. Takes `self` by value and a sync
/// closure that borrows the chunk `&str`. The closure maps the short-lived
/// borrow to an owned `R`; the accessor is handed back alongside `R` on
/// `Data` and the claim emerges on `Done`.
///
/// **Callers:** forked accessors must be driven concurrently (e.g. via
/// `select_probe!`).  Awaiting one fork to `Done` before polling another
/// will deadlock - the buffer cannot advance until all forks have consumed
/// the current chunk.
pub trait StrAccessOwned: Sized {
    type Claim;
    type Error: DeserializeError;

    /// Fork a sibling accessor at the same read position.
    ///
    /// Both `self` and the returned accessor are independent: each must be
    /// driven to `Done` (or dropped) before the underlying buffer can
    /// advance.  Neither reader replays data - both continue from the
    /// current position, consuming the same remaining chunks.
    ///
    /// # For implementors
    ///
    /// The forked accessor must hold the buffer open at its current position.
    /// All forks are woken when new data arrives; none may require another
    /// to complete first.
    fn fork(&mut self) -> Self;

    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`BytesAccess`]. Same closure pattern as
/// [`StrAccessOwned`].
///
/// **Callers:** forked accessors must be driven concurrently.
/// See [`StrAccessOwned`] for the deadlock hazard.
pub trait BytesAccessOwned: Sized {
    type Claim;
    type Error: DeserializeError;

    /// Fork a sibling accessor at the same read position.
    ///
    /// Both `self` and the returned accessor are independent: each must be
    /// driven to `Done` (or dropped) before the underlying buffer can
    /// advance.  Neither reader replays data - both continue from the
    /// current position, consuming the same remaining chunks.
    ///
    /// # For implementors
    ///
    /// Same contract as [`StrAccessOwned::fork`].
    fn fork(&mut self) -> Self;

    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

/// Owned counterpart to [`SeqAccess`](crate::SeqAccess).
///
/// **Callers:** forked sequence accessors must be driven concurrently.
/// See [`DeserializerOwned`] for the deadlock hazard.
pub trait SeqAccessOwned: Sized {
    type Error: DeserializeError;
    type SeqClaim;
    type ElemClaim;
    type Elem: SeqEntryOwned<Claim = Self::ElemClaim, Error = Self::Error>;

    /// Fork a sibling accessor at the same sequence position.
    ///
    /// # For implementors
    ///
    /// Same contract as [`StrAccessOwned::fork`].
    fn fork(&mut self) -> Self;

    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>;
}

/// Owned counterpart to [`SeqEntry`](crate::SeqEntry).
pub trait SeqEntryOwned {
    type Error: DeserializeError;
    type Claim;

    /// Fork a sibling element handle for the same sequence slot.
    fn fork(&mut self) -> Self
    where
        Self: Sized;

    async fn get<T: DeserializeOwned<Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

    /// Consume and discard the element without deserializing it.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ===========================================================================
// Map Access
// ===========================================================================

// ---------------------------------------------------------------------------
// Claim chain: MapKeyProbeOwned → MapKeyClaimOwned → MapValueProbeOwned → MapValueClaimOwned → next MapKeyProbeOwned
// ---------------------------------------------------------------------------

/// A key probe for a single map key. Forkable for racing multiple arms.
pub trait MapKeyProbeOwned: Sized {
    type Error: DeserializeError;
    type KeyClaim: MapKeyClaimOwned<Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn deserialize_key<K: DeserializeOwned<Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>;
}

/// Proof that a key was consumed. Converts into a value probe.
///
/// Separate type from `ValueClaim` so you can't use a key claim to
/// advance past a value without consuming it.
pub trait MapKeyClaimOwned: Sized {
    type Error: DeserializeError;
    type MapClaim;
    type ValueProbe: MapValueProbeOwned<MapClaim = Self::MapClaim, Error = Self::Error>;

    /// Consume this key claim and produce a value probe for the corresponding
    /// map value. Format-specific (e.g. JSON reads `:` and the value start token).
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error>;
}

/// A value probe that can deserialize a value or skip it.
pub trait MapValueProbeOwned: Sized {
    type Error: DeserializeError;
    type MapClaim;
    type ValueClaim: MapValueClaimOwned<MapClaim = Self::MapClaim, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn deserialize_value<V: DeserializeOwned<Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>;

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error>;
}

/// Proof that a value was consumed. Advances to the next key or ends the map.
pub trait MapValueClaimOwned: Sized {
    type Error: DeserializeError;
    type KeyProbe: MapKeyProbeOwned<Error = Self::Error>;
    type MapClaim;

    /// Consume this value claim and advance the map.
    ///
    /// `unsatisfied` is the number of arms that still require a value (required
    /// fields not yet matched). `open` is the number of arms still willing to
    /// run, including both unsatisfied required-field arms and virtual arms
    /// (skip-unknown, dup-detect, tag-inject) that are always active. When
    /// `open == 0`, the format may skip remaining entries and close the map
    /// early. When `unsatisfied == 0` but `open > 0`, the format should
    /// continue iterating to let virtual arms consume or inspect remaining keys.
    async fn next_key(
        self,
        unsatisfied: usize,
        open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Arm stack - tuple-stack of closure-based map field arms
// ---------------------------------------------------------------------------

/// Shorthand for the key probe type reachable from a DeserializerOwned.
pub type KP<D> = <<<D as DeserializerOwned>::Entry as EntryOwned>::Map as MapAccessOwned>::KeyProbe;
/// Shorthand for the value claim type reachable from a key probe type.
pub type VC<KP> = <<<KP as MapKeyProbeOwned>::KeyClaim as MapKeyClaimOwned>::ValueProbe as MapValueProbeOwned>::ValueClaim;
/// Shorthand for the value probe type reachable from a key probe type.
pub type VP<KP> = <<KP as MapKeyProbeOwned>::KeyClaim as MapKeyClaimOwned>::ValueProbe;
/// Shorthand for the value probe type reachable from a DeserializerOwned.
pub type VP2<D> = <<KP<D> as MapKeyProbeOwned>::KeyClaim as MapKeyClaimOwned>::ValueProbe;

/// A left-nested tuple stack of [`MapArmSlot`]s: `((MapArmBase, Slot0), Slot1)`.
///
/// The map impl drives the iteration loop. Each round it calls the
/// [`race_keys`] free function which forks the key probe, creates per-arm
/// key futures via [`init_race`](MapArmStackOwned::init_race), and polls them
/// flat via [`poll_race_one`](MapArmStackOwned::poll_race_one). On a hit,
/// [`dispatch_value`] converts the key claim to a value probe and polls
/// the winning arm's value callback via [`poll_dispatch`](MapArmStackOwned::poll_dispatch).
///
/// All poll methods are sync - recursion through `(Rest, Slot)` tuples is
/// ordinary call-stack recursion, not nested async state machines. This
/// avoids the compiler recursion depth limits that `async fn` nesting causes.
pub trait MapArmStackOwned<KP: MapKeyProbeOwned>: Sized {
    const SIZE: usize;

    /// Left-nested tuple of `Option<(K, V)>` for each arm.
    type Outputs;

    /// Number of arms that still require a value (required fields not yet matched).
    /// Virtual arms (skip-unknown, dup-detect, tag-inject) are excluded.
    fn unsatisfied_count(&self) -> usize;

    /// Number of arms still willing to run, including both unsatisfied
    /// required-field arms and always-active virtual arms.
    fn open_count(&self) -> usize;

    // --- race_keys (init/poll) ---
    //
    // The init/poll API is the primary interface. `(Rest, Slot)` and
    // `StackConcat` implement these directly, producing flat poll loops
    // instead of nested async state machines.
    //
    // The `async fn race_keys` / `dispatch_value` default methods wrap
    // init/poll for convenience. Wrappers (SkipUnknownOwned, DetectDuplicatesOwned,
    // TagInjectingStackOwned) override the async methods to add their virtual
    // arms while delegating inner arms via the init/poll path.

    /// Pinned state holding per-arm key futures for one round of racing.
    type RaceState;

    /// Fork `kp` for each unsatisfied arm, call each arm's key callback to
    /// create its future, and return the combined pinned state.
    fn init_race(&mut self, kp: KP) -> Self::RaceState;

    /// Poll a single arm's key future. **Sync** - recursion through
    /// `(Rest, Slot)` is ordinary function calls, not async nesting.
    #[allow(clippy::type_complexity)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>>;

    // --- dispatch_value (init/poll) ---

    /// Pinned state for dispatching the winning arm's value callback.
    /// Only one arm is dispatched per call (the race winner).
    type DispatchState;

    /// Create the dispatch state for the winning arm at `arm_index`.
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<KP>) -> Self::DispatchState;

    /// Poll the dispatch future.
    #[allow(clippy::type_complexity)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<KP>, ())>, KP::Error>>;

    // --- Provided async methods (convenience / wrapper override points) ---

    /// Race all unsatisfied arms' key callbacks against the given key probe.
    ///
    /// Default implementation wraps `init_race` + `poll_race_one` in a flat
    /// poll loop. Wrappers that add virtual arms override this.
    async fn race_keys(&mut self, kp: KP) -> Result<Probe<(usize, KP::KeyClaim)>, KP::Error> {
        if Self::SIZE == 0 {
            return Ok(Probe::Miss);
        }
        let mut race_state = core::pin::pin!(self.init_race(kp));
        core::future::poll_fn(|cx| {
            let mut all_miss = true;
            for i in 0..Self::SIZE {
                match self.poll_race_one(race_state.as_mut(), i, cx) {
                    Poll::Ready(Ok(Probe::Hit(v))) => return Poll::Ready(Ok(Probe::Hit(v))),
                    Poll::Ready(Err(e)) => return Poll::Ready(Err(e)),
                    Poll::Ready(Ok(Probe::Miss)) => {}
                    Poll::Pending => {
                        all_miss = false;
                    }
                }
            }
            if all_miss {
                return Poll::Ready(Ok(Probe::Miss));
            }
            Poll::Pending
        })
        .await
    }

    /// Dispatch the value probe to the arm at `arm_index`.
    ///
    /// Default implementation wraps `init_dispatch` + `poll_dispatch`.
    /// Wrappers that add virtual arms override this.
    async fn dispatch_value(
        &mut self,
        arm_index: usize,
        vp: VP<KP>,
    ) -> Result<Probe<(VC<KP>, ())>, KP::Error> {
        let dispatch_state = self.init_dispatch(arm_index, vp);
        let mut dispatch_state = core::pin::pin!(dispatch_state);
        core::future::poll_fn(|cx| self.poll_dispatch(dispatch_state.as_mut(), cx)).await
    }

    /// Extract all outputs.
    fn take_outputs(&mut self) -> Self::Outputs;
}

// --- MapArmBase impls ---

impl<KP: MapKeyProbeOwned> MapArmStackOwned<KP> for MapArmBase {
    const SIZE: usize = 0;
    type Outputs = ();

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        0
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        0
    }

    type RaceState = ();

    #[inline(always)]
    fn init_race(&mut self, _kp: KP) {}
    #[inline(always)]
    fn poll_race_one(
        &mut self,
        _state: Pin<&mut ()>,
        _arm_index: usize,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        unreachable!("poll_race_one called on MapArmBase (SIZE=0)")
    }

    type DispatchState = core::convert::Infallible;

    #[inline(always)]
    fn init_dispatch(&mut self, _arm_index: usize, _vp: VP<KP>) -> Self::DispatchState {
        unreachable!("init_dispatch called on MapArmBase")
    }
    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        _state: Pin<&mut Self::DispatchState>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<KP>, ())>, KP::Error>> {
        unreachable!("poll_dispatch called on MapArmBase")
    }

    #[inline(always)]
    fn take_outputs(&mut self) {}
}

// --- Recursive (Rest, Slot) impl ---
//
// `init_race` forks the key probe and creates per-arm key futures, storing
// them in `SlotRaceState`. `poll_race_one` polls a single arm by index -
// sync recursion through `self.0.poll_race_one()` for rest arms.
//
// `init_dispatch` creates a `SlotDispatchState` enum selecting this arm's
// val future or delegating to rest. `poll_dispatch` polls the active variant.

impl<KP, Rest, K, V, KeyFn, KeyFut, ValFn, ValFut> MapArmStackOwned<KP>
    for (Rest, MapArmSlot<K, V, KeyFn, ValFn>)
where
    KP: MapKeyProbeOwned,
    Rest: MapArmStackOwned<KP>,
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(VP<KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(VC<KP>, (K, V))>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    type Outputs = (Rest::Outputs, Option<(K, V)>);

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count() + if self.1.state.is_done() { 0 } else { 1 }
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.0.open_count() + if self.1.state.is_done() { 0 } else { 1 }
    }

    type RaceState = SlotRaceState<Rest::RaceState, KeyFut>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        // Always fork - fork() is cheap (shared buffer pointer copy).
        let rest_kp = kp.fork();
        let this_fut = if self.1.state.is_done() {
            None
        } else {
            Some((self.1.key_fn)(kp))
        };
        SlotRaceState {
            rest: self.0.init_race(rest_kp),
            this: this_fut,
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == Self::SIZE - 1 {
            // Poll this arm's key future.
            match poll_key_slot(projected.this, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, k)))) => {
                    self.1.state = ArmState::Key(k);
                    Poll::Ready(Ok(Probe::Hit((Self::SIZE - 1, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            // Delegate to rest - sync recursion.
            self.0.poll_race_one(projected.rest, arm_index, cx)
        }
    }

    type DispatchState = SlotDispatchState<Rest::DispatchState, ValFut>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            let k = match mem::replace(&mut self.1.state, ArmState::Empty) {
                ArmState::Key(k) => k,
                _ => unreachable!("init_dispatch called but arm not in Key state"),
            };
            SlotDispatchState::ThisArm((self.1.val_fn)(vp, k))
        } else {
            SlotDispatchState::Delegated(self.0.init_dispatch(arm_index, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<KP>, ())>, KP::Error>> {
        match state.project() {
            SlotDispatchProj::ThisArm(fut) => fut.poll(cx).map(|r| {
                r.map(|probe| match probe {
                    Probe::Hit((vc, (k, v))) => {
                        self.1.state = ArmState::Done(k, v);
                        Probe::Hit((vc, ()))
                    }
                    Probe::Miss => Probe::Miss,
                })
            }),
            SlotDispatchProj::Delegated(rest_state) => self.0.poll_dispatch(rest_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        let out = match mem::replace(&mut self.1.state, ArmState::Empty) {
            ArmState::Done(k, v) => Some((k, v)),
            _ => None,
        };
        (self.0.take_outputs(), out)
    }
}

/// `(Rest, VirtualArmSlot)` impl - virtual arm at index `Rest::SIZE`.
/// The virtual arm is always active (never satisfied) and produces no output.
impl<KP, Rest, K, KeyFn, KeyFut, ValFn, ValFut> MapArmStackOwned<KP>
    for (Rest, VirtualArmSlot<K, KeyFn, ValFn>)
where
    KP: MapKeyProbeOwned,
    Rest: MapArmStackOwned<KP>,
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(VP<KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(VC<KP>, ())>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    // Virtual arm produces no output - pass through rest's outputs.
    type Outputs = Rest::Outputs;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        // Virtual arm excluded - it doesn't represent a required field.
        self.0.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        // Virtual arm is always active - contributes 1 to open count.
        self.0.open_count() + 1
    }

    type RaceState = SlotRaceState<Rest::RaceState, KeyFut>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        // Virtual arm is always active - always create a future.
        let rest_kp = kp.fork();
        let this_fut = (self.1.key_fn)(kp);
        SlotRaceState {
            rest: self.0.init_race(rest_kp),
            this: Some(this_fut),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == Self::SIZE - 1 {
            match poll_key_slot(projected.this, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, k)))) => {
                    // Store K for dispatch.
                    self.1.pending_key = Some(k);
                    Poll::Ready(Ok(Probe::Hit((Self::SIZE - 1, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            self.0.poll_race_one(projected.rest, arm_index, cx)
        }
    }

    type DispatchState = SlotDispatchState<Rest::DispatchState, ValFut>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            let k = self
                .1
                .pending_key
                .take()
                .expect("init_dispatch on virtual arm without pending key");
            SlotDispatchState::ThisArm((self.1.val_fn)(vp, k))
        } else {
            SlotDispatchState::Delegated(self.0.init_dispatch(arm_index, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<KP>, ())>, KP::Error>> {
        match state.project() {
            SlotDispatchProj::ThisArm(fut) => fut.poll(cx),
            SlotDispatchProj::Delegated(rest_state) => self.0.poll_dispatch(rest_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        // Virtual arm produces no output.
        self.0.take_outputs()
    }
}

// --- SkipUnknownOwned macro ---
//
// Expands `SkipUnknownOwned(arms)` to `(arms, VirtualArmSlot::new(skip_key_fn, skip_val_fn))`.
// The closure types are inferred at the call site - no need to name them.

/// Wraps a [`MapArmStackOwned`] so that unknown map keys are silently consumed.
///
/// Expands to `(arms, VirtualArmSlot::new(...))` with a skip key/value arm.
#[macro_export]
macro_rules! SkipUnknownOwned {
    // 3-arg form: explicit KP/VP types for closure annotations (used by derive).
    ($inner:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbeOwned as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp: $kp| kp.deserialize_key::<$crate::Skip, _>(()),
                |vp: $vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbeOwned as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
    // 1-arg form: types inferred from context (for hand-written code).
    ($inner:expr) => {{
        use $crate::MapKeyProbeOwned as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp| kp.deserialize_key::<$crate::Skip, _>(()),
                |vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbeOwned as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
}

impl<KP, S, const M: usize, KeyFn, KeyFut, SkipFn, SkipFut> MapArmStackOwned<KP>
    for DetectDuplicatesOwned<S, M, KeyFn, SkipFn>
where
    KP: MapKeyProbeOwned,
    S: MapArmStackOwned<KP>,
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::MatchVals<usize>)>, KP::Error>>,
    SkipFn: FnMut(VP<KP>) -> SkipFut,
    SkipFut: Future<Output = Result<VC<KP>, KP::Error>>,
{
    const SIZE: usize = S::SIZE + 1;
    type Outputs = S::Outputs;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.inner.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.inner.open_count() + 1
    }

    type RaceState = WrapperRaceState<S::RaceState, KeyFut>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let dup_kp = kp.fork();
        let dup_fut = (self.key_fn)(dup_kp);
        WrapperRaceState {
            inner: self.inner.init_race(kp),
            virtual_arm: Some(dup_fut),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == Self::SIZE - 1 {
            // Dup arm.
            match poll_key_slot(projected.virtual_arm, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, matched)))) => {
                    self.dup = self.wire_names[matched.0].0;
                    Poll::Ready(Ok(Probe::Hit((Self::SIZE - 1, kc))))
                }
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            self.inner.poll_race_one(projected.inner, arm_index, cx)
        }
    }

    type DispatchState = WrapperDispatchState<S::DispatchState, SkipFut>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            WrapperDispatchState::Virtual((self.skip_fn)(vp))
        } else {
            WrapperDispatchState::Inner(self.inner.init_dispatch(arm_index, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<KP>, ())>, KP::Error>> {
        match state.project() {
            WrapperDispatchProj::Virtual(fut) => {
                // Poll the skip future, then return duplicate_field error.
                fut.poll(cx).map(|r| match r {
                    Ok(_vc) => Err(<KP::Error as crate::DeserializeError>::duplicate_field(
                        self.dup,
                    )),
                    Err(e) => Err(e),
                })
            }
            WrapperDispatchProj::Inner(inner_state) => self.inner.poll_dispatch(inner_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.inner.take_outputs()
    }
}

/// Convenience macro to construct a [`DetectDuplicatesOwned`] with inferred closure types.
///
/// `DetectDuplicatesOwned!(inner, wire_names, KP, VP)` expands to
/// `DetectDuplicatesOwned::new(inner, wire_names, key_fn, skip_fn)` with typed closures.
/// The KP and VP types are needed for closure annotations.
#[macro_export]
macro_rules! DetectDuplicatesOwned {
    ($inner:expr, $wire_names:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbeOwned as _;
        use $crate::MapValueProbeOwned as _;
        let __wn = $wire_names;
        $crate::DetectDuplicatesOwned::new(
            $inner,
            __wn,
            move |kp: $kp| kp.deserialize_key::<$crate::MatchVals<usize>, _>(__wn),
            |vp: $vp| vp.skip(),
        )
    }};
}

impl<'v, KP, S, const N: usize, TagKeyFn, TagKeyFut, TagValFn, TagValFut> MapArmStackOwned<KP>
    for TagInjectingStackOwned<'v, S, N, TagKeyFn, TagValFn>
where
    KP: MapKeyProbeOwned,
    S: MapArmStackOwned<KP>,
    TagKeyFn: FnMut(KP) -> TagKeyFut,
    TagKeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::Match)>, KP::Error>>,
    TagValFn: FnMut(VP<KP>) -> TagValFut,
    TagValFut: Future<Output = Result<Probe<(VC<KP>, crate::MatchVals<usize>)>, KP::Error>>,
{
    // Tag arm at index 0, inner arms at 1..SIZE.
    const SIZE: usize = S::SIZE + 1;
    type Outputs = S::Outputs;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.inner.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.inner.open_count() + 1
    }

    type RaceState = TagRaceState<TagKeyFut, S::RaceState>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let inner_kp = kp.fork();
        let tag_fut = (self.tag_key_fn)(kp);
        TagRaceState {
            tag_fut: Some(tag_fut),
            inner: self.inner.init_race(inner_kp),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index == 0 {
            // Tag arm (highest priority - polled first).
            match poll_key_slot(projected.tag_fut, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, _match)))) => Poll::Ready(Ok(Probe::Hit((0, kc)))),
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
            // Inner arms (indices 1..SIZE → inner indices 0..S::SIZE).
            match self.inner.poll_race_one(projected.inner, arm_index - 1, cx) {
                Poll::Ready(Ok(Probe::Hit((idx, kc)))) => {
                    Poll::Ready(Ok(Probe::Hit((idx + 1, kc))))
                }
                other => other,
            }
        }
    }

    type DispatchState = TagDispatchState<TagValFut, S::DispatchState>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<KP>) -> Self::DispatchState {
        if arm_index == 0 {
            TagDispatchState::Tag((self.tag_val_fn)(vp))
        } else {
            TagDispatchState::Inner(self.inner.init_dispatch(arm_index - 1, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<KP>, ())>, KP::Error>> {
        match state.project() {
            TagDispatchProj::Tag(fut) => fut.poll(cx).map(|r| {
                r.map(|probe| match probe {
                    Probe::Hit((vc, crate::MatchVals(idx))) => {
                        self.tag_value.set(Some(idx));
                        Probe::Hit((vc, ()))
                    }
                    Probe::Miss => Probe::Miss,
                })
            }),
            TagDispatchProj::Inner(inner_state) => self.inner.poll_dispatch(inner_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.inner.take_outputs()
    }
}

/// Convenience macro to construct a [`TagInjectingStackOwned`] with inferred closure types.
///
/// `TagInjectingStackOwned!(inner, tag_field, tag_candidates, tag_value, KP, VP)`
#[macro_export]
macro_rules! TagInjectingStackOwned {
    ($inner:expr, $tag_field:expr, $tag_candidates:expr, $tag_value:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbeOwned as _;
        use $crate::MapValueProbeOwned as _;
        let __tf = $tag_field;
        let __tc = $tag_candidates;
        $crate::TagInjectingStackOwned::new(
            $inner,
            __tf,
            __tc,
            $tag_value,
            move |kp: $kp| kp.deserialize_key::<$crate::Match, &str>(__tf),
            move |vp: $vp| vp.deserialize_value::<$crate::MatchVals<usize>, _>(__tc),
        )
    }};
}

impl<KP, A, B> MapArmStackOwned<KP> for StackConcat<A, B>
where
    KP: MapKeyProbeOwned,
    A: MapArmStackOwned<KP>,
    B: MapArmStackOwned<KP>,
{
    const SIZE: usize = A::SIZE + B::SIZE;
    type Outputs = (A::Outputs, B::Outputs);

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count() + self.1.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.0.open_count() + self.1.open_count()
    }

    type RaceState = ConcatRaceState<A::RaceState, B::RaceState>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
        let b_kp = kp.fork();
        ConcatRaceState {
            a: self.0.init_race(kp),
            b: self.1.init_race(b_kp),
        }
    }

    #[inline(always)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>> {
        let projected = state.project();
        if arm_index < A::SIZE {
            // Delegate to A - sync recursion.
            self.0.poll_race_one(projected.a, arm_index, cx)
        } else {
            // Delegate to B, offset index.
            match self.1.poll_race_one(projected.b, arm_index - A::SIZE, cx) {
                Poll::Ready(Ok(Probe::Hit((idx, kc)))) => {
                    Poll::Ready(Ok(Probe::Hit((A::SIZE + idx, kc))))
                }
                other => other,
            }
        }
    }

    type DispatchState = ConcatDispatchState<A::DispatchState, B::DispatchState>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<KP>) -> Self::DispatchState {
        if arm_index < A::SIZE {
            ConcatDispatchState::InA(self.0.init_dispatch(arm_index, vp))
        } else {
            ConcatDispatchState::InB(self.1.init_dispatch(arm_index - A::SIZE, vp))
        }
    }

    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<KP>, ())>, KP::Error>> {
        match state.project() {
            ConcatDispatchProj::InA(a_state) => self.0.poll_dispatch(a_state, cx),
            ConcatDispatchProj::InB(b_state) => self.1.poll_dispatch(b_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        (self.0.take_outputs(), self.1.take_outputs())
    }
}

// --- FlattenContOwned ---
//
// Continuation trait for flatten chaining. `FlattenMapAccessOwned::iterate` calls
// `cont.finish(map, combined_arms)` instead of `map.iterate(combined_arms)`
// directly. This lets intermediate flatten fields pass the growing arm stack
// to the next flatten field's `DeserializeOwned` impl rather than consuming
// the map immediately.
//
// Terminal: `FlattenTerminal` calls `map.iterate(SkipUnknownOwned(arms))`.
// Intermediate: generated struct that calls next flatten field's `deserialize_owned`
// with a new `FlattenDeserializerOwned` wrapping the combined arms.

/// Continuation for [`FlattenMapAccessOwned`]. Determines what happens after
/// the outer and inner arm stacks are combined via [`StackConcat`].
#[allow(async_fn_in_trait)]
pub trait FlattenContOwned<M: MapAccessOwned> {
    /// Extra result produced by the continuation (e.g., deserialized values
    /// from subsequent flatten fields). `()` for the terminal case.
    type Extra;

    /// Drive the combined arm stack to completion.
    ///
    /// `arms` is `StackConcat(outer_arms, inner_arms)` - the full set of arms
    /// from both the parent struct's fields and the current flatten field.
    async fn finish<Arms: MapArmStackOwned<M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs, Self::Extra)>, M::Error>;
}

impl<M: MapAccessOwned> FlattenContOwned<M> for crate::FlattenTerminal {
    type Extra = ();

    async fn finish<Arms: MapArmStackOwned<M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs, ())>, M::Error> {
        let wrapped = crate::SkipUnknownOwned!(arms, M::KeyProbe, VP<M::KeyProbe>);
        let (claim, out) = hit!(map.iterate(wrapped).await);
        Ok(Probe::Hit((claim, out, ())))
    }
}

#[cfg(feature = "alloc")]
impl<M: MapAccessOwned> FlattenContOwned<M> for crate::FlattenTerminalBoxed {
    type Extra = ();

    async fn finish<Arms: MapArmStackOwned<M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs, ())>, M::Error> {
        let wrapped = crate::SkipUnknownOwned!(arms, M::KeyProbe, VP<M::KeyProbe>);
        #[allow(clippy::type_complexity)]
        let r: Result<Probe<(M::MapClaim, Arms::Outputs)>, M::Error> =
            alloc::boxed::Box::pin(map.iterate(wrapped)).await;
        let (claim, out) = hit!(r);
        Ok(Probe::Hit((claim, out, ())))
    }
}

// --- FlattenDeserializerOwned ---
//
// A `DeserializerOwned` facade used by `#[strede(flatten)]`. The parent
// struct's derive holds:
//   1. The real `MapAccessOwned` for the outer map.
//   2. Its own arm stack for the non-flatten fields.
//   3. A continuation (`Cont`) that either terminates or chains to the next
//      flatten field.
//
// It passes `FlattenDeserializerOwned` to the flattened field's `DeserializeOwned`
// impl. That impl calls `entry → deserialize_map → iterate(inner_arms)`.
// `FlattenMapAccessOwned::iterate` intercepts `inner_arms`, prepends the outer arms
// via `StackConcat`, and calls `cont.finish(map, combined)`.
//
// The outer arms are passed via `core::cell::Cell<Option<OuterArms>>` so they
// can be moved out (and outputs moved back in) across the async boundary without
// borrow conflicts.

/// Facade `DeserializerOwned` used to implement `#[strede(flatten)]`.
///
/// Pass this as the deserializer to a flattened field's `DeserializeOwned`
/// impl. It intercepts `deserialize_map` → `iterate(inner_arms)`,
/// prepends the outer struct's arms via [`StackConcat`], and delegates to
/// the [`FlattenContOwned`] continuation. For the terminal case
/// ([`FlattenTerminal`]) this drives the real map's `iterate`; for
/// intermediate cases it chains to the next flatten field.
pub struct FlattenDeserializerOwned<'a, M, OuterArms, Cont>
where
    M: MapAccessOwned,
    OuterArms: MapArmStackOwned<M::KeyProbe>,
    Cont: FlattenContOwned<M>,
{
    pub map: M,
    pub outer_arms: &'a core::cell::Cell<Option<OuterArms>>,
    pub outer_outputs: &'a core::cell::Cell<Option<OuterArms::Outputs>>,
    pub cont: Cont,
    pub extra: &'a core::cell::Cell<Option<Cont::Extra>>,
}

impl<'a, M, OuterArms, Cont> FlattenDeserializerOwned<'a, M, OuterArms, Cont>
where
    M: MapAccessOwned,
    OuterArms: MapArmStackOwned<M::KeyProbe>,
    Cont: FlattenContOwned<M>,
{
    pub fn new(
        map: M,
        outer_arms: &'a core::cell::Cell<Option<OuterArms>>,
        outer_outputs: &'a core::cell::Cell<Option<OuterArms::Outputs>>,
        cont: Cont,
        extra: &'a core::cell::Cell<Option<Cont::Extra>>,
    ) -> Self {
        Self {
            map,
            outer_arms,
            outer_outputs,
            cont,
            extra,
        }
    }
}

impl<'a, M, OuterArms, Cont> DeserializerOwned for FlattenDeserializerOwned<'a, M, OuterArms, Cont>
where
    M: MapAccessOwned,
    OuterArms: MapArmStackOwned<M::KeyProbe>,
    Cont: FlattenContOwned<M>,
{
    type Error = M::Error;
    type Claim = M::MapClaim;
    type EntryClaim = M::MapClaim;
    type Entry = FlattenEntryOwned<'a, M, OuterArms, Cont>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<
                Output = Result<Probe<(<Self::Entry as EntryOwned>::Claim, R)>, Self::Error>,
            >,
    {
        // Only N=1 is supported - flatten entry handles cannot be forked.
        assert!(
            N == 1,
            "FlattenDeserializerOwned only supports entry<1, ...>"
        );
        let entry = FlattenEntryOwned {
            map: self.map,
            outer_arms: self.outer_arms,
            outer_outputs: self.outer_outputs,
            cont: self.cont,
            extra: self.extra,
        };
        let mut slot = Some(entry);
        let entries: [Self::Entry; N] = core::array::from_fn(|_| slot.take().unwrap());
        f(entries).await
    }
}

/// Entry handle produced by [`FlattenDeserializerOwned`].
pub struct FlattenEntryOwned<'a, M, OuterArms, Cont>
where
    M: MapAccessOwned,
    OuterArms: MapArmStackOwned<M::KeyProbe>,
    Cont: FlattenContOwned<M>,
{
    pub map: M,
    pub outer_arms: &'a core::cell::Cell<Option<OuterArms>>,
    pub outer_outputs: &'a core::cell::Cell<Option<OuterArms::Outputs>>,
    pub cont: Cont,
    pub extra: &'a core::cell::Cell<Option<Cont::Extra>>,
}

impl<'a, M, OuterArms, Cont> EntryOwned for FlattenEntryOwned<'a, M, OuterArms, Cont>
where
    M: MapAccessOwned,
    OuterArms: MapArmStackOwned<M::KeyProbe>,
    Cont: FlattenContOwned<M>,
{
    type Error = M::Error;
    type Claim = M::MapClaim;
    type StrChunks = crate::Never<'static, M::MapClaim, M::Error>;
    type BytesChunks = crate::Never<'static, M::MapClaim, M::Error>;
    type Map = FlattenMapAccessOwned<'a, M, OuterArms, Cont>;
    type Seq = crate::Never<'static, M::MapClaim, M::Error>;

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        Ok(Probe::Hit(FlattenMapAccessOwned {
            map: self.map,
            outer_arms: self.outer_arms,
            outer_outputs: self.outer_outputs,
            cont: self.cont,
            extra: self.extra,
        }))
    }

    // All other entry methods return Miss - flatten only supports map-shaped types.
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
    async fn deserialize_option<T: DeserializeOwned<Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error> {
        Ok(Probe::Miss)
    }
    async fn deserialize_value<T: DeserializeOwned<Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        Ok(Probe::Miss)
    }
    fn fork(&mut self) -> Self {
        panic!("FlattenEntryOwned::fork called; flatten only supports N=1 entry")
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        panic!("FlattenEntryOwned::skip called on flatten entry")
    }
}

/// [`MapAccessOwned`] produced by [`FlattenEntryOwned`].
///
/// When `iterate(inner_arms)` is called, it takes the outer arms from the
/// `Cell`, builds `StackConcat(outer_arms, inner_arms)`, and delegates to
/// `cont.finish(map, combined)`. The continuation either drives the real map
/// (terminal) or chains to the next flatten field (intermediate).
pub struct FlattenMapAccessOwned<'a, M, OuterArms, Cont>
where
    M: MapAccessOwned,
    OuterArms: MapArmStackOwned<M::KeyProbe>,
    Cont: FlattenContOwned<M>,
{
    pub map: M,
    pub outer_arms: &'a core::cell::Cell<Option<OuterArms>>,
    pub outer_outputs: &'a core::cell::Cell<Option<OuterArms::Outputs>>,
    pub cont: Cont,
    pub extra: &'a core::cell::Cell<Option<Cont::Extra>>,
}

impl<'a, M, OuterArms, Cont> MapAccessOwned for FlattenMapAccessOwned<'a, M, OuterArms, Cont>
where
    M: MapAccessOwned,
    OuterArms: MapArmStackOwned<M::KeyProbe>,
    Cont: FlattenContOwned<M>,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    fn fork(&mut self) -> Self {
        panic!("FlattenMapAccessOwned::fork not supported")
    }

    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        inner_arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        let outer_arms = self
            .outer_arms
            .take()
            .expect("FlattenMapAccessOwned::iterate called without outer arms");
        let combined = StackConcat(outer_arms, inner_arms);
        let (claim, (outer_out, inner_out), extra) =
            hit!(self.cont.finish(self.map, combined).await);
        self.outer_outputs.set(Some(outer_out));
        self.extra.set(Some(extra));
        Ok(Probe::Hit((claim, inner_out)))
    }
}

/// Experimental redesign of [`MapAccessOwned`].
///
/// Instead of a closure-based `iterate`, takes a [`MapArmStackOwned`] - a
/// left-nested tuple of [`MapArmSlot`]s. The map impl owns the iteration loop:
///
/// 1. Produce the first key probe (or detect empty map).
/// 2. Each round: call `arms.race_keys(kp)` to race key callbacks,
///    then on hit: `key_claim.into_value_probe()` → `arms.dispatch_value(idx, vp)`.
/// 3. `value_claim.next_key(unsatisfied, open)` → next round or done.
/// 4. Extract `arms.take_outputs()`.
pub trait MapAccessOwned: Sized {
    type Error: DeserializeError;
    type MapClaim;
    type KeyProbe: MapKeyProbeOwned<Error = Self::Error>;

    fn fork(&mut self) -> Self;

    /// Drive the map iteration with the given arm stack.
    ///
    /// Returns `Hit((MapClaim, Outputs))` on success, `Miss` if a value
    /// type mismatched or a required field was missing, `Err` on format errors.
    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Never impls - owned family
// ---------------------------------------------------------------------------

impl<'n, C, E: DeserializeError> StrAccessOwned for crate::Never<'n, C, E> {
    type Claim = C;
    type Error = E;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn next_str<R>(
        self,
        _f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> BytesAccessOwned for crate::Never<'n, C, E> {
    type Claim = C;
    type Error = E;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn next_bytes<R>(
        self,
        _f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> SeqAccessOwned for crate::Never<'n, C, E> {
    type Error = E;
    type SeqClaim = C;
    type ElemClaim = C;
    type Elem = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn next<const N: usize, F, Fut, R>(
        self,
        _f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::SeqClaim, R)>, Self::Error>>,
    {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> SeqEntryOwned for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn get<T: DeserializeOwned<Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapAccessOwned for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type KeyProbe = crate::Never<'n, C, E>;

    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        _arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapKeyProbeOwned for crate::Never<'n, C, E> {
    type Error = E;
    type KeyClaim = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_key<K: DeserializeOwned<Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapKeyClaimOwned for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type ValueProbe = crate::Never<'n, C, E>;
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapValueProbeOwned for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type ValueClaim = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_value<V: DeserializeOwned<Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error> {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        match self.0 {}
    }
}

impl<'n, C, E: DeserializeError> MapValueClaimOwned for crate::Never<'n, C, E> {
    type Error = E;
    type KeyProbe = crate::Never<'n, C, E>;
    type MapClaim = C;
    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        match self.0 {}
    }
}

// ---------------------------------------------------------------------------
// DeserializeOwned impls for primitives
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_owned_primitive {
    ($ty:ty, $method:ident) => {
        impl DeserializeOwned for $ty {
            async fn deserialize_owned<D: DeserializerOwned>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async { e.$method().await }).await
            }
        }
    };
}

impl_deserialize_owned_primitive!(bool, deserialize_bool);
impl_deserialize_owned_primitive!(u8, deserialize_u8);
impl_deserialize_owned_primitive!(u16, deserialize_u16);
impl_deserialize_owned_primitive!(u32, deserialize_u32);
impl_deserialize_owned_primitive!(u64, deserialize_u64);
impl_deserialize_owned_primitive!(u128, deserialize_u128);
impl_deserialize_owned_primitive!(i8, deserialize_i8);
impl_deserialize_owned_primitive!(i16, deserialize_i16);
impl_deserialize_owned_primitive!(i32, deserialize_i32);
impl_deserialize_owned_primitive!(i64, deserialize_i64);
impl_deserialize_owned_primitive!(i128, deserialize_i128);
impl_deserialize_owned_primitive!(f32, deserialize_f32);
impl_deserialize_owned_primitive!(f64, deserialize_f64);
impl_deserialize_owned_primitive!(char, deserialize_char);

impl<Extra: Copy, T: DeserializeOwned<Extra>> DeserializeOwned<Extra> for Option<T> {
    async fn deserialize_owned<D: DeserializerOwned>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async { e.deserialize_option::<T, Extra>(extra).await })
            .await
    }
}
