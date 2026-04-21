use core::future::Future;
use core::mem;
use core::pin::Pin;
use core::task::{Context, Poll};

use crate::map_arm::{
    ArmState, ConcatDispatchProj, ConcatDispatchState, ConcatRaceState, DetectDuplicatesOwned,
    MapArmBase, MapArmSlot, NextKey, SlotDispatchProj, SlotDispatchState, SlotRaceState,
    StackConcat, TagDispatchProj, TagDispatchState, TagInjectingStackOwned, TagRaceState,
    VirtualArmSlot, WrapperDispatchProj, WrapperDispatchState, WrapperRaceState, poll_key_slot,
};
use crate::{Chunk, DeserializeError, Probe};

// ---------------------------------------------------------------------------
// Deserialize - the "what"  (mirrors serde::Deserialize)
// ---------------------------------------------------------------------------

/// Types that can deserialize themselves from a [`Deserializer`] stream.
///
/// Implement this to make a type readable by any format that implements
/// [`Deserializer`].  The method drives the deserializer forward and returns
/// the fully constructed value, or a fatal format error.
///
/// The `Extra` type parameter is side-channel context passed into
/// [`Entry::deserialize_value`], [`SeqEntry::get`], [`MapKeyProbe::deserialize_key`], and
/// [`MapValueProbe::deserialize_value`] at the call site.  Defaults to `()` for types that
/// need no extra context.
pub trait Deserialize<'de, Extra = ()>: Sized {
    /// Deserialize `Self` from `d`, with caller-supplied side-channel `extra`.
    ///
    /// - `Ok(Probe::Hit((claim, value)))` - succeeded; thread `claim` back to the caller.
    /// - `Ok(Probe::Miss)` - token type didn't match; stream not advanced.
    /// - `Err(e)` - fatal format error (malformed data, I/O failure).
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error>;
}

// ---------------------------------------------------------------------------
// Deserializer - stream handle
// ---------------------------------------------------------------------------

/// A stream of tokens that can be decoded into Rust values.
///
/// The deserializer owns the stream and is the sole means of advancing it -
/// all advancement goes through [`Deserializer::entry`].  Type probing is done
/// through [`Entry`] handles handed to the closure.
pub trait Deserializer<'de>: Sized {
    /// Fatal error type produced by this format.
    type Error: DeserializeError;

    /// Proof-of-consumption token returned from [`Deserializer::entry`].
    type Claim: 'de;

    /// The claim type produced by entry handles ([`Entry::Claim`]).
    /// Distinct from `Claim` to allow implementations to use different claim
    /// types at the deserializer level vs the entry level (e.g. flatten facades).
    type EntryClaim: 'de;

    /// Owned handle for one item slot.  See [`Entry`].
    type Entry: Entry<'de, Claim = Self::EntryClaim, Error = Self::Error>;

    /// Advance to the next item in the stream.
    ///
    /// Passes `N` owned [`Entry`] handles to `f`.  When a probe inside `f`
    /// resolves `Ok(Probe::Hit((claim, r)))`, the winning arm returns it;
    /// `entry` verifies the claim and returns `Ok(Probe::Hit((claim, r)))`.
    /// Returns `Err(e)` if a fatal error occurs before or during `f`.
    /// `Ok(Probe::Miss)` propagates upward if no probe matched.
    /// `Pending` until a token is available.
    ///
    /// Use `N > 1` to race multiple probe arms via [`select_probe!`](crate::select_probe) without
    /// borrow conflicts.  Handles dropped without resolving do not advance
    /// the stream.
    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>;
}

// ---------------------------------------------------------------------------
// Entry - type-probing handle for one item slot
// ---------------------------------------------------------------------------

/// Owned handle for one item slot, passed into the closure of [`Deserializer::entry`].
///
/// Each probe consumes `self` and resolves to `Ok(Probe::Hit((Claim, T)))` when
/// the current token is of type `T`, `Ok(Probe::Miss)` if the type doesn't
/// match or the token is a different kind, or `Err(e)` on a fatal format error.
/// The `Claim` must be returned from the closure so `next` can advance the stream.
///
/// # For implementors
///
/// Type mismatches **must** return `Ok(Probe::Miss)` - never `Err`.  `Err` is
/// reserved for fatal format errors (malformed data, I/O failure).
///
/// Use `N > 1` in `entry` to race arms via [`select_probe!`](crate::select_probe):
///
/// ```rust,ignore
/// let value = d.entry(|[e1, e2, e3, e4]| async {
///     select_probe! {
///         e1.deserialize_bool(),
///         e2.deserialize_i64(),
///         async move {
///             let (c, v) = hit!(e3.deserialize_str().await);
///             Ok(Probe::Hit((c, Value::Str(v))))
///         },
///         async move {
///             let mut m = hit!(e4.deserialize_map().await);
///             let (c, v) = collect_map(m).await?;
///             Ok(Probe::Hit((c, Value::Map(v))))
///         },
///     }
/// }).await?;
/// ```
pub trait Entry<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Proof-of-consumption token.  Must be returned from the `next` closure
    /// alongside the caller's value so the deserializer can advance the stream.
    type Claim: 'de;

    type StrChunks: StrAccess<Claim = Self::Claim, Error = Self::Error>;
    type BytesChunks: BytesAccess<Claim = Self::Claim, Error = Self::Error>;
    type Map: MapAccess<'de, MapClaim = Self::Claim, Error = Self::Error>;
    type Seq: SeqAccess<'de, SeqClaim = Self::Claim, Error = Self::Error>;

    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_bool(self) -> Result<Probe<(Self::Claim, bool)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u8(self) -> Result<Probe<(Self::Claim, u8)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u16(self) -> Result<Probe<(Self::Claim, u16)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u32(self) -> Result<Probe<(Self::Claim, u32)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u64(self) -> Result<Probe<(Self::Claim, u64)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_u128(self) -> Result<Probe<(Self::Claim, u128)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i8(self) -> Result<Probe<(Self::Claim, i8)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i16(self) -> Result<Probe<(Self::Claim, i16)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i32(self) -> Result<Probe<(Self::Claim, i32)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i64(self) -> Result<Probe<(Self::Claim, i64)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_i128(self) -> Result<Probe<(Self::Claim, i128)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_f32(self) -> Result<Probe<(Self::Claim, f32)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_f64(self) -> Result<Probe<(Self::Claim, f64)>, Self::Error>;
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_char(self) -> Result<Probe<(Self::Claim, char)>, Self::Error>;
    /// Attempt a **zero-copy** string borrow.
    ///
    /// Hits only when the format can return the entire string as a single
    /// contiguous `&'de str` slice - i.e. with no escape sequences or
    /// transcoding.  Returns `Ok(Probe::Miss)` when the token is a string
    /// but cannot be represented that way (e.g. JSON `"\n"`), so that a
    /// concurrent [`Entry::deserialize_str_chunks`] arm can take over via
    /// `select_probe!`.  This makes `Cow<str>` callers easy to write: race
    /// the two methods and borrow when free, allocate only when necessary.
    ///
    /// `Ok(Probe::Miss)` on type mismatch *or* when zero-copy is unavailable;
    /// `Err` on fatal format error only.
    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error>;
    /// Begin streaming a string chunk-by-chunk.  The [`Entry::Claim`] is
    /// returned by [`StrAccess::next_str`] when the string is exhausted.
    /// Handles all strings including those with escape sequences.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error>;
    /// Attempt a **zero-copy** bytes borrow.  Same Miss semantics as
    /// [`Entry::deserialize_str`]: hits only when the entire value is
    /// available as a contiguous `&'de [u8]` slice; returns `Miss` otherwise
    /// so a [`Entry::deserialize_bytes_chunks`] arm can handle the rest.
    /// `Ok(Probe::Miss)` on type mismatch *or* when zero-copy is unavailable;
    /// `Err` on fatal format error only.
    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error>;
    /// Begin streaming a byte string chunk-by-chunk.  The [`Entry::Claim`] is
    /// returned by [`BytesAccess::next_bytes`] when the byte string is exhausted.
    /// Handles all byte strings including those requiring transcoding.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error>;
    /// Begin reading a map.  The [`Entry::Claim`] is returned by
    /// [`MapAccess::iterate`] when the map is exhausted.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error>;
    /// Begin reading a sequence.  The [`Entry::Claim`] is returned by
    /// [`SeqAccess::next`] when the sequence is exhausted.
    /// `Ok(Probe::Miss)` on type mismatch; `Err` on fatal format error only.
    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error>;

    /// Deserialize an optional value.
    ///
    /// - Null token → `Ok(Probe::Hit((claim, None)))`
    /// - Token matching `T::deserialize` → `Ok(Probe::Hit((claim, Some(v))))`
    /// - Token matching neither → `Ok(Probe::Miss)`
    /// - Fatal format error → `Err(e)`
    async fn deserialize_option<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>;

    /// Probe for a null token.
    ///
    /// - `Ok(Probe::Hit(claim))` - null token consumed.
    /// - `Ok(Probe::Miss)` - token is not null.
    /// - `Err(e)` - fatal format error.
    async fn deserialize_null(self) -> Result<Probe<Self::Claim>, Self::Error>;

    /// Delegate to `T::deserialize` from this entry handle, forwarding `extra`.
    ///
    /// Creates a sub-deserializer with the current token pre-loaded and
    /// calls `T::deserialize`.  Returns `Hit` if `T` matched, `Miss` if
    /// `T::deserialize` returned `Miss`.
    async fn deserialize_value<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

    /// Fork a sibling entry handle for the same item slot.
    ///
    /// Both `self` and the returned handle refer to the same slot.
    /// Whichever resolves a probe first claims the slot; the other
    /// becomes inert and may be dropped without advancing the stream.
    fn fork(&mut self) -> Self;

    /// Consume and discard the current token regardless of its type
    /// (scalar, string, map, or sequence).
    ///
    /// Always succeeds on well-formed input.  Returns the [`Claim`](Entry::Claim)
    /// so the stream can advance.  `Err` only on malformed data.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ---------------------------------------------------------------------------
// StrAccess
// ---------------------------------------------------------------------------

/// Streams a string in zero-copy chunks.  Obtained from [`Entry::deserialize_str_chunks`].
///
/// Strings are primitives - the type is already known, so no probing or
/// racing is needed.  This adapter exists solely for formats that cannot
/// deliver the value as a single contiguous slice (e.g. escape-sequence
/// synthesis in JSON).
///
/// ```rust,ignore
/// let mut chunks = hit!(e.deserialize_str_chunks().await);
/// let mut out = String::new();
/// let claim = loop {
///     match chunks.next_str(|s| out.push_str(s)).await? {
///         Chunk::Data((new, ())) => chunks = new,
///         Chunk::Done(claim) => break claim,
///     }
/// };
/// ```
pub trait StrAccess: Sized {
    /// Proof-of-consumption token, returned when the string is exhausted.
    /// Must match the enclosing [`Entry::Claim`].
    type Claim;
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Fork a sibling accessor at the same read position.
    ///
    /// Both `self` and the returned accessor are independent: advancing one
    /// does not affect the other.  Both start from the current chunk
    /// position and must each be driven to `Done` independently.
    fn fork(&mut self) -> Self;

    /// Advance to the next chunk, passing it to `f`.
    ///
    /// - `Ok(Chunk::Data((self, r)))` - next chunk processed; accessor returned for the next call.
    /// - `Ok(Chunk::Done(claim))` - string exhausted; claim is now valid.
    /// - `Err(e)` - fatal format error.
    /// - `Pending` - no data yet.
    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// BytesAccess
// ---------------------------------------------------------------------------

/// Streams a byte string in zero-copy chunks.  Obtained from [`Entry::deserialize_bytes_chunks`].
///
/// Byte strings are primitives - the type is already known, so no probing or
/// racing is needed.  This adapter exists solely for formats that cannot
/// deliver the value as a single contiguous slice.
///
/// ```rust,ignore
/// let mut chunks = hit!(e.deserialize_bytes_chunks().await);
/// let mut out = Vec::new();
/// let claim = loop {
///     match chunks.next_bytes(|b| out.extend_from_slice(b)).await? {
///         Chunk::Data((new, ())) => chunks = new,
///         Chunk::Done(claim) => break claim,
///     }
/// };
/// ```
pub trait BytesAccess: Sized {
    /// Proof-of-consumption token, returned when the byte string is exhausted.
    /// Must match the enclosing [`Entry::Claim`].
    type Claim;
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Fork a sibling accessor at the same read position.
    ///
    /// See [`StrAccess::fork`] for full semantics.
    fn fork(&mut self) -> Self;

    /// Advance to the next chunk, passing it to `f`.
    ///
    /// - `Ok(Chunk::Data((self, r)))` - next chunk processed; accessor returned for the next call.
    /// - `Ok(Chunk::Done(claim))` - byte string exhausted; claim is now valid.
    /// - `Err(e)` - fatal format error.
    /// - `Pending` - no data yet.
    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Map access - new design
// ---------------------------------------------------------------------------

/// A key probe for a single map key in the borrow family. Forkable for racing multiple arms.
pub trait MapKeyProbe<'de>: Sized {
    type Error: DeserializeError;
    type KeyClaim: MapKeyClaim<'de, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn deserialize_key<K: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>;
}

/// Proof that a key was consumed. Converts into a value probe.
pub trait MapKeyClaim<'de>: Sized {
    type Error: DeserializeError;
    type MapClaim: 'de;
    type ValueProbe: MapValueProbe<'de, MapClaim = Self::MapClaim, Error = Self::Error>;

    /// Consume this key claim and produce a value probe for the corresponding
    /// map value. Format-specific (e.g. JSON reads `:` and the value start token).
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error>;
}

/// A value probe that can deserialize a value or skip it (borrow family).
pub trait MapValueProbe<'de>: Sized {
    type Error: DeserializeError;
    type MapClaim: 'de;
    type ValueClaim: MapValueClaim<'de, MapClaim = Self::MapClaim, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    async fn deserialize_value<V: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>;

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error>;
}

/// Proof that a value was consumed. Advances to the next key or ends the map (borrow family).
pub trait MapValueClaim<'de>: Sized {
    type Error: DeserializeError;
    type KeyProbe: MapKeyProbe<'de, Error = Self::Error>;
    type MapClaim: 'de;

    async fn next_key(
        self,
        unsatisfied: usize,
        open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error>;
}

// ---------------------------------------------------------------------------
// MapAccess - iterates key-value pairs via arm stacks
// ---------------------------------------------------------------------------

/// Iterates the key-value pairs of a map.  Obtained from [`Entry::deserialize_map`].
pub trait MapAccess<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Proof-of-consumption token returned on map exhaustion; must match
    /// the enclosing [`Entry::Claim`].
    type MapClaim: 'de;

    type KeyProbe: MapKeyProbe<'de, Error = Self::Error>;

    fn fork(&mut self) -> Self;

    /// Drive the map iteration with the given arm stack.
    ///
    /// Returns `Hit((MapClaim, Outputs))` on success, `Miss` if a value
    /// type mismatched or a required field was missing, `Err` on format errors.
    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Type aliases for borrow map access
// ---------------------------------------------------------------------------

/// Shorthand for the key probe type reachable from a `Deserializer`.
pub type KP<'de, D> =
    <<<D as Deserializer<'de>>::Entry as Entry<'de>>::Map as MapAccess<'de>>::KeyProbe;

/// Shorthand for the value claim type reachable from a borrow key probe type.
pub type VC<'de, KP> =
    <<<KP as MapKeyProbe<'de>>::KeyClaim as MapKeyClaim<'de>>::ValueProbe as MapValueProbe<'de>>::ValueClaim;

/// Shorthand for the value probe type reachable from a borrow key probe type.
pub type VP<'de, KP> = <<KP as MapKeyProbe<'de>>::KeyClaim as MapKeyClaim<'de>>::ValueProbe;

/// Shorthand for the value probe type reachable directly from a `Deserializer`.
pub type VP2<'de, D> = <<KP<'de, D> as MapKeyProbe<'de>>::KeyClaim as MapKeyClaim<'de>>::ValueProbe;

// ---------------------------------------------------------------------------
// MapArmStack<'de, KP> - borrow-family arm stack
// ---------------------------------------------------------------------------

/// Borrow-family counterpart to [`crate::MapArmStackOwned`].
///
/// A left-nested tuple stack of [`MapArmSlot`]s parameterized by `'de`.
/// The map impl drives the iteration loop using this trait's `init_race` /
/// `poll_race_one` / `init_dispatch` / `poll_dispatch` methods.
pub trait MapArmStack<'de, KP: MapKeyProbe<'de>>: Sized {
    const SIZE: usize;

    /// Left-nested tuple of `Option<(K, V)>` for each arm.
    type Outputs;

    /// Number of arms that still require a value (required fields not yet matched).
    /// Virtual arms are excluded.
    fn unsatisfied_count(&self) -> usize;

    /// Number of arms still willing to run, including virtual arms.
    fn open_count(&self) -> usize;

    type RaceState;

    fn init_race(&mut self, kp: KP) -> Self::RaceState;
    #[allow(clippy::type_complexity)]
    fn poll_race_one(
        &mut self,
        state: Pin<&mut Self::RaceState>,
        arm_index: usize,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(usize, KP::KeyClaim)>, KP::Error>>;

    type DispatchState;

    fn init_dispatch(&mut self, arm_index: usize, vp: VP<'de, KP>) -> Self::DispatchState;
    #[allow(clippy::type_complexity)]
    fn poll_dispatch(
        &mut self,
        state: Pin<&mut Self::DispatchState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>>;

    /// Race all unsatisfied arms' key callbacks against the given key probe.
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
    async fn dispatch_value(
        &mut self,
        arm_index: usize,
        vp: VP<'de, KP>,
    ) -> Result<Probe<(VC<'de, KP>, ())>, KP::Error> {
        let dispatch_state = self.init_dispatch(arm_index, vp);
        let mut dispatch_state = core::pin::pin!(dispatch_state);
        core::future::poll_fn(|cx| self.poll_dispatch(dispatch_state.as_mut(), cx)).await
    }

    fn take_outputs(&mut self) -> Self::Outputs;
}

// --- MapArmBase impl ---

impl<'de, KP: MapKeyProbe<'de>> MapArmStack<'de, KP> for MapArmBase {
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
    fn init_dispatch(&mut self, _arm_index: usize, _vp: VP<'de, KP>) -> Self::DispatchState {
        unreachable!("init_dispatch called on MapArmBase")
    }
    #[inline(always)]
    fn poll_dispatch(
        &mut self,
        _state: Pin<&mut Self::DispatchState>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
        unreachable!("poll_dispatch called on MapArmBase")
    }

    #[inline(always)]
    fn take_outputs(&mut self) {}
}

// --- Recursive (Rest, Slot) impl ---

impl<'de, KP, Rest, K, V, KeyFn, KeyFut, ValFn, ValFut> MapArmStack<'de, KP>
    for (Rest, MapArmSlot<K, V, KeyFn, ValFn>)
where
    KP: MapKeyProbe<'de>,
    Rest: MapArmStack<'de, KP>,
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(VP<'de, KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(VC<'de, KP>, (K, V))>, KP::Error>>,
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
            self.0.poll_race_one(projected.rest, arm_index, cx)
        }
    }

    type DispatchState = SlotDispatchState<Rest::DispatchState, ValFut>;

    #[inline(always)]
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<'de, KP>) -> Self::DispatchState {
        if arm_index == Self::SIZE - 1 {
            let k = match core::mem::replace(&mut self.1.state, ArmState::Empty) {
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
    ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
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

// --- (Rest, VirtualArmSlot) impl ---

impl<'de, KP, Rest, K, KeyFn, KeyFut, ValFn, ValFut> MapArmStack<'de, KP>
    for (Rest, VirtualArmSlot<K, KeyFn, ValFn>)
where
    KP: MapKeyProbe<'de>,
    Rest: MapArmStack<'de, KP>,
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, K)>, KP::Error>>,
    ValFn: FnMut(VP<'de, KP>, K) -> ValFut,
    ValFut: Future<Output = Result<Probe<(VC<'de, KP>, ())>, KP::Error>>,
{
    const SIZE: usize = Rest::SIZE + 1;
    type Outputs = Rest::Outputs;

    #[inline(always)]
    fn unsatisfied_count(&self) -> usize {
        self.0.unsatisfied_count()
    }
    #[inline(always)]
    fn open_count(&self) -> usize {
        self.0.open_count() + 1
    }

    type RaceState = SlotRaceState<Rest::RaceState, KeyFut>;

    #[inline(always)]
    fn init_race(&mut self, mut kp: KP) -> Self::RaceState {
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
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<'de, KP>) -> Self::DispatchState {
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
    ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
        match state.project() {
            SlotDispatchProj::ThisArm(fut) => fut.poll(cx),
            SlotDispatchProj::Delegated(rest_state) => self.0.poll_dispatch(rest_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.0.take_outputs()
    }
}

// --- DetectDuplicatesOwned impl ---

impl<'de, KP, S, const M: usize, KeyFn, KeyFut, SkipFn, SkipFut> MapArmStack<'de, KP>
    for DetectDuplicatesOwned<S, M, KeyFn, SkipFn>
where
    KP: MapKeyProbe<'de>,
    S: MapArmStack<'de, KP>,
    KeyFn: FnMut(KP) -> KeyFut,
    KeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::MatchVals<usize>)>, KP::Error>>,
    SkipFn: FnMut(VP<'de, KP>) -> SkipFut,
    SkipFut: Future<Output = Result<VC<'de, KP>, KP::Error>>,
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
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<'de, KP>) -> Self::DispatchState {
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
    ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
        match state.project() {
            WrapperDispatchProj::Virtual(fut) => fut.poll(cx).map(|r| match r {
                Ok(_vc) => Err(<KP::Error as crate::DeserializeError>::duplicate_field(
                    self.dup,
                )),
                Err(e) => Err(e),
            }),
            WrapperDispatchProj::Inner(inner_state) => self.inner.poll_dispatch(inner_state, cx),
        }
    }

    #[inline(always)]
    fn take_outputs(&mut self) -> Self::Outputs {
        self.inner.take_outputs()
    }
}

// --- TagInjectingStackOwned impl ---

impl<'de, 'v, KP, S, const N: usize, TagKeyFn, TagKeyFut, TagValFn, TagValFut> MapArmStack<'de, KP>
    for TagInjectingStackOwned<'v, S, N, TagKeyFn, TagValFn>
where
    KP: MapKeyProbe<'de>,
    S: MapArmStack<'de, KP>,
    TagKeyFn: FnMut(KP) -> TagKeyFut,
    TagKeyFut: Future<Output = Result<Probe<(KP::KeyClaim, crate::Match)>, KP::Error>>,
    TagValFn: FnMut(VP<'de, KP>) -> TagValFut,
    TagValFut: Future<Output = Result<Probe<(VC<'de, KP>, crate::MatchVals<usize>)>, KP::Error>>,
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
            match poll_key_slot(projected.tag_fut, cx) {
                Poll::Ready(Ok(Probe::Hit((kc, _match)))) => Poll::Ready(Ok(Probe::Hit((0, kc)))),
                Poll::Ready(Ok(Probe::Miss)) => Poll::Ready(Ok(Probe::Miss)),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        } else {
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
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<'de, KP>) -> Self::DispatchState {
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
    ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
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

// --- StackConcat impl ---

impl<'de, KP, A, B> MapArmStack<'de, KP> for StackConcat<A, B>
where
    KP: MapKeyProbe<'de>,
    A: MapArmStack<'de, KP>,
    B: MapArmStack<'de, KP>,
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
            self.0.poll_race_one(projected.a, arm_index, cx)
        } else {
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
    fn init_dispatch(&mut self, arm_index: usize, vp: VP<'de, KP>) -> Self::DispatchState {
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
    ) -> Poll<Result<Probe<(VC<'de, KP>, ())>, KP::Error>> {
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

// ---------------------------------------------------------------------------
// Flatten infrastructure - borrow family
// ---------------------------------------------------------------------------

// Parallel to FlattenContOwned / FlattenDeserializerOwned / FlattenEntryOwned / FlattenMapAccessOwned
// in the owned family, but parameterized over MapAccess<'de> and using
// MapArmStack<'de, KP> instead of MapArmStackOwned<KP>.

/// Continuation for [`FlattenMapAccess`]. Determines what happens after
/// the outer and inner arm stacks are combined via [`StackConcat`].
///
/// Terminal case: [`FlattenTerminal`] calls `map.iterate(SkipUnknown!(arms))`.
/// Intermediate case: generated by the derive macro - calls the next flatten
/// field's `Deserialize::deserialize` with a new [`FlattenDeserializer`].
#[allow(async_fn_in_trait)]
pub trait FlattenCont<'de, M: MapAccess<'de>> {
    /// Extra result produced by the continuation (e.g., values from subsequent
    /// flatten fields). `()` for the terminal case.
    type Extra;

    /// Drive the combined arm stack to completion.
    ///
    /// `arms` is `StackConcat(outer_arms, inner_arms)`.
    async fn finish<Arms: MapArmStack<'de, M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs, Self::Extra)>, M::Error>;
}

impl<'de, M: MapAccess<'de>> FlattenCont<'de, M> for crate::FlattenTerminal {
    type Extra = ();

    async fn finish<Arms: MapArmStack<'de, M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs, ())>, M::Error> {
        let wrapped = crate::SkipUnknown!(arms, M::KeyProbe, VP<'de, M::KeyProbe>);
        let (claim, out) = crate::hit!(map.iterate(wrapped).await);
        Ok(Probe::Hit((claim, out, ())))
    }
}

#[cfg(feature = "alloc")]
impl<'de, M: MapAccess<'de>> FlattenCont<'de, M> for crate::FlattenTerminalBoxed {
    type Extra = ();

    async fn finish<Arms: MapArmStack<'de, M::KeyProbe>>(
        self,
        map: M,
        arms: Arms,
    ) -> Result<Probe<(M::MapClaim, Arms::Outputs, ())>, M::Error> {
        let wrapped = crate::SkipUnknown!(arms, M::KeyProbe, VP<'de, M::KeyProbe>);
        #[allow(clippy::type_complexity)]
        let r: Result<Probe<(M::MapClaim, Arms::Outputs)>, M::Error> =
            alloc::boxed::Box::pin(map.iterate(wrapped)).await;
        let (claim, out) = crate::hit!(r);
        Ok(Probe::Hit((claim, out, ())))
    }
}

/// Facade [`Deserializer<'de>`] used to implement `#[strede(flatten)]` for the borrow family.
///
/// Pass this as the deserializer to a flattened field's [`Deserialize`] impl. It
/// intercepts `deserialize_map` → `iterate(inner_arms)`, prepends the outer
/// struct's arms via [`StackConcat`], and delegates to the [`FlattenCont`]
/// continuation. For the terminal case ([`FlattenTerminal`]) this drives the
/// real map's `iterate`; for intermediate cases it chains to the next flatten field.
pub struct FlattenDeserializer<'a, 'de, M, OuterArms, Cont>
where
    M: MapAccess<'de>,
    OuterArms: MapArmStack<'de, M::KeyProbe>,
    Cont: FlattenCont<'de, M>,
{
    pub map: M,
    pub outer_arms: &'a core::cell::Cell<Option<OuterArms>>,
    pub outer_outputs: &'a core::cell::Cell<Option<OuterArms::Outputs>>,
    pub cont: Cont,
    pub extra: &'a core::cell::Cell<Option<Cont::Extra>>,
    pub _de: core::marker::PhantomData<&'de ()>,
}

impl<'a, 'de, M, OuterArms, Cont> FlattenDeserializer<'a, 'de, M, OuterArms, Cont>
where
    M: MapAccess<'de>,
    OuterArms: MapArmStack<'de, M::KeyProbe>,
    Cont: FlattenCont<'de, M>,
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
            _de: core::marker::PhantomData,
        }
    }
}

impl<'a, 'de, M, OuterArms, Cont> Deserializer<'de>
    for FlattenDeserializer<'a, 'de, M, OuterArms, Cont>
where
    M: MapAccess<'de>,
    OuterArms: MapArmStack<'de, M::KeyProbe>,
    Cont: FlattenCont<'de, M>,
{
    type Error = M::Error;
    type Claim = M::MapClaim;
    type EntryClaim = M::MapClaim;
    type Entry = FlattenEntry<'a, 'de, M, OuterArms, Cont>;

    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        assert!(N == 1, "FlattenDeserializer only supports entry<1, ...>");
        let entry = FlattenEntry {
            map: self.map,
            outer_arms: self.outer_arms,
            outer_outputs: self.outer_outputs,
            cont: self.cont,
            extra: self.extra,
            _de: core::marker::PhantomData,
        };
        let mut slot = Some(entry);
        let entries: [Self::Entry; N] = core::array::from_fn(|_| slot.take().unwrap());
        f(entries).await
    }
}

/// Entry handle produced by [`FlattenDeserializer`].
pub struct FlattenEntry<'a, 'de, M, OuterArms, Cont>
where
    M: MapAccess<'de>,
    OuterArms: MapArmStack<'de, M::KeyProbe>,
    Cont: FlattenCont<'de, M>,
{
    pub map: M,
    pub outer_arms: &'a core::cell::Cell<Option<OuterArms>>,
    pub outer_outputs: &'a core::cell::Cell<Option<OuterArms::Outputs>>,
    pub cont: Cont,
    pub extra: &'a core::cell::Cell<Option<Cont::Extra>>,
    pub _de: core::marker::PhantomData<&'de ()>,
}

impl<'a, 'de, M, OuterArms, Cont> Entry<'de> for FlattenEntry<'a, 'de, M, OuterArms, Cont>
where
    M: MapAccess<'de>,
    OuterArms: MapArmStack<'de, M::KeyProbe>,
    Cont: FlattenCont<'de, M>,
{
    type Error = M::Error;
    type Claim = M::MapClaim;
    type StrChunks = crate::Never<'static, M::MapClaim, M::Error>;
    type BytesChunks = crate::Never<'static, M::MapClaim, M::Error>;
    type Map = FlattenMapAccess<'a, 'de, M, OuterArms, Cont>;
    type Seq = crate::Never<'static, M::MapClaim, M::Error>;

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        Ok(Probe::Hit(FlattenMapAccess {
            map: self.map,
            outer_arms: self.outer_arms,
            outer_outputs: self.outer_outputs,
            cont: self.cont,
            extra: self.extra,
            _de: core::marker::PhantomData,
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
        panic!("FlattenEntry::fork called; flatten only supports N=1 entry")
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        panic!("FlattenEntry::skip called on flatten entry")
    }
}

/// [`MapAccess<'de>`] produced by [`FlattenEntry`].
///
/// When `iterate(inner_arms)` is called, it takes the outer arms from the
/// `Cell`, builds `StackConcat(outer_arms, inner_arms)`, and delegates to
/// `cont.finish(map, combined)`. The continuation either drives the real map
/// (terminal) or chains to the next flatten field (intermediate).
pub struct FlattenMapAccess<'a, 'de, M, OuterArms, Cont>
where
    M: MapAccess<'de>,
    OuterArms: MapArmStack<'de, M::KeyProbe>,
    Cont: FlattenCont<'de, M>,
{
    pub map: M,
    pub outer_arms: &'a core::cell::Cell<Option<OuterArms>>,
    pub outer_outputs: &'a core::cell::Cell<Option<OuterArms::Outputs>>,
    pub cont: Cont,
    pub extra: &'a core::cell::Cell<Option<Cont::Extra>>,
    pub _de: core::marker::PhantomData<&'de ()>,
}

impl<'a, 'de, M, OuterArms, Cont> MapAccess<'de> for FlattenMapAccess<'a, 'de, M, OuterArms, Cont>
where
    M: MapAccess<'de>,
    OuterArms: MapArmStack<'de, M::KeyProbe>,
    Cont: FlattenCont<'de, M>,
{
    type Error = M::Error;
    type MapClaim = M::MapClaim;
    type KeyProbe = M::KeyProbe;

    fn fork(&mut self) -> Self {
        panic!("FlattenMapAccess::fork not supported")
    }

    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        inner_arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        let outer_arms = self
            .outer_arms
            .take()
            .expect("FlattenMapAccess::iterate called without outer arms");
        let combined = StackConcat(outer_arms, inner_arms);
        let (claim, (outer_out, inner_out), extra) =
            crate::hit!(self.cont.finish(self.map, combined).await);
        self.outer_outputs.set(Some(outer_out));
        self.extra.set(Some(extra));
        Ok(Probe::Hit((claim, inner_out)))
    }
}

// ---------------------------------------------------------------------------
// Macros for borrow family
// ---------------------------------------------------------------------------

/// Wraps a [`MapArmStack`] so that unknown map keys are silently consumed (borrow family).
#[macro_export]
macro_rules! SkipUnknown {
    ($inner:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbe as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp: $kp| kp.deserialize_key::<$crate::Skip, _>(()),
                |vp: $vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbe as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
    ($inner:expr) => {{
        use $crate::MapKeyProbe as _;
        (
            $inner,
            $crate::VirtualArmSlot::new(
                |kp| kp.deserialize_key::<$crate::Skip, _>(()),
                |vp, _k: $crate::Skip| async move {
                    use $crate::MapValueProbe as _;
                    let vc = vp.skip().await?;
                    ::core::result::Result::Ok($crate::Probe::Hit((vc, ())))
                },
            ),
        )
    }};
}

/// Wraps a [`MapArmStack`] to return a duplicate-field error (borrow family).
#[macro_export]
macro_rules! DetectDuplicates {
    ($inner:expr, $wire_names:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbe as _;
        let __wn = $wire_names;
        $crate::DetectDuplicatesOwned::new(
            $inner,
            __wn,
            move |kp: $kp| kp.deserialize_key::<$crate::MatchVals<usize>, _>(__wn),
            |vp: $vp| vp.skip(),
        )
    }};
}

/// Wraps a [`MapArmStack`] to intercept a tag field (borrow family).
#[macro_export]
macro_rules! TagInjectingStack {
    ($inner:expr, $tag_field:expr, $tag_candidates:expr, $tag_value:expr, $kp:ty, $vp:ty) => {{
        use $crate::MapKeyProbe as _;
        use $crate::MapValueProbe as _;
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

// ---------------------------------------------------------------------------
// SeqAccess
// ---------------------------------------------------------------------------

/// Iterates the elements of a sequence.  Obtained from [`Entry::deserialize_seq`].
///
/// ```rust,ignore
/// let mut out = Vec::new();
/// loop {
///     match hit!(seq.next(|[e]| async {
///         let (claim, v) = hit!(e.get::<u32, ()>(()).await);
///         Ok(Probe::Hit((claim, v)))
///     }).await) {
///         Chunk::Data((n_seq, v)) => {
///             out.push(v)
///             seq = n_seq;
///         },
///         Chunk::Done(claim) => return Ok(Probe::Hit((claim, out))),
///     }
/// }
/// ```
pub trait SeqAccess<'de>: Sized {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;

    /// Proof-of-consumption token returned on sequence exhaustion; must match
    /// the enclosing [`Entry::Claim`].
    type SeqClaim: 'de;

    /// Proof-of-consumption token produced by [`SeqEntry`] probes; threaded
    /// back through the closure return value to advance the sequence.
    type ElemClaim: 'de;

    /// Owned handle for one sequence element.  See [`SeqEntry`].
    type Elem: SeqEntry<'de, Claim = Self::ElemClaim, Error = Self::Error>;

    /// Fork a sibling accessor at the same sequence position.
    ///
    /// See [`MapAccess::fork`] for full semantics.
    fn fork(&mut self) -> Self;

    /// Advance to the next element, passing `N` owned [`SeqEntry`] handles to `f`.
    ///
    /// Returns:
    /// - `Ok(Probe::Hit(Chunk::Data(r)))` - an element was consumed.
    /// - `Ok(Probe::Hit(Chunk::Done(claim)))` - sequence exhausted.
    /// - `Ok(Probe::Miss)` - closure returned Miss.
    /// - `Err(e)` - fatal format error.
    /// - `Pending` - no data yet.
    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>;
}

/// Owned handle for one element in a sequence.
pub trait SeqEntry<'de> {
    /// Fatal error type; must match the parent [`Deserializer::Error`].
    type Error: DeserializeError;
    /// Proof-of-consumption token; must match the enclosing [`SeqAccess::Claim`].
    type Claim: 'de;

    /// Deserialize the element as `T`, forwarding `extra` into `T::deserialize`.
    /// Returns `Ok(Probe::Hit((claim, value)))`; the claim must be returned from
    /// the closure passed to [`SeqAccess::next`] to advance the stream.
    /// `Ok(Probe::Miss)` if `T::deserialize` misses; `Err` on fatal format error only.
    async fn get<T: Deserialize<'de, Extra>, Extra>(
        self,
        extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>;

    /// Fork a sibling element handle for the same sequence slot.
    ///
    /// Both handles refer to the same element; whichever resolves `get`
    /// first claims it.
    fn fork(&mut self) -> Self
    where
        Self: Sized;

    /// Consume and discard the element without deserializing it.
    /// Returns the [`Claim`](SeqEntry::Claim) so the stream can advance.
    async fn skip(self) -> Result<Self::Claim, Self::Error>;
}

// ---------------------------------------------------------------------------
// Never impls - borrow family
// ---------------------------------------------------------------------------

impl<'n, C, E: DeserializeError> StrAccess for crate::Never<'n, C, E> {
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

impl<'n, C, E: DeserializeError> BytesAccess for crate::Never<'n, C, E> {
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

impl<'n, 'de, C: 'de, E: DeserializeError> SeqAccess<'de> for crate::Never<'n, C, E> {
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
        Fut: core::future::Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> SeqEntry<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type Claim = C;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn get<T: Deserialize<'de, Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error> {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> MapAccess<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type KeyProbe = crate::Never<'n, C, E>;

    fn fork(&mut self) -> Self {
        match self.0 {}
    }

    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        _arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> MapKeyProbe<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type KeyClaim = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_key<K: Deserialize<'de, Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error> {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> MapKeyClaim<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type ValueProbe = crate::Never<'n, C, E>;
    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> MapValueProbe<'de> for crate::Never<'n, C, E> {
    type Error = E;
    type MapClaim = C;
    type ValueClaim = crate::Never<'n, C, E>;
    fn fork(&mut self) -> Self {
        match self.0 {}
    }
    async fn deserialize_value<V: Deserialize<'de, Extra>, Extra>(
        self,
        _extra: Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error> {
        match self.0 {}
    }
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        match self.0 {}
    }
}

impl<'n, 'de, C: 'de, E: DeserializeError> MapValueClaim<'de> for crate::Never<'n, C, E> {
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
// Deserialize impls for primitives
// ---------------------------------------------------------------------------

macro_rules! impl_deserialize_primitive {
    ($ty:ty, $method:ident) => {
        impl<'de> Deserialize<'de> for $ty {
            async fn deserialize<D: Deserializer<'de>>(
                d: D,
                _extra: (),
            ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
                d.entry(|[e]| async { e.$method().await }).await
            }
        }
    };
}

impl_deserialize_primitive!(bool, deserialize_bool);
impl_deserialize_primitive!(u8, deserialize_u8);
impl_deserialize_primitive!(u16, deserialize_u16);
impl_deserialize_primitive!(u32, deserialize_u32);
impl_deserialize_primitive!(u64, deserialize_u64);
impl_deserialize_primitive!(u128, deserialize_u128);
impl_deserialize_primitive!(i8, deserialize_i8);
impl_deserialize_primitive!(i16, deserialize_i16);
impl_deserialize_primitive!(i32, deserialize_i32);
impl_deserialize_primitive!(i64, deserialize_i64);
impl_deserialize_primitive!(i128, deserialize_i128);
impl_deserialize_primitive!(f32, deserialize_f32);
impl_deserialize_primitive!(f64, deserialize_f64);
impl_deserialize_primitive!(char, deserialize_char);

impl<'de: 'a, 'a> Deserialize<'de> for &'a str {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_str()).await
    }
}

impl<'de: 'a, 'a> Deserialize<'de> for &'a [u8] {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        _extra: (),
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_bytes()).await
    }
}

impl<'de, Extra: Copy, T: Deserialize<'de, Extra>> Deserialize<'de, Extra> for Option<T> {
    async fn deserialize<D: Deserializer<'de>>(
        d: D,
        extra: Extra,
    ) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| e.deserialize_option::<T, Extra>(extra)).await
    }
}
