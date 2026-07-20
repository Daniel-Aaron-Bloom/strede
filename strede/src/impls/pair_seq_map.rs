//! Generic `MapAccessOwned` over "flat vector of alternating key/value slots"
//! wire formats (msgpack fixmap/map16/map32, CBOR major type 5, and similar).
//!
//! Formats whose maps carry no structural framing beyond a pair count —
//! nothing analogous to JSON's `{`, `:`, `,`, `}` — implement [`RawSlot`] once
//! for their existing claim/cursor type and get the full
//! `MapKeyProbeOwned` / `MapKeyClaimOwned` / `MapValueProbeOwned` /
//! `MapValueClaimOwned` / `MapAccessOwned` quintet for free via the blanket
//! impls in this module, instead of hand-rolling it per format.
//!
//! `RawSlot` is implemented directly on the format's own claim type (not a
//! wrapper) because [`EntryOwned`] requires `type Map: MapAccessOwned<MapClaim
//! = Self::Claim>` — the map's claim must be the exact same type as every
//! other claim the entry produces. `remaining_after` is therefore bookkeeping
//! the format's claim type must be able to carry and expose, not state this
//! module can own on the side.
//!
//! `remaining_after() == None` covers formats with indefinite-length maps
//! (e.g. CBOR's `Break`-terminated maps): no count is known up front, so the
//! wire format itself must signal completion via [`RawSlot::next_token_or_done`].

use crate::owned::{
    DeserializerOwned, MapAccessOwned, MapArmStackOwned, MapKeyClaimOwned, MapKeyProbeOwned,
    MapValueClaimOwned, MapValueProbeOwned,
};
use crate::{DeserializeError, DeserializeOwned, NextKey, Probe};

/// Outcome of [`RawSlot::next_token_or_done`].
pub enum PairStep<C, T> {
    /// Another key/value slot follows.
    More(C, T),
    /// No more slots; the map is exhausted.
    Done(C),
}

/// The primitive a format supplies to drive [`PairSeqMapAccess`].
///
/// `Self` doubles as the claim/cursor type threaded through key and value
/// probes — it must be the same type the format uses as its `EntryOwned::Claim`.
pub trait RawSlot: Sized {
    type Error: DeserializeError;
    type Token: Copy;
    type SubDeserializer: DeserializerOwned<Claim = Self, Error = Self::Error>;

    /// Independent cursor for racing multiple arms against the same slot.
    fn fork(&mut self) -> Self;

    /// Read the next raw slot, advancing the cursor. Only called when the
    /// caller already knows another slot exists (a definite-length counter
    /// has not yet reached zero).
    async fn next_token(self) -> Result<(Self, Self::Token), Self::Error>;

    /// Read the next slot, or detect the format's own end-of-map marker.
    /// Only called for maps of unknown length (`remaining_after() == None`).
    /// Formats with definite-length-only maps never hit this and can leave
    /// the default, which just forwards to [`RawSlot::next_token`].
    async fn next_token_or_done(self) -> Result<PairStep<Self, Self::Token>, Self::Error> {
        let (cursor, token) = self.next_token().await?;
        Ok(PairStep::More(cursor, token))
    }

    /// Build a sub-deserializer over a just-read slot for probing/dispatch.
    fn into_sub_deserializer(self, token: Self::Token) -> Self::SubDeserializer;

    /// Skip a just-read slot's payload entirely (`MapValueProbeOwned::skip`).
    async fn skip_token(self, token: Self::Token) -> Result<Self, Self::Error>;

    /// Key-value pairs this cursor still owes the map iteration; `None` means
    /// unknown length (a self-terminating wire format).
    fn remaining_after(&self) -> Option<usize>;

    /// Rebuild this cursor carrying an updated pairs-remaining count.
    fn with_remaining_after(self, remaining: Option<usize>) -> Self;
}

pub struct PairSeqMapAccess<C: RawSlot> {
    cursor: C,
    remaining_pairs: Option<usize>,
}

impl<C: RawSlot> PairSeqMapAccess<C> {
    /// `remaining_pairs = None` means the map's length is unknown up front
    /// (the format must self-terminate via [`RawSlot::next_token_or_done`]).
    pub fn new(cursor: C, remaining_pairs: Option<usize>) -> Self {
        Self {
            cursor,
            remaining_pairs,
        }
    }
}

pub struct PairSeqKeyProbe<C: RawSlot> {
    cursor: C,
    token: C::Token,
    remaining_after: Option<usize>,
}

impl<C: RawSlot> PairSeqKeyProbe<C> {
    /// Construct a key probe directly over an already-read token, without
    /// going through [`PairSeqMapAccess::iterate`]. Used by callers that
    /// drive a single key/value pair by hand (e.g. externally-tagged enum
    /// variant payloads, which are a single-key map).
    pub fn new(cursor: C, token: C::Token, remaining_after: Option<usize>) -> Self {
        Self {
            cursor,
            token,
            remaining_after,
        }
    }
}

pub struct PairSeqValueProbe<C: RawSlot> {
    cursor: C,
    token: C::Token,
    remaining_after: Option<usize>,
}

/// Read the next key slot, or signal the map is exhausted, given the pairs
/// still owed. Shared by [`PairSeqMapAccess::iterate`]'s first read and the
/// blanket [`MapValueClaimOwned::next_key`] impl below.
async fn next_pair<C: RawSlot>(
    cursor: C,
    remaining_pairs: Option<usize>,
) -> Result<PairStep<C, C::Token>, C::Error> {
    match remaining_pairs {
        Some(0) => Ok(PairStep::Done(cursor.with_remaining_after(Some(0)))),
        Some(n) => {
            let (cursor, token) = cursor.next_token().await?;
            Ok(PairStep::More(
                cursor.with_remaining_after(Some(n - 1)),
                token,
            ))
        }
        None => match cursor.next_token_or_done().await? {
            PairStep::Done(cursor) => Ok(PairStep::Done(cursor.with_remaining_after(None))),
            PairStep::More(cursor, token) => {
                Ok(PairStep::More(cursor.with_remaining_after(None), token))
            }
        },
    }
}

impl<C: RawSlot> MapAccessOwned for PairSeqMapAccess<C> {
    type Error = C::Error;
    type MapClaim = C;
    type KeyProbe = PairSeqKeyProbe<C>;

    async fn iterate<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        pair_seq_map_iterate(self, arms).await
    }

    async fn iterate_dyn<S: MapArmStackOwned<Self::KeyProbe>>(
        self,
        arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        pair_seq_map_iterate(self, arms).await
    }
}

/// Shared body for [`MapAccessOwned::iterate`] / [`MapAccessOwned::iterate_dyn`]
/// — msgpack/CBOR maps are wire-identical for structs and dynamic collections
/// (both are just "N pairs", definite or self-terminating), so both trait
/// methods delegate here rather than one calling the other.
async fn pair_seq_map_iterate<C: RawSlot, S: MapArmStackOwned<PairSeqKeyProbe<C>>>(
    map: PairSeqMapAccess<C>,
    mut arms: S,
) -> Result<Probe<(C, S::Outputs)>, C::Error> {
    let mut key_probe_opt = match next_pair(map.cursor, map.remaining_pairs).await? {
        PairStep::Done(cursor) => {
            return Ok(Probe::Hit((cursor, arms.take_outputs())));
        }
        PairStep::More(cursor, token) => {
            let remaining_after = cursor.remaining_after();
            Some(PairSeqKeyProbe {
                cursor,
                token,
                remaining_after,
            })
        }
    };

    loop {
        let key_probe = key_probe_opt.take().unwrap();

        let (arm_index, key_claim) = match arms.race_keys(key_probe).await? {
            Probe::Miss => return Ok(Probe::Miss),
            Probe::Hit(x) => x,
        };

        let value_probe = key_claim.into_value_probe().await?;

        let (value_claim, ()) = match arms.dispatch_value(arm_index, value_probe).await? {
            Probe::Miss => return Ok(Probe::Miss),
            Probe::Hit(x) => x,
        };

        match value_claim
            .next_key(arms.unsatisfied_count(), arms.open_count())
            .await?
        {
            NextKey::Done(map_claim) => {
                return Ok(Probe::Hit((map_claim, arms.take_outputs())));
            }
            NextKey::Entry(next_kp) => {
                key_probe_opt = Some(next_kp);
            }
        }
    }
}

impl<C: RawSlot> MapKeyProbeOwned for PairSeqKeyProbe<C> {
    type Error = C::Error;
    type KeyClaim = C;
    type KeySubDeserializer = C::SubDeserializer;

    fn fork(&mut self) -> Self {
        Self {
            cursor: self.cursor.fork(),
            token: self.token,
            remaining_after: self.remaining_after,
        }
    }

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: DeserializeOwned<Self::KeySubDeserializer>,
    {
        let sub = self.cursor.into_sub_deserializer(self.token);
        match K::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, k)) => Ok(Probe::Hit((
                claim.with_remaining_after(self.remaining_after),
                k,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

impl<C: RawSlot> MapValueProbeOwned for PairSeqValueProbe<C> {
    type Error = C::Error;
    type MapClaim = C;
    type ValueClaim = C;
    type ValueSubDeserializer = C::SubDeserializer;

    fn fork(&mut self) -> Self {
        Self {
            cursor: self.cursor.fork(),
            token: self.token,
            remaining_after: self.remaining_after,
        }
    }

    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeOwned<Self::ValueSubDeserializer>,
    {
        let sub = self.cursor.into_sub_deserializer(self.token);
        match V::deserialize_owned(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((
                claim.with_remaining_after(self.remaining_after),
                v,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let cursor = self.cursor.skip_token(self.token).await?;
        Ok(cursor.with_remaining_after(self.remaining_after))
    }
}

impl<C: RawSlot> MapKeyClaimOwned for C {
    type Error = C::Error;
    type MapClaim = C;
    type ValueProbe = PairSeqValueProbe<C>;

    async fn into_value_probe(self) -> Result<Self::ValueProbe, Self::Error> {
        let remaining_after = self.remaining_after();
        let (cursor, token) = self.next_token().await?;
        Ok(PairSeqValueProbe {
            cursor,
            token,
            remaining_after,
        })
    }
}

impl<C: RawSlot> MapValueClaimOwned for C {
    type Error = C::Error;
    type KeyProbe = PairSeqKeyProbe<C>;
    type MapClaim = C;

    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        let remaining_after = self.remaining_after();
        match next_pair(self, remaining_after).await? {
            PairStep::Done(cursor) => Ok(NextKey::Done(cursor)),
            PairStep::More(cursor, token) => {
                let remaining_after = cursor.remaining_after();
                Ok(NextKey::Entry(PairSeqKeyProbe {
                    cursor,
                    token,
                    remaining_after,
                }))
            }
        }
    }
}
