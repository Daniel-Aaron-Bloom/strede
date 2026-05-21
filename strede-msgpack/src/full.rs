//! In-memory borrow-family MessagePack deserializer.
//!
//! Implements [`strede::Deserializer`] over a borrowed `&[u8]` source.
//! Entry probe futures resolve immediately — `Ok(Probe::Hit(...))` when the
//! token type matches, `Ok(Probe::Miss)` when it does not.  `Pending` is
//! never reached for in-memory use.
//!
//! # Zero-copy strings and bytes
//!
//! MessagePack strings contain no escape sequences; the payload bytes are
//! valid UTF-8 directly in the source buffer.  [`Entry::deserialize_str`]
//! always returns a zero-copy `&'de str` when the token is a `Str`.
//!
//! [`Entry::deserialize_bytes`] accepts both `Bin` and `Str` tokens and
//! returns the raw bytes zero-copy.
//!
//! # Numbers
//!
//! Numbers are stored in binary form in msgpack; there is no text to parse.
//! [`Entry::deserialize_number_chunks`] always returns `Probe::Miss` — format-
//! specific `Deserialize` impls in [`crate::impls`] use the typed token value
//! directly via `MsgpackEntry::parse_num`.

use crate::{
    MsgpackError,
    token::{MsgpackToken, next_token, skip_value},
};
use strede::{
    BytesAccess, Chunk, Deserialize, DeserializeFromMap, DeserializeFromSeq, Deserializer, Entry,
    MapAccess, MapArmStack, MapKeyClaim, MapKeyProbe, MapValueClaim, MapValueProbe, Never, NextKey,
    Probe, SeqAccess, SeqEntry, StrAccess, hit, utils::repeat,
};

// ---------------------------------------------------------------------------
// MsgpackClaim — top-level proof of consumption
// ---------------------------------------------------------------------------

pub struct MsgpackClaim<'de> {
    pub(crate) src: &'de [u8],
    /// Remaining map key-value pairs after this claim (map context only); 0 otherwise.
    pub(crate) remaining_after: usize,
}

// ---------------------------------------------------------------------------
// MsgpackDeserializer
// ---------------------------------------------------------------------------

pub struct MsgpackDeserializer<'de> {
    src: &'de [u8],
}

impl<'de> MsgpackDeserializer<'de> {
    pub fn new(src: &'de [u8]) -> Self {
        Self { src }
    }
}

// ---------------------------------------------------------------------------
// MsgpackSubDeserializer
// ---------------------------------------------------------------------------

/// Sub-deserializer created internally for map keys/values, seq elements,
/// `deserialize_option`, and `deserialize_value`. Has a pre-loaded token.
pub struct MsgpackSubDeserializer<'de> {
    src: &'de [u8],
    pending_tok: MsgpackToken,
}

impl<'de> MsgpackSubDeserializer<'de> {
    #[inline(always)]
    pub(crate) fn new(src: &'de [u8], tok: MsgpackToken) -> Self {
        Self {
            src,
            pending_tok: tok,
        }
    }
}

// ---------------------------------------------------------------------------
// Deserializer impls
// ---------------------------------------------------------------------------

#[inline(always)]
async fn msgpack_root_next<'de, const N: usize, F, Fut, R>(
    mut de: MsgpackDeserializer<'de>,
    mut f: F,
) -> Result<Probe<(MsgpackClaim<'de>, R)>, MsgpackError>
where
    F: FnMut([MsgpackEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(MsgpackClaim<'de>, R)>, MsgpackError>>,
{
    let token = next_token(&mut de.src)?;
    let entry = MsgpackEntry { token, src: de.src };
    match f(repeat(entry, |e| e.clone())).await? {
        Probe::Hit((claim, r)) => {
            if !claim.src.is_empty() {
                return Err(MsgpackError::ExpectedEnd);
            }
            Ok(Probe::Hit((claim, r)))
        }
        Probe::Miss => Ok(Probe::Miss),
    }
}

impl<'de> Deserializer<'de> for MsgpackDeserializer<'de> {
    type Error = MsgpackError;
    type Claim = MsgpackClaim<'de>;
    type EntryClaim = MsgpackClaim<'de>;
    type Entry = MsgpackEntry<'de>;

    #[inline(always)]
    async fn entry<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        msgpack_root_next(self, f).await
    }
}

impl<'de> Deserializer<'de> for MsgpackSubDeserializer<'de> {
    type Error = MsgpackError;
    type Claim = MsgpackClaim<'de>;
    type EntryClaim = MsgpackClaim<'de>;
    type Entry = MsgpackEntry<'de>;

    #[inline(always)]
    async fn entry<const N: usize, F, Fut, R>(
        self,
        mut f: F,
    ) -> Result<Probe<(Self::Claim, R)>, Self::Error>
    where
        F: FnMut([Self::Entry; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::EntryClaim, R)>, Self::Error>>,
    {
        let entry = MsgpackEntry {
            token: self.pending_tok,
            src: self.src,
        };
        f(repeat(entry, |e| e.clone())).await
    }
}

// ---------------------------------------------------------------------------
// MsgpackEntry
// ---------------------------------------------------------------------------

pub struct MsgpackEntry<'de> {
    pub(crate) token: MsgpackToken,
    /// Buffer position after the header bytes — payload starts here.
    pub(crate) src: &'de [u8],
}

impl<'de> MsgpackEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            token: self.token,
            src: self.src,
        }
    }
}

impl<'de> Entry<'de> for MsgpackEntry<'de> {
    type Error = MsgpackError;
    type Claim = MsgpackClaim<'de>;
    type SubDeserializer = MsgpackSubDeserializer<'de>;
    type StrChunks = MsgpackStrAccess<'de>;
    type BytesChunks = MsgpackBytesAccess<'de>;
    type NumberChunks = Never<'de, MsgpackClaim<'de>, MsgpackError>;
    type Map = MsgpackMapAccess<'de>;
    type Seq = MsgpackSeqAccess<'de>;

    fn fork(&mut self) -> Self {
        self.clone()
    }

    // ---- Strings ------------------------------------------------------------

    async fn deserialize_str(self) -> Result<Probe<(Self::Claim, &'de str)>, Self::Error> {
        match self.token {
            MsgpackToken::Str(len) => {
                if self.src.len() < len {
                    return Err(MsgpackError::UnexpectedEnd);
                }
                let (payload, rest) = self.src.split_at(len);
                let s = core::str::from_utf8(payload).map_err(|_| MsgpackError::InvalidUtf8)?;
                Ok(Probe::Hit((
                    MsgpackClaim {
                        src: rest,
                        remaining_after: 0,
                    },
                    s,
                )))
            }
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_str_chunks(self) -> Result<Probe<Self::StrChunks>, Self::Error> {
        match self.token {
            MsgpackToken::Str(len) => {
                if self.src.len() < len {
                    return Err(MsgpackError::UnexpectedEnd);
                }
                Ok(Probe::Hit(MsgpackStrAccess {
                    src: self.src,
                    remaining: len,
                }))
            }
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Bytes --------------------------------------------------------------

    async fn deserialize_bytes(self) -> Result<Probe<(Self::Claim, &'de [u8])>, Self::Error> {
        match self.token {
            MsgpackToken::Bin(len) | MsgpackToken::Str(len) => {
                if self.src.len() < len {
                    return Err(MsgpackError::UnexpectedEnd);
                }
                let (payload, rest) = self.src.split_at(len);
                Ok(Probe::Hit((
                    MsgpackClaim {
                        src: rest,
                        remaining_after: 0,
                    },
                    payload,
                )))
            }
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_bytes_chunks(self) -> Result<Probe<Self::BytesChunks>, Self::Error> {
        match self.token {
            MsgpackToken::Bin(len) | MsgpackToken::Str(len) => {
                if self.src.len() < len {
                    return Err(MsgpackError::UnexpectedEnd);
                }
                Ok(Probe::Hit(MsgpackBytesAccess {
                    src: self.src,
                    remaining: len,
                }))
            }
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Numbers ------------------------------------------------------------

    async fn deserialize_number_chunks(self) -> Result<Probe<Self::NumberChunks>, Self::Error> {
        Ok(Probe::Miss)
    }

    // ---- Map / Seq ----------------------------------------------------------

    async fn deserialize_map(self) -> Result<Probe<Self::Map>, Self::Error> {
        match self.token {
            MsgpackToken::Map(count) => Ok(Probe::Hit(MsgpackMapAccess {
                src: self.src,
                remaining: count,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    async fn deserialize_seq(self) -> Result<Probe<Self::Seq>, Self::Error> {
        match self.token {
            MsgpackToken::Array(count) => Ok(Probe::Hit(MsgpackSeqAccess {
                src: self.src,
                remaining: count,
            })),
            _ => Ok(Probe::Miss),
        }
    }

    // ---- Option -------------------------------------------------------------

    #[inline(always)]
    async fn deserialize_option<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, Option<T>)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        match self.token {
            MsgpackToken::Nil => Ok(Probe::Hit((
                MsgpackClaim {
                    src: self.src,
                    remaining_after: 0,
                },
                None,
            ))),
            other => {
                let sub = MsgpackSubDeserializer::new(self.src, other);
                let (claim, v) = hit!(T::deserialize(sub, extra).await);
                Ok(Probe::Hit((claim, Some(v))))
            }
        }
    }

    // ---- Value / Map / Seq forwarding ---------------------------------------

    #[inline(always)]
    async fn deserialize_value<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = MsgpackSubDeserializer::new(self.src, self.token);
        T::deserialize(sub, extra).await
    }

    #[inline(always)]
    async fn deserialize_map_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>,
    {
        let map = hit!(Entry::deserialize_map(self).await);
        T::deserialize_from_map(map, extra).await
    }

    #[inline(always)]
    async fn deserialize_seq_into<T>(
        self,
        extra: T::Extra,
    ) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeq<'de, Self::Seq>,
    {
        let seq = hit!(Entry::deserialize_seq(self).await);
        T::deserialize_from_seq(seq, extra).await
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut src = self.src;
        skip_value(&mut src, self.token)?;
        Ok(MsgpackClaim {
            src,
            remaining_after: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// parse_num helper — used by format-specific Deserialize impls
// ---------------------------------------------------------------------------

impl<'de> MsgpackEntry<'de> {
    /// Extract a native msgpack number as type `T`.
    /// Returns `Probe::Miss` on token type mismatch or value out of range.
    pub(crate) async fn parse_num<T: ParseNum>(
        self,
    ) -> Result<Probe<(MsgpackClaim<'de>, T)>, MsgpackError> {
        let v = match self.token {
            MsgpackToken::UInt(n) => T::from_uint(n),
            MsgpackToken::Int(n) => T::from_int(n),
            MsgpackToken::Float32(f) => T::from_f32(f),
            MsgpackToken::Float64(f) => T::from_f64(f),
            _ => return Ok(Probe::Miss),
        };
        match v {
            Some(value) => Ok(Probe::Hit((
                MsgpackClaim {
                    src: self.src,
                    remaining_after: 0,
                },
                value,
            ))),
            None => Ok(Probe::Miss),
        }
    }
}

pub(crate) trait ParseNum: Sized {
    fn from_uint(v: u64) -> Option<Self>;
    fn from_int(v: i64) -> Option<Self>;
    fn from_f32(v: f32) -> Option<Self>;
    fn from_f64(v: f64) -> Option<Self>;
}

// ---------------------------------------------------------------------------
// MsgpackStrAccess / MsgpackBytesAccess
// ---------------------------------------------------------------------------

pub struct MsgpackStrAccess<'de> {
    src: &'de [u8],
    remaining: usize,
}

impl<'de> StrAccess for MsgpackStrAccess<'de> {
    type Claim = MsgpackClaim<'de>;
    type Error = MsgpackError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            src: self.src,
            remaining: self.remaining,
        }
    }

    async fn next_str<R>(
        self,
        f: impl FnOnce(&str) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(MsgpackClaim {
                src: self.src,
                remaining_after: 0,
            }));
        }
        if self.src.len() < self.remaining {
            return Err(MsgpackError::UnexpectedEnd);
        }
        let (payload, rest) = self.src.split_at(self.remaining);
        let s = core::str::from_utf8(payload).map_err(|_| MsgpackError::InvalidUtf8)?;
        let r = f(s);
        // All bytes in one chunk; next call will return Done.
        Ok(Chunk::Data((
            Self {
                src: rest,
                remaining: 0,
            },
            r,
        )))
    }
}

pub struct MsgpackBytesAccess<'de> {
    pub(crate) src: &'de [u8],
    pub(crate) remaining: usize,
}

impl<'de> BytesAccess for MsgpackBytesAccess<'de> {
    type Claim = MsgpackClaim<'de>;
    type Error = MsgpackError;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            src: self.src,
            remaining: self.remaining,
        }
    }

    async fn next_bytes<R>(
        self,
        f: impl FnOnce(&[u8]) -> R,
    ) -> Result<Chunk<(Self, R), Self::Claim>, Self::Error> {
        if self.remaining == 0 {
            return Ok(Chunk::Done(MsgpackClaim {
                src: self.src,
                remaining_after: 0,
            }));
        }
        if self.src.len() < self.remaining {
            return Err(MsgpackError::UnexpectedEnd);
        }
        let (payload, rest) = self.src.split_at(self.remaining);
        let r = f(payload);
        Ok(Chunk::Data((
            Self {
                src: rest,
                remaining: 0,
            },
            r,
        )))
    }
}

// ---------------------------------------------------------------------------
// Map access type chain
//
// MsgpackClaim serves as both KeyClaim and ValueClaim. remaining_after carries
// the remaining pair count through the iteration chain. This allows the inner
// sub-deserializer (MsgpackSubDeserializer, Claim = MsgpackClaim) to satisfy
// MapKeyProbe::KeySubDeserializer: Deserializer<Claim = KeyClaim = MsgpackClaim>
// and MapValueProbe::ValueSubDeserializer: Deserializer<Claim = ValueClaim = MsgpackClaim>.
// ---------------------------------------------------------------------------

pub struct MsgpackMapAccess<'de> {
    src: &'de [u8],
    remaining: usize,
}

// --- MapKeyClaim and MapValueClaim on MsgpackClaim ---

impl<'de> MapKeyClaim<'de> for MsgpackClaim<'de> {
    type Error = MsgpackError;
    type MapClaim = MsgpackClaim<'de>;
    type ValueProbe = MsgpackMapValueProbe<'de>;

    async fn into_value_probe(mut self) -> Result<Self::ValueProbe, Self::Error> {
        let value_tok = next_token(&mut self.src)?;
        Ok(MsgpackMapValueProbe {
            src: self.src,
            value_tok,
            remaining_after: self.remaining_after,
        })
    }
}

impl<'de> MapValueClaim<'de> for MsgpackClaim<'de> {
    type Error = MsgpackError;
    type KeyProbe = MsgpackMapKeyProbe<'de>;
    type MapClaim = MsgpackClaim<'de>;

    async fn next_key(
        self,
        _unsatisfied: usize,
        _open: usize,
    ) -> Result<NextKey<Self::KeyProbe, Self::MapClaim>, Self::Error> {
        if self.remaining_after == 0 {
            return Ok(NextKey::Done(MsgpackClaim {
                src: self.src,
                remaining_after: 0,
            }));
        }
        let mut src = self.src;
        let key_tok = next_token(&mut src)?;
        Ok(NextKey::Entry(MsgpackMapKeyProbe {
            src,
            key_tok,
            remaining_after: self.remaining_after - 1,
        }))
    }
}

// --- Key probe ---

pub struct MsgpackMapKeyProbe<'de> {
    src: &'de [u8],
    key_tok: MsgpackToken,
    /// Remaining pairs *after* this key-value pair completes.
    remaining_after: usize,
}

impl<'de> MsgpackMapKeyProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            key_tok: self.key_tok,
            remaining_after: self.remaining_after,
        }
    }
}

impl<'de> MapKeyProbe<'de> for MsgpackMapKeyProbe<'de> {
    type Error = MsgpackError;
    type KeyClaim = MsgpackClaim<'de>;
    type KeySubDeserializer = MsgpackSubDeserializer<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    async fn deserialize_key<K>(
        self,
        extra: K::Extra,
    ) -> Result<Probe<(Self::KeyClaim, K)>, Self::Error>
    where
        K: Deserialize<'de, Self::KeySubDeserializer>,
    {
        let sub = MsgpackSubDeserializer::new(self.src, self.key_tok);
        match K::deserialize(sub, extra).await? {
            Probe::Hit((claim, k)) => Ok(Probe::Hit((
                MsgpackClaim {
                    src: claim.src,
                    remaining_after: self.remaining_after,
                },
                k,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }
}

// --- Value probe ---

pub struct MsgpackMapValueProbe<'de> {
    /// Buffer position after the value header — value payload starts here.
    src: &'de [u8],
    value_tok: MsgpackToken,
    /// Pair count remaining *after* this value is consumed.
    remaining_after: usize,
}

impl<'de> MsgpackMapValueProbe<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            value_tok: self.value_tok,
            remaining_after: self.remaining_after,
        }
    }
}

impl<'de> MapValueProbe<'de> for MsgpackMapValueProbe<'de> {
    type Error = MsgpackError;
    type MapClaim = MsgpackClaim<'de>;
    type ValueClaim = MsgpackClaim<'de>;
    type ValueSubDeserializer = MsgpackSubDeserializer<'de>;
    type ValueMap = MsgpackMapAccess<'de>;
    type ValueSeq = MsgpackSeqAccess<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    #[inline(always)]
    async fn deserialize_value<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: Deserialize<'de, Self::ValueSubDeserializer>,
    {
        let remaining_after = self.remaining_after;
        let sub = MsgpackSubDeserializer::new(self.src, self.value_tok);
        match V::deserialize(sub, extra).await? {
            Probe::Hit((claim, v)) => Ok(Probe::Hit((
                MsgpackClaim {
                    src: claim.src,
                    remaining_after,
                },
                v,
            ))),
            Probe::Miss => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_map_into<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeFromMap<'de, Self::ValueMap>,
    {
        let remaining_after = self.remaining_after;
        match self.value_tok {
            MsgpackToken::Map(count) => {
                let map = MsgpackMapAccess {
                    src: self.src,
                    remaining: count,
                };
                match V::deserialize_from_map(map, extra).await? {
                    Probe::Hit((claim, v)) => Ok(Probe::Hit((
                        MsgpackClaim {
                            src: claim.src,
                            remaining_after,
                        },
                        v,
                    ))),
                    Probe::Miss => Ok(Probe::Miss),
                }
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn deserialize_seq_into<V>(
        self,
        extra: V::Extra,
    ) -> Result<Probe<(Self::ValueClaim, V)>, Self::Error>
    where
        V: DeserializeFromSeq<'de, Self::ValueSeq>,
    {
        let remaining_after = self.remaining_after;
        match self.value_tok {
            MsgpackToken::Array(count) => {
                let seq = MsgpackSeqAccess {
                    src: self.src,
                    remaining: count,
                };
                match V::deserialize_from_seq(seq, extra).await? {
                    Probe::Hit((claim, v)) => Ok(Probe::Hit((
                        MsgpackClaim {
                            src: claim.src,
                            remaining_after,
                        },
                        v,
                    ))),
                    Probe::Miss => Ok(Probe::Miss),
                }
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::ValueClaim, Self::Error> {
        let remaining_after = self.remaining_after;
        let mut src = self.src;
        skip_value(&mut src, self.value_tok)?;
        Ok(MsgpackClaim {
            src,
            remaining_after,
        })
    }
}

// --- MapAccess ---

impl<'de> MapAccess<'de> for MsgpackMapAccess<'de> {
    type Error = MsgpackError;
    type MapClaim = MsgpackClaim<'de>;
    type KeyProbe = MsgpackMapKeyProbe<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            src: self.src,
            remaining: self.remaining,
        }
    }

    async fn iterate<S: MapArmStack<'de, Self::KeyProbe>>(
        self,
        mut arms: S,
    ) -> Result<Probe<(Self::MapClaim, S::Outputs)>, Self::Error> {
        let mut src = self.src;
        let mut remaining = self.remaining;

        // Build the first key probe, or return early for empty maps.
        let mut key_probe_opt: Option<MsgpackMapKeyProbe<'de>> = if remaining == 0 {
            return Ok(Probe::Hit((
                MsgpackClaim {
                    src,
                    remaining_after: 0,
                },
                arms.take_outputs(),
            )));
        } else {
            let key_tok = next_token(&mut src)?;
            remaining -= 1;
            Some(MsgpackMapKeyProbe {
                src,
                key_tok,
                remaining_after: remaining,
            })
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
}

// ---------------------------------------------------------------------------
// MsgpackSeqAccess / MsgpackSeqEntry
// ---------------------------------------------------------------------------

pub struct MsgpackSeqAccess<'de> {
    pub(crate) src: &'de [u8],
    pub(crate) remaining: usize,
}

pub struct MsgpackSeqEntry<'de> {
    src: &'de [u8],
    elem_tok: MsgpackToken,
}

impl<'de> MsgpackSeqEntry<'de> {
    fn clone(&self) -> Self {
        Self {
            src: self.src,
            elem_tok: self.elem_tok,
        }
    }
}

#[inline(always)]
async fn msgpack_seq_next<'de, const N: usize, F, Fut, R>(
    seq: MsgpackSeqAccess<'de>,
    mut f: F,
) -> Result<Probe<Chunk<(MsgpackSeqAccess<'de>, R), MsgpackClaim<'de>>>, MsgpackError>
where
    F: FnMut([MsgpackSeqEntry<'de>; N]) -> Fut,
    Fut: core::future::Future<Output = Result<Probe<(MsgpackClaim<'de>, R)>, MsgpackError>>,
{
    if seq.remaining == 0 {
        return Ok(Probe::Hit(Chunk::Done(MsgpackClaim {
            src: seq.src,
            remaining_after: 0,
        })));
    }
    let mut src = seq.src;
    let elem_tok = next_token(&mut src)?;
    let entry = MsgpackSeqEntry { src, elem_tok };
    let new_seq = MsgpackSeqAccess {
        src,
        remaining: seq.remaining - 1,
    };
    let (claim, r) = hit!(f(repeat(entry, |e| e.clone())).await);
    let updated_seq = MsgpackSeqAccess {
        src: claim.src,
        remaining: new_seq.remaining,
    };
    Ok(Probe::Hit(Chunk::Data((updated_seq, r))))
}

impl<'de> SeqAccess<'de> for MsgpackSeqAccess<'de> {
    type Error = MsgpackError;
    type SeqClaim = MsgpackClaim<'de>;
    type ElemClaim = MsgpackClaim<'de>;
    type Elem = MsgpackSeqEntry<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        Self {
            src: self.src,
            remaining: self.remaining,
        }
    }

    #[inline(always)]
    async fn next<const N: usize, F, Fut, R>(
        self,
        f: F,
    ) -> Result<Probe<Chunk<(Self, R), Self::SeqClaim>>, Self::Error>
    where
        F: FnMut([Self::Elem; N]) -> Fut,
        Fut: core::future::Future<Output = Result<Probe<(Self::ElemClaim, R)>, Self::Error>>,
    {
        msgpack_seq_next(self, f).await
    }
}

impl<'de> SeqEntry<'de> for MsgpackSeqEntry<'de> {
    type Error = MsgpackError;
    type Claim = MsgpackClaim<'de>;
    type SubDeserializer = MsgpackSubDeserializer<'de>;
    type Map = MsgpackMapAccess<'de>;
    type Seq = MsgpackSeqAccess<'de>;

    #[inline(always)]
    fn fork(&mut self) -> Self {
        self.clone()
    }

    #[inline(always)]
    async fn get<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: Deserialize<'de, Self::SubDeserializer>,
    {
        let sub = MsgpackSubDeserializer::new(self.src, self.elem_tok);
        T::deserialize(sub, extra).await
    }

    #[inline(always)]
    async fn get_map_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromMap<'de, Self::Map>,
    {
        match self.elem_tok {
            MsgpackToken::Map(count) => {
                let map = MsgpackMapAccess {
                    src: self.src,
                    remaining: count,
                };
                T::deserialize_from_map(map, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn get_seq_into<T>(self, extra: T::Extra) -> Result<Probe<(Self::Claim, T)>, Self::Error>
    where
        T: DeserializeFromSeq<'de, Self::Seq>,
    {
        match self.elem_tok {
            MsgpackToken::Array(count) => {
                let seq = MsgpackSeqAccess {
                    src: self.src,
                    remaining: count,
                };
                T::deserialize_from_seq(seq, extra).await
            }
            _ => Ok(Probe::Miss),
        }
    }

    #[inline(always)]
    async fn skip(self) -> Result<Self::Claim, Self::Error> {
        let mut src = self.src;
        skip_value(&mut src, self.elem_tok)?;
        Ok(MsgpackClaim {
            src,
            remaining_after: 0,
        })
    }
}
