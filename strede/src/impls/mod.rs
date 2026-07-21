//! Built-in `Deserialize` / `DeserializeOwned` impls.
//!
//! Ships the universal-D utility impls that derive bound-emission depends on
//! (Match, MatchVals, Skip, Option, PhantomData), Cow<'de, str>, and collection
//! impls (Vec, Box<[T]>, HashMap, BTreeMap, HashSet, tuples). Each collection
//! emits both a universal `Deserialize<'de, D>` impl that delegates through
//! `Entry::deserialize_map_into` / `deserialize_seq_into` and a shape-specific
//! `DeserializeFromMap` / `DeserializeFromSeq` impl that owns the iteration body.

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::{
    Chunk, Probe,
    borrow::{
        BytesAccess, Deserialize, DeserializeFromSeq, Deserializer, Entry, SeqAccess, SeqEntry,
        StrAccess,
    },
    owned::{
        DeserializeFromSeqOwned, DeserializeOwned, DeserializerOwned, EntryOwned, SeqAccessOwned,
        SeqEntryOwned, StrAccessOwned,
    },
};
use core::marker::PhantomData;

mod flat_source;
mod maps;
mod pair_seq_map;
mod sequences;
pub mod string_enum;
mod strings;
mod tag_flatten;
mod utility;

pub use flat_source::{MapFieldProvider, MapFieldProviderOwned};
pub use pair_seq_map::{PairSeqKeyProbe, PairSeqMapAccess, PairSeqValueProbe, PairStep, RawSlot};
pub use tag_flatten::{TagAwareMap, TagAwareMapOwned};
pub use utility::{Match, MatchVals, Skip, UnwrapOrElse};
