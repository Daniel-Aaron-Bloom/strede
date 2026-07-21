//! Tag handling for CBOR semantic tags (major type 6).
//!
//! The `TagHandler` trait abstracts how a CBOR consumer deals with a tag number.
//! Built-in implementations cover the common cases; downstream crates can
//! implement the trait for custom strategies.
//!
//! `CborTag<T, H>` deserializes CBOR tags from the concrete CBOR deserializer
//! types (borrow and owned families).  It is NOT a generic `Deserialize<'de, D>`
//! for any D; it only works with CBOR-specific deserializers because tag-stripping
//! requires access to the raw token.
//!
//! The standard borrow/owned entry impls (`CborEntry`, `ChunkedCborEntry`) strip
//! all `Tag` tokens before dispatching probe methods, so normal `Deserialize`
//! impls never see tags.  User code that wants the tag number uses `CborTag<T, H>`.

// ---------------------------------------------------------------------------
// TagHandler trait
// ---------------------------------------------------------------------------

/// Controls how a `CborTag<T, H>` consumer reacts to CBOR tag numbers.
pub trait TagHandler: Sized {
    /// Called when tag number `n` is encountered.
    /// Return `Some(self)` to continue deserializing; `None` → `Probe::Miss`.
    fn handle(self, n: u64) -> Option<Self>;

    /// Called after the wrapped value is successfully deserialized.
    /// Return `true` to accept; `false` → `Probe::Miss`.
    fn finish(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Built-in TagHandler impls
// ---------------------------------------------------------------------------

/// Silently ignores all tag numbers.
#[derive(Clone, Copy, Default)]
pub struct Ignored;

impl TagHandler for Ignored {
    #[inline(always)]
    fn handle(self, _n: u64) -> Option<Self> {
        Some(Self)
    }
    #[inline(always)]
    fn finish(&self) -> bool {
        true
    }
}

/// Requires exactly tag number `N`.  Misses on any other tag or no tag.
#[derive(Clone, Copy)]
pub struct Required<const N: u64> {
    pub seen: bool,
}

impl<const N: u64> Required<N> {
    pub fn new() -> Self {
        Self { seen: false }
    }
}

impl<const N: u64> Default for Required<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: u64> TagHandler for Required<N> {
    fn handle(self, n: u64) -> Option<Self> {
        if n == N {
            Some(Self { seen: true })
        } else {
            None
        }
    }
    fn finish(&self) -> bool {
        self.seen
    }
}

/// Captures the most-recent tag number (if any).  Always succeeds.
#[derive(Clone, Copy, Default)]
pub struct Captured {
    pub tag: Option<u64>,
}

impl TagHandler for Captured {
    fn handle(self, n: u64) -> Option<Self> {
        Some(Self { tag: Some(n) })
    }
    fn finish(&self) -> bool {
        true
    }
}

/// Accepts tag `N` if present, but also accepts no tag.
#[derive(Clone, Copy, Default)]
pub struct Accepted<const N: u64>;

impl<const N: u64> TagHandler for Accepted<N> {
    fn handle(self, n: u64) -> Option<Self> {
        if n == N { Some(Self) } else { None }
    }
    fn finish(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// CborTag<T, H>
// ---------------------------------------------------------------------------

/// A wrapper that deserializes `T` after processing any leading CBOR tag(s)
/// with handler `H`.
///
/// `CborTag<T>` uses `Ignored` by default — tags are transparently stripped.
///
/// Implements `Deserialize<'de, CborDeserializer<'de>>` and the corresponding
/// `CborSubDeserializer<'de>` variant, as well as `DeserializeOwned` for the
/// chunked family.  The `Extra` type is `(H, T::Extra)`.
///
/// The tag-stripping loop reads raw tokens from the `src` pointer directly,
/// which is why this type can only be implemented for the concrete CBOR types
/// rather than the generic `Entry<'de>` trait.
pub struct CborTag<T, H: TagHandler = Ignored> {
    pub handler: H,
    pub value: T,
}
