//! Compile-time-only wire-name collision detection for `#[strede(flatten)]`.
//!
//! This is a *separate* mechanism from [`crate::MapFieldProvider::WireNames`] /
//! `wire_names()`, which is a runtime `(name, arm_index)` table consumed by
//! `DetectDuplicates` while actually deserializing. `Fields::NAMES` exists
//! purely so the derive macro can force a `const`-time assertion — it plays
//! no role in any runtime code path.
//!
//! `NAMES` is the full, transitively-unioned set of wire names/tags a type
//! contributes when used as a `#[strede(flatten)]` target — e.g. for a plain
//! struct, its own field names plus (recursively) every one of its own
//! flatten fields' contributions; for an externally-tagged enum, its variant
//! names only (a flattened externally-tagged enum never splices its variants'
//! *inner* fields into the parent map, so those never surface here).

use core::marker::PhantomData;

/// Implemented by every `#[derive(Deserialize)]` / `#[derive(DeserializeOwned)]`
/// struct and enum. See the module docs for what `NAMES` represents.
pub trait Fields {
    const NAMES: &'static [&'static str];
}

/// Byte-wise `str` equality, usable in `const fn` on stable Rust (`str`'s
/// `PartialEq` impl is not `const`).
pub const fn str_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Does any name in `a` also appear in `b`?
pub const fn slices_overlap(a: &[&str], b: &[&str]) -> bool {
    let mut i = 0;
    while i < a.len() {
        let mut j = 0;
        while j < b.len() {
            if str_eq(a[i], b[j]) {
                return true;
            }
            j += 1;
        }
        i += 1;
    }
    false
}

/// Does `a` contain the same name twice (e.g. two fields whose `rename`/
/// `alias` values collide within a single struct or enum variant)?
pub const fn has_duplicates(a: &[&str]) -> bool {
    let mut i = 0;
    while i < a.len() {
        let mut j = i + 1;
        while j < a.len() {
            if str_eq(a[i], a[j]) {
                return true;
            }
            j += 1;
        }
        i += 1;
    }
    false
}

/// Forces a compile-time "no duplicate names within `T` itself" check —
/// e.g. two fields on the same struct whose `rename`/`alias` values collide,
/// independent of any `#[strede(flatten)]` composition.
pub struct NoInternalDuplicates<T>(PhantomData<T>);

impl<T: Fields> NoInternalDuplicates<T> {
    pub const CHECK: () = {
        if has_duplicates(T::NAMES) {
            panic!("wire name collision: two fields/variants declare the same wire name");
        }
    };
}

/// Forces a compile-time pairwise disjointness check between two [`Fields`]
/// participants of the same `#[strede(flatten)]` scope.
///
/// Referencing `Disjoint::<A, B>::CHECK` from a top-level `const _: () = ...;`
/// item forces immediate, unconditional evaluation — used when both `A` and
/// `B` are concrete at the point the derive macro generates the reference.
///
/// When either `A` or `B` is still a generic type parameter at that point
/// (e.g. a `#[strede(flatten)]` field whose type is the container's own
/// generic parameter), evaluation is naturally deferred by the compiler until
/// something monomorphizes the reference with concrete types — which, for a
/// derive-generated reference embedded in `MapFieldProvider::wire_names()`'s
/// body, happens exactly when the container is actually used to deserialize.
/// A collision then surfaces as a compile error at that concrete
/// instantiation, not at the (still-abstract) generic definition.
pub struct Disjoint<A, B>(PhantomData<(A, B)>);

impl<A: Fields, B: Fields> Disjoint<A, B> {
    pub const CHECK: () = {
        if slices_overlap(A::NAMES, B::NAMES) {
            panic!(
                "wire name collision: two `#[strede(flatten)]` participants declare the same wire name"
            );
        }
    };
}
