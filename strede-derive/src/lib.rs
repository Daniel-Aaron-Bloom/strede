//! Derive macros for the `strede` deserialization framework.
//!
//! This crate provides two derives:
//!
//! - [`Deserialize`] — borrow family (`'de`); enables zero-copy borrowing from the source buffer.
//! - [`DeserializeOwned`] — owned family (`'s`); for chunked / streaming sources with no zero-copy borrows.
//!
//! Both derives are driven by `#[strede(...)]` attributes. All attributes described below are
//! recognized by both derives unless noted otherwise.
//!
//! # Container attributes
//!
//! Applied to the `struct` or `enum` itself.
//!
//! ## `#[strede(rename_all = "convention")]`
//!
//! Converts every field or variant name to the given case convention before it is used as
//! the wire key. An explicit `#[strede(rename)]` on a field/variant takes priority.
//! Aliases are always literal — they are never transformed.
//!
//! Supported conventions: `"lowercase"`, `"UPPERCASE"`, `"PascalCase"`, `"camelCase"`,
//! `"snake_case"`, `"SCREAMING_SNAKE_CASE"`, `"kebab-case"`, `"SCREAMING-KEBAB-CASE"`.
//!
//! ## `#[strede(tag = "field")]`
//!
//! Internally-tagged enum. The discriminant is a key _inside_ the map:
//! `{"type": "Move", "x": 1}`. All variant kinds are supported, but newtype/tuple variants
//! must themselves deserialize from a map (the tag facade only exposes `deserialize_map`).
//! Cannot be combined with `untagged` or `other`.
//!
//! ## `#[strede(tag = "t", content = "c")]`
//!
//! Adjacently-tagged enum. The outer map contains exactly the tag field and a content
//! field: `{"t": "Move", "c": {"x": 1}}`. Key order is irrelevant. Unit variants have no
//! content field. Cannot be combined with `untagged` or `other`.
//!
//! ## `#[strede(allow_unknown_fields)]`
//!
//! On a struct. Unknown map keys are silently skipped. Required fields that are absent
//! still return `Probe::Miss`; duplicate fields still return `Err`.
//!
//! ## `#[strede(crate = "path")]`
//!
//! Overrides the crate root used in generated code (default: `::strede`). Useful when
//! strede is re-exported under a different name.
//!
//! ## `#[strede(bound = "T: MyTrait")]`
//!
//! Replaces *all* auto-generated where-clause predicates in the `impl` block. An empty
//! string (`bound = ""`) suppresses every auto-generated bound. When set at the container
//! level, field-level `bound` annotations are ignored.
//!
//! # Field attributes
//!
//! ## `#[strede(rename = "wire_name")]`
//!
//! Changes the wire key for this field or variant. The Rust identifier is unchanged.
//!
//! ## `#[strede(alias = "alt_name")]`
//!
//! Adds an extra wire name that is accepted during deserialization. Repeatable.
//! Cannot be used on untagged variants.
//!
//! ## `#[strede(default)]` / `#[strede(default = "expr")]`
//!
//! If the field is absent from the data, uses `Default::default()` or evaluates `expr`.
//! `expr` may be a function path (called with no arguments) or a value expression
//! (`42`, `String::new()`, `vec![]`).
//!
//! ## `#[strede(skip_deserializing)]`
//!
//! The field's key is treated as unknown and always takes its default value.
//! Must be combined with `default` or `default = "fn"`.
//!
//! ## `#[strede(deserialize_with = "path")]`
//!
//! Uses a custom function instead of `T::deserialize` for the **borrow** family.
//! The function must have the same signature as `Deserialize::deserialize`.
//!
//! ## `#[strede(deserialize_owned_with = "path")]`
//!
//! Same as `deserialize_with` but for the **owned** family.
//!
//! ## `#[strede(with = "module")]`
//!
//! Shorthand: uses `module::deserialize` for the borrow family and
//! `module::deserialize_owned` for the owned family.
//!
//! ## `#[strede(from = "FromType")]`
//!
//! Deserializes `FromType` and converts to `Self` (container) or `FieldType` (field)
//! via `From::from`. Mutually exclusive with `try_from` and `deserialize_with` / `with`.
//!
//! ## `#[strede(try_from = "FromType")]`
//!
//! Like `from`, but uses `TryFrom::try_from`. A failed conversion returns `Probe::Miss`
//! rather than an error.
//!
//! ## `#[strede(bound = "T: MyTrait")]`
//!
//! At the field level, replaces only the predicate emitted for this field's type.
//! Ignored when a container-level `bound` is present.
//!
//! ## `#[strede(flatten)]`
//!
//! Merges the field's map keys into the parent struct's map iteration — no wrapping
//! map token in the data. Multiple flatten fields per struct are supported. Unknown keys
//! (not claimed by the outer struct or any flattened type) are silently skipped. Cannot
//! be combined with `rename`, `alias`, `default`, `skip_deserializing`,
//! `deserialize_with`, `from`, or `try_from`.
//!
//! ## `#[strede(borrow)]` / `#[strede(borrow = "'a + 'b")]`
//!
//! **Borrow family only.** Controls which `'de: 'lifetime` bounds are emitted for a
//! field.
//!
//! - No attribute (default): emits `'de: 'a` for each lifetime appearing directly in a
//!   `&'a T`, `&'a mut T`, or `Cow<'a, T>` at the top level of the field type.
//! - `#[strede(borrow)]`: emits `'de: 'a` for every lifetime found anywhere in the
//!   field type, including those inside nested generics.
//! - `#[strede(borrow = "'a + 'b")]`: emits `'de: 'a` only for the explicitly listed
//!   lifetimes.
//!
//! # Enum variant attributes
//!
//! ## `#[strede(untagged)]`
//!
//! On an enum or individual variant. Untagged variants are deserialized by shape
//! (trying each in declaration order) rather than by name tag. Tagged variants are tried
//! first; untagged variants act as sequential fallback. Unit variants match null;
//! newtype/tuple/struct variants use `deserialize_value`.
//!
//! ## `#[strede(other)]`
//!
//! On a **unit** variant. Acts as catch-all: returned when no tagged variant matches the
//! discriminant, instead of `Probe::Miss`. For map-keyed (non-unit) variants, the
//! unknown key's value is consumed before returning. At most one per enum. Cannot be
//! combined with `rename`, `alias`, `untagged`, or coexist with `untagged` variants on
//! the same enum.
//!
//! # Struct attributes
//!
//! ## `#[strede(transparent)]`
//!
//! On a struct with exactly one non-skipped field. The struct deserializes as its inner
//! field directly — no map/seq wrapper. Works on named structs and tuple structs
//! (newtypes). Skipped fields use their defaults.

extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod borrow;
mod common;
mod owned;

/// Derive `Deserialize` for the **borrow family** (`'de` lifetime).
///
/// The generated impl borrows directly from the source buffer wherever possible —
/// `&str`, `&[u8]`, and `Cow<'_, str>` fields are zero-copy. The `'de` lifetime
/// is threaded through the impl and all associated types.
///
/// All shared `#[strede(...)]` attributes apply. In addition:
///
/// - [`#[strede(borrow)]`][crate] / `#[strede(borrow = "'a + 'b")]` — controls
///   which `'de: 'lifetime` outlives bounds are generated for a field (see crate docs).
/// - [`#[strede(deserialize_with = "path")]`][crate] — custom deserializer function
///   for the borrow family only (see crate docs).
///
/// # Enum representations
///
/// When no tagging attribute is present the representation is **externally tagged**:
/// unit variants match a bare string (`"Ping"`); non-unit variants match a single-key
/// map whose key is the variant name (`{"Move": ...}`). Unknown variant names return
/// `Probe::Miss`.
#[proc_macro_derive(Deserialize, attributes(strede))]
pub fn derive_deserialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    borrow::expand(input)
        .unwrap_or_else(|e| e.into_compile_error())
        .into()
}

/// Derive `DeserializeOwned` for the **owned family** (`'s` lifetime).
///
/// The generated impl reads from a chunked or streaming source. There are no
/// zero-copy borrows — string and byte fields are always accumulated through chunk
/// accessors. This family is used with the `chunked` JSON deserializer and any
/// streaming format implementation.
///
/// All shared `#[strede(...)]` attributes apply. In addition:
///
/// - [`#[strede(deserialize_owned_with = "path")]`][crate] — custom deserializer
///   function for the owned family only (see crate docs).
///
/// Attributes that are **not** applicable to this derive:
///
/// - `#[strede(borrow)]` — has no meaning without a `'de` borrow lifetime.
/// - `#[strede(deserialize_with)]` — use `deserialize_owned_with` or `with` instead.
///
/// # Enum representations
///
/// Same as [`Deserialize`]: externally tagged by default; `#[strede(tag)]` and
/// `#[strede(tag, content)]` select internally-tagged and adjacently-tagged respectively.
#[proc_macro_derive(DeserializeOwned, attributes(strede))]
pub fn derive_deserialize_owned(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    owned::expand(input)
        .unwrap_or_else(|e| e.into_compile_error())
        .into()
}
