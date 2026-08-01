use convert_case::{Case, Casing};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Fields, Token};

// ---------------------------------------------------------------------------
// Generics-lift helpers
// ---------------------------------------------------------------------------

/// Insert `'de` (if absent) and `__D: Deserializer<'de>` into `impl_gen`.
/// Call between cloning `input.generics` and `split_for_impl`.
pub fn insert_de_and_d_borrow(impl_gen: &mut syn::Generics, krate: &syn::Path) {
    let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
    if !has_de {
        impl_gen.params.insert(0, syn::parse_quote!('de));
    }
    impl_gen
        .params
        .push(syn::parse_quote!(__D: #krate::Deserializer<'de>));
}

/// Insert `__D: DeserializerOwned` into `impl_gen` (owned mirror).
pub fn insert_d_owned(impl_gen: &mut syn::Generics, krate: &syn::Path) {
    impl_gen
        .params
        .push(syn::parse_quote!(__D: #krate::DeserializerOwned));
}

/// Insert `'de` (if absent) and `__M: MapAccess<'de>` into `impl_gen`.
/// Used for `DeserializeFromMap` impl emission.
pub fn insert_de_and_m_borrow(impl_gen: &mut syn::Generics, krate: &syn::Path) {
    let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
    if !has_de {
        impl_gen.params.insert(0, syn::parse_quote!('de));
    }
    impl_gen
        .params
        .push(syn::parse_quote!(__M: #krate::MapAccess<'de>));
}

/// Insert `__M: MapAccessOwned` into `impl_gen` (owned mirror).
pub fn insert_m_owned(impl_gen: &mut syn::Generics, krate: &syn::Path) {
    impl_gen
        .params
        .push(syn::parse_quote!(__M: #krate::MapAccessOwned));
}

// ---------------------------------------------------------------------------
// Bound shapes (D3/D4)
// ---------------------------------------------------------------------------

/// Where a field is consumed; determines which probe-trait projection to use
/// when emitting the auto field bound.
#[derive(Copy, Clone)]
pub enum FieldContext {
    /// `<T as Deserialize<'de, __D>>::deserialize(d, ())` — top-level / transparent / `from`.
    Direct,
    /// `__vp.deserialize_value::<T>(())` on `VP2<'de, __D>`.
    MapValue,
    /// `__se.get::<T>(())` on `SE<'de, __D>`.
    SeqElem,
}

/// Auto-generated bound for one field in the borrow family.
///
/// Borrow family always assumes `'de` is in scope. `d_ident` names the
/// in-scope `Deserializer<'de>` type param (usually `__D`, but helper impls
/// with their own generic — e.g. tuple-variant seq helpers — may use a
/// different name like `__D2` to avoid colliding with an outer `__D`).
pub fn field_bound_borrow(
    krate: &syn::Path,
    ty: &syn::Type,
    ctx: FieldContext,
    d_ident: &syn::Ident,
) -> syn::WherePredicate {
    match ctx {
        FieldContext::Direct => syn::parse_quote!(
            #ty: #krate::Deserialize<'de, #d_ident, Extra = ()>
        ),
        FieldContext::MapValue => syn::parse_quote!(
            #ty: #krate::Deserialize<
                'de,
                <#krate::borrow::VP2<'de, #d_ident> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                Extra = ()
            >
        ),
        FieldContext::SeqElem => syn::parse_quote!(
            #ty: #krate::Deserialize<
                'de,
                <#krate::borrow::SE<'de, #d_ident> as #krate::SeqEntry<'de>>::SubDeserializer,
                Extra = ()
            >
        ),
    }
}

/// Auto-generated bound for one field in the owned family.
///
/// `d_ident` names the in-scope `DeserializerOwned` type param — see
/// `field_bound_borrow` for why this isn't always `__D`.
pub fn field_bound_owned(
    krate: &syn::Path,
    ty: &syn::Type,
    ctx: FieldContext,
    d_ident: &syn::Ident,
) -> syn::WherePredicate {
    match ctx {
        FieldContext::Direct => syn::parse_quote!(
            #ty: #krate::DeserializeOwned<#d_ident, Extra = ()>
        ),
        FieldContext::MapValue => syn::parse_quote!(
            #ty: #krate::DeserializeOwned<
                <#krate::owned::VP2<#d_ident> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                Extra = ()
            >
        ),
        FieldContext::SeqElem => syn::parse_quote!(
            #ty: #krate::DeserializeOwned<
                <#krate::owned::SE<#d_ident> as #krate::SeqEntryOwned>::SubDeserializer,
                Extra = ()
            >
        ),
    }
}

/// Auto-generated bound for a type parameter on the user's struct/enum.
pub fn type_param_bound_borrow(krate: &syn::Path, ident: &syn::Ident) -> syn::WherePredicate {
    syn::parse_quote!(#ident: #krate::Deserialize<'de, __D, Extra = ()>)
}

/// Auto-generated bound for a type parameter on the user's struct/enum (owned).
pub fn type_param_bound_owned(krate: &syn::Path, ident: &syn::Ident) -> syn::WherePredicate {
    syn::parse_quote!(#ident: #krate::DeserializeOwned<__D, Extra = ()>)
}

// ---------------------------------------------------------------------------
// Type-tree generic-param detection (D6)
// ---------------------------------------------------------------------------

/// Returns true if `ty` has a universal `Deserialize<'de, D> for ty` blanket
/// impl shipped in `strede` core. Derive must skip the field bound for these
/// types — an explicit where-clause `ty: Deserialize<'de, X>` would conflict
/// with the blanket impl and cause "multiple impls or where clauses satisfying"
/// errors.
///
/// Covers the top-level types only — `&str`, `&[u8]`, `Cow<'_, str>`,
/// `PhantomData<…>`. Composite types like `Option<&str>` still need the
/// where-clause bound because the universal Option impl has inner-T bounds
/// that the trait solver checks separately.
pub fn has_universal_blanket(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Reference(r) => match &*r.elem {
            syn::Type::Path(p) => p.path.is_ident("str"),
            syn::Type::Slice(s) => matches!(&*s.elem, syn::Type::Path(p) if p.path.is_ident("u8")),
            _ => false,
        },
        syn::Type::Path(p) => p.path.segments.last().is_some_and(|seg| {
            let n = seg.ident.to_string();
            matches!(n.as_str(), "Cow" | "PhantomData")
        }),
        _ => false,
    }
}

/// Whether a field is flattened.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlattenMode {
    /// Not flattened.
    None,
    /// `#[strede(flatten)]`.
    Plain,
}

// ---------------------------------------------------------------------------
// Variant classification
// ---------------------------------------------------------------------------

pub enum VariantKind<'a> {
    Unit,
    Newtype(&'a syn::Type),
    Tuple(&'a syn::punctuated::Punctuated<syn::Field, Token![,]>),
    Struct(&'a syn::punctuated::Punctuated<syn::Field, Token![,]>),
}

pub struct ClassifiedVariant<'a> {
    pub variant: &'a syn::Variant,
    pub kind: VariantKind<'a>,
    pub wire_name: String,
    pub aliases: Vec<String>,
    pub untagged: bool,
    pub other: bool,
    /// Index in the original enum variant list (used for helper type naming).
    pub index: usize,
}

pub fn classify_variants<'a>(
    data: &'a syn::DataEnum,
    container_attrs: &ContainerAttrs,
) -> syn::Result<Vec<ClassifiedVariant<'a>>> {
    let mut out = Vec::new();
    for (index, variant) in data.variants.iter().enumerate() {
        let vattrs = parse_variant_attrs(&variant.attrs)?;

        if vattrs.untagged && vattrs.rename.is_some() {
            return Err(syn::Error::new_spanned(
                variant,
                "cannot combine #[strede(untagged)] and #[strede(rename)] on the same variant",
            ));
        }

        if vattrs.untagged && !vattrs.aliases.is_empty() {
            return Err(syn::Error::new_spanned(
                variant,
                "cannot use #[strede(alias)] on an untagged variant",
            ));
        }

        if vattrs.other && vattrs.rename.is_some() {
            return Err(syn::Error::new_spanned(
                variant,
                "cannot combine #[strede(other)] with #[strede(rename)]",
            ));
        }

        if vattrs.other && !vattrs.aliases.is_empty() {
            return Err(syn::Error::new_spanned(
                variant,
                "cannot combine #[strede(other)] with #[strede(alias)]",
            ));
        }

        if vattrs.other && vattrs.untagged {
            return Err(syn::Error::new_spanned(
                variant,
                "cannot combine #[strede(other)] with #[strede(untagged)]",
            ));
        }

        if vattrs.other && out.iter().any(|cv: &ClassifiedVariant| cv.other) {
            return Err(syn::Error::new_spanned(
                variant,
                "at most one variant may be #[strede(other)]",
            ));
        }

        let kind = match &variant.fields {
            Fields::Unit => VariantKind::Unit,
            Fields::Unnamed(f) if f.unnamed.len() == 1 => VariantKind::Newtype(&f.unnamed[0].ty),
            Fields::Named(f) => VariantKind::Struct(&f.named),
            Fields::Unnamed(f) => VariantKind::Tuple(&f.unnamed),
        };

        if vattrs.other && !matches!(kind, VariantKind::Unit) {
            return Err(syn::Error::new_spanned(
                variant,
                "#[strede(other)] can only be applied to unit variants",
            ));
        }

        let untagged = container_attrs.untagged || vattrs.untagged;
        let wire_name = wire_name(&variant.ident, &vattrs.rename, container_attrs.rename_all);

        out.push(ClassifiedVariant {
            variant,
            kind,
            wire_name,
            aliases: vattrs.aliases,
            untagged,
            other: vattrs.other,
            index,
        });
    }

    // `other` cannot coexist with untagged variants - the fallback semantics conflict.
    if out.iter().any(|cv| cv.other) && out.iter().any(|cv| cv.untagged) {
        let other_variant = out.iter().find(|cv| cv.other).unwrap();
        return Err(syn::Error::new_spanned(
            other_variant.variant,
            "cannot use #[strede(other)] alongside #[strede(untagged)] variants",
        ));
    }

    Ok(out)
}

/// Return the ident of the `#[strede(other)]` catch-all variant, if present.
pub fn other_variant<'a>(classified: &'a [ClassifiedVariant]) -> Option<&'a syn::Ident> {
    classified
        .iter()
        .find(|cv| cv.other)
        .map(|cv| &cv.variant.ident)
}

/// Collect all types used across all enum variant fields (for generic bounds).
pub fn all_field_types(data: &syn::DataEnum) -> Vec<&syn::Type> {
    let mut types = Vec::new();
    for variant in &data.variants {
        match &variant.fields {
            Fields::Unnamed(f) => {
                for field in &f.unnamed {
                    types.push(&field.ty);
                }
            }
            Fields::Named(f) => {
                for field in &f.named {
                    types.push(&field.ty);
                }
            }
            Fields::Unit => {}
        }
    }
    types
}

// ---------------------------------------------------------------------------
// Attribute parsing
// ---------------------------------------------------------------------------

/// Supported case conventions for `#[strede(rename_all = "...")]`.
#[derive(Copy, Clone)]
pub enum RenameAll {
    /// `"lowercase"`
    Lower,
    /// `"UPPERCASE"`
    Upper,
    /// `"PascalCase"`
    Pascal,
    /// `"camelCase"`
    Camel,
    /// `"snake_case"`
    Snake,
    /// `"SCREAMING_SNAKE_CASE"`
    ScreamingSnake,
    /// `"kebab-case"`
    Kebab,
    /// `"SCREAMING-KEBAB-CASE"`
    ScreamingKebab,
}

impl RenameAll {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "lowercase" => Some(Self::Lower),
            "UPPERCASE" => Some(Self::Upper),
            "PascalCase" => Some(Self::Pascal),
            "camelCase" => Some(Self::Camel),
            "snake_case" => Some(Self::Snake),
            "SCREAMING_SNAKE_CASE" => Some(Self::ScreamingSnake),
            "kebab-case" => Some(Self::Kebab),
            "SCREAMING-KEBAB-CASE" => Some(Self::ScreamingKebab),
            _ => None,
        }
    }

    pub fn apply(self, s: &str) -> String {
        let case = match self {
            Self::Lower => Case::Lower,
            Self::Upper => Case::Upper,
            Self::Pascal => Case::Pascal,
            Self::Camel => Case::Camel,
            Self::Snake => Case::Snake,
            Self::ScreamingSnake => Case::UpperSnake,
            Self::Kebab => Case::Kebab,
            Self::ScreamingKebab => Case::UpperKebab,
        };
        s.to_case(case)
    }
}

pub struct ContainerAttrs {
    pub untagged: bool,
    pub allow_unknown_fields: bool,
    pub transparent: bool,
    pub rename_all: Option<RenameAll>,
    pub crate_path: syn::Path,
    /// When `Some`, replaces all auto-generated where-clause predicates in the
    /// outer `impl` block.  An empty `Vec` suppresses all bounds.
    pub bound: Option<Vec<syn::WherePredicate>>,
    /// `#[strede(from = "FromType")]` - deserialize `FromType`, then call `Self::from(v)`.
    pub from: Option<syn::Type>,
    /// `#[strede(try_from = "FromType")]` - deserialize `FromType`, then call
    /// `Self::try_from(v).ok()`, returning `Probe::Miss` on failure.
    pub try_from: Option<syn::Type>,
    /// `#[strede(tag = "field")]` - internally tagged enum; the named field in the map
    /// is the variant discriminant.
    pub tag: Option<String>,
    /// `#[strede(content = "field")]` - adjacently tagged enum; the named field holds
    /// the variant payload. Requires `tag` to also be set.
    pub content: Option<String>,
}

pub struct VariantAttrs {
    pub rename: Option<String>,
    pub aliases: Vec<String>,
    pub untagged: bool,
    pub other: bool,
}

/// Controls how `'de: 'lifetime` bounds are inferred for borrow-family derives.
#[derive(Clone)]
pub enum BorrowAttr {
    /// `#[strede(borrow)]` - emit `'de: 'a` for every lifetime in the field type.
    All,
    /// `#[strede(borrow = "'a, 'b")]` - emit `'de: 'a` only for the listed lifetimes.
    Explicit(Vec<syn::Lifetime>),
}

pub struct FieldAttrs {
    pub rename: Option<String>,
    pub aliases: Vec<String>,
    pub default: Option<DefaultAttr>,
    pub skip_deserializing: bool,
    pub flatten: FlattenMode,
    pub deserialize_with: Option<syn::ExprPath>,
    pub deserialize_owned_with: Option<syn::ExprPath>,
    /// When `Some`, replaces the auto-generated bound for this field's type.
    /// An empty `Vec` suppresses the bound entirely.
    pub bound: Option<Vec<syn::WherePredicate>>,
    /// `#[strede(from = "FromType")]` - deserialize `FromType`, then call `FieldType::from(v)`.
    pub from: Option<syn::Type>,
    /// `#[strede(try_from = "FromType")]` - deserialize `FromType`, then call
    /// `FieldType::try_from(v).ok()`, returning `Probe::Miss` on failure.
    pub try_from: Option<syn::Type>,
    /// Controls `'de: 'a` bound inference for the borrow-family derive.
    pub borrow: Option<BorrowAttr>,
}

pub enum DefaultAttr {
    /// `#[strede(default)]` - calls `Default::default()`
    Trait,
    /// `#[strede(default = "expr")]` - evaluates `expr` via `DefaultWrapper`.
    /// If `expr` is a function path it gets called; otherwise the value is used as-is.
    Expr(syn::Expr),
}

fn parse_borrow_lifetimes(lit: &syn::LitStr) -> syn::Result<Vec<syn::Lifetime>> {
    let s = lit.value();
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }
    // Split on '+' or ',' (supports both "'a + 'b" and "'a, 'b").
    let mut lifetimes = Vec::new();
    for part in s.split(['+', ',']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let lt: syn::Lifetime = syn::parse_str(part).map_err(|e| syn::Error::new(lit.span(), e))?;
        lifetimes.push(lt);
    }
    Ok(lifetimes)
}

fn parse_bound_predicates(lit: &syn::LitStr) -> syn::Result<Vec<syn::WherePredicate>> {
    let s = lit.value();
    let s = s.trim();
    if s.is_empty() {
        return Ok(vec![]);
    }
    let wc: syn::WhereClause =
        syn::parse_str(&format!("where {s}")).map_err(|e| syn::Error::new(lit.span(), e))?;
    Ok(wc.predicates.into_iter().collect())
}

pub fn parse_container_attrs(attrs: &[syn::Attribute]) -> syn::Result<ContainerAttrs> {
    let mut untagged = false;
    let mut allow_unknown_fields = false;
    let mut transparent = false;
    let mut rename_all: Option<RenameAll> = None;
    let mut crate_path: Option<syn::Path> = None;
    let mut bound: Option<Vec<syn::WherePredicate>> = None;
    let mut from: Option<syn::Type> = None;
    let mut try_from: Option<syn::Type> = None;
    let mut tag: Option<String> = None;
    let mut content: Option<String> = None;
    for attr in attrs {
        if !attr.path().is_ident("strede") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("untagged") {
                untagged = true;
                Ok(())
            } else if meta.path.is_ident("allow_unknown_fields") {
                allow_unknown_fields = true;
                Ok(())
            } else if meta.path.is_ident("transparent") {
                transparent = true;
                Ok(())
            } else if meta.path.is_ident("rename_all") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                let s = s.value();
                rename_all = Some(RenameAll::from_str(&s).ok_or_else(|| {
                    meta.error(format!(
                        "unknown rename_all value {s:?}; expected one of: \
                         lowercase, UPPERCASE, PascalCase, camelCase, snake_case, \
                         SCREAMING_SNAKE_CASE, kebab-case, SCREAMING-KEBAB-CASE",
                    ))
                })?);
                Ok(())
            } else if meta.path.is_ident("crate") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                crate_path = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("bound") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                bound = Some(parse_bound_predicates(&lit)?);
                Ok(())
            } else if meta.path.is_ident("from") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                from = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("try_from") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                try_from = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("tag") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                tag = Some(s.value());
                Ok(())
            } else if meta.path.is_ident("content") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                content = Some(s.value());
                Ok(())
            } else {
                Err(meta.error("unknown strede attribute"))
            }
        })?;
    }
    if from.is_some() && try_from.is_some() {
        // Find the attr span for a better error location.
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cannot use both #[strede(from)] and #[strede(try_from)] on the same item",
        ));
    }
    if content.is_some() && tag.is_none() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[strede(content)] requires #[strede(tag)] to also be set",
        ));
    }
    Ok(ContainerAttrs {
        untagged,
        allow_unknown_fields,
        transparent,
        rename_all,
        crate_path: crate_path.unwrap_or_else(|| syn::parse_quote!(::strede)),
        bound,
        from,
        try_from,
        tag,
        content,
    })
}

pub fn parse_variant_attrs(attrs: &[syn::Attribute]) -> syn::Result<VariantAttrs> {
    let mut rename = None;
    let mut aliases = Vec::new();
    let mut untagged = false;
    let mut other = false;
    for attr in attrs {
        if !attr.path().is_ident("strede") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                rename = Some(s.value());
                Ok(())
            } else if meta.path.is_ident("alias") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                aliases.push(s.value());
                Ok(())
            } else if meta.path.is_ident("untagged") {
                untagged = true;
                Ok(())
            } else if meta.path.is_ident("other") {
                other = true;
                Ok(())
            } else {
                Err(meta.error("unknown strede attribute"))
            }
        })?;
    }
    Ok(VariantAttrs {
        rename,
        aliases,
        untagged,
        other,
    })
}

pub fn parse_field_attrs(attrs: &[syn::Attribute]) -> syn::Result<FieldAttrs> {
    let mut rename = None;
    let mut aliases = Vec::new();
    let mut default = None;
    let mut skip_deserializing = false;
    let mut flatten = FlattenMode::None;
    let mut deserialize_with = None;
    let mut deserialize_owned_with = None;
    let mut with_module: Option<syn::ExprPath> = None;
    let mut bound: Option<Vec<syn::WherePredicate>> = None;
    let mut from: Option<syn::Type> = None;
    let mut try_from: Option<syn::Type> = None;
    let mut borrow: Option<BorrowAttr> = None;
    for attr in attrs {
        if !attr.path().is_ident("strede") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("rename") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                rename = Some(s.value());
                Ok(())
            } else if meta.path.is_ident("alias") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                aliases.push(s.value());
                Ok(())
            } else if meta.path.is_ident("default") {
                if meta.input.peek(syn::Token![=]) {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    let expr: syn::Expr = s.parse()?;
                    default = Some(DefaultAttr::Expr(expr));
                } else {
                    default = Some(DefaultAttr::Trait);
                }
                Ok(())
            } else if meta.path.is_ident("skip_deserializing") {
                skip_deserializing = true;
                Ok(())
            } else if meta.path.is_ident("flatten") {
                if meta.input.peek(syn::token::Paren) {
                    // The only legacy paren form was `flatten(boxed)`, which existed
                    // to break the old continuation-chain async state machine. The
                    // current `MapFieldProvider`-based codegen runs a single
                    // `iterate` future, so the boxing workaround is no longer needed.
                    let inner;
                    syn::parenthesized!(inner in meta.input);
                    let ident: syn::Ident = inner.parse()?;
                    return Err(syn::Error::new_spanned(
                        ident,
                        "`#[strede(flatten(...))]` is no longer supported; use plain \
                         `#[strede(flatten)]` instead. The `flatten(boxed)` workaround \
                         was removed alongside the old continuation-chain codegen.",
                    ));
                }
                flatten = FlattenMode::Plain;
                Ok(())
            } else if meta.path.is_ident("deserialize_with") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                deserialize_with = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("deserialize_owned_with") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                deserialize_owned_with = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("with") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                with_module = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("bound") {
                let value = meta.value()?;
                let lit: syn::LitStr = value.parse()?;
                bound = Some(parse_bound_predicates(&lit)?);
                Ok(())
            } else if meta.path.is_ident("from") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                from = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("try_from") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                try_from = Some(s.parse()?);
                Ok(())
            } else if meta.path.is_ident("borrow") {
                if meta.input.peek(syn::Token![=]) {
                    let value = meta.value()?;
                    let lit: syn::LitStr = value.parse()?;
                    borrow = Some(BorrowAttr::Explicit(parse_borrow_lifetimes(&lit)?));
                } else {
                    borrow = Some(BorrowAttr::All);
                }
                Ok(())
            } else {
                Err(meta.error("unknown strede attribute"))
            }
        })?;
    }

    // `with = "module"` expands to both deserialize paths.
    if let Some(module) = with_module {
        if deserialize_with.is_none() {
            deserialize_with = Some(syn::parse_quote!(#module::deserialize));
        }
        if deserialize_owned_with.is_none() {
            deserialize_owned_with = Some(syn::parse_quote!(#module::deserialize_owned));
        }
    }

    if from.is_some() && try_from.is_some() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "cannot use both #[strede(from)] and #[strede(try_from)] on the same field",
        ));
    }
    let has_from = from.is_some() || try_from.is_some();
    if has_from && (deserialize_with.is_some() || deserialize_owned_with.is_some()) {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[strede(from)] / #[strede(try_from)] cannot be combined with \
             #[strede(deserialize_with)] / #[strede(deserialize_owned_with)] / #[strede(with)]",
        ));
    }
    if flatten != FlattenMode::None {
        if rename.is_some() || !aliases.is_empty() {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[strede(flatten)] cannot be combined with #[strede(rename)] or #[strede(alias)]",
            ));
        }
        if default.is_some() || skip_deserializing {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[strede(flatten)] cannot be combined with #[strede(default)] or #[strede(skip_deserializing)]",
            ));
        }
        if deserialize_with.is_some() || deserialize_owned_with.is_some() || has_from {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[strede(flatten)] cannot be combined with #[strede(deserialize_with)] / #[strede(deserialize_owned_with)] / #[strede(with)] / #[strede(from)] / #[strede(try_from)]",
            ));
        }
    }

    Ok(FieldAttrs {
        rename,
        aliases,
        default,
        skip_deserializing,
        flatten,
        deserialize_with,
        deserialize_owned_with,
        bound,
        from,
        try_from,
        borrow,
    })
}

/// Extend `wc` with `'de: 'a` bounds for one field in the borrow family.
///
/// Priority:
/// 1. If `field_bound` is `Some(preds)`: use those (empty = suppress).
/// 2. Else if `has_custom_deserializer`: skip (the wrapper handles its own bound).
/// 3. Else: extract lifetimes from the type and emit `'de: 'lt` for each.
pub fn apply_borrow_field_bound(
    wc: &mut syn::WhereClause,
    ty: &syn::Type,
    field_bound: &Option<Vec<syn::WherePredicate>>,
    has_custom_deserializer: bool,
    borrow_attr: &Option<BorrowAttr>,
) {
    match field_bound {
        Some(preds) => wc.predicates.extend(preds.iter().cloned()),
        None if has_custom_deserializer => {}
        None => {
            for lt in borrow_lifetimes(ty, borrow_attr) {
                wc.predicates.push(syn::parse_quote!('de: #lt));
            }
        }
    }
}

/// Extend `wc` with the appropriate predicates for one field.
///
/// Priority:
/// 1. If `field_bound` is `Some(preds)`: use those (empty = suppress).
/// 2. Else if `has_custom_deserializer`: skip (the wrapper handles its own bound).
/// 3. Else: call `auto_pred` and push the result.
pub fn apply_field_bound(
    wc: &mut syn::WhereClause,
    ty: &syn::Type,
    field_bound: &Option<Vec<syn::WherePredicate>>,
    has_custom_deserializer: bool,
    auto_pred: impl FnOnce(&syn::Type) -> syn::WherePredicate,
) {
    match field_bound {
        Some(preds) => wc.predicates.extend(preds.iter().cloned()),
        None if has_custom_deserializer => {}
        // Skip auto-bound for universal-blanket types ((`&str`, `&[u8]`, `Cow`,
        // `PhantomData`) — an explicit Deserialize predicate would conflict with
        // the blanket impl (E0283).
        None if has_universal_blanket(ty) => {}
        None => wc.predicates.push(auto_pred(ty)),
    }
}

pub fn wire_name(
    ident: &syn::Ident,
    rename: &Option<String>,
    rename_all: Option<RenameAll>,
) -> String {
    if let Some(r) = rename {
        r.clone()
    } else {
        let s = ident.to_string();
        match rename_all {
            Some(ra) => ra.apply(&s),
            None => s,
        }
    }
}

pub struct ClassifiedField {
    pub wire_name: String,
    pub aliases: Vec<String>,
    pub default: Option<DefaultAttr>,
    pub skip_deserializing: bool,
    pub flatten: FlattenMode,
    pub deserialize_with: Option<syn::ExprPath>,
    pub deserialize_owned_with: Option<syn::ExprPath>,
    /// When `Some`, replaces the auto-generated bound for this field's type.
    pub bound: Option<Vec<syn::WherePredicate>>,
    pub from: Option<syn::Type>,
    pub try_from: Option<syn::Type>,
    /// Controls `'de: 'a` bound inference for the borrow-family derive.
    pub borrow: Option<BorrowAttr>,
}

/// Classify struct fields, extracting wire names, default, and skip attributes.
pub fn classify_fields(
    fields: &syn::punctuated::Punctuated<syn::Field, Token![,]>,
    rename_all: Option<RenameAll>,
) -> syn::Result<Vec<ClassifiedField>> {
    fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let attrs = parse_field_attrs(&f.attrs)?;
            if attrs.skip_deserializing && attrs.default.is_none() {
                return Err(syn::Error::new_spanned(
                    f,
                    "#[strede(skip_deserializing)] requires a default value; \
                     add #[strede(default)] or #[strede(default = \"fn_name\")]",
                ));
            }
            // Named fields use their ident; unnamed (tuple) fields use their index.
            let wn = match f.ident.as_ref() {
                Some(ident) => wire_name(ident, &attrs.rename, rename_all),
                None => i.to_string(),
            };
            Ok(ClassifiedField {
                wire_name: wn,
                aliases: attrs.aliases,
                default: attrs.default,
                skip_deserializing: attrs.skip_deserializing,
                flatten: attrs.flatten,
                deserialize_with: attrs.deserialize_with,
                deserialize_owned_with: attrs.deserialize_owned_with,
                bound: attrs.bound,
                from: attrs.from,
                try_from: attrs.try_from,
                borrow: attrs.borrow,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Borrow-family lifetime extraction
// ---------------------------------------------------------------------------

/// Collect all lifetimes that appear anywhere in a type.
fn all_lifetimes_in_type(ty: &syn::Type) -> Vec<syn::Lifetime> {
    let mut out = Vec::new();
    collect_lifetimes_recursive(ty, &mut out);
    out
}

fn collect_lifetimes_recursive(ty: &syn::Type, out: &mut Vec<syn::Lifetime>) {
    match ty {
        syn::Type::Reference(r) => {
            if let Some(lt) = &r.lifetime
                && !out.iter().any(|l| l.ident == lt.ident)
            {
                out.push(lt.clone());
            }
            collect_lifetimes_recursive(&r.elem, out);
        }
        syn::Type::Path(p) => {
            if let Some(qself) = &p.qself {
                collect_lifetimes_recursive(&qself.ty, out);
            }
            for seg in &p.path.segments {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    for arg in &args.args {
                        match arg {
                            syn::GenericArgument::Lifetime(lt) => {
                                if !out.iter().any(|l| l.ident == lt.ident) {
                                    out.push(lt.clone());
                                }
                            }
                            syn::GenericArgument::Type(t) => {
                                collect_lifetimes_recursive(t, out);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        syn::Type::Tuple(t) => {
            for elem in &t.elems {
                collect_lifetimes_recursive(elem, out);
            }
        }
        syn::Type::Array(a) => collect_lifetimes_recursive(&a.elem, out),
        syn::Type::Slice(s) => collect_lifetimes_recursive(&s.elem, out),
        syn::Type::Paren(p) => collect_lifetimes_recursive(&p.elem, out),
        syn::Type::Group(g) => collect_lifetimes_recursive(&g.elem, out),
        _ => {}
    }
}

/// How a `#[strede(flatten)]` field's type relates to the container's own
/// generic type parameters, for compile-time wire-name-collision checking
/// (see `strede::Fields` / `strede::Disjoint`).
pub enum FlattenTier {
    /// Type doesn't mention any container type parameter — its `Fields::NAMES`
    /// is knowable directly, right now, at derive-expansion time.
    Concrete,
    /// Type *is* (exactly) one of the container's own type parameters. Its
    /// `Fields::NAMES` isn't known here, but a `T: Fields` bound lets the
    /// check be deferred to whenever the container is actually monomorphized
    /// with a concrete substitution for `T`.
    BareParam(syn::Ident),
    /// Type mentions a container type parameter but isn't a bare occurrence
    /// of it (e.g. `Box<T>`, `Vec<T>`) — there's no sensible `Fields` bound to
    /// add for a wrapper like this, so the pair can't be checked at all.
    /// Same (unchecked) behavior as before this feature existed.
    Unprovable,
}

/// One participant in a struct/enum's flatten wire-name-collision-check
/// scope: either the container's own direct fields/tag/candidates (a
/// synthesized marker type), or a `#[strede(flatten)]` field's own type.
pub struct FieldsParticipant {
    /// Tokens for a type implementing `strede::Fields` (or, for `generic`
    /// participants, a still-abstract type parameter bound to `Fields`).
    pub ty_tokens: TokenStream2,
    /// `true` for a [`FlattenTier::BareParam`] participant: the check
    /// against it must be deferred (embedded in a real, later-monomorphized
    /// fn body) rather than asserted unconditionally at derive-expansion time.
    pub generic: bool,
}

/// Build every pairwise disjointness check among `participants` (typically:
/// index 0 is the container's own fields, the rest are flatten fields).
///
/// Returns `(unconditional, deferred)`:
/// - `unconditional`: `const _: () = ...;` items, fired immediately and
///   unconditionally wherever spliced in — used for pairs where both sides
///   are concrete at derive-expansion time.
/// - `deferred`: `let _: () = ...;` statements to splice into a real,
///   already-generic function body (e.g. `wire_names()`) so the check fires
///   once a still-abstract participant is actually monomorphized.
pub fn build_fields_checks(
    krate: &syn::Path,
    participants: &[FieldsParticipant],
    owned: bool,
) -> (TokenStream2, TokenStream2) {
    let mut unconditional = TokenStream2::new();
    let mut deferred = TokenStream2::new();
    for i in 0..participants.len() {
        for j in (i + 1)..participants.len() {
            let a = &participants[i].ty_tokens;
            let b = &participants[j].ty_tokens;
            if participants[i].generic || participants[j].generic {
                deferred.extend(quote! {
                    let _: () = #krate::Disjoint::<#a, #b, #owned>::CHECK;
                });
            } else {
                unconditional.extend(quote! {
                    const _: () = #krate::Disjoint::<#a, #b, #owned>::CHECK;
                });
            }
        }
    }
    (unconditional, deferred)
}

/// Build the `impl strede::Fields<OWNED> for #own_marker` block plus its
/// unconditional no-internal-duplicates check, for a synthesized marker type
/// representing a container's own literal wire names (fields, tag, or
/// candidate-variant names — never flatten children). `owned` selects the
/// `Fields<false>` (borrow, `#[derive(Deserialize)]`) or `Fields<true>`
/// (owned, `#[derive(DeserializeOwned)]`) instantiation — see `Fields`'s own
/// docs for why this must differ between the two derives on the same type.
pub fn build_own_fields_impl(
    krate: &syn::Path,
    own_marker: &syn::Ident,
    own_names_tokens: &[TokenStream2],
    owned: bool,
) -> TokenStream2 {
    quote! {
        #[allow(non_camel_case_types)]
        struct #own_marker;
        impl #krate::Fields<#owned> for #own_marker {
            const NAMES: &'static [&'static str] = &[ #( #own_names_tokens ),* ];
        }
        const _: () = #krate::NoInternalDuplicates::<#own_marker, #owned>::CHECK;
    }
}

/// Build a full, transitively-unioned `impl strede::Fields<OWNED> for #name #ty_generics`
/// — only valid when every flatten participant is [`FlattenTier::Concrete`]
/// (checked by the caller; `concrete_flatten_types` must contain only those).
/// Uses plain non-generic array-length arithmetic per concrete participant,
/// so it needs no unstable `generic_const_exprs` support: every type named in
/// `concrete_flatten_types` is concrete relative to this impl. `owned` selects
/// `Fields<false>` vs `Fields<true>`, matching `build_own_fields_impl`.
#[allow(clippy::too_many_arguments)]
pub fn build_concrete_fields_impl(
    krate: &syn::Path,
    impl_generics: &syn::ImplGenerics,
    name: &syn::Ident,
    ty_generics: &syn::TypeGenerics,
    where_clause: Option<&syn::WhereClause>,
    own_marker: &syn::Ident,
    concrete_flatten_types: &[TokenStream2],
    owned: bool,
) -> TokenStream2 {
    let copy_loops = concrete_flatten_types.iter().map(|ty| {
        quote! {
            {
                let __src = <#ty as #krate::Fields<#owned>>::NAMES;
                let mut __j = 0usize;
                while __j < __src.len() {
                    __arr[__i] = __src[__j];
                    __i += 1;
                    __j += 1;
                }
            }
        }
    });
    quote! {
        impl #impl_generics #krate::Fields<#owned> for #name #ty_generics #where_clause {
            const NAMES: &'static [&'static str] = {
                const __OWN: &[&str] = <#own_marker as #krate::Fields<#owned>>::NAMES;
                const __N: usize = __OWN.len() #( + <#concrete_flatten_types as #krate::Fields<#owned>>::NAMES.len() )*;
                const __ARR: [&str; __N] = {
                    let mut __arr: [&str; __N] = [""; __N];
                    let mut __i = 0usize;
                    while __i < __OWN.len() {
                        __arr[__i] = __OWN[__i];
                        __i += 1;
                    }
                    #( #copy_loops )*
                    __arr
                };
                &__ARR
            };
        }
    }
}

/// Classify a `#[strede(flatten)]` field's type against the container's own
/// generic type parameters. See [`FlattenTier`] for what each case means.
pub fn classify_flatten_tier(ty: &syn::Type, params: &[syn::Ident]) -> FlattenTier {
    if let syn::Type::Path(p) = ty
        && p.qself.is_none()
        && p.path.leading_colon.is_none()
        && p.path.segments.len() == 1
    {
        let seg = &p.path.segments[0];
        if matches!(seg.arguments, syn::PathArguments::None)
            && let Some(param) = params.iter().find(|ident| **ident == seg.ident)
        {
            return FlattenTier::BareParam(param.clone());
        }
    }
    if type_mentions_param(ty, params) {
        FlattenTier::Unprovable
    } else {
        FlattenTier::Concrete
    }
}

/// Does `ty` mention any of `params` anywhere in its type tree (including
/// nested generics, e.g. `Vec<T>`, `Option<Box<T>>`)?
///
/// Used to classify a `#[strede(flatten)]` field's type as "concrete at
/// derive-expansion time" (can build a compile-time wire-name set for it) vs.
/// "still abstract" (the field's type is, or depends on, one of the
/// container's own generic type parameters — its wire names cannot be known
/// until the container itself is monomorphized with a concrete substitution).
pub fn type_mentions_param(ty: &syn::Type, params: &[syn::Ident]) -> bool {
    let mut found = false;
    walk_type_idents(ty, &mut |ident| {
        if params.iter().any(|p| p == ident) {
            found = true;
        }
    });
    found
}

fn walk_type_idents(ty: &syn::Type, f: &mut impl FnMut(&syn::Ident)) {
    match ty {
        syn::Type::Path(p) => {
            if let Some(qself) = &p.qself {
                walk_type_idents(&qself.ty, f);
            }
            // A bare single-segment path (e.g. `T`) is the common case for a
            // container's own type parameter appearing directly as a field type.
            if p.qself.is_none()
                && p.path.leading_colon.is_none()
                && let Some(seg) = p.path.segments.last()
                && p.path.segments.len() == 1
            {
                f(&seg.ident);
            }
            for seg in &p.path.segments {
                match &seg.arguments {
                    syn::PathArguments::AngleBracketed(args) => {
                        for arg in &args.args {
                            if let syn::GenericArgument::Type(t) = arg {
                                walk_type_idents(t, f);
                            }
                        }
                    }
                    syn::PathArguments::Parenthesized(args) => {
                        for arg in &args.inputs {
                            walk_type_idents(&arg.ty, f);
                        }
                        if let syn::ReturnType::Type(_, t) = &args.output {
                            walk_type_idents(t, f);
                        }
                    }
                    syn::PathArguments::None => {}
                }
            }
        }
        syn::Type::Reference(r) => walk_type_idents(&r.elem, f),
        syn::Type::Tuple(t) => {
            for elem in &t.elems {
                walk_type_idents(elem, f);
            }
        }
        syn::Type::Array(a) => walk_type_idents(&a.elem, f),
        syn::Type::Slice(s) => walk_type_idents(&s.elem, f),
        syn::Type::Paren(p) => walk_type_idents(&p.elem, f),
        syn::Type::Group(g) => walk_type_idents(&g.elem, f),
        syn::Type::Ptr(p) => walk_type_idents(&p.elem, f),
        _ => {}
    }
}

/// Collect lifetimes from the "obvious" borrowing positions in a type:
/// `&'a T`, `&'a mut T`, and `Cow<'a, T>`.  Does not recurse into nested
/// generics beyond the outermost layer - only picks up lifetimes that are
/// directly visible at the top-level type structure.
fn auto_borrow_lifetimes(ty: &syn::Type) -> Vec<syn::Lifetime> {
    let mut out = Vec::new();
    auto_borrow_lifetimes_inner(ty, &mut out);
    out
}

fn auto_borrow_lifetimes_inner(ty: &syn::Type, out: &mut Vec<syn::Lifetime>) {
    match ty {
        syn::Type::Reference(r) => {
            if let Some(lt) = &r.lifetime
                && !out.iter().any(|l| l.ident == lt.ident)
            {
                out.push(lt.clone());
            }
        }
        syn::Type::Path(p) => {
            // Check for Cow<'a, ...>
            if let Some(seg) = p.path.segments.last()
                && seg.ident == "Cow"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
            {
                for arg in &args.args {
                    if let syn::GenericArgument::Lifetime(lt) = arg
                        && !out.iter().any(|l| l.ident == lt.ident)
                    {
                        out.push(lt.clone());
                    }
                }
            }
        }
        _ => {}
    }
}

/// Determine which `'de: 'a` bounds to emit for a field type in the borrow family.
///
/// Returns a list of lifetimes for which `'de: 'lifetime` should be added.
/// Excludes `'de` itself (no need for `'de: 'de`).
pub fn borrow_lifetimes(ty: &syn::Type, borrow_attr: &Option<BorrowAttr>) -> Vec<syn::Lifetime> {
    let lifetimes = match borrow_attr {
        Some(BorrowAttr::All) => all_lifetimes_in_type(ty),
        Some(BorrowAttr::Explicit(lts)) => lts.clone(),
        None => auto_borrow_lifetimes(ty),
    };
    // Filter out 'de - `'de: 'de` is trivially true.
    lifetimes
        .into_iter()
        .filter(|lt| lt.ident != "de")
        .collect()
}
