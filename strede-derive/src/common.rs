use convert_case::{Case, Casing};
use syn::{Fields, Token};

/// Whether a field is flattened and, if so, whether it opts into heap allocation
/// to break deeply-nested async state-machine chains that would overflow the stack.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlattenMode {
    /// Not flattened.
    None,
    /// `#[strede(flatten)]` - zero-alloc; may stack-overflow with 3+ flatten fields.
    Plain,
    /// `#[strede(flatten(boxed))]` - heap-allocates to break the async chain.
    Boxed,
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
                rename_all = Some(RenameAll::from_str(&s.value()).ok_or_else(|| {
                    meta.error(format!(
                        "unknown rename_all value {:?}; expected one of: \
                         lowercase, UPPERCASE, PascalCase, camelCase, snake_case, \
                         SCREAMING_SNAKE_CASE, kebab-case, SCREAMING-KEBAB-CASE",
                        s.value()
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
                    // flatten(boxed)
                    let inner;
                    syn::parenthesized!(inner in meta.input);
                    let ident: syn::Ident = inner.parse()?;
                    if ident != "boxed" {
                        return Err(syn::Error::new_spanned(ident, "expected `flatten(boxed)`"));
                    }
                    flatten = FlattenMode::Boxed;
                } else {
                    flatten = FlattenMode::Plain;
                }
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
                "#[strede(flatten)] cannot be combined with #[strede(deserialize_with)] / #[strede(with)] / #[strede(from)] / #[strede(try_from)]",
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
