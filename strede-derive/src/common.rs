use convert_case::{Case, Casing};
use syn::{Fields, Token};

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

    // `other` cannot coexist with untagged variants — the fallback semantics conflict.
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
    /// `#[strede(from = "FromType")]` — deserialize `FromType`, then call `Self::from(v)`.
    pub from: Option<syn::Type>,
    /// `#[strede(try_from = "FromType")]` — deserialize `FromType`, then call
    /// `Self::try_from(v).ok()`, returning `Probe::Miss` on failure.
    pub try_from: Option<syn::Type>,
}

pub struct VariantAttrs {
    pub rename: Option<String>,
    pub aliases: Vec<String>,
    pub untagged: bool,
    pub other: bool,
}

pub struct FieldAttrs {
    pub rename: Option<String>,
    pub aliases: Vec<String>,
    pub default: Option<DefaultAttr>,
    pub skip_deserializing: bool,
    pub deserialize_with: Option<syn::ExprPath>,
    pub deserialize_owned_with: Option<syn::ExprPath>,
    /// When `Some`, replaces the auto-generated bound for this field's type.
    /// An empty `Vec` suppresses the bound entirely.
    pub bound: Option<Vec<syn::WherePredicate>>,
    /// `#[strede(from = "FromType")]` — deserialize `FromType`, then call `FieldType::from(v)`.
    pub from: Option<syn::Type>,
    /// `#[strede(try_from = "FromType")]` — deserialize `FromType`, then call
    /// `FieldType::try_from(v).ok()`, returning `Probe::Miss` on failure.
    pub try_from: Option<syn::Type>,
}

pub enum DefaultAttr {
    /// `#[strede(default)]` — calls `Default::default()`
    Trait,
    /// `#[strede(default = "expr")]` — evaluates `expr` via `DefaultWrapper`.
    /// If `expr` is a function path it gets called; otherwise the value is used as-is.
    Expr(syn::Expr),
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
    Ok(ContainerAttrs {
        untagged,
        allow_unknown_fields,
        transparent,
        rename_all,
        crate_path: crate_path.unwrap_or_else(|| syn::parse_quote!(::strede)),
        bound,
        from,
        try_from,
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
    let mut deserialize_with = None;
    let mut deserialize_owned_with = None;
    let mut with_module: Option<syn::ExprPath> = None;
    let mut bound: Option<Vec<syn::WherePredicate>> = None;
    let mut from: Option<syn::Type> = None;
    let mut try_from: Option<syn::Type> = None;
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

    Ok(FieldAttrs {
        rename,
        aliases,
        default,
        skip_deserializing,
        deserialize_with,
        deserialize_owned_with,
        bound,
        from,
        try_from,
    })
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
    pub deserialize_with: Option<syn::ExprPath>,
    pub deserialize_owned_with: Option<syn::ExprPath>,
    /// When `Some`, replaces the auto-generated bound for this field's type.
    pub bound: Option<Vec<syn::WherePredicate>>,
    pub from: Option<syn::Type>,
    pub try_from: Option<syn::Type>,
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
                deserialize_with: attrs.deserialize_with,
                deserialize_owned_with: attrs.deserialize_owned_with,
                bound: attrs.bound,
                from: attrs.from,
                try_from: attrs.try_from,
            })
        })
        .collect()
}
