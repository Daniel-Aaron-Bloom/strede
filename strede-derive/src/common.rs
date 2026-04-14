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
    pub untagged: bool,
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

        let kind = match &variant.fields {
            Fields::Unit => VariantKind::Unit,
            Fields::Unnamed(f) if f.unnamed.len() == 1 => VariantKind::Newtype(&f.unnamed[0].ty),
            Fields::Named(f) => VariantKind::Struct(&f.named),
            Fields::Unnamed(f) => VariantKind::Tuple(&f.unnamed),
        };

        let untagged = container_attrs.untagged || vattrs.untagged;
        let wire_name = wire_name(&variant.ident, &vattrs.rename);

        out.push(ClassifiedVariant {
            variant,
            kind,
            wire_name,
            untagged,
            index,
        });
    }
    Ok(out)
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

pub struct ContainerAttrs {
    pub untagged: bool,
    pub crate_path: syn::Path,
}

pub struct VariantAttrs {
    pub rename: Option<String>,
    pub untagged: bool,
}

pub struct FieldAttrs {
    pub rename: Option<String>,
    pub default: Option<DefaultAttr>,
    pub skip_deserializing: bool,
    pub deserialize_with: Option<syn::ExprPath>,
    pub deserialize_owned_with: Option<syn::ExprPath>,
}

pub enum DefaultAttr {
    /// `#[strede(default)]` — calls `Default::default()`
    Trait,
    /// `#[strede(default = "path")]` — calls `path()`
    Path(syn::ExprPath),
}

pub fn parse_container_attrs(attrs: &[syn::Attribute]) -> syn::Result<ContainerAttrs> {
    let mut untagged = false;
    let mut crate_path: Option<syn::Path> = None;
    for attr in attrs {
        if !attr.path().is_ident("strede") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("untagged") {
                untagged = true;
                Ok(())
            } else if meta.path.is_ident("crate") {
                let value = meta.value()?;
                let s: syn::LitStr = value.parse()?;
                crate_path = Some(s.parse()?);
                Ok(())
            } else {
                Err(meta.error("unknown strede attribute"))
            }
        })?;
    }
    Ok(ContainerAttrs {
        untagged,
        crate_path: crate_path.unwrap_or_else(|| syn::parse_quote!(::strede)),
    })
}

pub fn parse_variant_attrs(attrs: &[syn::Attribute]) -> syn::Result<VariantAttrs> {
    let mut rename = None;
    let mut untagged = false;
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
            } else if meta.path.is_ident("untagged") {
                untagged = true;
                Ok(())
            } else {
                Err(meta.error("unknown strede attribute"))
            }
        })?;
    }
    Ok(VariantAttrs { rename, untagged })
}

pub fn parse_field_attrs(attrs: &[syn::Attribute]) -> syn::Result<FieldAttrs> {
    let mut rename = None;
    let mut default = None;
    let mut skip_deserializing = false;
    let mut deserialize_with = None;
    let mut deserialize_owned_with = None;
    let mut with_module: Option<syn::ExprPath> = None;
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
            } else if meta.path.is_ident("default") {
                if meta.input.peek(syn::Token![=]) {
                    let value = meta.value()?;
                    let s: syn::LitStr = value.parse()?;
                    let path: syn::ExprPath = s.parse()?;
                    default = Some(DefaultAttr::Path(path));
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

    Ok(FieldAttrs {
        rename,
        default,
        skip_deserializing,
        deserialize_with,
        deserialize_owned_with,
    })
}

pub fn wire_name(ident: &syn::Ident, rename: &Option<String>) -> String {
    rename.clone().unwrap_or_else(|| ident.to_string())
}

pub struct ClassifiedField {
    pub wire_name: String,
    pub default: Option<DefaultAttr>,
    pub skip_deserializing: bool,
    pub deserialize_with: Option<syn::ExprPath>,
    pub deserialize_owned_with: Option<syn::ExprPath>,
}

/// Classify struct fields, extracting wire names, default, and skip attributes.
pub fn classify_fields(
    fields: &syn::punctuated::Punctuated<syn::Field, Token![,]>,
) -> syn::Result<Vec<ClassifiedField>> {
    fields
        .iter()
        .map(|f| {
            let attrs = parse_field_attrs(&f.attrs)?;
            if attrs.skip_deserializing && attrs.default.is_none() {
                return Err(syn::Error::new_spanned(
                    f,
                    "#[strede(skip_deserializing)] requires a default value; \
                     add #[strede(default)] or #[strede(default = \"fn_name\")]",
                ));
            }
            Ok(ClassifiedField {
                wire_name: wire_name(f.ident.as_ref().unwrap(), &attrs.rename),
                default: attrs.default,
                skip_deserializing: attrs.skip_deserializing,
                deserialize_with: attrs.deserialize_with,
                deserialize_owned_with: attrs.deserialize_owned_with,
            })
        })
        .collect()
}
