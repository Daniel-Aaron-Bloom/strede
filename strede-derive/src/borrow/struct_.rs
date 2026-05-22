use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

use super::gen_container_from_borrow;
use crate::common::{
    DefaultAttr, FieldContext, apply_borrow_field_bound, borrow_lifetimes, classify_fields,
    field_bound_borrow, has_universal_blanket, insert_de_and_d_borrow, mentions_type_param,
    type_param_bound_borrow,
};

/// Generate wrapper newtypes for fields with `deserialize_with`, `from`, or `try_from` (borrow family).
fn gen_deserialize_with_wrappers_borrow(
    field_names: &[&syn::Ident],
    field_types: &[&syn::Type],
    classified: &[&crate::common::ClassifiedField],
    krate: &syn::Path,
) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for ((name, ty), cf) in field_names
        .iter()
        .zip(field_types.iter())
        .zip(classified.iter())
    {
        if let Some(path) = &cf.deserialize_with {
            let wrapper = format_ident!("__DeWith_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl<'de, __D: #krate::Deserializer<'de>> #krate::Deserialize<'de, __D> for #wrapper
                where #ty: 'de
                {
                    type Extra = ();
                    async fn deserialize(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        let (__claim, __v) = #krate::hit!(#path(d, ()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, Self(__v))))
                    }
                }
            });
        } else if let Some(from_ty) = &cf.from {
            let wrapper = format_ident!("__DeFrom_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl<'de, __D: #krate::Deserializer<'de>> #krate::Deserialize<'de, __D> for #wrapper
                where #from_ty: #krate::Deserialize<'de, __D, Extra = ()>
                {
                    type Extra = ();
                    async fn deserialize(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        let (__claim, __v) = #krate::hit!(<#from_ty as #krate::Deserialize<'de, __D>>::deserialize(d, ()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, Self(
                            <#ty as ::core::convert::From<#from_ty>>::from(__v)
                        ))))
                    }
                }
            });
        } else if let Some(try_from_ty) = &cf.try_from {
            let wrapper = format_ident!("__DeTryFrom_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl<'de, __D: #krate::Deserializer<'de>> #krate::Deserialize<'de, __D> for #wrapper
                where #try_from_ty: #krate::Deserialize<'de, __D, Extra = ()>
                {
                    type Extra = ();
                    async fn deserialize(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        let (__claim, __v) = #krate::hit!(<#try_from_ty as #krate::Deserialize<'de, __D>>::deserialize(d, ()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, Self(
                            #krate::or_miss!(<#ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__v).ok())
                        ))))
                    }
                }
            });
        }
    }
    tokens
}

pub(super) fn expand(
    input: DeriveInput,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
) -> syn::Result<TokenStream2> {
    let allow_unknown_fields = container_attrs.allow_unknown_fields;
    let transparent = container_attrs.transparent;
    let name = &input.ident;

    // ---- container-level from / try_from ------------------------------------
    if let Some(ref from_ty) = container_attrs.from {
        return gen_container_from_borrow(&input, krate, container_attrs, from_ty, false);
    }
    if let Some(ref try_from_ty) = container_attrs.try_from {
        return gen_container_from_borrow(&input, krate, container_attrs, try_from_ty, true);
    }

    // Determine struct kind and extract fields.
    enum StructKind<'a> {
        Named(&'a syn::punctuated::Punctuated<syn::Field, syn::Token![,]>),
        Tuple(&'a syn::punctuated::Punctuated<syn::Field, syn::Token![,]>),
    }
    let kind = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => StructKind::Named(&f.named),
            Fields::Unnamed(f) => StructKind::Tuple(&f.unnamed),
            Fields::Unit => {
                // Unit struct: deserialize from null via `deserialize_value::<()>`.
                let (_, ty_generics, _) = input.generics.split_for_impl();
                let mut impl_gen = input.generics.clone();
                insert_de_and_d_borrow(&mut impl_gen, krate);
                impl_gen
                    .make_where_clause()
                    .predicates
                    .push(syn::parse_quote!(
                        (): #krate::Deserialize<
                            'de,
                            <__D::Entry as #krate::Entry<'de>>::SubDeserializer,
                            Extra = ()
                        >
                    ));
                let (impl_generics, _, where_clause) = impl_gen.split_for_impl();
                return Ok(quote! {
                    #[allow(unreachable_code)]
                    const _: () = {
                        use #krate::{DefaultValue as _, Deserializer as _, Entry as _};

                        impl #impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #where_clause {
                            type Extra = ();
                            async fn deserialize(
                                d: __D,
                                _extra: (),
                            ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                            {
                                d.entry(|[__e]| async {
                                    let (__c, _) = #krate::hit!(__e.deserialize_value::<()>(()).await);
                                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name)))
                                }).await
                            }
                        }
                    };
                });
            }
        },
        _ => unreachable!(),
    };
    let is_named = matches!(kind, StructKind::Named(_));
    let fields = match &kind {
        StructKind::Named(f) | StructKind::Tuple(f) => *f,
    };

    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
    let classified_fields = classify_fields(fields, container_attrs.rename_all)?;

    // Validation.
    if transparent && allow_unknown_fields {
        return Err(syn::Error::new_spanned(
            name,
            "cannot combine #[strede(transparent)] and #[strede(allow_unknown_fields)]",
        ));
    }
    if !is_named && allow_unknown_fields {
        return Err(syn::Error::new_spanned(
            name,
            "#[strede(allow_unknown_fields)] is not supported on tuple structs",
        ));
    }

    // ty_generics: original struct type params.
    let (_, ty_generics, _) = input.generics.split_for_impl();

    // ---- transparent --------------------------------------------------------

    if transparent {
        let non_skipped: Vec<_> = field_types
            .iter()
            .zip(classified_fields.iter())
            .enumerate()
            .filter(|(_, (_, cf))| !cf.skip_deserializing)
            .collect();
        if non_skipped.len() != 1 {
            return Err(syn::Error::new_spanned(
                name,
                "#[strede(transparent)] requires exactly one non-skipped field",
            ));
        }
        let (transparent_idx, (transparent_ty, transparent_cf)) = non_skipped[0];

        // Build generics with Deserialize bound only for the transparent field.
        let mut impl_gen = input.generics.clone();
        insert_de_and_d_borrow(&mut impl_gen, krate);
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(type_param_bound_borrow(krate, ident));
                }
                let has_custom = transparent_cf.deserialize_with.is_some()
                    || transparent_cf.from.is_some()
                    || transparent_cf.try_from.is_some();
                apply_borrow_field_bound(
                    wc,
                    transparent_ty,
                    &transparent_cf.bound,
                    has_custom,
                    &transparent_cf.borrow,
                );
                // Transparent reads the inner field via direct `<T as Deserialize>::deserialize` — need Direct bound.
                if transparent_cf.bound.is_none() && !has_custom {
                    wc.predicates.push(field_bound_borrow(
                        krate,
                        transparent_ty,
                        FieldContext::Direct,
                    ));
                }
                if let Some(ft) = &transparent_cf.from {
                    for lt in borrow_lifetimes(ft, &None) {
                        wc.predicates.push(syn::parse_quote!('de: #lt));
                    }
                    wc.predicates
                        .push(field_bound_borrow(krate, ft, FieldContext::Direct));
                } else if let Some(ft) = &transparent_cf.try_from {
                    for lt in borrow_lifetimes(ft, &None) {
                        wc.predicates.push(syn::parse_quote!('de: #lt));
                    }
                    wc.predicates
                        .push(field_bound_borrow(krate, ft, FieldContext::Direct));
                }
            }
        }
        let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

        // Generate deserialize expression for the transparent field.
        let de_expr = if let Some(path) = &transparent_cf.deserialize_with {
            quote! { #krate::hit!(#path(d, ()).await) }
        } else if let Some(from_ty) = &transparent_cf.from {
            quote! {
                {
                    let (__claim, __tmp) = #krate::hit!(<#from_ty as #krate::Deserialize<'de, __D>>::deserialize(d, ()).await);
                    (__claim, <#transparent_ty as ::core::convert::From<#from_ty>>::from(__tmp))
                }
            }
        } else if let Some(try_from_ty) = &transparent_cf.try_from {
            quote! {
                {
                    let (__claim, __tmp) = #krate::hit!(<#try_from_ty as #krate::Deserialize<'de, __D>>::deserialize(d, ()).await);
                    (__claim, #krate::or_miss!(<#transparent_ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__tmp).ok()))
                }
            }
        } else {
            quote! { #krate::hit!(<#transparent_ty as #krate::Deserialize<'de, __D>>::deserialize(d, ()).await) }
        };

        // Generate default expressions for all fields.
        let field_exprs: Vec<TokenStream2> = classified_fields
            .iter()
            .enumerate()
            .map(|(i, cf)| {
                if i == transparent_idx {
                    quote! { __v }
                } else {
                    match &cf.default {
                        Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                        Some(DefaultAttr::Expr(expr)) => {
                            quote! { #krate::DefaultWrapper(#expr).value() }
                        }
                        None => unreachable!("validated: non-transparent fields must be skipped"),
                    }
                }
            })
            .collect();

        // Construct the struct.
        let construct = if is_named {
            let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
            quote! { #name { #( #field_names: #field_exprs, )* } }
        } else {
            quote! { #name( #( #field_exprs, )* ) }
        };

        return Ok(quote! {
            #[allow(unreachable_code)]
            const _: () = {
                use #krate::{DefaultValue as _, Deserialize as _, Deserializer as _};

                impl #impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #where_clause {
                    type Extra = ();
                    async fn deserialize(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        let (__claim, __v) = #de_expr;
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #construct)))
                    }
                }
            };
        });
    }

    // ---- tuple struct (seq-based) -------------------------------------------

    if !is_named {
        let acc_names: Vec<_> = (0..field_types.len())
            .map(|i| format_ident!("__f{}", i))
            .collect();

        // Non-skipped field types (and their classified attrs) for generic bounds.
        let de_field_types_and_cfs: Vec<_> = field_types
            .iter()
            .zip(classified_fields.iter())
            .filter(|(_, cf)| !cf.skip_deserializing)
            .collect();

        let mut impl_gen = input.generics.clone();
        insert_de_and_d_borrow(&mut impl_gen, krate);
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(type_param_bound_borrow(krate, ident));
                }
                for (ty, cf) in &de_field_types_and_cfs {
                    let has_custom =
                        cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                    apply_borrow_field_bound(wc, ty, &cf.bound, has_custom, &cf.borrow);
                    if cf.bound.is_none() && !has_custom {
                        // Read via `__se.get::<T>(())` — SeqElem context.
                        let bound_ty = cf.from.as_ref().or(cf.try_from.as_ref()).unwrap_or(*ty);
                        // Skip for types with universal blanket impls — explicit bound would
                        // conflict with the blanket and trigger E0283 ambiguity.
                        if !has_universal_blanket(bound_ty) {
                            wc.predicates.push(field_bound_borrow(
                                krate,
                                bound_ty,
                                FieldContext::SeqElem,
                            ));
                        }
                    }
                    if let Some(ft) = &cf.from {
                        for lt in borrow_lifetimes(ft, &None) {
                            wc.predicates.push(syn::parse_quote!('de: #lt));
                        }
                    } else if let Some(ft) = &cf.try_from {
                        for lt in borrow_lifetimes(ft, &None) {
                            wc.predicates.push(syn::parse_quote!('de: #lt));
                        }
                    }
                }
            }
        }
        let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

        let seq_reads: Vec<TokenStream2> = acc_names
            .iter()
            .zip(field_types.iter())
            .zip(classified_fields.iter())
            .map(|((acc, ty), cf)| {
                if cf.skip_deserializing {
                    let default_expr = match &cf.default {
                        Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                        Some(DefaultAttr::Expr(expr)) => quote! { #krate::DefaultWrapper(#expr).value() },
                        None => unreachable!("validated in classify_fields"),
                    };
                    quote! { let #acc = #default_expr; }
                } else if let Some(from_ty) = &cf.from {
                    quote! {
                        let __v = #krate::hit!(__seq.next(|[__se]| async {
                            __se.get::<#from_ty>(()).await
                        }).await);
                        let (__seq_back, __raw) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                        let #acc = <#ty as ::core::convert::From<#from_ty>>::from(__raw);
                    }
                } else if let Some(try_from_ty) = &cf.try_from {
                    quote! {
                        let __v = #krate::hit!(__seq.next(|[__se]| async {
                            __se.get::<#try_from_ty>(()).await
                        }).await);
                        let (__seq_back, __raw) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                        let #acc = #krate::or_miss!(<#ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__raw).ok());
                    }
                } else {
                    quote! {
                        let __v = #krate::hit!(__seq.next(|[__se]| async {
                            __se.get::<#ty>(()).await
                        }).await);
                        let (__seq_back, #acc) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                    }
                }
            })
            .collect();

        return Ok(quote! {
            #[allow(unreachable_code)]
            const _: () = {
                use #krate::{
                    Deserialize as _, Deserializer as _, Entry as _,
                    SeqAccess as _, SeqEntry as _,
                };

                impl #impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #where_clause {
                    type Extra = ();
                    async fn deserialize(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        d.entry(|[__e]| async {
                            let mut __seq = #krate::hit!(__e.deserialize_seq().await);

                            #( #seq_reads )*

                            // Expect sequence exhaustion.
                            let __v = #krate::hit!(__seq.next::<1, _, _, ()>(|[__se]| async {
                                ::core::result::Result::Ok(#krate::Probe::Miss)
                            }).await);
                            let __claim = #krate::or_miss!(__v.done());
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __claim,
                                #name( #( #acc_names, )* ),
                            )))
                        }).await
                    }
                }
            };
        });
    }

    // ---- named struct (map-based) -------------------------------------------

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();

    // Filtered views: only non-skipped, non-flatten fields participate as regular arms.
    let de_field_names: Vec<_> = field_names
        .iter()
        .zip(classified_fields.iter())
        .filter(|(_, cf)| !cf.skip_deserializing && cf.flatten == crate::common::FlattenMode::None)
        .map(|(n, _)| *n)
        .collect();
    let de_field_types: Vec<_> = field_types
        .iter()
        .zip(classified_fields.iter())
        .filter(|(_, cf)| !cf.skip_deserializing && cf.flatten == crate::common::FlattenMode::None)
        .map(|(t, _)| *t)
        .collect();

    let de_classified: Vec<_> = classified_fields
        .iter()
        .filter(|cf| !cf.skip_deserializing && cf.flatten == crate::common::FlattenMode::None)
        .collect();

    // All flatten fields, in declaration order.
    let flatten_fields: Vec<(
        &syn::Ident,
        &syn::Type,
        crate::common::FlattenMode,
        &Option<crate::common::BorrowAttr>,
    )> = field_names
        .iter()
        .zip(field_types.iter())
        .zip(classified_fields.iter())
        .filter(|(_, cf)| cf.flatten != crate::common::FlattenMode::None)
        .map(|((n, t), cf)| (*n, *t, cf.flatten, &cf.borrow))
        .collect();

    // For deserialize_with / from / try_from.
    let de_value_types: Vec<TokenStream2> = de_field_names
        .iter()
        .zip(de_classified.iter())
        .map(|(name, cf)| {
            if cf.deserialize_with.is_some() {
                let wrapper = format_ident!("__DeWith_{}", name);
                quote! { #wrapper }
            } else if cf.from.is_some() {
                let wrapper = format_ident!("__DeFrom_{}", name);
                quote! { #wrapper }
            } else if cf.try_from.is_some() {
                let wrapper = format_ident!("__DeTryFrom_{}", name);
                quote! { #wrapper }
            } else {
                let ty = &de_field_types[de_field_names.iter().position(|n| *n == *name).unwrap()];
                quote! { #ty }
            }
        })
        .collect();
    let de_value_conversions: Vec<TokenStream2> = de_classified
        .iter()
        .map(|cf| {
            if cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some() {
                quote! { .0 }
            } else {
                quote! {}
            }
        })
        .collect();

    let de_with_wrappers = gen_deserialize_with_wrappers_borrow(
        &de_field_names,
        &de_field_types,
        &de_classified,
        krate,
    );

    // D6 — detect generic-T flatten fields. On stable, we cannot emit the bound
    // `T: Deserialize<'de, FlattenDeserializer<…>, Extra = ()>` because the
    // outer-arms type is Voldemort. Emit a clear compile_error directing users
    // to `#[strede(bound)]` or the `nightly-flatten` feature.
    let type_param_idents: Vec<syn::Ident> = input
        .generics
        .type_params()
        .map(|tp| tp.ident.clone())
        .collect();
    for (fname, flat_ty, _, _) in &flatten_fields {
        // Skip if the field has an explicit bound override.
        let fcf = classified_fields
            .iter()
            .zip(field_names.iter())
            .find(|(_, n)| **n == *fname)
            .map(|(cf, _)| cf);
        let has_explicit_bound = fcf.map(|cf| cf.bound.is_some()).unwrap_or(false);
        if !has_explicit_bound
            && !type_param_idents.is_empty()
            && mentions_type_param(flat_ty, &type_param_idents)
        {
            let msg =
                "`#[strede(flatten)]` on a field whose type mentions a struct type parameter \
                 is not supported on stable Rust. Add `#[strede(bound = \"...\")]` to the \
                 field, or enable the `nightly-flatten` feature."
                    .to_string();
            return Ok(quote! {
                ::core::compile_error!(#msg);
            });
        }
    }

    // Compute the dup_wire_names count up front so we can include the right MatchVals bound.
    let dup_wire_names_count_pre: usize = de_classified.iter().map(|cf| 1 + cf.aliases.len()).sum();

    // Build impl generics.
    let mut impl_gen = input.generics.clone();
    insert_de_and_d_borrow(&mut impl_gen, krate);
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in input.generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_borrow(krate, ident));
            }
            for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                let has_custom =
                    cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                apply_borrow_field_bound(wc, ty, &cf.bound, has_custom, &cf.borrow);
                if cf.bound.is_none() && !has_custom {
                    // Regular field — consumed via `__vp.deserialize_value::<T>(())`.
                    let bound_ty = cf.from.as_ref().or(cf.try_from.as_ref()).unwrap_or(*ty);
                    if !has_universal_blanket(bound_ty) {
                        wc.predicates.push(field_bound_borrow(
                            krate,
                            bound_ty,
                            FieldContext::MapValue,
                        ));
                    }
                }
            }
            for (_, flat_ty, _, flat_borrow) in &flatten_fields {
                for lt in borrow_lifetimes(flat_ty, flat_borrow) {
                    wc.predicates.push(syn::parse_quote!('de: #lt));
                }
                // Flatten field — D6 option α: no bound emitted (rustc resolves at use site).
            }
            // Map iteration uses Match/Skip key probes.
            // dup_wire_names_count > 0 means DetectDuplicates is in play, needing MatchVals<usize>.
            let dup_n = if dup_wire_names_count_pre > 0 {
                Some(dup_wire_names_count_pre)
            } else {
                None
            };
            let _ = dup_n; // universal Match/MatchVals/Skip impls cover the key bounds
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    // For non-flatten structs, the Deserialize<'de, __D> impl delegates through
    // DeserializeFromMap so its where-clause only needs Self: DeserializeFromMap
    // rather than propagating every field type's ValueSubDeserializer chain. This
    // prevents unbounded projection chains when this struct appears as a nested type.
    let (de_impl_generics, de_where_clause) = if flatten_fields.is_empty() {
        let mut impl_gen_d2 = input.generics.clone();
        insert_de_and_d_borrow(&mut impl_gen_d2, krate);
        {
            let wc = impl_gen_d2.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(type_param_bound_borrow(krate, ident));
                }
                wc.predicates.push(syn::parse_quote!(
                    #name #ty_generics: #krate::DeserializeFromMap<
                        'de,
                        <__D::Entry as #krate::Entry<'de>>::Map,
                        Extra = ()
                    >
                ));
            }
        }
        let (ig, _, wc) = impl_gen_d2.split_for_impl();
        (quote! { #ig }, quote! { #wc })
    } else {
        (quote! { #impl_generics }, quote! { #where_clause })
    };

    // Build one MapArmSlot per non-skipped, non-flatten field.
    // `kp_ty` / `vp_ty` are the KeyProbe and ValueProbe type tokens used
    // inside the slot closures — they differ between the Deserialize (__D)
    // and DeserializeFromMap (__M) emissions.
    let build_arm_slots = |kp_ty: TokenStream2, vp_ty: TokenStream2| -> Vec<TokenStream2> {
        de_field_names
            .iter()
            .zip(de_classified.iter())
            .zip(de_value_types.iter().zip(de_value_conversions.iter()))
            .map(|((_, cf), (vt, vc))| {
                let mut wire_names: Vec<&str> = vec![cf.wire_name.as_str()];
                for alias in &cf.aliases {
                    wire_names.push(alias.as_str());
                }
                let key_fn = if wire_names.len() == 1 {
                    let wname = wire_names[0];
                    quote! {
                        |mut __kp: #kp_ty, _i: usize| {
                            __kp.deserialize_key::<#krate::Match>(#wname)
                        }
                    }
                } else {
                    quote! {
                        |mut __kp: #kp_ty, _i: usize| {
                            __kp.deserialize_key::<#krate::MatchVals<(), _>>([#( (#wire_names, ()), )*])
                        }
                    }
                };
                let val_fn = quote! {
                    |__vp: #vp_ty, __k| async move {
                        let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#vt>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v #vc))))
                    }
                };
                quote! { #krate::MapArmSlot::new(#key_fn, #val_fn) }
            })
            .collect()
    };

    let arm_slots: Vec<TokenStream2> = build_arm_slots(
        quote! { #krate::borrow::KP<'de, __D> },
        quote! { #krate::borrow::VP2<'de, __D> },
    );
    let arm_slots_m: Vec<TokenStream2> = build_arm_slots(
        quote! { <__M as #krate::MapAccess<'de>>::KeyProbe },
        quote! { #krate::borrow::VP<'de, <__M as #krate::MapAccess<'de>>::KeyProbe> },
    );

    // dup detection wire names array.
    let dup_wire_names: Vec<TokenStream2> = de_classified
        .iter()
        .enumerate()
        .flat_map(|(arm_idx, cf)| {
            let mut entries: Vec<TokenStream2> = vec![];
            let primary = &cf.wire_name;
            entries.push(quote! { (#primary, #arm_idx) });
            for alias in &cf.aliases {
                entries.push(quote! { (#alias, #arm_idx) });
            }
            entries
        })
        .collect();
    let dup_wire_names_count = dup_wire_names.len();

    // Output accumulator names and pattern (for regular, non-flatten fields).
    let de_out_names: Vec<syn::Ident> = de_field_names
        .iter()
        .map(|n| format_ident!("__out_{}", n))
        .collect();
    let output_pat = {
        let mut pat = quote! { () };
        for out_name in &de_out_names {
            pat = quote! { (#pat, #out_name) };
        }
        pat
    };

    // Field finalizers for regular (non-flatten) fields.
    let regular_field_finalizers: Vec<TokenStream2> = field_names
        .iter()
        .zip(classified_fields.iter())
        .filter(|(_, cf)| cf.flatten == crate::common::FlattenMode::None)
        .map(|(fname, cf)| {
            if cf.skip_deserializing {
                let default_expr = match &cf.default {
                    Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                    Some(DefaultAttr::Expr(expr)) => {
                        quote! { #krate::DefaultWrapper(#expr).value() }
                    }
                    None => unreachable!("validated in classify_fields"),
                };
                return quote! { let #fname = #default_expr; };
            }
            let out_name = format_ident!("__out_{}", fname);
            let none_branch = match &cf.default {
                Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                Some(DefaultAttr::Expr(expr)) => quote! { #krate::DefaultWrapper(#expr).value() },
                None => quote! { return ::core::result::Result::Ok(#krate::Probe::Miss) },
            };
            quote! {
                let #fname = match #out_name {
                    ::core::option::Option::Some((_k, __v)) => __v,
                    ::core::option::Option::None => #none_branch,
                };
            }
        })
        .collect();

    // Generate deserialization body - two paths: with flatten or without.
    let mut flatten_cont_structs = TokenStream2::new();
    let deser_body = if !flatten_fields.is_empty() {
        // Outer arms - same shape as regular path but applied to the FlattenMapAccess wrapper.
        let outer_arms_expr = {
            let mut expr = quote! { #krate::MapArmBase };
            for slot in &arm_slots {
                expr = quote! { (#expr, #slot) };
            }
            if dup_wire_names_count > 0 {
                expr = quote! {
                    {
                        let __wn = [#( #dup_wire_names, )*];
                        #krate::DetectDuplicatesOwned::new(
                            #expr,
                            __wn,
                            move |__kp: #krate::borrow::KP<'de, __D>, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                            |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
                        )
                    }
                };
            }
            // No skip-unknown here: keys not claimed by outer fall through to
            // the inner type's arms via `StackConcat` inside `FlattenMapAccess`.
            expr
        };

        let n_flat = flatten_fields.len();
        let first_flat_name = flatten_fields[0].0;
        let first_flat_ty = flatten_fields[0].1;

        // If any flatten field requests boxed mode, use FlattenTerminalBoxed at
        // the terminal to heap-allocate the deepest iterate future.
        let any_boxed = flatten_fields
            .iter()
            .any(|(_, _, mode, _)| *mode == crate::common::FlattenMode::Boxed);
        let terminal_expr = if any_boxed {
            quote! { #krate::FlattenTerminalBoxed }
        } else {
            quote! { #krate::FlattenTerminal }
        };

        // Cont-struct generics: outer struct's generics plus '__cont.
        let mut cont_struct_gens = input.generics.clone();
        cont_struct_gens
            .params
            .insert(0, syn::parse_quote!('__cont));
        let (cont_struct_impl_gens, cont_struct_ty_gens, _) = cont_struct_gens.split_for_impl();

        // Cont-impl generics: cont_struct_gens + 'de + __M.
        let mut cont_impl_gens = cont_struct_gens.clone();
        let has_de = cont_impl_gens.lifetimes().any(|l| l.lifetime.ident == "de");
        if !has_de {
            cont_impl_gens.params.insert(0, syn::parse_quote!('de));
        }
        cont_impl_gens
            .params
            .push(syn::parse_quote!(__M: #krate::MapAccess<'de>));
        // Bounds on cont impls mirror the outer DFM impl bounds: each
        // type-param + each regular non-flatten field type must be
        // `Deserialize<'de, ValueSubDe(__M::KeyProbe), Extra = ()>`. This
        // permits the inner flatten target's DFM call (which projects through
        // the same KeyProbe) to resolve.
        {
            let wc = cont_impl_gens.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(syn::parse_quote!(
                        #ident: #krate::Deserialize<
                            'de,
                            <#krate::borrow::VP<'de, <__M as #krate::MapAccess<'de>>::KeyProbe>
                             as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                            Extra = ()
                        >
                    ));
                }
                for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                    let has_custom =
                        cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                    apply_borrow_field_bound(wc, ty, &cf.bound, has_custom, &cf.borrow);
                    if cf.bound.is_none() && !has_custom {
                        let bound_ty = cf.from.as_ref().or(cf.try_from.as_ref()).unwrap_or(*ty);
                        if !has_universal_blanket(bound_ty) {
                            wc.predicates.push(syn::parse_quote!(
                                #bound_ty: #krate::Deserialize<
                                    'de,
                                    <#krate::borrow::VP<'de, <__M as #krate::MapAccess<'de>>::KeyProbe>
                                     as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                                    Extra = ()
                                >
                            ));
                        }
                    }
                }
                // Flatten lifetime predicates.
                for (_, flat_ty, _, flat_borrow) in &flatten_fields {
                    for lt in borrow_lifetimes(flat_ty, flat_borrow) {
                        wc.predicates.push(syn::parse_quote!('de: #lt));
                    }
                }
            }
        }
        let (cont_impl_impl_gens, _, cont_impl_where) = cont_impl_gens.split_for_impl();

        // Emit (N - 1) cont structs: cont_after_f0 .. cont_after_f_(N-2).
        // cont_after_f_i recurses into f_(i+1).
        for i in 0..n_flat.saturating_sub(1) {
            let cont_name = format_ident!("__FlatContB_after_{}", flatten_fields[i].0);
            let next_field_name = flatten_fields[i + 1].0;
            let next_field_ty = flatten_fields[i + 1].1;
            let result_stash_ident = format_ident!("__result_{}", next_field_name);

            // Cell fields: one per future flatten result (i+1 .. n_flat).
            let cell_fields: Vec<TokenStream2> = flatten_fields[i + 1..]
                .iter()
                .map(|(fname, fty, _, _)| {
                    let cell_ident = format_ident!("__result_{}", fname);
                    quote! {
                        #cell_ident: &'__cont ::core::cell::Cell<::core::option::Option<#fty>>
                    }
                })
                .collect();

            // Next cont expression - either FlattenTerminal or next intermediate.
            let next_cont_expr = if i + 1 < n_flat - 1 {
                let next_cont_name = format_ident!("__FlatContB_after_{}", flatten_fields[i + 1].0);
                let next_args: Vec<TokenStream2> = flatten_fields[i + 2..]
                    .iter()
                    .map(|(fname, _, _, _)| {
                        let cell_ident = format_ident!("__result_{}", fname);
                        quote! { #cell_ident: self.#cell_ident }
                    })
                    .collect();
                quote! {
                    #next_cont_name {
                        #( #next_args, )*
                        __phantom: ::core::marker::PhantomData,
                    }
                }
            } else {
                terminal_expr.clone()
            };

            flatten_cont_structs.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #cont_name #cont_struct_impl_gens {
                    #( #cell_fields, )*
                    __phantom: ::core::marker::PhantomData<&'__cont ()>,
                }

                impl #cont_impl_impl_gens #krate::FlattenCont<'de, __M>
                    for #cont_name #cont_struct_ty_gens
                #cont_impl_where
                {
                    async fn finish<__Arms: #krate::MapArmStack<'de, <__M as #krate::MapAccess<'de>>::KeyProbe>>(
                        self,
                        __map: __M,
                        __arms: __Arms,
                    ) -> ::core::result::Result<
                        #krate::Probe<(<__M as #krate::MapAccess<'de>>::MapClaim, __Arms::Outputs)>,
                        <__M as #krate::MapAccess<'de>>::Error,
                    > {
                        let __next_outputs_cell: ::core::cell::Cell<
                            ::core::option::Option<__Arms::Outputs>,
                        > = ::core::cell::Cell::new(::core::option::Option::None);
                        let __next_fma = #krate::FlattenMapAccess::new(
                            __map,
                            __arms,
                            &__next_outputs_cell,
                            #next_cont_expr,
                        );
                        let (__claim, __next_val) = #krate::hit!(
                            <#next_field_ty as #krate::DeserializeFromMap<'de, _>>
                                ::deserialize_from_map(__next_fma, ()).await
                        );
                        self.#result_stash_ident
                            .set(::core::option::Option::Some(__next_val));
                        let __out = match __next_outputs_cell.take() {
                            ::core::option::Option::Some(__v) => __v,
                            ::core::option::Option::None => {
                                return ::core::result::Result::Ok(#krate::Probe::Miss);
                            }
                        };
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __out)))
                    }
                }
            });
        }

        // Result cell declarations for flatten fields [1..].
        let extra_result_cell_decls: Vec<TokenStream2> = flatten_fields[1..]
            .iter()
            .map(|(fname, fty, _, _)| {
                let cell_ident = format_ident!("__result_{}", fname);
                quote! {
                    let #cell_ident: ::core::cell::Cell<::core::option::Option<#fty>>
                        = ::core::cell::Cell::new(::core::option::Option::None);
                }
            })
            .collect();

        // First cont expression: FlattenTerminal for N=1, else cont_after_f0.
        let first_cont_expr = if n_flat == 1 {
            terminal_expr.clone()
        } else {
            let first_cont_name = format_ident!("__FlatContB_after_{}", first_flat_name);
            let first_cont_args: Vec<TokenStream2> = flatten_fields[1..]
                .iter()
                .map(|(fname, _, _, _)| {
                    let cell_ident = format_ident!("__result_{}", fname);
                    quote! { #cell_ident: &#cell_ident }
                })
                .collect();
            quote! {
                #first_cont_name {
                    #( #first_cont_args, )*
                    __phantom: ::core::marker::PhantomData,
                }
            }
        };

        // Recover subsequent flatten field results from cells.
        let extra_result_recovers: Vec<TokenStream2> = flatten_fields[1..]
            .iter()
            .map(|(fname, _, _, _)| {
                let cell_ident = format_ident!("__result_{}", fname);
                quote! {
                    let #fname = match #cell_ident.take() {
                        ::core::option::Option::Some(__v) => __v,
                        ::core::option::Option::None => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                    };
                }
            })
            .collect();

        quote! {
            d.entry(|[__e]| async {
                let __map = #krate::hit!(__e.deserialize_map().await);
                let __outer_arms = #outer_arms_expr;
                let __outer_outputs_cell: ::core::cell::Cell<::core::option::Option<_>>
                    = ::core::cell::Cell::new(::core::option::Option::None);
                #( #extra_result_cell_decls )*
                let __flatten_map = #krate::FlattenMapAccess::new(
                    __map,
                    __outer_arms,
                    &__outer_outputs_cell,
                    #first_cont_expr,
                );
                let (__claim, #first_flat_name) = #krate::hit!(
                    <#first_flat_ty as #krate::DeserializeFromMap<'de, _>>::deserialize_from_map(
                        __flatten_map,
                        (),
                    ).await
                );
                let #output_pat = match __outer_outputs_cell.take() {
                    ::core::option::Option::Some(__o) => __o,
                    ::core::option::Option::None => {
                        return ::core::result::Result::Ok(#krate::Probe::Miss);
                    }
                };
                #( #extra_result_recovers )*
                #( #regular_field_finalizers )*
                ::core::result::Result::Ok(
                    #krate::Probe::Hit((__claim, #name { #( #field_names, )* }))
                )
            }).await
        }
    } else {
        // Regular path: no flatten fields.
        quote! {
            d.entry(|[__e]| async {
                __e.deserialize_map_into::<Self>(()).await
            }).await
        }
    };

    // -----------------------------------------------------------------
    // DeserializeFromMap impl emission (regular non-flatten path only)
    // -----------------------------------------------------------------
    let dfm_impl = if flatten_fields.is_empty() {
        // Build arms_expr_m using __M-rooted projections.
        let arms_expr_m = {
            let mut expr = quote! { #krate::MapArmBase };
            for slot in &arm_slots_m {
                expr = quote! { (#expr, #slot) };
            }
            expr = quote! {
                {
                    let __wn = [#( #dup_wire_names, )*];
                    #krate::DetectDuplicatesOwned::new(
                        #expr,
                        __wn,
                        move |__kp: <__M as #krate::MapAccess<'de>>::KeyProbe, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                        |__vp: #krate::borrow::VP<'de, <__M as #krate::MapAccess<'de>>::KeyProbe>| __vp.skip(),
                    )
                }
            };
            if allow_unknown_fields {
                expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                    |__kp: <__M as #krate::MapAccess<'de>>::KeyProbe, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
                    |__vp: #krate::borrow::VP<'de, <__M as #krate::MapAccess<'de>>::KeyProbe>, _k: #krate::Skip| async move {
                        use #krate::MapValueProbe as _;
                        let __vc = __vp.skip().await?;
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                    },
                )) };
            }
            expr
        };

        // Build impl generics for DeserializeFromMap.
        let mut impl_gen_m = input.generics.clone();
        crate::common::insert_de_and_m_borrow(&mut impl_gen_m, krate);
        {
            let wc = impl_gen_m.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(syn::parse_quote!(
                        #ident: #krate::Deserialize<
                            'de,
                            <#krate::borrow::VP<'de, <__M as #krate::MapAccess<'de>>::KeyProbe>
                             as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                            Extra = ()
                        >
                    ));
                }
                for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                    let has_custom =
                        cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                    apply_borrow_field_bound(wc, ty, &cf.bound, has_custom, &cf.borrow);
                    if cf.bound.is_none() && !has_custom {
                        let bound_ty = cf.from.as_ref().or(cf.try_from.as_ref()).unwrap_or(*ty);
                        if !has_universal_blanket(bound_ty) {
                            wc.predicates.push(syn::parse_quote!(
                                #bound_ty: #krate::Deserialize<
                                    'de,
                                    <#krate::borrow::VP<'de, <__M as #krate::MapAccess<'de>>::KeyProbe>
                                     as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                                    Extra = ()
                                >
                            ));
                        }
                    }
                }
            }
        }
        let (impl_generics_m, _, where_clause_m) = impl_gen_m.split_for_impl();

        quote! {
            impl #impl_generics_m #krate::DeserializeFromMap<'de, __M> for #name #ty_generics #where_clause_m {
                type Extra = ();
                async fn deserialize_from_map(
                    __map: __M,
                    _extra: (),
                ) -> ::core::result::Result<
                    #krate::Probe<(<__M as #krate::MapAccess<'de>>::MapClaim, Self)>,
                    <__M as #krate::MapAccess<'de>>::Error,
                > {
                    let __arms = #arms_expr_m;
                    let (__claim, #output_pat) = #krate::hit!(__map.iterate(__arms).await);
                    #( #regular_field_finalizers )*
                    ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name { #( #field_names, )* })))
                }
            }
        }
    } else {
        TokenStream2::new()
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #de_with_wrappers
            #flatten_cont_structs

            impl #de_impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #de_where_clause {
                type Extra = ();
                async fn deserialize(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    #deser_body
                }
            }

            #dfm_impl
        };
    })
}
