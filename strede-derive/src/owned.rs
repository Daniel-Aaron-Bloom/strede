use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

use crate::common::{
    ClassifiedVariant, DefaultAttr, VariantKind, all_field_types, apply_field_bound,
    classify_fields, classify_variants, other_variant, parse_container_attrs,
};

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let container_attrs = parse_container_attrs(&input.attrs)?;
    let krate = &container_attrs.crate_path;
    match &input.data {
        Data::Struct(_) => expand_owned_struct(input, krate, &container_attrs),
        Data::Enum(_) => expand_owned_enum(input, krate),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "DeserializeOwned can only be derived for structs and enums",
        )),
    }
}

/// Generate an owned-family impl that deserializes `from_ty` and converts to `Self`
/// via `From` (`is_try = false`) or `TryFrom` with `or_miss!` (`is_try = true`).
fn gen_container_from_owned(
    input: &DeriveInput,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    from_ty: &syn::Type,
    is_try: bool,
) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let (_, ty_generics, _) = input.generics.split_for_impl();
    let mut impl_gen = input.generics.clone();
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            wc.predicates
                .push(syn::parse_quote!(#from_ty: #krate::DeserializeOwned));
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    let convert = if is_try {
        quote! { #krate::or_miss!(<#name #ty_generics as ::core::convert::TryFrom<#from_ty>>::try_from(__v).ok()) }
    } else {
        quote! { <#name #ty_generics as ::core::convert::From<#from_ty>>::from(__v) }
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{DeserializeOwned as _, DeserializerOwned as _};

            impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    let (__c, __v) = #krate::hit!(<#from_ty as #krate::DeserializeOwned>::deserialize_owned(d, ()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, #convert)))
                }
            }
        };
    })
}

fn expand_owned_struct(
    input: DeriveInput,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
) -> syn::Result<TokenStream2> {
    let allow_unknown_fields = container_attrs.allow_unknown_fields;
    let transparent = container_attrs.transparent;
    let name = &input.ident;

    // ---- container-level from / try_from ------------------------------------
    if let Some(ref from_ty) = container_attrs.from {
        return gen_container_from_owned(&input, krate, container_attrs, from_ty, false);
    }
    if let Some(ref try_from_ty) = container_attrs.try_from {
        return gen_container_from_owned(&input, krate, container_attrs, try_from_ty, true);
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
                let (_, ty_generics, _) = input.generics.split_for_impl();
                let (impl_generics, _, where_clause) = input.generics.split_for_impl();
                return Ok(quote! {
                    #[allow(unreachable_code)]
                    const _: () = {
                        use #krate::{DefaultValue as _, DeserializerOwned as _, EntryOwned as _};

                        impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                            async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                                d: __D,
                                _extra: (),
                            ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                            where
                                __D::Error: #krate::DeserializeError,
                            {
                                d.entry(|[__e]| async {
                                    let __c = #krate::hit!(__e.deserialize_null().await);
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

        // Build generics with DeserializeOwned bound only for the transparent field.
        let mut impl_gen = input.generics.clone();
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                let has_custom = transparent_cf.deserialize_owned_with.is_some()
                    || transparent_cf.from.is_some()
                    || transparent_cf.try_from.is_some();
                apply_field_bound(
                    wc,
                    transparent_ty,
                    &transparent_cf.bound,
                    has_custom,
                    |ty| syn::parse_quote!(#ty: #krate::DeserializeOwned),
                );
                if let Some(ft) = &transparent_cf.from {
                    wc.predicates
                        .push(syn::parse_quote!(#ft: #krate::DeserializeOwned));
                } else if let Some(ft) = &transparent_cf.try_from {
                    wc.predicates
                        .push(syn::parse_quote!(#ft: #krate::DeserializeOwned));
                }
            }
        }
        let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

        // Generate deserialize expression for the transparent field.
        let de_expr = if let Some(path) = &transparent_cf.deserialize_owned_with {
            quote! { #krate::hit!(#path(d).await) }
        } else if let Some(from_ty) = &transparent_cf.from {
            quote! {
                {
                    let (__c, __tmp) = #krate::hit!(<#from_ty as #krate::DeserializeOwned>::deserialize_owned(d, ()).await);
                    (__c, <#transparent_ty as ::core::convert::From<#from_ty>>::from(__tmp))
                }
            }
        } else if let Some(try_from_ty) = &transparent_cf.try_from {
            quote! {
                {
                    let (__c, __tmp) = #krate::hit!(<#try_from_ty as #krate::DeserializeOwned>::deserialize_owned(d, ()).await);
                    (__c, #krate::or_miss!(<#transparent_ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__tmp).ok()))
                }
            }
        } else {
            quote! { #krate::hit!(<#transparent_ty as #krate::DeserializeOwned>::deserialize_owned(d, ()).await) }
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
                use #krate::{DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _};

                impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                    async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__c, __v) = #de_expr;
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #construct)))
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
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for (ty, cf) in &de_field_types_and_cfs {
                    let has_custom = cf.deserialize_owned_with.is_some()
                        || cf.from.is_some()
                        || cf.try_from.is_some();
                    apply_field_bound(
                        wc,
                        ty,
                        &cf.bound,
                        has_custom,
                        |t| syn::parse_quote!(#t: #krate::DeserializeOwned),
                    );
                    if let Some(ft) = &cf.from {
                        wc.predicates
                            .push(syn::parse_quote!(#ft: #krate::DeserializeOwned));
                    } else if let Some(ft) = &cf.try_from {
                        wc.predicates
                            .push(syn::parse_quote!(#ft: #krate::DeserializeOwned));
                    }
                }
            }
        }
        let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

        // Generate sequential seq reads for non-skipped fields,
        // and default expressions for skipped fields.
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
                            __se.get::<#from_ty, _>(()).await
                        }).await);
                        let (__seq_back, __raw) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                        let #acc = <#ty as ::core::convert::From<#from_ty>>::from(__raw);
                    }
                } else if let Some(try_from_ty) = &cf.try_from {
                    quote! {
                        let __v = #krate::hit!(__seq.next(|[__se]| async {
                            __se.get::<#try_from_ty, _>(()).await
                        }).await);
                        let (__seq_back, __raw) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                        let #acc = #krate::or_miss!(<#ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__raw).ok());
                    }
                } else {
                    quote! {
                        let __v = #krate::hit!(__seq.next(|[__se]| async {
                            __se.get::<#ty, _>(()).await
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
                    DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                    SeqAccessOwned as _, SeqEntryOwned as _,
                };

                impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                    async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        d.entry(|[__e]| async {
                            let __seq = #krate::hit!(__e.deserialize_seq().await);

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

    // ---- named struct (map-based, existing codegen) -------------------------

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();

    // Filtered views: only non-skipped fields participate in deserialization.
    // Split into regular (arm-based) and flatten fields.
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

    // All flatten fields, in declaration order (with their mode).
    let flatten_fields: Vec<(&syn::Ident, &syn::Type, crate::common::FlattenMode)> = field_names
        .iter()
        .zip(field_types.iter())
        .zip(classified_fields.iter())
        .filter(|(_, cf)| cf.flatten != crate::common::FlattenMode::None)
        .map(|((n, t), cf)| (*n, *t, cf.flatten))
        .collect();

    // For deserialize_owned_with / from / try_from: compute value types and conversions.
    let de_classified: Vec<_> = classified_fields
        .iter()
        .filter(|cf| !cf.skip_deserializing && cf.flatten == crate::common::FlattenMode::None)
        .collect();
    let de_value_types: Vec<TokenStream2> = de_field_names
        .iter()
        .zip(de_classified.iter())
        .map(|(name, cf)| {
            if cf.deserialize_owned_with.is_some() {
                let wrapper = format_ident!("__DeOwnedWith_{}", name);
                quote! { #wrapper }
            } else if cf.from.is_some() {
                let wrapper = format_ident!("__DeOwnedFrom_{}", name);
                quote! { #wrapper }
            } else if cf.try_from.is_some() {
                let wrapper = format_ident!("__DeOwnedTryFrom_{}", name);
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
            if cf.deserialize_owned_with.is_some() || cf.from.is_some() || cf.try_from.is_some() {
                quote! { .0 }
            } else {
                quote! {}
            }
        })
        .collect();

    let de_with_wrappers = gen_deserialize_with_wrappers_owned(
        &de_field_names,
        &de_field_types,
        &de_classified,
        krate,
    );

    // Build impl generics: add DeserializeOwned bounds.
    let mut impl_gen = input.generics.clone();
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                let has_custom = cf.deserialize_owned_with.is_some()
                    || cf.from.is_some()
                    || cf.try_from.is_some();
                apply_field_bound(
                    wc,
                    ty,
                    &cf.bound,
                    has_custom,
                    |t| syn::parse_quote!(#t: #krate::DeserializeOwned),
                );
            }
            // Add DeserializeOwned bounds for all flatten field types.
            for (_, flat_ty, _) in &flatten_fields {
                wc.predicates
                    .push(syn::parse_quote!(#flat_ty: #krate::DeserializeOwned));
            }
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    // Output binding names - one per non-skipped, non-flatten field.
    let de_out_names: Vec<syn::Ident> = de_field_names
        .iter()
        .map(|n| format_ident!("__out_{}", n))
        .collect();

    // Build one MapArmSlot per non-skipped, non-flatten field.
    let arm_slots: Vec<TokenStream2> = de_field_names
        .iter()
        .zip(de_classified.iter())
        .zip(de_value_types.iter().zip(de_value_conversions.iter()))
        .map(|((_, cf), (vt, vc))| {
            let mut wire_names: Vec<&str> = vec![cf.wire_name.as_str()];
            for alias in &cf.aliases {
                wire_names.push(alias.as_str());
            }

            let key_fn = if wire_names.len() == 1 {
                let name = wire_names[0];
                quote! {
                    |mut __kp: #krate::owned::KP<__D>| {
                        __kp.deserialize_key::<#krate::Match, &str>(#name)
                    }
                }
            } else {
                quote! {
                    |mut __kp: #krate::owned::KP<__D>| {
                        __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wire_names, ()), )*])
                    }
                }
            };

            let val_fn = quote! {
                |__vp: #krate::owned::VP2<__D>, __k| async move {
                    let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#vt, _>(()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v #vc))))
                }
            };

            quote! { #krate::MapArmSlot::new(#key_fn, #val_fn) }
        })
        .collect();

    // Build the flat (wire_name, arm_index) array for DetectDuplicatesOwned.
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

    // Output pattern for regular (non-flatten) fields.
    let output_pat = {
        let mut pat = quote! { () };
        for out_name in &de_out_names {
            pat = quote! { (#pat, #out_name) };
        }
        pat
    };

    // Field finalizers for non-skipped, non-flatten fields.
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

    // Generate the deserialization body - two paths: with flatten or without.
    let mut flatten_cont_structs = TokenStream2::new();
    let deser_body = if !flatten_fields.is_empty() {
        // If any flatten field is flatten(boxed), use Box::pin everywhere to break the
        // deeply-nested async chain produced by StackConcat.
        let use_boxed = flatten_fields
            .iter()
            .any(|(_, _, mode)| *mode == crate::common::FlattenMode::Boxed);

        // If flatten(boxed) is used but the alloc feature of strede-derive is not
        // enabled (strede forwards its alloc feature to strede-derive/alloc), error
        // at macro-expansion time with a clear message.
        if use_boxed && !cfg!(feature = "alloc") {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "`#[strede(flatten(boxed))]` requires the `alloc` feature of the `strede` crate",
            ));
        }

        // Flatten path: outer arms go into a Cell, FlattenDeserializerOwned drives the map.
        // For multiple flatten fields, we generate continuation structs that chain
        // each flatten field to the next.
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
                            move |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                            |__vp: #krate::owned::VP2<__D>| __vp.skip(),
                        )
                    }
                };
            }
            // No SkipUnknownOwned here - FlattenTerminal applies it to the combined stack.
            expr
        };

        // Generate continuation structs for intermediate flatten fields (all but the last).
        // The last flatten field uses FlattenTerminal directly.
        // Chain: flatten_fields[0] → cont_0 → flatten_fields[1] → cont_1 → ... → FlattenTerminal
        //
        // Each intermediate continuation struct captures result cells for all
        // *subsequent* flatten fields and calls the next flatten field's
        // deserialize_owned inside its finish() method.
        // For N flatten fields, we generate N-1 continuation structs.
        // cont_i handles flatten_fields[i] and chains to flatten_fields[i+1].
        for i in 0..flatten_fields.len().saturating_sub(1) {
            let cont_name = format_ident!("__FlatCont_{}", flatten_fields[i].0);
            let next_flat_ty = flatten_fields[i + 1].1;

            // This continuation needs result cells for all flatten fields after [i].
            // flatten_fields[i+1..] results are stored in cells.
            let result_cell_fields: Vec<TokenStream2> = flatten_fields[i + 1..]
                .iter()
                .map(|(fname, fty, _)| {
                    let cell_name = format_ident!("__result_{}", fname);
                    quote! { #cell_name: &'__cont ::core::cell::Cell<::core::option::Option<#fty>> }
                })
                .collect();

            // Build the continuation that chains to the next flatten field.
            let next_cont_expr = if i + 1 < flatten_fields.len() - 1 {
                // Intermediate: chain to another continuation.
                let next_cont_name = format_ident!("__FlatCont_{}", flatten_fields[i + 1].0);
                let next_cell_args: Vec<TokenStream2> = flatten_fields[i + 2..]
                    .iter()
                    .map(|(fname, _, _)| {
                        let cell_name = format_ident!("__result_{}", fname);
                        quote! { #cell_name: self.#cell_name }
                    })
                    .collect();
                quote! { #next_cont_name { #( #next_cell_args, )* } }
            } else {
                // Terminal: the next flatten field is the last one.
                if use_boxed {
                    quote! { #krate::FlattenTerminalBoxed }
                } else {
                    quote! { #krate::FlattenTerminal }
                }
            };

            let next_extra_cell = if i + 1 < flatten_fields.len() - 1 {
                let next_cont_name = format_ident!("__FlatCont_{}", flatten_fields[i + 1].0);
                quote! {
                    let __next_extra_cell: ::core::cell::Cell<
                        ::core::option::Option<<#next_cont_name<'_> as #krate::FlattenContOwned<__M>>::Extra>
                    > = ::core::cell::Cell::new(::core::option::Option::None);
                }
            } else {
                quote! {
                    let __next_extra_cell: ::core::cell::Cell<::core::option::Option<()>>
                        = ::core::cell::Cell::new(::core::option::Option::None);
                }
            };

            let next_result_name = format_ident!("__result_{}", flatten_fields[i + 1].0);

            let finish_body = quote! {
                let __arms_cell = ::core::cell::Cell::new(
                    ::core::option::Option::Some(__arms)
                );
                let __out_cell = ::core::cell::Cell::new(
                    ::core::option::Option::None
                );
                #next_extra_cell

                let (__claim, __next_val) = #krate::hit!(
                    <#next_flat_ty as #krate::DeserializeOwned>::deserialize_owned(
                        #krate::FlattenDeserializerOwned::new(
                            __map,
                            &__arms_cell,
                            &__out_cell,
                            #next_cont_expr,
                            &__next_extra_cell,
                        ),
                        ()
                    ).await
                );

                self.#next_result_name.set(::core::option::Option::Some(__next_val));
                let __out = __out_cell.take().unwrap();
                ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __out, ())))
            };

            // When using boxed mode, wrap the finish body in Box::pin to break the
            // deeply-nested async chain and prevent stack overflow.
            let finish_impl = if use_boxed {
                quote! { #krate::Box::pin(async move { #finish_body }).await }
            } else {
                finish_body
            };

            flatten_cont_structs.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #cont_name<'__cont> {
                    #( #result_cell_fields, )*
                }

                impl<'__cont, __M> #krate::FlattenContOwned<__M> for #cont_name<'__cont>
                where
                    __M: #krate::MapAccessOwned,
                    #next_flat_ty: #krate::DeserializeOwned,
                {
                    type Extra = ();

                    async fn finish<__Arms: #krate::MapArmStackOwned<__M::KeyProbe>>(
                        self,
                        __map: __M,
                        __arms: __Arms,
                    ) -> ::core::result::Result<
                        #krate::Probe<(__M::MapClaim, __Arms::Outputs, ())>,
                        __M::Error,
                    > {
                        #finish_impl
                    }
                }
            });
        }

        // The first flatten field is the one called from the parent.
        let first_flat_name = flatten_fields[0].0;
        let first_flat_ty = flatten_fields[0].1;
        let _ = flatten_fields[0].2; // mode (used via use_boxed)

        // Build the continuation and extra cell for the first flatten field.
        let terminal_expr = if use_boxed {
            quote! { #krate::FlattenTerminalBoxed }
        } else {
            quote! { #krate::FlattenTerminal }
        };
        let (first_cont_expr, first_extra_cell_decl) = if flatten_fields.len() == 1 {
            // Single flatten field: use terminal directly.
            (
                terminal_expr,
                quote! {
                    let __first_extra_cell: ::core::cell::Cell<::core::option::Option<()>>
                        = ::core::cell::Cell::new(::core::option::Option::None);
                },
            )
        } else {
            let first_cont_name = format_ident!("__FlatCont_{}", first_flat_name);
            let first_cell_args: Vec<TokenStream2> = flatten_fields[1..]
                .iter()
                .map(|(fname, _, _)| {
                    let cell_name = format_ident!("__result_{}", fname);
                    quote! { #cell_name: &#cell_name }
                })
                .collect();
            (
                quote! { #first_cont_name { #( #first_cell_args, )* } },
                quote! {
                    let __first_extra_cell: ::core::cell::Cell<::core::option::Option<()>>
                        = ::core::cell::Cell::new(::core::option::Option::None);
                },
            )
        };

        // Result cell declarations for flatten fields [1..] (the first's result comes
        // directly from deserialize_owned).
        let result_cell_decls: Vec<TokenStream2> = flatten_fields[1..]
            .iter()
            .map(|(fname, fty, _)| {
                let cell_name = format_ident!("__result_{}", fname);
                quote! {
                    let #cell_name: ::core::cell::Cell<::core::option::Option<#fty>>
                        = ::core::cell::Cell::new(::core::option::Option::None);
                }
            })
            .collect();

        // Recover subsequent flatten field results from cells.
        let result_recovers: Vec<TokenStream2> = flatten_fields[1..]
            .iter()
            .map(|(fname, _, _)| {
                let cell_name = format_ident!("__result_{}", fname);
                quote! {
                    let #fname = match #cell_name.take() {
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

                let __outer_arms_cell = ::core::cell::Cell::new(
                    ::core::option::Option::Some(#outer_arms_expr)
                );
                let __outer_outputs_cell = ::core::cell::Cell::new(
                    ::core::option::Option::None
                );
                #( #result_cell_decls )*
                #first_extra_cell_decl

                let (__claim, #first_flat_name) = #krate::hit!(
                    <#first_flat_ty as #krate::DeserializeOwned>::deserialize_owned(
                        #krate::FlattenDeserializerOwned::new(
                            __map,
                            &__outer_arms_cell,
                            &__outer_outputs_cell,
                            #first_cont_expr,
                            &__first_extra_cell,
                        ),
                        ()
                    ).await
                );

                // Recover outer outputs written back by FlattenMapAccessOwned::iterate.
                let #output_pat = match __outer_outputs_cell.take() {
                    ::core::option::Option::Some(__o) => __o,
                    ::core::option::Option::None => {
                        return ::core::result::Result::Ok(#krate::Probe::Miss);
                    }
                };

                // Recover subsequent flatten field results from cells.
                #( #result_recovers )*

                #( #regular_field_finalizers )*

                ::core::result::Result::Ok(
                    #krate::Probe::Hit((__claim, #name { #( #field_names, )* }))
                )
            }).await
        }
    } else {
        // Regular path: no flatten fields.
        let arms_expr = {
            let mut expr = quote! { #krate::MapArmBase };
            for slot in &arm_slots {
                expr = quote! { (#expr, #slot) };
            }
            expr = quote! {
                {
                    let __wn = [#( #dup_wire_names, )*];
                    #krate::DetectDuplicatesOwned::new(
                        #expr,
                        __wn,
                        move |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                        |__vp: #krate::owned::VP2<__D>| __vp.skip(),
                    )
                }
            };
            if allow_unknown_fields {
                expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                    |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                    |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                        let __vc = __vp.skip().await?;
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                    },
                )) };
            }
            expr
        };

        quote! {
            d.entry(|[__e]| async {
                let __map = #krate::hit!(__e.deserialize_map().await);

                let __arms = #arms_expr;
                let (__claim, #output_pat) = #krate::hit!(__map.iterate(__arms).await);

                #( #regular_field_finalizers )*

                ::core::result::Result::Ok(
                    #krate::Probe::Hit((__claim, #name { #( #field_names, )* }))
                )
            }).await
        }
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapKeyProbeOwned as _, MapAccessOwned as _, SeqAccessOwned as _,
                SeqEntryOwned as _, StrAccessOwned as _, MapValueProbeOwned as _,
            };

            #de_with_wrappers
            #flatten_cont_structs

            impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    #deser_body
                }
            }
        };
    })
}

/// Generate wrapper newtypes for fields with `deserialize_owned_with`, `from`, or `try_from` (owned family).
fn gen_deserialize_with_wrappers_owned(
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
        if let Some(path) = &cf.deserialize_owned_with {
            let wrapper = format_ident!("__DeOwnedWith_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl #krate::DeserializeOwned for #wrapper {
                    async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__c, __v) = #krate::hit!(#path(d).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, Self(__v))))
                    }
                }
            });
        } else if let Some(from_ty) = &cf.from {
            let wrapper = format_ident!("__DeOwnedFrom_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl #krate::DeserializeOwned for #wrapper
                where #from_ty: #krate::DeserializeOwned
                {
                    async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__c, __v) = #krate::hit!(<#from_ty as #krate::DeserializeOwned>::deserialize_owned(d, ()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, Self(
                            <#ty as ::core::convert::From<#from_ty>>::from(__v)
                        ))))
                    }
                }
            });
        } else if let Some(try_from_ty) = &cf.try_from {
            let wrapper = format_ident!("__DeOwnedTryFrom_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl #krate::DeserializeOwned for #wrapper
                where #try_from_ty: #krate::DeserializeOwned
                {
                    async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__c, __v) = #krate::hit!(<#try_from_ty as #krate::DeserializeOwned>::deserialize_owned(d, ()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, Self(
                            #krate::or_miss!(<#ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__v).ok())
                        ))))
                    }
                }
            });
        }
    }
    tokens
}

// ---------------------------------------------------------------------------
// Owned-family enum derive
// ---------------------------------------------------------------------------

fn expand_owned_enum(input: DeriveInput, krate: &syn::Path) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let data = match &input.data {
        Data::Enum(d) => d,
        _ => unreachable!(),
    };

    let container_attrs = parse_container_attrs(&input.attrs)?;

    if let Some(ref from_ty) = container_attrs.from {
        return gen_container_from_owned(&input, krate, &container_attrs, from_ty, false);
    }
    if let Some(ref try_from_ty) = container_attrs.try_from {
        return gen_container_from_owned(&input, krate, &container_attrs, try_from_ty, true);
    }

    let classified = classify_variants(data, &container_attrs)?;

    let field_types = all_field_types(data);

    let (_, ty_generics, _) = input.generics.split_for_impl();

    let mut impl_gen = input.generics.clone();
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for ty in &field_types {
                wc.predicates
                    .push(syn::parse_quote!(#ty: #krate::DeserializeOwned));
            }
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    let has_tagged_unit = classified
        .iter()
        .any(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit));
    let has_tagged_nonunit = classified
        .iter()
        .any(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit));
    let has_untagged = classified.iter().any(|cv| cv.untagged);

    // Generate __VariantKey for tagged variant name matching.
    // Only include tagged variants in the key matcher.
    let variant_count: usize = classified.iter().filter(|cv| !cv.untagged).count();
    let variant_key_sentinel = variant_count;

    let variant_candidates: Vec<(String, usize)> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .enumerate()
        .flat_map(|(local_idx, cv)| {
            let mut pairs = vec![(cv.wire_name.clone(), local_idx)];
            for alias in &cv.aliases {
                pairs.push((alias.clone(), local_idx));
            }
            pairs
        })
        .collect();

    if let Some(ref tag_field) = container_attrs.tag {
        if let Some(ref content_field) = container_attrs.content {
            return expand_owned_enum_adjacent_tagged(
                name,
                &classified,
                tag_field,
                content_field,
                &variant_candidates,
                variant_key_sentinel,
                krate,
                &container_attrs,
                &impl_generics,
                &ty_generics,
                where_clause,
            );
        }
        return expand_owned_enum_internally_tagged(
            name,
            &classified,
            tag_field,
            &variant_candidates,
            variant_key_sentinel,
            krate,
            &container_attrs,
            &impl_generics,
            &ty_generics,
            where_clause,
        );
    }

    let body = if !has_untagged {
        if has_tagged_unit && !has_tagged_nonunit {
            expand_owned_enum_unit_only(name, &classified, variant_key_sentinel, krate)?
        } else if !has_tagged_unit && has_tagged_nonunit {
            expand_owned_enum_map_only(
                name,
                &classified,
                &variant_candidates,
                variant_key_sentinel,
                krate,
            )?
        } else {
            expand_owned_enum_mixed(
                name,
                &classified,
                &variant_candidates,
                variant_key_sentinel,
                krate,
            )?
        }
    } else if !has_tagged_unit && !has_tagged_nonunit {
        expand_owned_enum_untagged_only(name, &classified, krate)?
    } else {
        expand_owned_enum_with_untagged(
            name,
            &classified,
            &variant_candidates,
            variant_key_sentinel,
            krate,
        )?
    };

    // For tuple variants, generate per-variant __TupleVariantOwnedN types.
    let tuple_variant_helpers = gen_tuple_variant_helpers_owned(&classified, krate);
    // For struct variants, generate per-variant __VariantOwnedN types.
    let struct_variant_helpers =
        gen_struct_variant_helpers_owned(&classified, krate, container_attrs.rename_all);

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                SeqAccessOwned as _, SeqEntryOwned as _, StrAccessOwned as _,
            };

            #tuple_variant_helpers
            #struct_variant_helpers

            impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    #body
                }
            }
        };
    })
}

/// Build tokens for using `MatchVals<usize>` / `UnwrapOrElse<MatchVals<usize>>` as a key
/// deserializer at a `.key()` or `deserialize_value` call site.
///
/// Returns `(key_type, extra_expr, idx_access)`:
/// - `key_type`  - the type to annotate `__k` with in the closure
/// - `extra_expr` - the extra value to pass as the first arg to `.key()`
/// - `idx_access` - how to extract the matched `usize` from `__k` in the closure body
///
/// If `sentinel` is `None`, an unknown string produces `Probe::Miss` (via `MatchVals`).
/// If `sentinel` is `Some(s)`, an unknown string produces `s` (via `UnwrapOrElse`).
fn key_matcher_tokens(
    candidates: &[(String, usize)],
    sentinel: Option<usize>,
    krate: &syn::Path,
) -> (TokenStream2, TokenStream2, TokenStream2) {
    let keys: Vec<&str> = candidates.iter().map(|(s, _)| s.as_str()).collect();
    let indices: Vec<usize> = candidates.iter().map(|(_, i)| *i).collect();
    let indices_lit: Vec<proc_macro2::Literal> = indices
        .iter()
        .map(|i| proc_macro2::Literal::usize_suffixed(*i))
        .collect();
    let count = proc_macro2::Literal::usize_suffixed(candidates.len());
    let array_expr = quote! {
        {
            let __arr: [(&'static str, usize); #count] = [ #( (#keys, #indices_lit), )* ];
            __arr
        }
    };

    match sentinel {
        None => (
            quote! { #krate::MatchVals<usize> },
            array_expr,
            quote! { __k.0 },
        ),
        Some(s) => {
            let s_lit = proc_macro2::Literal::usize_suffixed(s);
            (
                quote! { #krate::UnwrapOrElse<#krate::MatchVals<usize>> },
                quote! { (async || #krate::MatchVals(#s_lit), #array_expr) },
                quote! { __k.0.0 },
            )
        }
    }
}

/// Generate helper tuple struct definitions and DeserializeOwned impls for tuple variants (owned family).
fn gen_tuple_variant_helpers_owned(
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for cv in classified.iter() {
        if let VariantKind::Tuple(fields) = &cv.kind {
            let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
            let field_count = field_types.len();
            let acc_names: Vec<_> = (0..field_count)
                .map(|i| format_ident!("__f{}", i))
                .collect();

            // Generate the sequential seq.next calls for each element.
            let seq_reads: Vec<_> = acc_names
                .iter()
                .zip(field_types.iter())
                .map(|(acc, ty)| {
                    quote! {
                        let __v = #krate::hit!(__seq.next(|[__se]| async {
                            __se.get::<#ty, _>(()).await
                        }).await);
                        let (__seq_back, #acc) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                    }
                })
                .collect();

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name( #( #field_types, )* );

                impl #krate::DeserializeOwned for #helper_name
                where
                    #( #field_types: #krate::DeserializeOwned, )*
                {
                    async fn deserialize_owned<__D2: #krate::DeserializerOwned>(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.entry(|[__e]| async {
                            let __seq = #krate::hit!(__e.deserialize_seq().await);

                            #( #seq_reads )*

                            // Expect sequence exhaustion.
                            let __v = #krate::hit!(__seq.next::<1, _, _, ()>(|[__se]| async {
                                ::core::result::Result::Ok(#krate::Probe::Miss)
                            }).await);
                            let __claim = #krate::or_miss!(__v.done());
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __claim,
                                #helper_name( #( #acc_names, )* ),
                            )))
                        }).await
                    }
                }
            });
        }
    }
    tokens
}

fn expand_owned_enum_unit_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_key_sentinel: usize,
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let candidates: Vec<(String, usize)> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .enumerate()
        .flat_map(|(local_idx, cv)| {
            let mut pairs = vec![(cv.wire_name.clone(), local_idx)];
            for alias in &cv.aliases {
                pairs.push((alias.clone(), local_idx));
            }
            pairs
        })
        .collect();

    let has_other = other_variant(classified).is_some();
    let sentinel = if has_other {
        Some(variant_key_sentinel)
    } else {
        None
    };
    let (key_type, key_extra, key_idx) = key_matcher_tokens(&candidates, sentinel, krate);

    let unit_match_arms: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .enumerate()
        .filter_map(|(idx, cv)| {
            if matches!(cv.kind, VariantKind::Unit) {
                let vname = &cv.variant.ident;
                Some(quote! {
                    #idx => ::core::result::Result::Ok(
                        #krate::Probe::Hit((__claim, #name::#vname))
                    ),
                })
            } else {
                None
            }
        })
        .collect();

    let unit_wildcard = match other_variant(classified) {
        Some(vname) => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname))),
        },
        None => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        },
    };

    Ok(quote! {
        d.entry(|[__e]| async {
            let (__claim, __k) = #krate::hit!(
                __e.deserialize_value::<#key_type, _>(#key_extra).await
            );
            let __matched = #key_idx;
            match __matched {
                #( #unit_match_arms )*
                #unit_wildcard
            }
        }).await
    })
}

fn expand_owned_enum_map_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_candidates: &[(String, usize)],
    variant_key_sentinel: usize,
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let map_body = gen_owned_enum_map_body(
        name,
        classified,
        variant_candidates,
        variant_key_sentinel,
        krate,
    );
    Ok(quote! {
        d.entry(|[__e]| async {
            #map_body
        }).await
    })
}

fn expand_owned_enum_mixed(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_candidates: &[(String, usize)],
    variant_key_sentinel: usize,
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    // Build candidates for known unit str variants only (no sentinel - unknown strings
    // are handled by a dedicated arm below when has_other is true).
    let tagged_units: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit) && !cv.other)
        .collect();

    let unit_candidates: Vec<(String, usize)> = tagged_units
        .iter()
        .enumerate()
        .flat_map(|(local_idx, cv)| {
            let mut pairs = vec![(cv.wire_name.clone(), local_idx)];
            for alias in &cv.aliases {
                pairs.push((alias.clone(), local_idx));
            }
            pairs
        })
        .collect();
    // No sentinel - plain MatchVals; if the token is not a matching str it returns Miss
    // and the next arm handles it.
    let (unit_key_type, unit_key_extra, unit_key_idx) =
        key_matcher_tokens(&unit_candidates, None, krate);

    let unit_match_arms: Vec<_> = tagged_units
        .iter()
        .enumerate()
        .map(|(local_idx, cv)| {
            let vname = &cv.variant.ident;
            quote! {
                #local_idx => {
                    ::core::result::Result::Ok(
                        #krate::Probe::Hit((__unit_claim, #name::#vname))
                    )
                }
            }
        })
        .collect();

    let map_body = gen_owned_enum_map_body(
        name,
        classified,
        variant_candidates,
        variant_key_sentinel,
        krate,
    );

    let has_other = other_variant(classified).is_some();

    if has_other {
        let other_vname = other_variant(classified).unwrap();
        // 3 arms: known str, unknown str drain → other, map
        Ok(quote! {
            d.entry(|[__e1, __e2, __e3]| async {
                #krate::select_probe! {
                    // Arm 1: known unit str variants
                    async move {
                        match __e1.deserialize_value::<#unit_key_type, _>(#unit_key_extra).await? {
                            #krate::Probe::Hit((__unit_claim, __k)) => {
                                let __matched = #unit_key_idx;
                                match __matched {
                                    #( #unit_match_arms, )*
                                    _ => unreachable!(),
                                }
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    },
                    // Arm 2: drain any remaining string → other
                    async move {
                        let mut __str_acc = #krate::hit!(__e2.deserialize_str_chunks().await);
                        let __claim = loop {
                            match __str_acc.next_str(|_| {}).await? {
                                #krate::Chunk::Data((__c, ())) => { __str_acc = __c; }
                                #krate::Chunk::Done(__c) => break __c,
                            }
                        };
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#other_vname)))
                    },
                    // Arm 3: map (non-unit variants)
                    async move {
                        let __e = __e3;
                        #map_body
                    },
                    @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }).await
        })
    } else {
        // 2 arms: known str, map
        Ok(quote! {
            d.entry(|[__e1, __e2]| async {
                #krate::select_probe! {
                    // Arm 1: known unit str variants
                    async move {
                        match __e1.deserialize_value::<#unit_key_type, _>(#unit_key_extra).await? {
                            #krate::Probe::Hit((__unit_claim, __k)) => {
                                let __matched = #unit_key_idx;
                                match __matched {
                                    #( #unit_match_arms, )*
                                    _ => unreachable!(),
                                }
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    },
                    // Arm 2: map (non-unit variants)
                    async move {
                        let __e = __e2;
                        #map_body
                    },
                    @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }).await
        })
    }
}

/// Generate the body that reads a single-key map for non-unit variant dispatch (owned family).
///
/// Takes `__e: EntryOwned` by name and calls `deserialize_map`, then builds one
/// `MapArmSlot` per non-unit tagged variant. The map must contain exactly one key-value pair.
fn gen_owned_enum_map_body(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    _variant_candidates: &[(String, usize)],
    _variant_key_sentinel: usize,
    krate: &syn::Path,
) -> TokenStream2 {
    let tagged_nonunit: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit))
        .collect();

    // One arm per non-unit tagged variant.
    let arm_slots: Vec<TokenStream2> = tagged_nonunit
        .iter()
        .map(|cv| {
            let vname = &cv.variant.ident;
            let mut wire_names: Vec<&str> = vec![cv.wire_name.as_str()];
            for alias in &cv.aliases {
                wire_names.push(alias.as_str());
            }
            let key_fn = if wire_names.len() == 1 {
                let wn = wire_names[0];
                quote! {
                    |mut __kp: #krate::owned::KP<__D>| {
                        __kp.deserialize_key::<#krate::Match, &str>(#wn)
                    }
                }
            } else {
                quote! {
                    |mut __kp: #krate::owned::KP<__D>| {
                        __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wire_names, ()), )*])
                    }
                }
            };
            let val_fn = match &cv.kind {
                VariantKind::Newtype(ty) => quote! {
                    |__vp: #krate::owned::VP2<__D>, __k| async move {
                        let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ty, _>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, #name::#vname(__v)))))
                    }
                },
                VariantKind::Struct(fields) => {
                    let helper_name = format_ident!("__VariantOwned{}", cv.index);
                    let field_names: Vec<_> = fields.iter()
                        .map(|f| f.ident.as_ref().unwrap())
                        .collect();
                    quote! {
                        |__vp: #krate::owned::VP2<__D>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name, _>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __vc,
                                (__k, #name::#vname { #( #field_names: __v.#field_names, )* })
                            )))
                        }
                    }
                },
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    let field_indices: Vec<syn::Index> = (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        |__vp: #krate::owned::VP2<__D>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name, _>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __vc,
                                (__k, #name::#vname( #( __v.#field_indices, )* ))
                            )))
                        }
                    }
                },
                VariantKind::Unit => unreachable!(),
            };
            quote! { #krate::MapArmSlot::new(#key_fn, #val_fn) }
        })
        .collect();

    // Output bindings - one per arm.
    let out_names: Vec<syn::Ident> = tagged_nonunit
        .iter()
        .enumerate()
        .map(|(i, _)| format_ident!("__out_v{}", i))
        .collect();
    let output_pat = {
        let mut pat = quote! { () };
        for out in &out_names {
            pat = quote! { (#pat, #out) };
        }
        pat
    };

    // dup wire names for DetectDuplicatesOwned.
    let dup_wire_names: Vec<TokenStream2> = tagged_nonunit
        .iter()
        .enumerate()
        .flat_map(|(arm_idx, cv)| {
            let mut entries: Vec<TokenStream2> = vec![];
            let primary = &cv.wire_name;
            entries.push(quote! { (#primary, #arm_idx) });
            for alias in &cv.aliases {
                entries.push(quote! { (#alias, #arm_idx) });
            }
            entries
        })
        .collect();

    let has_other = other_variant(classified).is_some();
    let arms_expr = {
        let mut expr = quote! { #krate::MapArmBase };
        for slot in &arm_slots {
            expr = quote! { (#expr, #slot) };
        }
        expr = quote! {{
            let __wn = [#( #dup_wire_names, )*];
            #krate::DetectDuplicatesOwned::new(
                #expr,
                __wn,
                move |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                |__vp: #krate::owned::VP2<__D>| __vp.skip(),
            )
        }};
        if has_other {
            expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                    let __vc = __vp.skip().await?;
                    ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                },
            )) };
        }
        expr
    };

    // Result extraction: find the one Some output; Miss if none.
    let other_arm = match other_variant(classified) {
        Some(vname) => {
            quote! { ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname))) }
        }
        None => quote! { ::core::result::Result::Ok(#krate::Probe::Miss) },
    };
    let result_arms: Vec<TokenStream2> = out_names
        .iter()
        .map(|out| {
            quote! {
                if let ::core::option::Option::Some((_k, __v)) = #out {
                    return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)));
                }
            }
        })
        .collect();

    quote! {
        let __map = #krate::hit!(__e.deserialize_map().await);
        let __arms = #arms_expr;
        let (__claim, #output_pat) = #krate::hit!(__map.iterate(__arms).await);
        #( #result_arms )*
        #other_arm
    }
}

/// Generate helper struct definitions and DeserializeOwned impls for struct variants (owned family).
fn gen_struct_variant_helpers_owned(
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    rename_all: Option<crate::common::RenameAll>,
) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for cv in classified.iter() {
        if let VariantKind::Struct(fields) = &cv.kind {
            let helper_name = format_ident!("__VariantOwned{}", cv.index);
            let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
            let cf = match classify_fields(fields, rename_all) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // One arm slot per field.
            let arm_slots: Vec<TokenStream2> = field_names
                .iter()
                .zip(cf.iter())
                .zip(field_types.iter())
                .map(|((_, f), ft)| {
                    let mut wire_names: Vec<&str> = vec![f.wire_name.as_str()];
                    for alias in &f.aliases {
                        wire_names.push(alias.as_str());
                    }
                    let key_fn = if wire_names.len() == 1 {
                        let name = wire_names[0];
                        quote! {
                            |mut __kp: #krate::owned::KP<__D2>| {
                                __kp.deserialize_key::<#krate::Match, &str>(#name)
                            }
                        }
                    } else {
                        quote! {
                            |mut __kp: #krate::owned::KP<__D2>| {
                                __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wire_names, ()), )*])
                            }
                        }
                    };
                    let val_fn = quote! {
                        |__vp: #krate::owned::VP2<__D2>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ft, _>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                        }
                    };
                    quote! { #krate::MapArmSlot::new(#key_fn, #val_fn) }
                })
                .collect();

            // Output bindings and destructure pattern.
            let out_names: Vec<syn::Ident> = field_names
                .iter()
                .map(|n| format_ident!("__out_{}", n))
                .collect();
            let output_pat = {
                let mut pat = quote! { () };
                for out_name in &out_names {
                    pat = quote! { (#pat, #out_name) };
                }
                pat
            };

            // DetectDuplicatesOwned wire names array.
            let dup_wire_names: Vec<TokenStream2> = cf
                .iter()
                .enumerate()
                .flat_map(|(arm_idx, f)| {
                    let mut entries: Vec<TokenStream2> = vec![];
                    let primary = &f.wire_name;
                    entries.push(quote! { (#primary, #arm_idx) });
                    for alias in &f.aliases {
                        entries.push(quote! { (#alias, #arm_idx) });
                    }
                    entries
                })
                .collect();

            let arms_expr = {
                let mut expr = quote! { #krate::MapArmBase };
                for slot in &arm_slots {
                    expr = quote! { (#expr, #slot) };
                }
                quote! {
                    {
                        let __wn = [#( #dup_wire_names, )*];
                        #krate::DetectDuplicatesOwned::new(
                            #expr,
                            __wn,
                            move |__kp: #krate::owned::KP<__D2>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                            |__vp: #krate::owned::VP2<__D2>| __vp.skip(),
                        )
                    }
                }
            };

            // Finalizers: extract from Option<(K, V)>, Miss on absent (no defaults here).
            let field_finalizers: Vec<TokenStream2> = field_names
                .iter()
                .map(|fname| {
                    let out_name = format_ident!("__out_{}", fname);
                    quote! {
                        let #fname = match #out_name {
                            ::core::option::Option::Some((_k, __v)) => __v,
                            ::core::option::Option::None => {
                                return ::core::result::Result::Ok(#krate::Probe::Miss)
                            }
                        };
                    }
                })
                .collect();

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name {
                    #( #field_names: #field_types, )*
                }

                impl #krate::DeserializeOwned for #helper_name
                where
                    #( #field_types: #krate::DeserializeOwned, )*
                {
                    async fn deserialize_owned<__D2: #krate::DeserializerOwned>(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.entry(|[__e]| async {
                            let __map = #krate::hit!(__e.deserialize_map().await);
                            let __arms = #arms_expr;
                            let (__claim, #output_pat) = #krate::hit!(__map.iterate(__arms).await);
                            #( #field_finalizers )*
                            ::core::result::Result::Ok(
                                #krate::Probe::Hit((__claim, #helper_name { #( #field_names, )* }))
                            )
                        }).await
                    }
                }
            });
        }
    }
    tokens
}

/// All untagged - try each variant by shape (owned family).
fn expand_owned_enum_untagged_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let n_handles = classified.len();
    let handle_names: Vec<_> = (0..n_handles).map(|i| format_ident!("__e{}", i)).collect();

    let refs: Vec<_> = classified.iter().collect();
    let probe_chain = gen_untagged_probe_chain_owned(name, &refs, &handle_names, krate);

    Ok(quote! {
        d.entry(|[#( #handle_names ),*]| async {
            #probe_chain
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await
    })
}

/// Mixed tagged + untagged (owned family).
fn expand_owned_enum_with_untagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_candidates: &[(String, usize)],
    variant_key_sentinel: usize,
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let has_tagged_unit = classified
        .iter()
        .any(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit));
    let has_tagged_nonunit = classified
        .iter()
        .any(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit));
    let untagged_count = classified.iter().filter(|cv| cv.untagged).count();

    let mut handle_idx = 0usize;
    let str_handle = if has_tagged_unit {
        let h = format_ident!("__e{}", handle_idx);
        handle_idx += 1;
        Some(h)
    } else {
        None
    };
    let map_handle = if has_tagged_nonunit {
        let h = format_ident!("__e{}", handle_idx);
        handle_idx += 1;
        Some(h)
    } else {
        None
    };
    let untagged_handles: Vec<_> = (0..untagged_count)
        .map(|i| format_ident!("__e{}", handle_idx + i))
        .collect();
    let n_handles = handle_idx + untagged_count;
    let all_handles: Vec<_> = (0..n_handles).map(|i| format_ident!("__e{}", i)).collect();

    // Tagged str section (unit variants via str_chunks).
    let str_section = if let Some(h) = &str_handle {
        let tagged_units: Vec<_> = classified
            .iter()
            .filter(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit))
            .collect();
        let unit_candidates: Vec<(String, usize)> = tagged_units
            .iter()
            .enumerate()
            .flat_map(|(local_idx, cv)| {
                let mut pairs = vec![(cv.wire_name.clone(), local_idx)];
                for alias in &cv.aliases {
                    pairs.push((alias.clone(), local_idx));
                }
                pairs
            })
            .collect();
        // No other variant in the with_untagged case - unknown string = Miss
        let (unit_key_type, unit_key_extra, unit_key_idx) =
            key_matcher_tokens(&unit_candidates, None, krate);
        let unit_match_arms: Vec<_> = tagged_units
            .iter()
            .enumerate()
            .map(|(local_idx, cv)| {
                let vname = &cv.variant.ident;
                quote! {
                    #local_idx => {
                        return ::core::result::Result::Ok(
                            #krate::Probe::Hit((__unit_claim, #name::#vname))
                        );
                    }
                }
            })
            .collect();

        quote! {
            match #h.deserialize_value::<#unit_key_type, _>(#unit_key_extra).await? {
                #krate::Probe::Hit((__unit_claim, __k)) => {
                    let __matched = #unit_key_idx;
                    match __matched {
                        #( #unit_match_arms )*
                        _ => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                    }
                }
                #krate::Probe::Miss => {}
            }
        }
    } else {
        quote! {}
    };

    // Tagged map section.
    let map_section = if let Some(h) = &map_handle {
        let map_body = gen_owned_enum_map_body(
            name,
            classified,
            variant_candidates,
            variant_key_sentinel,
            krate,
        );
        quote! {
            {
                let __e = #h;
                // Run the map body in an async closure so its `return`s exit
                // the closure rather than the outer entry closure.
                let __map_result: ::core::result::Result<
                    #krate::Probe<_>,
                    _,
                > = (async move { #map_body }).await;
                match __map_result {
                    ::core::result::Result::Ok(#krate::Probe::Hit(__v)) => {
                        return ::core::result::Result::Ok(#krate::Probe::Hit(__v));
                    }
                    ::core::result::Result::Err(__err) => {
                        return ::core::result::Result::Err(__err);
                    }
                    ::core::result::Result::Ok(#krate::Probe::Miss) => {}
                }
            }
        }
    } else {
        quote! {}
    };

    // Untagged section.
    let untagged_classified: Vec<_> = classified.iter().filter(|cv| cv.untagged).collect();
    let untagged_section =
        gen_untagged_probe_chain_owned(name, &untagged_classified, &untagged_handles, krate);

    Ok(quote! {
        d.entry(|[#( #all_handles ),*]| async {
            #str_section
            #map_section
            #untagged_section
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await
    })
}

/// Generate untagged probe chain for owned family.
fn gen_untagged_probe_chain_owned(
    name: &syn::Ident,
    variants: &[&ClassifiedVariant],
    handles: &[syn::Ident],
    krate: &syn::Path,
) -> TokenStream2 {
    let mut arms = TokenStream2::new();
    for (i, cv) in variants.iter().enumerate() {
        let handle = &handles[i];
        let vname = &cv.variant.ident;
        let arm = match &cv.kind {
            VariantKind::Unit => {
                quote! {
                    match #handle.deserialize_null().await? {
                        #krate::Probe::Hit(__c) => {
                            return ::core::result::Result::Ok(
                                #krate::Probe::Hit((__c, #name::#vname))
                            );
                        }
                        #krate::Probe::Miss => {}
                    }
                }
            }
            VariantKind::Newtype(ty) => {
                quote! {
                    match #handle.deserialize_value::<#ty, _>(()).await? {
                        #krate::Probe::Hit((__c, __v)) => {
                            return ::core::result::Result::Ok(
                                #krate::Probe::Hit((__c, #name::#vname(__v)))
                            );
                        }
                        #krate::Probe::Miss => {}
                    }
                }
            }
            VariantKind::Tuple(fields) => {
                let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                let field_indices: Vec<syn::Index> =
                    (0..fields.len()).map(syn::Index::from).collect();
                quote! {
                    match #handle.deserialize_value::<#helper_name, _>(()).await? {
                        #krate::Probe::Hit((__c, __v)) => {
                            return ::core::result::Result::Ok(
                                #krate::Probe::Hit((__c, #name::#vname( #( __v.#field_indices, )* )))
                            );
                        }
                        #krate::Probe::Miss => {}
                    }
                }
            }
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__VariantOwned{}", cv.index);
                let field_names: Vec<_> =
                    fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                quote! {
                    match #handle.deserialize_value::<#helper_name, _>(()).await? {
                        #krate::Probe::Hit((__c, __v)) => {
                            return ::core::result::Result::Ok(
                                #krate::Probe::Hit((__c, #name::#vname { #( #field_names: __v.#field_names, )* }))
                            );
                        }
                        #krate::Probe::Miss => {}
                    }
                }
            }
        };
        arms.extend(arm);
    }
    arms
}

/// Generate a `DeserializeOwned` impl for an internally tagged enum (`#[strede(tag = "field")]`).
///
/// Phase 1: unit variants only. Non-unit variants produce a compile-time error.
#[allow(clippy::too_many_arguments)]
fn expand_owned_enum_internally_tagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    variant_candidates: &[(String, usize)],
    variant_key_sentinel: usize,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    impl_generics: impl quote::ToTokens,
    ty_generics: impl quote::ToTokens,
    where_clause: impl quote::ToTokens,
) -> syn::Result<TokenStream2> {
    let has_nonunit = classified
        .iter()
        .any(|cv| !matches!(cv.kind, VariantKind::Unit));

    let (body, helpers) = if !has_nonunit {
        (
            expand_owned_internally_tagged_unit_only(
                name,
                classified,
                tag_field,
                variant_candidates,
                variant_key_sentinel,
                krate,
            )?,
            quote! {},
        )
    } else {
        expand_owned_internally_tagged_with_nonunit(
            name,
            classified,
            tag_field,
            variant_candidates,
            variant_key_sentinel,
            krate,
            container_attrs,
        )?
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapKeyProbeOwned as _, MapAccessOwned as _, SeqAccessOwned as _,
                SeqEntryOwned as _, StrAccessOwned as _, MapValueProbeOwned as _,
            };

            #helpers

            impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    #body
                }
            }
        };
    })
}

/// Unit-only internally-tagged enum (owned family).
fn expand_owned_internally_tagged_unit_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    variant_candidates: &[(String, usize)],
    _variant_key_sentinel: usize,
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    // Build MatchVals<usize> extra for the tag value: [("VariantA", 0), ("VariantB", 1), ...]
    let val_extra_entries: Vec<TokenStream2> = variant_candidates
        .iter()
        .map(|(wire_name, idx)| quote! { (#wire_name, #idx) })
        .collect();
    let val_extra_count = val_extra_entries.len();

    // One arm: key = tag_field, value = MatchVals<usize> over variant candidates.
    let arm_slot = quote! {
        #krate::MapArmSlot::new(
            |mut __kp: #krate::owned::KP<__D>| {
                __kp.deserialize_key::<#krate::Match, &str>(#tag_field)
            },
            |__vp: #krate::owned::VP2<__D>, __k| async move {
                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#krate::MatchVals<usize>, _>(
                    [#( #val_extra_entries, )*]
                ).await);
                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
            },
        )
    };

    // dup wire names for DetectDuplicatesOwned (only tag_field is a real arm).
    let arms_expr = quote! {{
        let __wn = [(#tag_field, 0usize)];
        let __inner = #krate::DetectDuplicatesOwned::new(
            (#krate::MapArmBase, #arm_slot),
            __wn,
            move |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
            |__vp: #krate::owned::VP2<__D>| __vp.skip(),
        );
        (__inner, #krate::VirtualArmSlot::new(
            |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip, _>(()),
            |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                let __vc = __vp.skip().await?;
                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
            },
        ))
    }};

    // Build match arms: idx → variant name.
    let unit_match_arms: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .enumerate()
        .filter_map(|(idx, cv)| {
            if matches!(cv.kind, VariantKind::Unit) {
                let vname = &cv.variant.ident;
                Some(quote! {
                    #krate::MatchVals(#idx) => ::core::result::Result::Ok(
                        #krate::Probe::Hit((__claim, #name::#vname))
                    ),
                })
            } else {
                None
            }
        })
        .collect();

    let unit_wildcard = match other_variant(classified) {
        Some(vname) => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname))),
        },
        None => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        },
    };

    let _ = val_extra_count;

    Ok(quote! {
        d.entry(|[__e]| async {
            let __map = #krate::hit!(__e.deserialize_map().await);
            let __arms = #arms_expr;
            let (__claim, ((), __out_0)) = #krate::hit!(__map.iterate(__arms).await);
            match __out_0 {
                ::core::option::Option::Some((_k, __matched)) => match __matched {
                    #( #unit_match_arms )*
                    #unit_wildcard
                },
                ::core::option::Option::None => {
                    // Tag field was not found.
                    ::core::result::Result::Ok(#krate::Probe::Miss)
                }
            }
        }).await
    })
}

/// Internally-tagged enum with non-unit variants (owned family).
///
/// Each non-unit variant is raced concurrently via `select_probe!`. Each arm
/// gets a `TagAwareDeserializerOwned` facade that injects a tag-capture arm
/// into the variant's field arm stack and validates the captured tag index
/// matches that variant before returning `Hit`.
fn expand_owned_internally_tagged_with_nonunit(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    variant_candidates: &[(String, usize)],
    _variant_key_sentinel: usize,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
) -> syn::Result<(TokenStream2, TokenStream2)> {
    // Collect tagged (non-untagged) variants with their local indices.
    let tagged: Vec<(usize, &ClassifiedVariant)> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .enumerate()
        .collect();

    // Separate into unit and non-unit variants.
    let unit_variants: Vec<_> = tagged
        .iter()
        .filter(|(_, cv)| matches!(cv.kind, VariantKind::Unit))
        .collect();
    let nonunit_variants: Vec<_> = tagged
        .iter()
        .filter(|(_, cv)| !matches!(cv.kind, VariantKind::Unit))
        .collect();

    let nonunit_count = nonunit_variants.len();
    let tag_candidates_count = variant_candidates.len();

    // --- Generate struct variant helpers ---
    let struct_helpers =
        gen_struct_variant_helpers_owned(classified, krate, container_attrs.rename_all);
    let tuple_helpers = gen_tuple_variant_helpers_owned(classified, krate);

    // Tag candidates array literal: [("VariantA", 0), ("VariantB", 1), ...]
    // variant_candidates is Vec<(String, usize)> = all (wire_name, local_idx) pairs.
    let tag_cands_entries: Vec<TokenStream2> = variant_candidates
        .iter()
        .map(|(wire_name, idx)| quote! { (#wire_name, #idx) })
        .collect();

    // Build one select_probe arm per non-unit variant.
    // Each arm gets a pre-forked map variable so all forks are created before
    // any async blocks capture them (avoiding multiple &mut borrows on __map).
    let mut fork_stmts: Vec<TokenStream2> = Vec::new();
    let mut select_arms: Vec<TokenStream2> = Vec::new();

    for (arm_i, &(local_idx, cv)) in nonunit_variants.iter().enumerate() {
        let vname = &cv.variant.ident;
        let fork_ident = format_ident!("__map_{}", arm_i);

        let (de_type, variant_construction) = match &cv.kind {
            VariantKind::Newtype(ty) => (quote! { #ty }, quote! { #name::#vname(__v) }),
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__VariantOwned{}", cv.index);
                let field_names: Vec<_> =
                    fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                (
                    quote! { #helper_name },
                    quote! { #name::#vname { #( #field_names: __v.#field_names, )* } },
                )
            }
            VariantKind::Tuple(_) => {
                let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                let field_indices: Vec<syn::Index> = match &cv.kind {
                    VariantKind::Tuple(fields) => (0..fields.len()).map(syn::Index::from).collect(),
                    _ => unreachable!(),
                };
                (
                    quote! { #helper_name },
                    quote! { #name::#vname( #( __v.#field_indices, )* ) },
                )
            }
            VariantKind::Unit => unreachable!(),
        };

        fork_stmts.push(quote! {
            let #fork_ident = __map.fork();
        });

        select_arms.push(quote! {
            async move {
                let __d = #krate::tag_facade::TagAwareDeserializerOwned::new(
                    #fork_ident,
                    #tag_field,
                    [#( #tag_cands_entries, )*],
                    #local_idx,
                    __tag_value,
                );
                match <#de_type as #krate::DeserializeOwned>::deserialize_owned(__d, ()).await? {
                    #krate::Probe::Hit((__c, __v)) =>
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #variant_construction))),
                    #krate::Probe::Miss =>
                        ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }
        });
    }

    // Unit variant match arms: after all non-unit arms miss, check tag_value.
    // These are returned if the tag matched a unit variant index.
    let unit_match_arms: Vec<_> = unit_variants
        .iter()
        .map(|&(local_idx, cv)| {
            let vname = &cv.variant.ident;
            quote! {
                ::core::option::Option::Some(#local_idx) => {
                    // Unit variant: drain the remaining map entries (all unknown since
                    // the tag was already consumed by the non-unit arms' TagInjectingStackOwned)
                    // and use the map claim to finalize the entry.
                    return match __map.fork().iterate(
                        (#krate::MapArmBase, #krate::VirtualArmSlot::new(
                        |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                        |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                            let __vc = __vp.skip().await?;
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                        },
                    ))
                    ).await? {
                        #krate::Probe::Hit((__c, _)) =>
                            ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname))),
                        #krate::Probe::Miss =>
                            ::core::result::Result::Ok(#krate::Probe::Miss),
                    };
                }
            }
        })
        .collect();

    let other_arm = match other_variant(classified) {
        Some(vname) => quote! {
            _ => {
                // `other` variant: drain the map and return the catch-all variant.
                return match __map.fork().iterate(
                    (#krate::MapArmBase, #krate::VirtualArmSlot::new(
                        |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                        |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                            let __vc = __vp.skip().await?;
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                        },
                    ))
                ).await? {
                    #krate::Probe::Hit((__c, _)) =>
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname))),
                    #krate::Probe::Miss =>
                        ::core::result::Result::Ok(#krate::Probe::Miss),
                };
            }
        },
        None => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        },
    };

    let _ = nonunit_count;
    let _ = tag_candidates_count;

    let body = quote! {
        d.entry(|[__e]| async {
            let mut __map = #krate::hit!(__e.deserialize_map().await);
            let __tag_value: ::core::cell::Cell<::core::option::Option<usize>> =
                ::core::cell::Cell::new(::core::option::Option::None);
            let __tag_value = &__tag_value;

            // Fork the map once per non-unit variant before entering select_probe!
            // so that each async arm captures an owned fork, not &mut __map.
            #( #fork_stmts )*

            // Race all non-unit variant arms. Each arm uses a TagAwareDeserializerOwned
            // that injects the tag arm and checks expected_variant before returning Hit.
            let __result = #krate::select_probe! {
                #( #select_arms, )*
                @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
            };
            if let ::core::result::Result::Ok(#krate::Probe::Hit(__v)) = __result {
                return ::core::result::Result::Ok(#krate::Probe::Hit(__v));
            }
            if let ::core::result::Result::Err(__e) = __result {
                return ::core::result::Result::Err(__e);
            }

            // All non-unit arms missed. Check tag_value for unit variants.
            match __tag_value.get() {
                #( #unit_match_arms )*
                #other_arm
            }
        }).await
    };

    let helpers = quote! {
        #struct_helpers
        #tuple_helpers
    };

    Ok((body, helpers))
}

// ---------------------------------------------------------------------------
// Adjacent-tagged enum derive  (#[strede(tag = "t", content = "c")])
// ---------------------------------------------------------------------------

/// Generate a `DeserializeOwned` impl for an adjacently tagged enum.
///
/// Wire format: `{"t": "VariantName", "c": <payload>}` (key order-independent).
/// Unit variants have no content field: `{"t": "VariantName"}`.
#[allow(clippy::too_many_arguments)]
fn expand_owned_enum_adjacent_tagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    content_field: &str,
    variant_candidates: &[(String, usize)],
    _variant_key_sentinel: usize,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    impl_generics: impl quote::ToTokens,
    ty_generics: impl quote::ToTokens,
    where_clause: impl quote::ToTokens,
) -> syn::Result<TokenStream2> {
    let has_nonunit = classified
        .iter()
        .any(|cv| !matches!(cv.kind, VariantKind::Unit));

    let (body, helpers) = if !has_nonunit {
        // Unit-only: same as internally-tagged (no content field needed).
        (
            expand_owned_internally_tagged_unit_only(
                name,
                classified,
                tag_field,
                variant_candidates,
                _variant_key_sentinel,
                krate,
            )?,
            quote! {},
        )
    } else {
        expand_owned_adjacent_tagged_with_nonunit(
            name,
            classified,
            tag_field,
            content_field,
            variant_candidates,
            krate,
            container_attrs,
        )?
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapKeyProbeOwned as _, MapAccessOwned as _, SeqAccessOwned as _,
                SeqEntryOwned as _, StrAccessOwned as _, MapValueProbeOwned as _,
            };

            #helpers

            impl #impl_generics #krate::DeserializeOwned for #name #ty_generics #where_clause {
                async fn deserialize_owned<__D: #krate::DeserializerOwned>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    #body
                }
            }
        };
    })
}

/// Adjacent-tagged enum with at least one non-unit variant.
///
/// For each non-unit variant, forks the outer map and runs `iterate` with a
/// two-slot arm stack (tag slot + content slot) wrapped in `SkipUnknownOwned`.
/// On success, checks the tag index matches the expected variant.
///
/// Unit variants are handled as a fallback after all non-unit arms miss.
fn expand_owned_adjacent_tagged_with_nonunit(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    content_field: &str,
    variant_candidates: &[(String, usize)],
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
) -> syn::Result<(TokenStream2, TokenStream2)> {
    let tagged: Vec<(usize, &ClassifiedVariant)> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .enumerate()
        .collect();

    let unit_variants: Vec<_> = tagged
        .iter()
        .filter(|(_, cv)| matches!(cv.kind, VariantKind::Unit))
        .collect();
    let nonunit_variants: Vec<_> = tagged
        .iter()
        .filter(|(_, cv)| !matches!(cv.kind, VariantKind::Unit))
        .collect();

    let struct_helpers =
        gen_struct_variant_helpers_owned(classified, krate, container_attrs.rename_all);
    let tuple_helpers = gen_tuple_variant_helpers_owned(classified, krate);

    // Tag candidates array: [("VariantA", 0), ("VariantB", 1), ...]
    let tag_cands_entries: Vec<TokenStream2> = variant_candidates
        .iter()
        .map(|(wire_name, idx)| quote! { (#wire_name, #idx) })
        .collect();
    let tag_cands_count = variant_candidates.len();

    // The dup-detection array covers tag_field (arm 0) and content_field (arm 1).
    let dup_wire_names = quote! {
        [(#tag_field, 0usize), (#content_field, 1usize)]
    };

    let mut fork_stmts: Vec<TokenStream2> = Vec::new();
    let mut select_arms: Vec<TokenStream2> = Vec::new();

    for (arm_i, &(local_idx, cv)) in nonunit_variants.iter().enumerate() {
        let vname = &cv.variant.ident;
        let fork_ident = format_ident!("__map_{}", arm_i);

        let (de_type, variant_construction) = match &cv.kind {
            VariantKind::Newtype(ty) => (quote! { #ty }, quote! { #name::#vname(__v) }),
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__VariantOwned{}", cv.index);
                let field_names: Vec<_> =
                    fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                (
                    quote! { #helper_name },
                    quote! { #name::#vname { #( #field_names: __v.#field_names, )* } },
                )
            }
            VariantKind::Tuple(_) => {
                let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                let field_indices: Vec<syn::Index> = match &cv.kind {
                    VariantKind::Tuple(fields) => (0..fields.len()).map(syn::Index::from).collect(),
                    _ => unreachable!(),
                };
                (
                    quote! { #helper_name },
                    quote! { #name::#vname( #( __v.#field_indices, )* ) },
                )
            }
            VariantKind::Unit => unreachable!(),
        };

        fork_stmts.push(quote! {
            let #fork_ident = __map.fork();
        });

        select_arms.push(quote! {
            async move {
                // Two-slot arm stack: tag slot + content slot, with dup detection + skip unknown.
                let __arms = {
                    let __inner_arms = (
                        (#krate::MapArmBase,
                         #krate::MapArmSlot::new(
                             |mut __kp: #krate::owned::KP<__D>| {
                                 __kp.deserialize_key::<#krate::Match, &str>(#tag_field)
                             },
                             |__vp: #krate::owned::VP2<__D>, __k| async move {
                                 let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                     #krate::MatchVals<usize>,
                                     [(&'static str, usize); #tag_cands_count]
                                 >([#( #tag_cands_entries, )*]).await);
                                 ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                             },
                         )),
                        #krate::MapArmSlot::new(
                            |mut __kp: #krate::owned::KP<__D>| {
                                __kp.deserialize_key::<#krate::Match, &str>(#content_field)
                            },
                            |__vp: #krate::owned::VP2<__D>, __k| async move {
                                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#de_type, ()>(()).await);
                                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                            },
                        )
                    );
                    let __wn = #dup_wire_names;
                    let __dd = #krate::DetectDuplicatesOwned::new(
                        __inner_arms,
                        __wn,
                        move |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                        |__vp: #krate::owned::VP2<__D>| __vp.skip(),
                    );
                    (__dd, #krate::VirtualArmSlot::new(
                        |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                        |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                            let __vc = __vp.skip().await?;
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                        },
                    ))
                };
                let (__claim, (((), __opt_tag), __opt_content)) =
                    #krate::hit!(#fork_ident.iterate(__arms).await);
                match (__opt_tag, __opt_content) {
                    (
                        ::core::option::Option::Some((_, #krate::MatchVals(#local_idx))),
                        ::core::option::Option::Some((_, __v)),
                    ) => ::core::result::Result::Ok(
                        #krate::Probe::Hit((__claim, #variant_construction))
                    ),
                    _ => ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }
        });
    }

    // Unit variant match arms (fallback after all non-unit arms miss).
    // Iterate the outer map with just the tag slot + SkipUnknownOwned.
    let unit_match_arms: Vec<_> = unit_variants
        .iter()
        .map(|&(local_idx, cv)| {
            let vname = &cv.variant.ident;
            quote! {
                ::core::option::Option::Some((_, #krate::MatchVals(#local_idx))) => {
                    return ::core::result::Result::Ok(
                        #krate::Probe::Hit((__unit_claim, #name::#vname))
                    );
                }
            }
        })
        .collect();

    let other_arm = match other_variant(classified) {
        Some(vname) => quote! {
            _ => {
                return ::core::result::Result::Ok(
                    #krate::Probe::Hit((__unit_claim, #name::#vname))
                );
            }
        },
        None => quote! {
            _ => return ::core::result::Result::Ok(#krate::Probe::Miss),
        },
    };

    let unit_fallback = if !unit_match_arms.is_empty() || other_variant(classified).is_some() {
        quote! {
            // Unit variant fallback: iterate the outer map looking for the tag field.
            let __unit_arms = (
                (#krate::MapArmBase,
                 #krate::MapArmSlot::new(
                     |mut __kp: #krate::owned::KP<__D>| {
                         __kp.deserialize_key::<#krate::Match, &str>(#tag_field)
                     },
                     |__vp: #krate::owned::VP2<__D>, __k| async move {
                         let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                             #krate::MatchVals<usize>,
                             [(&'static str, usize); #tag_cands_count]
                         >([#( #tag_cands_entries, )*]).await);
                         ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                     },
                 )),
                #krate::VirtualArmSlot::new(
                    |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                    |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                        let __vc = __vp.skip().await?;
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                    },
                ),
            );
            let (__unit_claim, ((), __opt_unit_tag)) =
                #krate::hit!(__map.fork().iterate(__unit_arms).await);
            match __opt_unit_tag {
                #( #unit_match_arms )*
                #other_arm
            }
        }
    } else {
        quote! {
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }
    };

    let body = quote! {
        d.entry(|[__e]| async {
            let mut __map = #krate::hit!(__e.deserialize_map().await);

            // Fork the map once per non-unit variant.
            #( #fork_stmts )*

            // Race all non-unit variant arms.
            let __result = #krate::select_probe! {
                #( #select_arms, )*
                @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
            };
            if let ::core::result::Result::Ok(#krate::Probe::Hit(__v)) = __result {
                return ::core::result::Result::Ok(#krate::Probe::Hit(__v));
            }
            if let ::core::result::Result::Err(__e) = __result {
                return ::core::result::Result::Err(__e);
            }

            // All non-unit arms missed - check for unit variants.
            #unit_fallback
        }).await
    };

    let helpers = quote! {
        #struct_helpers
        #tuple_helpers
    };

    Ok((body, helpers))
}
