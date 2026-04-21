use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

use crate::common::{
    ClassifiedVariant, DefaultAttr, VariantKind, all_field_types, apply_borrow_field_bound,
    borrow_lifetimes, classify_fields, classify_variants, other_variant, parse_container_attrs,
};

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let container_attrs = parse_container_attrs(&input.attrs)?;
    let krate = &container_attrs.crate_path;
    match &input.data {
        Data::Struct(_) => expand_struct(input, krate, &container_attrs),
        Data::Enum(_) => expand_enum(input, krate),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "Deserialize can only be derived for structs and enums",
        )),
    }
}

/// Generate a borrow-family impl that deserializes `from_ty` and converts to `Self`
/// via `From` (`is_try = false`) or `TryFrom` with `or_miss!` (`is_try = true`).
fn gen_container_from_borrow(
    input: &DeriveInput,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    from_ty: &syn::Type,
    is_try: bool,
) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let (_, ty_generics, _) = input.generics.split_for_impl();
    let mut impl_gen = input.generics.clone();
    let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
    if !has_de {
        impl_gen.params.insert(0, syn::parse_quote!('de));
    }
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            wc.predicates
                .push(syn::parse_quote!(#from_ty: #krate::Deserialize<'de>));
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
            use #krate::{Deserialize as _, Deserializer as _};

            impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::Deserializer<'de>>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    let (__claim, __v) = #krate::hit!(<#from_ty as #krate::Deserialize<'de>>::deserialize(d, ()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #convert)))
                }
            }
        };
    })
}

fn expand_struct(
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
                // Unit struct: deserialize from null.
                let (_, ty_generics, _) = input.generics.split_for_impl();
                let mut impl_gen = input.generics.clone();
                let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
                if !has_de {
                    impl_gen.params.insert(0, syn::parse_quote!('de));
                }
                let (impl_generics, _, where_clause) = impl_gen.split_for_impl();
                return Ok(quote! {
                    #[allow(unreachable_code)]
                    const _: () = {
                        use #krate::{DefaultValue as _, Deserializer as _, Entry as _};

                        impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                            async fn deserialize<__D: #krate::Deserializer<'de>>(
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

        // Build generics with Deserialize bound only for the transparent field.
        let mut impl_gen = input.generics.clone();
        let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
        if !has_de {
            impl_gen.params.insert(0, syn::parse_quote!('de));
        }
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates
                        .push(syn::parse_quote!(#ident: #krate::Deserialize<'de>));
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
                if let Some(ft) = &transparent_cf.from {
                    for lt in borrow_lifetimes(ft, &None) {
                        wc.predicates.push(syn::parse_quote!('de: #lt));
                    }
                } else if let Some(ft) = &transparent_cf.try_from {
                    for lt in borrow_lifetimes(ft, &None) {
                        wc.predicates.push(syn::parse_quote!('de: #lt));
                    }
                }
            }
        }
        let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

        // Generate deserialize expression for the transparent field.
        let de_expr = if let Some(path) = &transparent_cf.deserialize_with {
            quote! { #krate::hit!(#path(d).await) }
        } else if let Some(from_ty) = &transparent_cf.from {
            quote! {
                {
                    let (__claim, __tmp) = #krate::hit!(<#from_ty as #krate::Deserialize<'de>>::deserialize(d, ()).await);
                    (__claim, <#transparent_ty as ::core::convert::From<#from_ty>>::from(__tmp))
                }
            }
        } else if let Some(try_from_ty) = &transparent_cf.try_from {
            quote! {
                {
                    let (__claim, __tmp) = #krate::hit!(<#try_from_ty as #krate::Deserialize<'de>>::deserialize(d, ()).await);
                    (__claim, #krate::or_miss!(<#transparent_ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__tmp).ok()))
                }
            }
        } else {
            quote! { #krate::hit!(<#transparent_ty as #krate::Deserialize<'de>>::deserialize(d, ()).await) }
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

                impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                    async fn deserialize<__D: #krate::Deserializer<'de>>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
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
        let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
        if !has_de {
            impl_gen.params.insert(0, syn::parse_quote!('de));
        }
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates
                        .push(syn::parse_quote!(#ident: #krate::Deserialize<'de>));
                }
                for (ty, cf) in &de_field_types_and_cfs {
                    let has_custom =
                        cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                    apply_borrow_field_bound(wc, ty, &cf.bound, has_custom, &cf.borrow);
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
                    Deserialize as _, Deserializer as _, Entry as _,
                    SeqAccess as _, SeqEntry as _,
                };

                impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                    async fn deserialize<__D: #krate::Deserializer<'de>>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
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

    // Build impl generics.
    let mut impl_gen = input.generics.clone();
    let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
    if !has_de {
        impl_gen.params.insert(0, syn::parse_quote!('de));
    }
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in input.generics.type_params() {
                let ident = &tp.ident;
                wc.predicates
                    .push(syn::parse_quote!(#ident: #krate::Deserialize<'de>));
            }
            for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                let has_custom =
                    cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                apply_borrow_field_bound(wc, ty, &cf.bound, has_custom, &cf.borrow);
            }
            for (_, flat_ty, _, flat_borrow) in &flatten_fields {
                for lt in borrow_lifetimes(flat_ty, flat_borrow) {
                    wc.predicates.push(syn::parse_quote!('de: #lt));
                }
            }
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

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
                let wname = wire_names[0];
                quote! {
                    |mut __kp: #krate::borrow::KP<'de, __D>| {
                        __kp.deserialize_key::<#krate::Match, &str>(#wname)
                    }
                }
            } else {
                quote! {
                    |mut __kp: #krate::borrow::KP<'de, __D>| {
                        __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wire_names, ()), )*])
                    }
                }
            };

            let val_fn = quote! {
                |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                    let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#vt, _>(()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v #vc))))
                }
            };

            quote! { #krate::MapArmSlot::new(#key_fn, #val_fn) }
        })
        .collect();

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
        let use_boxed = flatten_fields
            .iter()
            .any(|(_, _, mode, _)| *mode == crate::common::FlattenMode::Boxed);

        if use_boxed && !cfg!(feature = "alloc") {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "`#[strede(flatten(boxed))]` requires the `alloc` feature of the `strede` crate",
            ));
        }

        // Outer arms - same as regular path but without SkipUnknownOwned (FlattenTerminal applies it).
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
                            move |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                            |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
                        )
                    }
                };
            }
            expr
        };

        // Generate N-1 intermediate continuation structs for N flatten fields.
        // cont_i handles flatten_fields[i] and chains to flatten_fields[i+1].
        for i in 0..flatten_fields.len().saturating_sub(1) {
            let cont_name = format_ident!("__FlatContB_{}", flatten_fields[i].0);
            let next_flat_ty = flatten_fields[i + 1].1;

            let result_cell_fields: Vec<TokenStream2> = flatten_fields[i + 1..]
                .iter()
                .map(|(fname, fty, _, _)| {
                    let cell_name = format_ident!("__result_{}", fname);
                    quote! { #cell_name: &'__cont ::core::cell::Cell<::core::option::Option<#fty>> }
                })
                .collect();

            let next_cont_expr = if i + 1 < flatten_fields.len() - 1 {
                let next_cont_name = format_ident!("__FlatContB_{}", flatten_fields[i + 1].0);
                let next_cell_args: Vec<TokenStream2> = flatten_fields[i + 2..]
                    .iter()
                    .map(|(fname, _, _, _)| {
                        let cell_name = format_ident!("__result_{}", fname);
                        quote! { #cell_name: self.#cell_name }
                    })
                    .collect();
                quote! { #next_cont_name { #( #next_cell_args, )* } }
            } else {
                if use_boxed {
                    quote! { #krate::FlattenTerminaloxed }
                } else {
                    quote! { #krate::FlattenTerminal }
                }
            };

            let next_extra_cell = if i + 1 < flatten_fields.len() - 1 {
                let next_cont_name = format_ident!("__FlatContB_{}", flatten_fields[i + 1].0);
                quote! {
                    let __next_extra_cell: ::core::cell::Cell<
                        ::core::option::Option<<#next_cont_name<'_> as #krate::FlattenCont<'de, __M>>::Extra>
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
                    <#next_flat_ty as #krate::Deserialize<'de>>::deserialize(
                        #krate::FlattenDeserializer::new(
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

                impl<'__cont, 'de, __M> #krate::FlattenCont<'de, __M> for #cont_name<'__cont>
                where
                    __M: #krate::MapAccess<'de>,
                    #next_flat_ty: #krate::Deserialize<'de>,
                {
                    type Extra = ();

                    async fn finish<__Arms: #krate::MapArmStack<'de, __M::KeyProbe>>(
                        self,
                        __map: __M,
                        __arms: __Arms,
                    ) -> ::core::result::Result<
                        #krate::Probe<(__M::Claim, __Arms::Outputs, ())>,
                        __M::Error,
                    > {
                        #finish_impl
                    }
                }
            });
        }

        let first_flat_name = flatten_fields[0].0;
        let first_flat_ty = flatten_fields[0].1;

        let terminal_expr = if use_boxed {
            quote! { #krate::FlattenTerminaloxed }
        } else {
            quote! { #krate::FlattenTerminal }
        };
        let (first_cont_expr, first_extra_cell_decl) = if flatten_fields.len() == 1 {
            (
                terminal_expr,
                quote! {
                    let __first_extra_cell: ::core::cell::Cell<::core::option::Option<()>>
                        = ::core::cell::Cell::new(::core::option::Option::None);
                },
            )
        } else {
            let first_cont_name = format_ident!("__FlatContB_{}", first_flat_name);
            let first_cell_args: Vec<TokenStream2> = flatten_fields[1..]
                .iter()
                .map(|(fname, _, _, _)| {
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

        // Result cell declarations for flatten fields [1..].
        let result_cell_decls: Vec<TokenStream2> = flatten_fields[1..]
            .iter()
            .map(|(fname, fty, _, _)| {
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
            .map(|(fname, _, _, _)| {
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
                    <#first_flat_ty as #krate::Deserialize<'de>>::deserialize(
                        #krate::FlattenDeserializer::new(
                            __map,
                            &__outer_arms_cell,
                            &__outer_outputs_cell,
                            #first_cont_expr,
                            &__first_extra_cell,
                        ),
                        ()
                    ).await
                );

                let #output_pat = match __outer_outputs_cell.take() {
                    ::core::option::Option::Some(__o) => __o,
                    ::core::option::Option::None => {
                        return ::core::result::Result::Ok(#krate::Probe::Miss);
                    }
                };

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
                        move |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                        |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
                    )
                }
            };
            if allow_unknown_fields {
                expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                    |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                    |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                        use #krate::MapValueProbe as _;
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
                ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name { #( #field_names, )* })))
            }).await
        }
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

            impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::Deserializer<'de>>(
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

/// Generate per-field finalization: extract from Option, applying defaults where configured.
fn gen_field_finalizers(
    field_names: &[&syn::Ident],
    acc_names: &[syn::Ident],
    classified_fields: &[crate::common::ClassifiedField],
    krate: &syn::Path,
) -> Vec<TokenStream2> {
    field_names
        .iter()
        .zip(acc_names.iter())
        .zip(classified_fields.iter())
        .map(|((fname, acc), cf)| {
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
            let none_branch = match &cf.default {
                Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                Some(DefaultAttr::Expr(expr)) => quote! { #krate::DefaultWrapper(#expr).value() },
                None => quote! { return ::core::result::Result::Ok(#krate::Probe::Miss) },
            };
            quote! {
                let #fname = match #acc {
                    ::core::option::Option::Some((_k, __v)) => __v,
                    ::core::option::Option::None => #none_branch,
                };
            }
        })
        .collect()
}

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
                impl<'de> #krate::Deserialize<'de> for #wrapper
                where #ty: 'de
                {
                    async fn deserialize<__D: #krate::Deserializer<'de>>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__claim, __v) = #krate::hit!(#path(d).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, Self(__v))))
                    }
                }
            });
        } else if let Some(from_ty) = &cf.from {
            let wrapper = format_ident!("__DeFrom_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl<'de> #krate::Deserialize<'de> for #wrapper
                where #from_ty: #krate::Deserialize<'de>
                {
                    async fn deserialize<__D: #krate::Deserializer<'de>>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__claim, __v) = #krate::hit!(<#from_ty as #krate::Deserialize<'de>>::deserialize(d, ()).await);
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
                impl<'de> #krate::Deserialize<'de> for #wrapper
                where #try_from_ty: #krate::Deserialize<'de>
                {
                    async fn deserialize<__D: #krate::Deserializer<'de>>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__claim, __v) = #krate::hit!(<#try_from_ty as #krate::Deserialize<'de>>::deserialize(d, ()).await);
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

// ---------------------------------------------------------------------------
// Borrow-family enum derive
// ---------------------------------------------------------------------------

fn expand_enum(input: DeriveInput, krate: &syn::Path) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let data = match &input.data {
        Data::Enum(d) => d,
        _ => unreachable!(),
    };

    let container_attrs = parse_container_attrs(&input.attrs)?;

    if let Some(ref from_ty) = container_attrs.from {
        return gen_container_from_borrow(&input, krate, &container_attrs, from_ty, false);
    }
    if let Some(ref try_from_ty) = container_attrs.try_from {
        return gen_container_from_borrow(&input, krate, &container_attrs, try_from_ty, true);
    }

    let classified = classify_variants(data, &container_attrs)?;

    let field_types = all_field_types(data);

    // ty_generics: original type params.
    let (_, ty_generics, _) = input.generics.split_for_impl();

    // Build impl generics: prepend 'de, add 'de: 'a bounds for field type lifetimes.
    let mut impl_gen = input.generics.clone();
    let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
    if !has_de {
        impl_gen.params.insert(0, syn::parse_quote!('de));
    }
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in input.generics.type_params() {
                let ident = &tp.ident;
                wc.predicates
                    .push(syn::parse_quote!(#ident: #krate::Deserialize<'de>));
            }
            for ty in &field_types {
                for lt in borrow_lifetimes(ty, &None) {
                    wc.predicates.push(syn::parse_quote!('de: #lt));
                }
            }
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    if let Some(ref tag_field) = container_attrs.tag {
        if let Some(ref content_field) = container_attrs.content {
            return expand_enum_adjacent_tagged_borrow(
                name,
                &classified,
                tag_field,
                content_field,
                krate,
                &container_attrs,
                &impl_generics,
                &ty_generics,
                where_clause,
            );
        }
        return expand_enum_internally_tagged(
            name,
            &classified,
            tag_field,
            krate,
            &container_attrs,
            &impl_generics,
            &ty_generics,
            where_clause,
        );
    }

    let has_tagged_unit = classified
        .iter()
        .any(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit));
    let has_tagged_nonunit = classified
        .iter()
        .any(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit));
    let has_untagged = classified.iter().any(|cv| cv.untagged);

    let body = if !has_untagged {
        if has_tagged_unit && !has_tagged_nonunit {
            expand_enum_unit_only(name, &classified, krate)?
        } else if !has_tagged_unit && has_tagged_nonunit {
            expand_enum_map_only(name, &classified, krate)?
        } else {
            expand_enum_mixed(name, &classified, krate)?
        }
    } else if !has_tagged_unit && !has_tagged_nonunit {
        expand_enum_untagged_only(name, &classified, krate)?
    } else {
        expand_enum_with_untagged(name, &classified, krate)?
    };

    let tuple_variant_helpers = gen_tuple_variant_helpers_borrow(&classified, krate);
    let struct_variant_helpers =
        gen_struct_variant_helpers_borrow(&classified, krate, container_attrs.rename_all);

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #tuple_variant_helpers
            #struct_variant_helpers

            impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::Deserializer<'de>>(
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

/// Generate the str match arms for tagged unit variants.
fn unit_str_match_arms(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> TokenStream2 {
    let arms: Vec<_> = classified
        .iter()
        .filter_map(|cv| {
            if !cv.untagged && matches!(cv.kind, VariantKind::Unit) {
                let vname = &cv.variant.ident;
                let vstr = &cv.wire_name;
                let aliases = &cv.aliases;
                Some(quote! {
                    #vstr #( | #aliases )* => ::core::result::Result::Ok(
                        #krate::Probe::Hit((__claim, #name::#vname))
                    ),
                })
            } else {
                None
            }
        })
        .collect();

    let wildcard = match other_variant(classified) {
        Some(vname) => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname))),
        },
        None => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        },
    };
    quote! {
        match __s {
            #( #arms )*
            #wildcard
        }
    }
}

#[allow(dead_code)]
fn nonunit_map_arm_slots(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> (Vec<TokenStream2>, Vec<TokenStream2>) {
    let mut arm_slots = Vec::new();
    let mut wire_names = Vec::new();
    let mut arm_idx = 0usize;

    for cv in classified.iter() {
        if cv.untagged {
            continue;
        }
        let vname = &cv.variant.ident;
        let vstr = &cv.wire_name;
        let aliases = &cv.aliases;

        match &cv.kind {
            VariantKind::Unit => {}
            VariantKind::Newtype(ty) => {
                wire_names.push(quote! { (#vstr, #arm_idx) });
                for alias in aliases {
                    wire_names.push(quote! { (#alias, #arm_idx) });
                }
                let mut wnames: Vec<&str> = vec![vstr.as_str()];
                for a in aliases {
                    wnames.push(a.as_str());
                }
                let key_fn = if wnames.len() == 1 {
                    quote! { |mut __kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Match, &str>(#vstr) }
                } else {
                    quote! { |mut __kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wnames, ()), )*]) }
                };
                arm_slots.push(quote! {
                    #krate::MapArmSlot::new(
                        #key_fn,
                        |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ty, _>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, #name::#vname(__v)))))
                        }
                    )
                });
                arm_idx += 1;
            }
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__Variant{}", cv.index);
                let field_names: Vec<_> =
                    fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                wire_names.push(quote! { (#vstr, #arm_idx) });
                for alias in aliases {
                    wire_names.push(quote! { (#alias, #arm_idx) });
                }
                let mut wnames: Vec<&str> = vec![vstr.as_str()];
                for a in aliases {
                    wnames.push(a.as_str());
                }
                let key_fn = if wnames.len() == 1 {
                    quote! { |mut __kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Match, &str>(#vstr) }
                } else {
                    quote! { |mut __kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wnames, ()), )*]) }
                };
                arm_slots.push(quote! {
                    #krate::MapArmSlot::new(
                        #key_fn,
                        |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name, _>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, #name::#vname { #( #field_names: __v.#field_names, )* }))))
                        }
                    )
                });
                arm_idx += 1;
            }
            VariantKind::Tuple(fields) => {
                let helper_name = format_ident!("__TupleVariant{}", cv.index);
                let field_indices: Vec<syn::Index> =
                    (0..fields.len()).map(syn::Index::from).collect();
                wire_names.push(quote! { (#vstr, #arm_idx) });
                for alias in aliases {
                    wire_names.push(quote! { (#alias, #arm_idx) });
                }
                let mut wnames: Vec<&str> = vec![vstr.as_str()];
                for a in aliases {
                    wnames.push(a.as_str());
                }
                let key_fn = if wnames.len() == 1 {
                    quote! { |mut __kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Match, &str>(#vstr) }
                } else {
                    quote! { |mut __kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wnames, ()), )*]) }
                };
                arm_slots.push(quote! {
                    #krate::MapArmSlot::new(
                        #key_fn,
                        |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name, _>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, #name::#vname( #( __v.#field_indices, )* )))))
                        }
                    )
                });
                arm_idx += 1;
            }
        }
    }

    (arm_slots, wire_names)
}

/// Generate helper tuple struct definitions and Deserialize impls for tuple variants (borrow family).
fn gen_tuple_variant_helpers_borrow(
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for cv in classified.iter() {
        if let VariantKind::Tuple(fields) = &cv.kind {
            let helper_name = format_ident!("__TupleVariant{}", cv.index);
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
            let field_count = field_types.len();
            let acc_names: Vec<_> = (0..field_count)
                .map(|i| format_ident!("__f{}", i))
                .collect();

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

            // Collect 'de: 'a bounds for all field types.
            let mut helper_bounds: Vec<syn::WherePredicate> = Vec::new();
            for fty in &field_types {
                for lt in borrow_lifetimes(fty, &None) {
                    helper_bounds.push(syn::parse_quote!('de: #lt));
                }
            }

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name( #( #field_types, )* );

                impl<'de> #krate::Deserialize<'de> for #helper_name
                where
                    #( #helper_bounds, )*
                {
                    async fn deserialize<__D2: #krate::Deserializer<'de>>(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.entry(|[__e]| async {
                            let mut __seq = #krate::hit!(__e.deserialize_seq().await);

                            #( #seq_reads )*

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

/// Generate helper struct definitions and Deserialize impls for struct variants (borrow family).
fn gen_struct_variant_helpers_borrow(
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    rename_all: Option<crate::common::RenameAll>,
) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for cv in classified.iter() {
        if let VariantKind::Struct(fields) = &cv.kind {
            let helper_name = format_ident!("__Variant{}", cv.index);
            let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
            let cf = match classify_fields(fields, rename_all) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let de_classified: Vec<_> = cf.iter().filter(|f| !f.skip_deserializing).collect();

            // Build arm slots for helper struct.
            let arm_slots: Vec<TokenStream2> = field_names
                .iter()
                .zip(field_types.iter())
                .zip(de_classified.iter())
                .map(|((fname, fty), dcf)| {
                    let mut wnames: Vec<&str> = vec![dcf.wire_name.as_str()];
                    for a in &dcf.aliases { wnames.push(a.as_str()); }
                    let key_fn = if wnames.len() == 1 {
                        let wn = wnames[0];
                        quote! { |mut __kp: #krate::borrow::KP<'de, __D2>| __kp.deserialize_key::<#krate::Match, &str>(#wn) }
                    } else {
                        quote! { |mut __kp: #krate::borrow::KP<'de, __D2>| __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wnames, ()), )*]) }
                    };
                    let out_name = format_ident!("__out_{}", fname);
                    let _ = out_name;
                    quote! {
                        #krate::MapArmSlot::new(
                            #key_fn,
                            |__vp: #krate::borrow::VP2<'de, __D2>, __k| async move {
                                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#fty, _>(()).await);
                                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                            }
                        )
                    }
                })
                .collect();

            let dup_wire_names2: Vec<TokenStream2> = de_classified
                .iter()
                .enumerate()
                .flat_map(|(idx, dcf)| {
                    let wn = &dcf.wire_name;
                    let mut entries: Vec<TokenStream2> = vec![quote! { (#wn, #idx) }];
                    for alias in &dcf.aliases {
                        entries.push(quote! { (#alias, #idx) });
                    }
                    entries
                })
                .collect();

            let de_out_names: Vec<syn::Ident> = field_names
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

            let field_finalizers = gen_field_finalizers(&field_names, &de_out_names, &cf, krate);

            let arms_expr = {
                let mut expr = quote! { #krate::MapArmBase };
                for slot in &arm_slots {
                    expr = quote! { (#expr, #slot) };
                }
                quote! {
                    {
                        let __wn = [#( #dup_wire_names2, )*];
                        #krate::DetectDuplicatesOwned::new(
                            #expr,
                            __wn,
                            move |__kp: #krate::borrow::KP<'de, __D2>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                            |__vp: #krate::borrow::VP2<'de, __D2>| __vp.skip(),
                        )
                    }
                }
            };

            // Collect 'de: 'a bounds for all field types.
            let mut helper_bounds: Vec<syn::WherePredicate> = Vec::new();
            for fty in &field_types {
                for lt in borrow_lifetimes(fty, &None) {
                    helper_bounds.push(syn::parse_quote!('de: #lt));
                }
            }

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name {
                    #( #field_names: #field_types, )*
                }

                impl<'de> #krate::Deserialize<'de> for #helper_name
                where
                    #( #helper_bounds, )*
                {
                    async fn deserialize<__D2: #krate::Deserializer<'de>>(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.entry(|[__e]| async {
                            let __map = #krate::hit!(__e.deserialize_map().await);
                            let __arms = #arms_expr;
                            match __map.iterate(__arms).await? {
                                #krate::Probe::Hit((__claim, #output_pat)) => {
                                    #( #field_finalizers )*
                                    ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #helper_name { #( #field_names, )* })))
                                }
                                #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                            }
                        }).await
                    }
                }
            });
        }
    }
    tokens
}

fn expand_enum_unit_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let str_match = unit_str_match_arms(name, classified, krate);
    Ok(quote! {
        d.entry(|[__e]| async {
            let (__claim, __s) = #krate::hit!(__e.deserialize_str().await);
            #str_match
        }).await
    })
}

fn expand_enum_map_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let map_body = gen_enum_map_body_borrow(name, classified, krate);
    Ok(quote! {
        d.entry(|[__e]| async {
            let __map = #krate::hit!(__e.deserialize_map().await);
            #map_body
        }).await
    })
}

fn expand_enum_mixed(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let str_match = unit_str_match_arms(name, classified, krate);
    let map_body = gen_enum_map_body_borrow(name, classified, krate);
    Ok(quote! {
        d.entry(|[__e1, __e2]| async {
            #krate::select_probe! {
                async move {
                    let (__claim, __s) = #krate::hit!(__e1.deserialize_str().await);
                    #str_match
                },
                async move {
                    let __map = #krate::hit!(__e2.deserialize_map().await);
                    #map_body
                },
            }
        }).await
    })
}

/// All untagged - try each variant by shape.
fn expand_enum_untagged_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let n_handles = classified.len();
    let handle_names: Vec<_> = (0..n_handles).map(|i| format_ident!("__e{}", i)).collect();

    let refs: Vec<_> = classified.iter().collect();
    let probe_chain = gen_untagged_probe_chain_borrow(name, &refs, &handle_names, krate);

    Ok(quote! {
        d.entry(|[#( #handle_names ),*]| async {
            #probe_chain
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await
    })
}

/// Mixed tagged + untagged - try tagged first, then untagged fallback.
fn expand_enum_with_untagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
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

    let str_section = if let Some(h) = &str_handle {
        let str_match = unit_str_match_arms(name, classified, krate);
        quote! {
            match #h.deserialize_str().await? {
                #krate::Probe::Hit((__claim, __s)) => {
                    return #str_match;
                }
                #krate::Probe::Miss => {}
            }
        }
    } else {
        quote! {}
    };

    let map_section = if let Some(h) = &map_handle {
        let map_body = gen_enum_map_body_borrow(name, classified, krate);
        quote! {
            match #h.deserialize_map().await? {
                #krate::Probe::Hit(__map) => {
                    let __map_result = { #map_body };
                    match __map_result {
                        ::core::result::Result::Ok(#krate::Probe::Hit(__v)) => {
                            return ::core::result::Result::Ok(#krate::Probe::Hit(__v));
                        }
                        ::core::result::Result::Err(__e) => {
                            return ::core::result::Result::Err(__e);
                        }
                        ::core::result::Result::Ok(#krate::Probe::Miss) => {}
                    }
                }
                #krate::Probe::Miss => {}
            }
        }
    } else {
        quote! {}
    };

    let untagged_classified: Vec<_> = classified.iter().filter(|cv| cv.untagged).collect();
    let untagged_section =
        gen_untagged_probe_chain_borrow(name, &untagged_classified, &untagged_handles, krate);

    Ok(quote! {
        d.entry(|[#( #all_handles ),*]| async {
            #str_section
            #map_section
            #untagged_section
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await
    })
}

/// Generate untagged probe chain for borrow family.
fn gen_untagged_probe_chain_borrow(
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
                let helper_name = format_ident!("__TupleVariant{}", cv.index);
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
                let helper_name = format_ident!("__Variant{}", cv.index);
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

/// Generate the body that reads a single-key map for non-unit variant dispatch (borrow family).
/// Uses `iterate` with arm slots.
/// NOTE: assumes `__map` is already in scope (bound by the caller).
fn gen_enum_map_body_borrow(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> TokenStream2 {
    let tagged_nonunit: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit))
        .collect();

    if tagged_nonunit.is_empty() {
        return quote! {
            ::core::result::Result::Ok(#krate::Probe::Miss)
        };
    }

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
                    |mut __kp: #krate::borrow::KP<'de, __D>| {
                        __kp.deserialize_key::<#krate::Match, &str>(#wn)
                    }
                }
            } else {
                quote! {
                    |mut __kp: #krate::borrow::KP<'de, __D>| {
                        __kp.deserialize_key::<#krate::MatchVals<()>, _>([#( (#wire_names, ()), )*])
                    }
                }
            };
            let val_fn = match &cv.kind {
                VariantKind::Newtype(ty) => quote! {
                    |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                        let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ty, _>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, #name::#vname(__v)))))
                    }
                },
                VariantKind::Struct(fields) => {
                    let helper_name = format_ident!("__Variant{}", cv.index);
                    let field_names: Vec<_> = fields.iter()
                        .map(|f| f.ident.as_ref().unwrap())
                        .collect();
                    quote! {
                        |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name, _>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __vc,
                                (__k, #name::#vname { #( #field_names: __v.#field_names, )* })
                            )))
                        }
                    }
                },
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
                    let field_indices: Vec<syn::Index> = (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
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
                move |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
            )
        }};
        if has_other {
            expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                    use #krate::MapValueProbe as _;
                    let __vc = __vp.skip().await?;
                    ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                },
            )) };
        }
        expr
    };

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
        {
            let __arms = #arms_expr;
            let (__claim, #output_pat) = #krate::hit!(__map.iterate(__arms).await);
            #( #result_arms )*
            #other_arm
        }
    }
}

/// Generate a `Deserialize` impl for an internally tagged enum (`#[strede(tag = "field")]`).
///
/// Supports unit variants, newtype variants, tuple variants, and struct variants.
#[allow(clippy::too_many_arguments)]
fn expand_enum_internally_tagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    impl_generics: impl quote::ToTokens,
    ty_generics: impl quote::ToTokens,
    where_clause: impl quote::ToTokens,
) -> syn::Result<TokenStream2> {
    // All (wire_name, local_idx) pairs including aliases, for every non-untagged variant.
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

    let has_nonunit = classified
        .iter()
        .any(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit));

    let (body, helpers) = if !has_nonunit {
        (
            expand_borrow_internally_tagged_unit_only(
                name,
                classified,
                tag_field,
                &variant_candidates,
                krate,
            )?,
            quote! {},
        )
    } else {
        expand_borrow_internally_tagged_with_nonunit(
            name,
            classified,
            tag_field,
            &variant_candidates,
            krate,
            container_attrs,
        )?
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #helpers

            impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::Deserializer<'de>>(
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

/// Unit-only internally-tagged enum (borrow family).
fn expand_borrow_internally_tagged_unit_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    variant_candidates: &[(String, usize)],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let str_match = unit_str_match_arms(name, classified, krate);
    let tag_cands: Vec<TokenStream2> = variant_candidates
        .iter()
        .map(|(wire_name, idx)| quote! { (#wire_name, #idx) })
        .collect();
    let n_cands = tag_cands.len();

    Ok(quote! {
        d.entry(|[__e]| async {
            let mut __map = #krate::hit!(__e.deserialize_map().await);
            let __tag_cell: ::core::cell::Cell<::core::option::Option<usize>> =
                ::core::cell::Cell::new(::core::option::Option::None);

            let __tag_candidates: [(&'static str, usize); #n_cands] = [#( #tag_cands, )*];
            let __arms = #krate::SkipUnknown!(
                #krate::TagInjectingStack!(
                    #krate::MapArmBase,
                    #tag_field,
                    __tag_candidates,
                    &__tag_cell,
                    #krate::borrow::KP<'de, __D>,
                    #krate::borrow::VP2<'de, __D>
                ),
                #krate::borrow::KP<'de, __D>,
                #krate::borrow::VP2<'de, __D>
            );
            match __map.iterate(__arms).await? {
                #krate::Probe::Hit((__claim, ())) => {
                    let __tag_idx = match __tag_cell.get() {
                        ::core::option::Option::Some(__i) => __i,
                        ::core::option::Option::None => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                    };
                    let __tag_candidates2: [(&'static str, usize); #n_cands] = [#( #tag_cands, )*];
                    let __s = match __tag_candidates2.iter().find(|(_, i)| *i == __tag_idx) {
                        ::core::option::Option::Some((s, _)) => *s,
                        ::core::option::Option::None => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                    };
                    #str_match
                }
                #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
            }
        }).await
    })
}

/// Internally-tagged enum with at least one non-unit variant (borrow family).
///
/// Each non-unit variant is raced concurrently via `select_probe!`. Each arm
/// gets a `TagAwareDeserializer` facade that injects a tag-capture arm into the
/// variant's field arm stack and validates the captured tag index matches that
/// variant before returning `Hit`. Unit variants are checked as a fallback.
fn expand_borrow_internally_tagged_with_nonunit(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
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
        gen_struct_variant_helpers_borrow(classified, krate, container_attrs.rename_all);
    let tuple_helpers = gen_tuple_variant_helpers_borrow(classified, krate);

    let tag_cands_entries: Vec<TokenStream2> = variant_candidates
        .iter()
        .map(|(wire_name, idx)| quote! { (#wire_name, #idx) })
        .collect();
    let tag_cands_count = variant_candidates.len();

    let mut fork_stmts: Vec<TokenStream2> = Vec::new();
    let mut select_arms: Vec<TokenStream2> = Vec::new();

    for (arm_i, &(local_idx, cv)) in nonunit_variants.iter().enumerate() {
        let vname = &cv.variant.ident;
        let fork_ident = format_ident!("__map_{}", arm_i);

        let (de_type, variant_construction) = match &cv.kind {
            VariantKind::Newtype(ty) => (quote! { #ty }, quote! { #name::#vname(__v) }),
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__Variant{}", cv.index);
                let field_names: Vec<_> =
                    fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                (
                    quote! { #helper_name },
                    quote! { #name::#vname { #( #field_names: __v.#field_names, )* } },
                )
            }
            VariantKind::Tuple(_) => {
                let helper_name = format_ident!("__TupleVariant{}", cv.index);
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
                let __d = #krate::tag_facade::TagAwareDeserializer::new(
                    #fork_ident,
                    #tag_field,
                    [#( #tag_cands_entries, )*],
                    #local_idx,
                    __tag_value,
                );
                match <#de_type as #krate::Deserialize<'de>>::deserialize(__d, ()).await? {
                    #krate::Probe::Hit((__c, __v)) =>
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #variant_construction))),
                    #krate::Probe::Miss =>
                        ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }
        });
    }

    let unit_match_arms: Vec<_> = unit_variants
        .iter()
        .map(|&(local_idx, cv)| {
            let vname = &cv.variant.ident;
            quote! {
                ::core::option::Option::Some(#local_idx) => {
                    return match __map.fork().iterate(
                        (#krate::MapArmBase, #krate::VirtualArmSlot::new(
                            |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                            |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                                use #krate::MapValueProbe as _;
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
                return match __map.fork().iterate(
                    (#krate::MapArmBase, #krate::VirtualArmSlot::new(
                        |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                        |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                            use #krate::MapValueProbe as _;
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

    let _ = tag_cands_count;

    let body = quote! {
        d.entry(|[__e]| async {
            let mut __map = #krate::hit!(__e.deserialize_map().await);
            let __tag_value: ::core::cell::Cell<::core::option::Option<usize>> =
                ::core::cell::Cell::new(::core::option::Option::None);
            let __tag_value = &__tag_value;

            #( #fork_stmts )*

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

/// Generate a `Deserialize` impl for an adjacently tagged enum (borrow family).
///
/// Wire format: `{"t": "VariantName", "c": <payload>}` (key order-independent).
/// Unit variants have no content field: `{"t": "VariantName"}`.
#[allow(clippy::too_many_arguments)]
fn expand_enum_adjacent_tagged_borrow(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    content_field: &str,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    impl_generics: impl quote::ToTokens,
    ty_generics: impl quote::ToTokens,
    where_clause: impl quote::ToTokens,
) -> syn::Result<TokenStream2> {
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

    let has_nonunit = classified
        .iter()
        .any(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit));

    let (body, helpers) = if !has_nonunit {
        // Unit-only: same as internally tagged (no content field needed).
        (
            expand_borrow_internally_tagged_unit_only(
                name,
                classified,
                tag_field,
                &variant_candidates,
                krate,
            )?,
            quote! {},
        )
    } else {
        expand_borrow_adjacent_tagged_with_nonunit(
            name,
            classified,
            tag_field,
            content_field,
            &variant_candidates,
            krate,
            container_attrs,
        )?
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #helpers

            impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::Deserializer<'de>>(
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

/// Adjacent-tagged enum with at least one non-unit variant (borrow family).
fn expand_borrow_adjacent_tagged_with_nonunit(
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
        gen_struct_variant_helpers_borrow(classified, krate, container_attrs.rename_all);
    let tuple_helpers = gen_tuple_variant_helpers_borrow(classified, krate);

    let tag_cands_entries: Vec<TokenStream2> = variant_candidates
        .iter()
        .map(|(wire_name, idx)| quote! { (#wire_name, #idx) })
        .collect();
    let tag_cands_count = variant_candidates.len();

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
                let helper_name = format_ident!("__Variant{}", cv.index);
                let field_names: Vec<_> =
                    fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                (
                    quote! { #helper_name },
                    quote! { #name::#vname { #( #field_names: __v.#field_names, )* } },
                )
            }
            VariantKind::Tuple(_) => {
                let helper_name = format_ident!("__TupleVariant{}", cv.index);
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
                let __arms = {
                    let __inner_arms = (
                        (#krate::MapArmBase,
                         #krate::MapArmSlot::new(
                             |mut __kp: #krate::borrow::KP<'de, __D>| {
                                 __kp.deserialize_key::<#krate::Match, &str>(#tag_field)
                             },
                             |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                                 let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                     #krate::MatchVals<usize>,
                                     [(&'static str, usize); #tag_cands_count]
                                 >([#( #tag_cands_entries, )*]).await);
                                 ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                             },
                         )),
                        #krate::MapArmSlot::new(
                            |mut __kp: #krate::borrow::KP<'de, __D>| {
                                __kp.deserialize_key::<#krate::Match, &str>(#content_field)
                            },
                            |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#de_type, ()>(()).await);
                                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                            },
                        )
                    );
                    let __wn = #dup_wire_names;
                    let __dd = #krate::DetectDuplicatesOwned::new(
                        __inner_arms,
                        __wn,
                        move |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<usize>, _>(__wn),
                        |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
                    );
                    (__dd, #krate::VirtualArmSlot::new(
                        |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                        |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                            use #krate::MapValueProbe as _;
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

    // Unit variant fallback: iterate the outer map looking for the tag field only.
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
            let __unit_arms = (
                (#krate::MapArmBase,
                 #krate::MapArmSlot::new(
                     |mut __kp: #krate::borrow::KP<'de, __D>| {
                         __kp.deserialize_key::<#krate::Match, &str>(#tag_field)
                     },
                     |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                         let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                             #krate::MatchVals<usize>,
                             [(&'static str, usize); #tag_cands_count]
                         >([#( #tag_cands_entries, )*]).await);
                         ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                     },
                 )),
                #krate::VirtualArmSlot::new(
                    |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip, _>(()),
                    |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                        use #krate::MapValueProbe as _;
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

            #( #fork_stmts )*

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

            #unit_fallback
        }).await
    };

    let helpers = quote! {
        #struct_helpers
        #tuple_helpers
    };

    Ok((body, helpers))
}
