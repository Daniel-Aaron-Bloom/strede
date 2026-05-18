use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

use super::gen_container_from_owned;
use crate::common::{
    DefaultAttr, FieldContext, apply_field_bound, classify_fields, field_bound_owned,
    insert_d_owned, mentions_type_param, type_param_bound_owned,
};

pub(super) fn expand_owned(
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
                let mut impl_gen = input.generics.clone();
                insert_d_owned(&mut impl_gen, krate);
                impl_gen
                    .make_where_clause()
                    .predicates
                    .push(syn::parse_quote!(
                        (): #krate::DeserializeOwned<
                            <__D::Entry as #krate::EntryOwned>::SubDeserializer,
                            Extra = ()
                        >
                    ));
                let (impl_generics, _, where_clause) = impl_gen.split_for_impl();
                return Ok(quote! {
                    #[allow(unreachable_code)]
                    const _: () = {
                        use #krate::{DefaultValue as _, DeserializerOwned as _, EntryOwned as _};

                        impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                            type Extra = ();
                            async fn deserialize_owned(
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

        // Build generics with DeserializeOwned bound only for the transparent field.
        let mut impl_gen = input.generics.clone();
        insert_d_owned(&mut impl_gen, krate);
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(type_param_bound_owned(krate, ident));
                }
                let has_custom = transparent_cf.deserialize_owned_with.is_some()
                    || transparent_cf.from.is_some()
                    || transparent_cf.try_from.is_some();
                apply_field_bound(
                    wc,
                    transparent_ty,
                    &transparent_cf.bound,
                    has_custom,
                    |ty| field_bound_owned(krate, ty, FieldContext::Direct),
                );
                if let Some(ft) = &transparent_cf.from {
                    wc.predicates
                        .push(field_bound_owned(krate, ft, FieldContext::Direct));
                } else if let Some(ft) = &transparent_cf.try_from {
                    wc.predicates
                        .push(field_bound_owned(krate, ft, FieldContext::Direct));
                }
            }
        }
        let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

        // Generate deserialize expression for the transparent field.
        let de_expr = if let Some(path) = &transparent_cf.deserialize_owned_with {
            quote! { #krate::hit!(#path(d, ()).await) }
        } else if let Some(from_ty) = &transparent_cf.from {
            quote! {
                {
                    let (__c, __tmp) = #krate::hit!(<#from_ty as #krate::DeserializeOwned<__D>>::deserialize_owned(d, ()).await);
                    (__c, <#transparent_ty as ::core::convert::From<#from_ty>>::from(__tmp))
                }
            }
        } else if let Some(try_from_ty) = &transparent_cf.try_from {
            quote! {
                {
                    let (__c, __tmp) = #krate::hit!(<#try_from_ty as #krate::DeserializeOwned<__D>>::deserialize_owned(d, ()).await);
                    (__c, #krate::or_miss!(<#transparent_ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__tmp).ok()))
                }
            }
        } else {
            quote! { #krate::hit!(<#transparent_ty as #krate::DeserializeOwned<__D>>::deserialize_owned(d, ()).await) }
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

                impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                    type Extra = ();
                    async fn deserialize_owned(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
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
        insert_d_owned(&mut impl_gen, krate);
        {
            let wc = impl_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(type_param_bound_owned(krate, ident));
                }
                for (ty, cf) in &de_field_types_and_cfs {
                    let has_custom = cf.deserialize_owned_with.is_some()
                        || cf.from.is_some()
                        || cf.try_from.is_some();
                    apply_field_bound(wc, ty, &cf.bound, has_custom, |t| {
                        field_bound_owned(krate, t, FieldContext::SeqElem)
                    });
                    if let Some(ft) = &cf.from {
                        wc.predicates
                            .push(field_bound_owned(krate, ft, FieldContext::SeqElem));
                    } else if let Some(ft) = &cf.try_from {
                        wc.predicates
                            .push(field_bound_owned(krate, ft, FieldContext::SeqElem));
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
                    DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                    SeqAccessOwned as _, SeqEntryOwned as _,
                };

                impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                    type Extra = ();
                    async fn deserialize_owned(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
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

    // D6 — detect generic-T flatten fields on stable.
    let type_param_idents: Vec<syn::Ident> = input
        .generics
        .type_params()
        .map(|tp| tp.ident.clone())
        .collect();
    for (fname, flat_ty, _) in &flatten_fields {
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
            let msg = "`#[strede(flatten)]` on a field whose type mentions a struct type \
                       parameter is not supported on stable Rust. Add `#[strede(bound = \
                       \"...\")]` to the field, or enable the `nightly-flatten` feature.";
            return Ok(quote! {
                ::core::compile_error!(#msg);
            });
        }
    }

    // Build impl generics.
    let mut impl_gen = input.generics.clone();
    insert_d_owned(&mut impl_gen, krate);
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in input.generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_owned(krate, ident));
            }
            for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                let has_custom = cf.deserialize_owned_with.is_some()
                    || cf.from.is_some()
                    || cf.try_from.is_some();
                apply_field_bound(wc, ty, &cf.bound, has_custom, |t| {
                    field_bound_owned(krate, t, FieldContext::MapValue)
                });
            }
            // D6 option α: no bound emitted for flatten field types; rustc resolves at use site.
            // Map iteration uses Match/Skip key probes — add their bounds.
            let dup_n: usize = de_classified.iter().map(|cf| 1 + cf.aliases.len()).sum();
            let _ = dup_n; // universal Match/MatchVals/Skip impls cover the key bounds
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    // Output binding names - one per non-skipped, non-flatten field.
    let de_out_names: Vec<syn::Ident> = de_field_names
        .iter()
        .map(|n| format_ident!("__out_{}", n))
        .collect();

    // Build one MapArmSlot per non-skipped, non-flatten field, parameterized
    // by the KP/VP type tokens (different for Deserialize __D vs DeserializeFromMap __M).
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
                    let name = wire_names[0];
                    quote! {
                        |mut __kp: #kp_ty| {
                            __kp.deserialize_key::<#krate::Match>(#name)
                        }
                    }
                } else {
                    quote! {
                        |mut __kp: #kp_ty| {
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
        quote! { #krate::owned::KP<__D> },
        quote! { #krate::owned::VP2<__D> },
    );
    let arm_slots_m: Vec<TokenStream2> = build_arm_slots(
        quote! { <__M as #krate::MapAccessOwned>::KeyProbe },
        quote! { #krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe> },
    );

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
                            move |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                            |__vp: #krate::owned::VP2<__D>| __vp.skip(),
                        )
                    }
                };
            }
            // No skip-unknown: unclaimed keys fall through to the inner type's
            // arms via `StackConcat` inside `FlattenMapAccessOwned`.
            expr
        };

        let n_flat = flatten_fields.len();
        let first_flat_name = flatten_fields[0].0;
        let first_flat_ty = flatten_fields[0].1;

        let any_boxed = flatten_fields
            .iter()
            .any(|(_, _, mode)| *mode == crate::common::FlattenMode::Boxed);
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

        // Cont-impl generics: cont_struct_gens + __M.
        let mut cont_impl_gens = cont_struct_gens.clone();
        cont_impl_gens
            .params
            .push(syn::parse_quote!(__M: #krate::MapAccessOwned));
        {
            let wc = cont_impl_gens.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(syn::parse_quote!(
                        #ident: #krate::DeserializeOwned<
                            <#krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>
                             as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                            Extra = ()
                        >
                    ));
                }
                for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                    let has_custom = cf.deserialize_owned_with.is_some()
                        || cf.from.is_some()
                        || cf.try_from.is_some();
                    if cf.bound.is_none() && !has_custom {
                        let bound_ty = cf.from.as_ref().or(cf.try_from.as_ref()).unwrap_or(*ty);
                        wc.predicates.push(syn::parse_quote!(
                            #bound_ty: #krate::DeserializeOwned<
                                <#krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>
                                 as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                }
            }
        }
        let (cont_impl_impl_gens, _, cont_impl_where) = cont_impl_gens.split_for_impl();

        // Emit (N - 1) cont structs.
        for i in 0..n_flat.saturating_sub(1) {
            let cont_name = format_ident!("__FlatContO_after_{}", flatten_fields[i].0);
            let next_field_name = flatten_fields[i + 1].0;
            let next_field_ty = flatten_fields[i + 1].1;
            let result_stash_ident = format_ident!("__result_{}", next_field_name);

            let cell_fields: Vec<TokenStream2> = flatten_fields[i + 1..]
                .iter()
                .map(|(fname, fty, _)| {
                    let cell_ident = format_ident!("__result_{}", fname);
                    quote! {
                        #cell_ident: &'__cont ::core::cell::Cell<::core::option::Option<#fty>>
                    }
                })
                .collect();

            let next_cont_expr = if i + 1 < n_flat - 1 {
                let next_cont_name = format_ident!("__FlatContO_after_{}", flatten_fields[i + 1].0);
                let next_args: Vec<TokenStream2> = flatten_fields[i + 2..]
                    .iter()
                    .map(|(fname, _, _)| {
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

                impl #cont_impl_impl_gens #krate::FlattenContOwned<__M>
                    for #cont_name #cont_struct_ty_gens
                #cont_impl_where
                {
                    async fn finish<__Arms: #krate::MapArmStackOwned<<__M as #krate::MapAccessOwned>::KeyProbe>>(
                        self,
                        __map: __M,
                        __arms: __Arms,
                    ) -> ::core::result::Result<
                        #krate::Probe<(<__M as #krate::MapAccessOwned>::MapClaim, __Arms::Outputs)>,
                        <__M as #krate::MapAccessOwned>::Error,
                    > {
                        let __next_outputs_cell: ::core::cell::Cell<
                            ::core::option::Option<__Arms::Outputs>,
                        > = ::core::cell::Cell::new(::core::option::Option::None);
                        let __next_fma = #krate::FlattenMapAccessOwned::new(
                            __map,
                            __arms,
                            &__next_outputs_cell,
                            #next_cont_expr,
                        );
                        let (__claim, __next_val) = #krate::hit!(
                            <#next_field_ty as #krate::DeserializeFromMapOwned<_>>
                                ::deserialize_from_map_owned(__next_fma, ()).await
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

        let extra_result_cell_decls: Vec<TokenStream2> = flatten_fields[1..]
            .iter()
            .map(|(fname, fty, _)| {
                let cell_ident = format_ident!("__result_{}", fname);
                quote! {
                    let #cell_ident: ::core::cell::Cell<::core::option::Option<#fty>>
                        = ::core::cell::Cell::new(::core::option::Option::None);
                }
            })
            .collect();

        let first_cont_expr = if n_flat == 1 {
            terminal_expr.clone()
        } else {
            let first_cont_name = format_ident!("__FlatContO_after_{}", first_flat_name);
            let first_cont_args: Vec<TokenStream2> = flatten_fields[1..]
                .iter()
                .map(|(fname, _, _)| {
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

        let extra_result_recovers: Vec<TokenStream2> = flatten_fields[1..]
            .iter()
            .map(|(fname, _, _)| {
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
                let __flatten_map = #krate::FlattenMapAccessOwned::new(
                    __map,
                    __outer_arms,
                    &__outer_outputs_cell,
                    #first_cont_expr,
                );
                let (__claim, #first_flat_name) = #krate::hit!(
                    <#first_flat_ty as #krate::DeserializeFromMapOwned<_>>::deserialize_from_map_owned(
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
                        move |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                        |__vp: #krate::owned::VP2<__D>| __vp.skip(),
                    )
                }
            };
            if allow_unknown_fields {
                expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                    |__kp: #krate::owned::KP<__D>| __kp.deserialize_key::<#krate::Skip>(()),
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

    // -----------------------------------------------------------------
    // DeserializeFromMapOwned impl emission (regular non-flatten path only)
    // -----------------------------------------------------------------
    let dfm_impl = if flatten_fields.is_empty() {
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
                        move |__kp: <__M as #krate::MapAccessOwned>::KeyProbe| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                        |__vp: #krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>| __vp.skip(),
                    )
                }
            };
            if allow_unknown_fields {
                expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                    |__kp: <__M as #krate::MapAccessOwned>::KeyProbe| __kp.deserialize_key::<#krate::Skip>(()),
                    |__vp: #krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>, _k: #krate::Skip| async move {
                        use #krate::MapValueProbeOwned as _;
                        let __vc = __vp.skip().await?;
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                    },
                )) };
            }
            expr
        };

        let mut impl_gen_m = input.generics.clone();
        crate::common::insert_m_owned(&mut impl_gen_m, krate);
        {
            let wc = impl_gen_m.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(syn::parse_quote!(
                        #ident: #krate::DeserializeOwned<
                            <#krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>
                             as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                            Extra = ()
                        >
                    ));
                }
                for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                    let has_custom = cf.deserialize_owned_with.is_some()
                        || cf.from.is_some()
                        || cf.try_from.is_some();
                    if cf.bound.is_none() && !has_custom {
                        let bound_ty = cf.from.as_ref().or(cf.try_from.as_ref()).unwrap_or(*ty);
                        wc.predicates.push(syn::parse_quote!(
                            #bound_ty: #krate::DeserializeOwned<
                                <#krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>
                                 as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                }
            }
        }
        let (impl_generics_m, _, where_clause_m) = impl_gen_m.split_for_impl();

        quote! {
            impl #impl_generics_m #krate::DeserializeFromMapOwned<__M> for #name #ty_generics #where_clause_m {
                type Extra = ();
                async fn deserialize_from_map_owned(
                    __map: __M,
                    _extra: (),
                ) -> ::core::result::Result<
                    #krate::Probe<(<__M as #krate::MapAccessOwned>::MapClaim, Self)>,
                    <__M as #krate::MapAccessOwned>::Error,
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
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapKeyProbeOwned as _, MapAccessOwned as _, SeqAccessOwned as _,
                SeqEntryOwned as _, StrAccessOwned as _, MapValueProbeOwned as _,
            };

            #de_with_wrappers
            #flatten_cont_structs

            impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize_owned(
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
                impl<__D: #krate::DeserializerOwned> #krate::DeserializeOwned<__D> for #wrapper {
                    type Extra = ();
                    async fn deserialize_owned(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        let (__c, __v) = #krate::hit!(#path(d, ()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, Self(__v))))
                    }
                }
            });
        } else if let Some(from_ty) = &cf.from {
            let wrapper = format_ident!("__DeOwnedFrom_{}", name);
            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #wrapper(#ty);
                impl<__D: #krate::DeserializerOwned> #krate::DeserializeOwned<__D> for #wrapper
                where #from_ty: #krate::DeserializeOwned<__D, Extra = ()>
                {
                    type Extra = ();
                    async fn deserialize_owned(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        let (__c, __v) = #krate::hit!(<#from_ty as #krate::DeserializeOwned<__D>>::deserialize_owned(d, ()).await);
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
                impl<__D: #krate::DeserializerOwned> #krate::DeserializeOwned<__D> for #wrapper
                where #try_from_ty: #krate::DeserializeOwned<__D, Extra = ()>
                {
                    type Extra = ();
                    async fn deserialize_owned(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    {
                        let (__c, __v) = #krate::hit!(<#try_from_ty as #krate::DeserializeOwned<__D>>::deserialize_owned(d, ()).await);
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
