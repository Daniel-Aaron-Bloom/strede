use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput};

use super::gen_container_from_owned;
use crate::common::{
    ClassifiedVariant, FieldContext, VariantKind, all_field_types, classify_fields,
    classify_variants, field_bound_owned, has_universal_blanket, insert_d_owned, other_variant,
    parse_container_attrs, type_param_bound_owned,
};

/// Insert `__E: EnumAccessOwned` into `impl_gen`.
/// Used for `DeserializeFromEnumOwned` impl emission.
fn insert_e_owned(impl_gen: &mut syn::Generics, krate: &syn::Path) {
    impl_gen
        .params
        .push(syn::parse_quote!(__E: #krate::EnumAccessOwned));
}

pub(super) fn expand_owned(input: DeriveInput, krate: &syn::Path) -> syn::Result<TokenStream2> {
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
    let d_ident = format_ident!("__D");

    let (_, ty_generics, _) = input.generics.split_for_impl();

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
            for ty in &field_types {
                if !has_universal_blanket(ty) {
                    wc.predicates.push(field_bound_owned(
                        krate,
                        ty,
                        FieldContext::MapValue,
                        &d_ident,
                    ));
                }
                // Untagged variants dispatch via `__e.deserialize_value::<T>(())` on Entry.
                if classified.iter().any(|cv| cv.untagged) && !has_universal_blanket(ty) {
                    wc.predicates.push(syn::parse_quote!(
                        #ty: #krate::DeserializeOwned<
                            <__D::Entry as #krate::EntryOwned>::SubDeserializer,
                            Extra = ()
                        >
                    ));
                }
            }
            // Untagged unit variants dispatch via `deserialize_value::<()>`.
            let has_untagged_unit = classified
                .iter()
                .any(|cv| cv.untagged && matches!(cv.kind, VariantKind::Unit));
            if has_untagged_unit {
                wc.predicates.push(syn::parse_quote!(
                    (): #krate::DeserializeOwned<
                        <__D::Entry as #krate::EntryOwned>::SubDeserializer,
                        Extra = ()
                    >
                ));
            }
            // Map iteration uses Match/Skip key probes.
            // For enums: map iteration only happens for non-unit non-untagged variants;
            // unit-only enums dispatch via string matching, not maps.
            let dup_n: usize = classified
                .iter()
                .filter(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit))
                .map(|cv| 1 + cv.aliases.len())
                .sum();
            let _ = dup_n; // universal Match/MatchVals/Skip impls cover the key bounds

            // Adjacent-tagged non-unit variants dispatch via
            // `__vp.deserialize_value::<HelperT>(())` on the content slot — see
            // the borrow-family note for why the helper-as-type bound is needed.
            let is_adjacent = container_attrs.tag.is_some() && container_attrs.content.is_some();
            let is_internal = container_attrs.tag.is_some() && container_attrs.content.is_none();
            if is_adjacent {
                for cv in &classified {
                    if cv.untagged {
                        continue;
                    }
                    let helper_ty: syn::Type = match &cv.kind {
                        VariantKind::Struct(_) => {
                            let id = format_ident!("__VariantOwned{}", cv.index);
                            syn::parse_quote!(#id)
                        }
                        VariantKind::Tuple(_) => {
                            let id = format_ident!("__TupleVariantOwned{}", cv.index);
                            syn::parse_quote!(#id)
                        }
                        VariantKind::Newtype(_) | VariantKind::Unit => continue,
                    };
                    wc.predicates.push(field_bound_owned(
                        krate,
                        &helper_ty,
                        FieldContext::MapValue,
                        &d_ident,
                    ));
                }
            }
            // Internally-tagged newtype: see borrow-family comment.
            if is_internal {
                let n_cands: usize = classified
                    .iter()
                    .filter(|cv| !cv.untagged)
                    .map(|cv| 1 + cv.aliases.len())
                    .sum();
                for cv in &classified {
                    if cv.untagged {
                        continue;
                    }
                    if let VariantKind::Newtype(ty) = &cv.kind {
                        wc.predicates.push(syn::parse_quote!(
                            for<'__v> #ty: #krate::DeserializeFromMapOwned<
                                #krate::TagAwareMapOwned<
                                    '__v,
                                    <__D::Entry as #krate::EntryOwned>::Map,
                                    [(&'static str, usize); #n_cands],
                                >,
                                Extra = (),
                            >
                        ));
                    }
                }
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

    if !has_untagged {
        // Externally-tagged (no untagged variants): use the new
        // DeserializeFromEnumOwned + DeserializeOwned two-impl approach.
        return expand_owned_enum_external_tagged(
            name,
            &classified,
            krate,
            &container_attrs,
            &input.generics,
        );
    }

    let body = if !has_tagged_unit && !has_tagged_nonunit {
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

            impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize_owned(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
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
            quote! { #krate::MatchVals<usize, _> },
            array_expr,
            quote! { __k.0 },
        ),
        Some(s) => {
            let s_lit = proc_macro2::Literal::usize_suffixed(s);
            (
                quote! { #krate::UnwrapOrElse<#krate::MatchVals<usize, _>, _> },
                quote! { (async || #krate::MatchVals(#s_lit, ::core::marker::PhantomData), #array_expr) },
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
                            __se.get::<#ty>(()).await
                        }).await);
                        let (__seq_back, #acc) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                    }
                })
                .collect();

            let helper_d_ident = format_ident!("__D2");
            let helper_bounds: Vec<syn::WherePredicate> = field_types
                .iter()
                .map(|fty| field_bound_owned(krate, fty, FieldContext::SeqElem, &helper_d_ident))
                .collect();

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name( #( #field_types, )* );

                impl<__D2: #krate::DeserializerOwned> #krate::DeserializeOwned<__D2> for #helper_name
                where
                    #( #helper_bounds, )*
                {
                    type Extra = ();
                    async fn deserialize_owned(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
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

/// Generate `DeserializeFromEnumOwned<__E>` + `DeserializeOwned<__D>` impls for
/// externally-tagged enums (no `#[strede(tag)]` / `#[strede(untagged)]`).
///
/// The `DeserializeFromEnumOwned` impl drives variant dispatch via `EnumAccessOwned::iterate`
/// with an arm stack. The `DeserializeOwned` impl delegates via `deserialize_enum_into`.
fn expand_owned_enum_external_tagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> syn::Result<TokenStream2> {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    // --- Build impl generics for DeserializeFromEnumOwned<__E> ---
    let mut enum_impl_gen = orig_generics.clone();
    insert_e_owned(&mut enum_impl_gen, krate);

    // Collect payload types for non-unit non-other tagged variants.
    let payload_types: Vec<syn::Type> = classified
        .iter()
        .filter(|cv| !cv.untagged && !cv.other && !matches!(cv.kind, VariantKind::Unit))
        .map(|cv| match &cv.kind {
            VariantKind::Newtype(ty) => syn::parse_quote!(#ty),
            VariantKind::Struct(_) => {
                let id = format_ident!("__VariantOwned{}", cv.index);
                syn::parse_quote!(#id)
            }
            VariantKind::Tuple(_) => {
                let id = format_ident!("__TupleVariantOwned{}", cv.index);
                syn::parse_quote!(#id)
            }
            VariantKind::Unit => unreachable!(),
        })
        .collect();

    {
        let wc = enum_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_owned(krate, ident));
            }
            // Payload types must be DeserializeOwned via the PayloadDeserializer.
            for pty in &payload_types {
                wc.predicates.push(syn::parse_quote!(
                    #pty: #krate::DeserializeOwned<
                        <__E::VariantProbe as #krate::EnumVariantProbeOwned>::PayloadDeserializer,
                        Extra = ()
                    >
                ));
            }
        }
    }
    let (enum_impl_generics, _, enum_where_clause) = enum_impl_gen.split_for_impl();

    // --- Build impl generics for DeserializeOwned<__D> ---
    let mut de_impl_gen = orig_generics.clone();
    insert_d_owned(&mut de_impl_gen, krate);
    {
        let wc = de_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_owned(krate, ident));
            }
            // Require that Self: DeserializeFromEnumOwned for the Entry's Enum type.
            wc.predicates.push(syn::parse_quote!(
                #name #ty_generics: #krate::DeserializeFromEnumOwned<
                    <__D::Entry as #krate::EntryOwned>::Enum,
                    Extra = ()
                >
            ));
        }
    }
    let (de_impl_generics, _, de_where_clause) = de_impl_gen.split_for_impl();

    // --- Build arm slots ---
    // Non-untagged variants (excluding `other`) get one arm each.
    let tagged_non_other: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && !cv.other)
        .collect();

    let arm_slots: Vec<TokenStream2> = tagged_non_other
        .iter()
        .enumerate()
        .map(|(arm_local_idx, cv)| {
            let mut candidates: Vec<(&str, usize)> = vec![(cv.wire_name.as_str(), arm_local_idx)];
            for alias in &cv.aliases {
                candidates.push((alias.as_str(), arm_local_idx));
            }
            let cands_tokens: Vec<TokenStream2> = candidates
                .iter()
                .map(|(wn, idx)| quote! { (#wn, #idx) })
                .collect();

        let cv_idx = cv.index;
            match &cv.kind {
                VariantKind::Unit => {
                    quote! {
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccessOwned>::VariantProbe| async move {
                            let __vp2 = __vp.fork();
                            let (__claim, _) = #krate::hit!(#krate::select_probe! {
                                __vp.deserialize_unit_by_name([#( #cands_tokens, )*]),
                                __vp2.deserialize_unit_by_index(#cv_idx),
                            });
                            ::core::result::Result::Ok(#krate::Probe::Hit((__claim, ())))
                        })
                    }
                }
                VariantKind::Newtype(ty) => {
                    quote! {
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccessOwned>::VariantProbe| async move {
                            let __vp2 = __vp.fork();
                            let (__claim, _, __v) = #krate::hit!(#krate::select_probe! {
                                __vp.deserialize_payload_by_name::<#ty, _>([#( #cands_tokens, )*], ()),
                                __vp2.deserialize_payload_by_index::<#ty>(#cv_idx, ()),
                            });
                            ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)))
                        })
                    }
                }
                VariantKind::Struct(_) => {
                    let helper_name = format_ident!("__VariantOwned{}", cv.index);
                    quote! {
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccessOwned>::VariantProbe| async move {
                            let __vp2 = __vp.fork();
                            let (__claim, _, __v) = #krate::hit!(#krate::select_probe! {
                                __vp.deserialize_payload_by_name::<#helper_name, _>([#( #cands_tokens, )*], ()),
                                __vp2.deserialize_payload_by_index::<#helper_name>(#cv_idx, ()),
                            });
                            ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)))
                        })
                    }
                }
                VariantKind::Tuple(_) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    quote! {
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccessOwned>::VariantProbe| async move {
                            let __vp2 = __vp.fork();
                            let (__claim, _, __v) = #krate::hit!(#krate::select_probe! {
                                __vp.deserialize_payload_by_name::<#helper_name, _>([#( #cands_tokens, )*], ()),
                                __vp2.deserialize_payload_by_index::<#helper_name>(#cv_idx, ()),
                            });
                            ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)))
                        })
                    }
                }
            }
        })
        .collect();

    // Build the left-nested arm stack expression.
    let arms_expr = {
        let mut expr = quote! { #krate::EnumArmBase };
        for slot in &arm_slots {
            expr = quote! { (#expr, #slot) };
        }
        expr
    };

    // Build output pattern: left-nested (((), out0), out1), ...
    let out_names: Vec<syn::Ident> = tagged_non_other
        .iter()
        .enumerate()
        .map(|(i, _)| format_ident!("__out_ev{}", i))
        .collect();
    let output_pat = {
        let mut pat = quote! { () };
        for out in &out_names {
            pat = quote! { (#pat, #out) };
        }
        pat
    };

    // Build result extraction: check each output option and construct variant.
    let other_arm = match other_variant(classified) {
        Some(vname) => {
            quote! { ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname))) }
        }
        None => quote! { ::core::result::Result::Ok(#krate::Probe::Miss) },
    };

    let result_arms: Vec<TokenStream2> = tagged_non_other
        .iter()
        .enumerate()
        .map(|(i, cv)| {
            let out = &out_names[i];
            let vname = &cv.variant.ident;
            match &cv.kind {
                VariantKind::Unit => quote! {
                    if let ::core::option::Option::Some(()) = #out {
                        return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname)));
                    }
                },
                VariantKind::Newtype(_) => quote! {
                    if let ::core::option::Option::Some(__v) = #out {
                        return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname(__v))));
                    }
                },
                VariantKind::Struct(fields) => {
                    let field_names: Vec<_> =
                        fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                    quote! {
                        if let ::core::option::Option::Some(__v) = #out {
                            return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname { #( #field_names: __v.#field_names, )* })));
                        }
                    }
                }
                VariantKind::Tuple(fields) => {
                    let field_indices: Vec<syn::Index> =
                        (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        if let ::core::option::Option::Some(__v) = #out {
                            return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname( #( __v.#field_indices, )* ))));
                        }
                    }
                }
            }
        })
        .collect();

    // Helper types for tuple and struct variants.
    let tuple_variant_helpers = gen_tuple_variant_helpers_owned(classified, krate);
    let struct_variant_helpers =
        gen_struct_variant_helpers_owned(classified, krate, container_attrs.rename_all);

    let deserialize_from_enum_body = quote! {
        let __arms = #arms_expr;
        match __e.iterate(__arms).await? {
            #krate::Probe::Hit((__claim, #output_pat)) => {
                #( #result_arms )*
                #other_arm
            }
            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
        }
    };

    // Build the DeserializeOwned body. When an `other` variant is present, use two
    // entry handles: the first calls `deserialize_enum_into`, and the second is a
    // fallback that `skip_other()`s the value (which iterate returned Miss without
    // consuming) and returns the `other` variant.
    let deserialize_owned_body = match other_variant(classified) {
        Some(other_vname) => quote! {
            d.entry(|[__e1, __e2]| async {
                match __e1.deserialize_enum_into::<Self>(()).await? {
                    #krate::Probe::Hit(__v) => ::core::result::Result::Ok(#krate::Probe::Hit(__v)),
                    #krate::Probe::Miss => {
                        // No arm matched — consume the value and return the `other` variant.
                        let __claim = __e2.skip_other().await?;
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#other_vname)))
                    }
                }
            }).await
        },
        None => quote! {
            d.entry(|[__e]| async {
                __e.deserialize_enum_into::<Self>(()).await
            }).await
        },
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializeFromEnumOwned as _,
                DeserializerOwned as _, EntryOwned as _, EnumAccessOwned as _,
                EnumVariantProbeOwned as _, MapAccessOwned as _, MapKeyProbeOwned as _,
                MapValueProbeOwned as _, SeqAccessOwned as _, SeqEntryOwned as _,
                StrAccessOwned as _,
            };

            #tuple_variant_helpers
            #struct_variant_helpers

            impl #enum_impl_generics #krate::DeserializeFromEnumOwned<__E>
                for #name #ty_generics
                #enum_where_clause
            {
                type Extra = ();
                async fn deserialize_from_enum_owned(
                    __e: __E,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__E::Claim, Self)>, __E::Error>
                {
                    #deserialize_from_enum_body
                }
            }

            impl #de_impl_generics #krate::DeserializeOwned<__D>
                for #name #ty_generics
                #de_where_clause
            {
                type Extra = ();
                async fn deserialize_owned(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    #deserialize_owned_body
                }
            }
        };
    })
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
                    |mut __kp: #krate::owned::KP<__D>, _i: usize| {
                        __kp.deserialize_key::<#krate::Match>(#wn)
                    }
                }
            } else {
                quote! {
                    |mut __kp: #krate::owned::KP<__D>, _i: usize| {
                        __kp.deserialize_key::<#krate::MatchVals<(), _>>([#( (#wire_names, ()), )*])
                    }
                }
            };
            let val_fn = match &cv.kind {
                VariantKind::Newtype(ty) => quote! {
                    |__vp: #krate::owned::VP2<__D>, __k| async move {
                        let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ty>(()).await);
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
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name>(()).await);
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
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name>(()).await);
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

    // dup wire names for DetectDuplicates.
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
            #krate::DetectDuplicates::new(
                #expr,
                __wn,
                move |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                |__vp: #krate::owned::VP2<__D>| __vp.skip(),
            )
        }};
        if has_other {
            expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
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

            // One arm slot per field. KP/VP projected from __M2 directly.
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
                            |mut __kp: <__M2 as #krate::MapAccessOwned>::KeyProbe, __i: usize| async move {
                                let __kp2 = __kp.fork();
                                #krate::select_probe! {
                                    __kp.deserialize_key::<#krate::Match>(#name),
                                    async move {
                                        let (__kc, ()) = #krate::hit!(__kp2.deserialize_key_by_index(__i).await);
                                        ::core::result::Result::Ok(#krate::Probe::Hit((__kc, #krate::Match)))
                                    },
                                }
                            }
                        }
                    } else {
                        quote! {
                            |mut __kp: <__M2 as #krate::MapAccessOwned>::KeyProbe, __i: usize| async move {
                                let __kp2 = __kp.fork();
                                #krate::select_probe! {
                                    __kp.deserialize_key::<#krate::MatchVals<(), _>>([#( (#wire_names, ()), )*]),
                                    async move {
                                        let (__kc, ()) = #krate::hit!(__kp2.deserialize_key_by_index(__i).await);
                                        ::core::result::Result::Ok(#krate::Probe::Hit((__kc, #krate::MatchVals((), ::core::marker::PhantomData))))
                                    },
                                }
                            }
                        }
                    };
                    let val_fn = quote! {
                        |__vp: #krate::owned::VP<<__M2 as #krate::MapAccessOwned>::KeyProbe>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ft>(()).await);
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

            // DetectDuplicates wire names array.
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
                        #krate::DetectDuplicates::new(
                            #expr,
                            __wn,
                            move |__kp: <__M2 as #krate::MapAccessOwned>::KeyProbe, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                            |__vp: #krate::owned::VP<<__M2 as #krate::MapAccessOwned>::KeyProbe>| __vp.skip(),
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

                // Shape-specific impl — used by tagged-enum dispatch via
                // `TagAwareMapOwned` + `deserialize_from_map`.
                impl<__M2: #krate::MapAccessOwned> #krate::DeserializeFromMapOwned<__M2> for #helper_name
                where
                    #(
                        #field_types: #krate::DeserializeOwned<
                            <#krate::owned::VP<<__M2 as #krate::MapAccessOwned>::KeyProbe> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                            Extra = ()
                        >,
                    )*
                {
                    type Extra = ();
                    async fn deserialize_from_map_owned(
                        __map: __M2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(<__M2 as #krate::MapAccessOwned>::MapClaim, Self)>, <__M2 as #krate::MapAccessOwned>::Error>
                    {
                        let __arms = #arms_expr;
                        let (__claim, #output_pat) = #krate::hit!(__map.iterate(__arms).await);
                        #( #field_finalizers )*
                        ::core::result::Result::Ok(
                            #krate::Probe::Hit((__claim, #helper_name { #( #field_names, )* }))
                        )
                    }
                }

                // Universal entry point — used when this helper is the value of
                // a map key (adjacent-tagged content field, externally-tagged
                // single-key value).
                impl<__D2: #krate::DeserializerOwned> #krate::DeserializeOwned<__D2> for #helper_name
                where
                    #helper_name: #krate::DeserializeFromMapOwned<<__D2::Entry as #krate::EntryOwned>::Map, Extra = ()>,
                {
                    type Extra = ();
                    async fn deserialize_owned(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    {
                        d.entry(|[__e]| async {
                            __e.deserialize_map_into::<Self>(()).await
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
            match #h.deserialize_value::<#unit_key_type>(#unit_key_extra).await? {
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
                    match #handle.deserialize_value::<()>(()).await? {
                        #krate::Probe::Hit((__c, _)) => {
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
                    match #handle.deserialize_value::<#ty>(()).await? {
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
                    match #handle.deserialize_value::<#helper_name>(()).await? {
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
                    match #handle.deserialize_value::<#helper_name>(()).await? {
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

            impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize_owned(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
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
            |mut __kp: #krate::owned::KP<__D>, _i: usize| {
                __kp.deserialize_key::<#krate::Match>(#tag_field)
            },
            |__vp: #krate::owned::VP2<__D>, __k| async move {
                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#krate::MatchVals<usize, _>>(
                    [#( #val_extra_entries, )*]
                ).await);
                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
            },
        )
    };

    // dup wire names for DetectDuplicates (only tag_field is a real arm).
    let arms_expr = quote! {{
        let __wn = [(#tag_field, 0usize)];
        let __inner = #krate::DetectDuplicates::new(
            (#krate::MapArmBase, #arm_slot),
            __wn,
            move |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
            |__vp: #krate::owned::VP2<__D>| __vp.skip(),
        );
        (__inner, #krate::VirtualArmSlot::new(
            |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
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
                    #krate::MatchVals(#idx, _) => ::core::result::Result::Ok(
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

    let mut all_entry_idents: Vec<syn::Ident> = Vec::new();
    let mut select_arms: Vec<TokenStream2> = Vec::new();

    for (arm_i, &(local_idx, cv)) in nonunit_variants.iter().enumerate() {
        let vname = &cv.variant.ident;
        let entry_ident = format_ident!("__e_{}", arm_i);
        all_entry_idents.push(entry_ident.clone());

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

        select_arms.push(quote! {
            async move {
                let __map = #krate::hit!(#entry_ident.deserialize_map().await);
                let __tag_cell: ::core::cell::Cell<::core::option::Option<usize>> =
                    ::core::cell::Cell::new(::core::option::Option::None);
                let __m = #krate::TagAwareMapOwned::new(
                    __map,
                    #tag_field,
                    [#( #tag_cands_entries, )*],
                    #local_idx,
                    &__tag_cell,
                );
                match <#de_type as #krate::DeserializeFromMapOwned<_>>::deserialize_from_map_owned(__m, ()).await? {
                    #krate::Probe::Hit((__c, __v)) =>
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #variant_construction))),
                    #krate::Probe::Miss =>
                        ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }
        });
    }

    // Unit arm: handles all unit variants and the `other` fallback.
    let needs_unit_arm = !unit_variants.is_empty() || other_variant(classified).is_some();
    if needs_unit_arm {
        let unit_entry_ident = format_ident!("__e_unit");
        all_entry_idents.push(unit_entry_ident.clone());

        // Reuse the unit-only logic to build the arm body.
        let unit_only_body = expand_owned_internally_tagged_unit_only(
            name,
            classified,
            tag_field,
            variant_candidates,
            _variant_key_sentinel,
            krate,
        )?;

        // The unit_only_body is a full `d.entry(|[__e]| async { ... }).await` expression.
        // We need just the inner map iteration logic. Instead, inline the map-iteration
        // directly so it can be used inside a select_probe! arm.
        let val_extra_entries: Vec<TokenStream2> = variant_candidates
            .iter()
            .map(|(wire_name, idx)| quote! { (#wire_name, #idx) })
            .collect();
        let val_extra_count = val_extra_entries.len();

        let arm_slot = quote! {
            #krate::MapArmSlot::new(
                |mut __kp: #krate::owned::KP<__D>, _i: usize| {
                    __kp.deserialize_key::<#krate::Match>(#tag_field)
                },
                |__vp: #krate::owned::VP2<__D>, __k| async move {
                    let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#krate::MatchVals<usize, _>>(
                        [#( #val_extra_entries, )*]
                    ).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                },
            )
        };

        let arms_expr = quote! {{
            let __wn = [(#tag_field, 0usize)];
            let __inner = #krate::DetectDuplicates::new(
                (#krate::MapArmBase, #arm_slot),
                __wn,
                move |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                |__vp: #krate::owned::VP2<__D>| __vp.skip(),
            );
            (__inner, #krate::VirtualArmSlot::new(
                |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
                |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                    let __vc = __vp.skip().await?;
                    ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                },
            ))
        }};

        // Build match arms for unit variants only.
        let unit_match_inner: Vec<_> = unit_variants
            .iter()
            .filter_map(|&(local_idx, cv)| {
                if matches!(cv.kind, VariantKind::Unit) {
                    let vname = &cv.variant.ident;
                    Some(quote! {
                        #krate::MatchVals(#local_idx, _) => ::core::result::Result::Ok(
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
        let _ = unit_only_body;

        select_arms.push(quote! {
            async move {
                let __map = #krate::hit!(#unit_entry_ident.deserialize_map().await);
                let __arms = #arms_expr;
                let (__claim, ((), __out_0)) = #krate::hit!(__map.iterate(__arms).await);
                match __out_0 {
                    ::core::option::Option::Some((_k, __matched)) => match __matched {
                        #( #unit_match_inner )*
                        #unit_wildcard
                    },
                    ::core::option::Option::None => {
                        ::core::result::Result::Ok(#krate::Probe::Miss)
                    }
                }
            }
        });
    }

    let _ = nonunit_count;
    let _ = tag_candidates_count;

    let body = quote! {
        d.entry(|[#( #all_entry_idents, )*]| async {
            #krate::select_probe! {
                #( #select_arms, )*
                @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
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

            impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize_owned(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
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

    let mut all_entry_idents: Vec<syn::Ident> = Vec::new();
    let mut select_arms: Vec<TokenStream2> = Vec::new();

    for (arm_i, &(local_idx, cv)) in nonunit_variants.iter().enumerate() {
        let vname = &cv.variant.ident;
        let entry_ident = format_ident!("__e_{}", arm_i);
        all_entry_idents.push(entry_ident.clone());

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

        select_arms.push(quote! {
            async move {
                let mut __map = #krate::hit!(#entry_ident.deserialize_map().await);
                // Two-slot arm stack: tag slot + content slot, with dup detection + skip unknown.
                let __arms = {
                    let __inner_arms = (
                        (#krate::MapArmBase,
                         #krate::MapArmSlot::new(
                             |mut __kp: #krate::owned::KP<__D>, _i: usize| {
                                 __kp.deserialize_key::<#krate::Match>(#tag_field)
                             },
                             |__vp: #krate::owned::VP2<__D>, __k| async move {
                                 let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                     #krate::MatchVals<usize, [(&'static str, usize); #tag_cands_count]>
                                 >([#( #tag_cands_entries, )*]).await);
                                 ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                             },
                         )),
                        #krate::MapArmSlot::new(
                            |mut __kp: #krate::owned::KP<__D>, _i: usize| {
                                __kp.deserialize_key::<#krate::Match>(#content_field)
                            },
                            |__vp: #krate::owned::VP2<__D>, __k| async move {
                                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#de_type>(()).await);
                                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                            },
                        )
                    );
                    let __wn = #dup_wire_names;
                    let __dd = #krate::DetectDuplicates::new(
                        __inner_arms,
                        __wn,
                        move |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                        |__vp: #krate::owned::VP2<__D>| __vp.skip(),
                    );
                    (__dd, #krate::VirtualArmSlot::new(
                        |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
                        |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                            let __vc = __vp.skip().await?;
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                        },
                    ))
                };
                let (__claim, (((), __opt_tag), __opt_content)) =
                    #krate::hit!(__map.iterate(__arms).await);
                match (__opt_tag, __opt_content) {
                    (
                        ::core::option::Option::Some((_, #krate::MatchVals(#local_idx, _))),
                        ::core::option::Option::Some((_, __v)),
                    ) => ::core::result::Result::Ok(
                        #krate::Probe::Hit((__claim, #variant_construction))
                    ),
                    _ => ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }
        });
    }

    // Unit variant arms (as a select arm that handles all unit/other fallback).
    let unit_match_arms: Vec<_> = unit_variants
        .iter()
        .map(|&(local_idx, cv)| {
            let vname = &cv.variant.ident;
            quote! {
                ::core::option::Option::Some((_, #krate::MatchVals(#local_idx, _))) => {
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

    let needs_unit_arm = !unit_match_arms.is_empty() || other_variant(classified).is_some();
    if needs_unit_arm {
        let unit_entry_ident = format_ident!("__e_unit");
        all_entry_idents.push(unit_entry_ident.clone());

        select_arms.push(quote! {
            async move {
                let mut __map = #krate::hit!(#unit_entry_ident.deserialize_map().await);
                // Unit variant arm: iterate the outer map looking for the tag field.
                let __unit_arms = (
                    (#krate::MapArmBase,
                     #krate::MapArmSlot::new(
                         |mut __kp: #krate::owned::KP<__D>, _i: usize| {
                             __kp.deserialize_key::<#krate::Match>(#tag_field)
                         },
                         |__vp: #krate::owned::VP2<__D>, __k| async move {
                             let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                 #krate::MatchVals<usize, [(&'static str, usize); #tag_cands_count]>
                             >([#( #tag_cands_entries, )*]).await);
                             ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                         },
                     )),
                    #krate::VirtualArmSlot::new(
                        |__kp: #krate::owned::KP<__D>, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
                        |__vp: #krate::owned::VP2<__D>, _k: #krate::Skip| async move {
                            let __vc = __vp.skip().await?;
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, ())))
                        },
                    ),
                );
                let (__unit_claim, ((), __opt_unit_tag)) =
                    #krate::hit!(__map.iterate(__unit_arms).await);
                match __opt_unit_tag {
                    #( #unit_match_arms )*
                    #other_arm
                }
            }
        });
    }

    let body = quote! {
        d.entry(|[#( #all_entry_idents, )*]| async {
            #krate::select_probe! {
                #( #select_arms, )*
                @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
            }
        }).await
    };

    let helpers = quote! {
        #struct_helpers
        #tuple_helpers
    };

    Ok((body, helpers))
}
