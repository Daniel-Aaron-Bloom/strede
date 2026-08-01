use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput};

use super::gen_container_from_owned;
use crate::common::{
    ClassifiedVariant, DefaultAttr, FieldContext, VariantKind, all_field_types, classify_fields,
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
            // Untagged struct/tuple variants dispatch via
            // `deserialize_value::<HelperT>(())` - the helper type itself is the
            // payload, not its individual field types. See the borrow-family
            // comment for the full rationale (same previously-uncovered gap).
            for cv in &classified {
                if !cv.untagged {
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
                wc.predicates.push(syn::parse_quote!(
                    #helper_ty: #krate::DeserializeOwned<
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
                &input.generics,
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
            &input.generics,
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

    let is_pure_untagged = !has_tagged_unit && !has_tagged_nonunit;
    let body = if is_pure_untagged {
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
    // `MapFieldProviderOwned` impl so a purely-untagged enum can be used as a
    // `#[strede(flatten)]` field's type - see
    // `gen_enum_candidate_map_field_provider_untagged_owned`.
    let flatten_provider = if is_pure_untagged {
        gen_enum_candidate_map_field_provider_untagged_owned(
            name,
            &classified,
            krate,
            &container_attrs,
            &input.generics,
        )
    } else {
        TokenStream2::new()
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapAccessOwned as _, MapKeyProbeOwned as _, MapValueProbeOwned as _,
                SeqAccessOwned as _, SeqEntryOwned as _, StrAccessOwned as _,
            };

            #tuple_variant_helpers
            #struct_variant_helpers
            #flatten_provider

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

    // `MapFieldProviderOwned` impl so this enum can be used as a
    // `#[strede(flatten)]` field's type — see `gen_enum_map_field_provider_owned`.
    let map_field_provider_impl =
        gen_enum_map_field_provider_owned(name, classified, krate, container_attrs, orig_generics);

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
            #map_field_provider_impl

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

/// Emit `MapFieldProviderOwned<__KP2>` for an externally-tagged enum so it can
/// be used as a `#[strede(flatten)]` field's type — owned-family mirror of
/// `gen_enum_map_field_provider_borrow` (see its doc comment for the design).
fn gen_enum_map_field_provider_owned(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> TokenStream2 {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    // Same candidate set as expand_owned_enum_external_tagged's arm_slots.
    let tagged_non_other: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && !cv.other)
        .collect();
    let n_cands: usize = tagged_non_other.iter().map(|cv| 1 + cv.aliases.len()).sum();

    // --- impl generics: __KP2: MapKeyProbeOwned ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbeOwned));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::DeserializeOwned<
                        <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            let has_unit = tagged_non_other
                .iter()
                .any(|cv| matches!(cv.kind, VariantKind::Unit));
            if has_unit {
                // Unit variant's wire payload is `()` ("VariantName": null).
                wc.predicates.push(syn::parse_quote!(
                    (): #krate::DeserializeOwned<
                        <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            for cv in &tagged_non_other {
                let pty: Option<syn::Type> = match &cv.kind {
                    VariantKind::Newtype(ty) => Some(syn::parse_quote!(#ty)),
                    VariantKind::Struct(_) => {
                        let id = format_ident!("__VariantOwned{}", cv.index);
                        Some(syn::parse_quote!(#id))
                    }
                    VariantKind::Tuple(_) => {
                        let id = format_ident!("__TupleVariantOwned{}", cv.index);
                        Some(syn::parse_quote!(#id))
                    }
                    VariantKind::Unit => None,
                };
                if let Some(pty) = pty {
                    wc.predicates.push(syn::parse_quote!(
                        #pty: #krate::DeserializeOwned<
                            <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                            Extra = ()
                        >
                    ));
                }
            }
        }
    }
    let (mfp_impl_generics, _, mfp_where_clause) = mfp_impl_gen.split_for_impl();

    // Candidates for the MatchVals key race: (wire_name, local_variant_idx).
    let match_cands_tokens: Vec<TokenStream2> = tagged_non_other
        .iter()
        .enumerate()
        .flat_map(|(i, cv)| {
            let mut v = vec![{
                let wn = &cv.wire_name;
                quote! { (#wn, #i) }
            }];
            for alias in &cv.aliases {
                v.push(quote! { (#alias, #i) });
            }
            v
        })
        .collect();

    // wire_names() candidates: every wire name/alias, all at (relative) arm index 0 —
    // this flatten field only ever contributes one arm, regardless of variant count.
    let wire_names_tokens: Vec<TokenStream2> = tagged_non_other
        .iter()
        .flat_map(|cv| {
            let mut v = vec![{
                let wn = &cv.wire_name;
                quote! { (#wn, 0usize) }
            }];
            for alias in &cv.aliases {
                v.push(quote! { (#alias, 0usize) });
            }
            v
        })
        .collect();

    let val_arms: Vec<TokenStream2> = tagged_non_other
        .iter()
        .enumerate()
        .map(|(i, cv)| {
            let vname = &cv.variant.ident;
            match &cv.kind {
                VariantKind::Unit => quote! {
                    #i => {
                        let (__vc, ()) = #krate::hit!(__vp.deserialize_value::<()>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, #name::#vname))))
                    }
                },
                VariantKind::Newtype(ty) => quote! {
                    #i => {
                        let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ty>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, #name::#vname(__v)))))
                    }
                },
                VariantKind::Struct(fields) => {
                    let helper_name = format_ident!("__VariantOwned{}", cv.index);
                    let field_names: Vec<_> =
                        fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                    quote! {
                        #i => {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __vc,
                                (__k, #name::#vname { #( #field_names: __v.#field_names, )* }),
                            )))
                        }
                    }
                }
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    let field_indices: Vec<syn::Index> =
                        (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        #i => {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __vc,
                                (__k, #name::#vname( #( __v.#field_indices, )* )),
                            )))
                        }
                    }
                }
            }
        })
        .collect();

    quote! {
        impl #mfp_impl_generics #krate::MapFieldProviderOwned<__KP2> for #name #ty_generics
            #mfp_where_clause
        {
            // `(MapArmBase, MapArmSlot)`'s real Outputs shape is
            // `(MapArmBase::Outputs, Option<(K, V)>)` = `((), Option<(K, V)>)` —
            // one left-nesting level per arm, base contributes `()`.
            type Outputs = (
                (),
                ::core::option::Option<(
                    #krate::MatchVals<usize, [(&'static str, usize); #n_cands]>,
                    #name #ty_generics,
                )>,
            );
            const ARMS: usize = 1usize;
            type WireNames = [(&'static str, usize); #n_cands];

            fn wire_names() -> Self::WireNames {
                [#( #wire_names_tokens, )*]
            }

            fn make_arms() -> impl #krate::MapArmStackOwned<__KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                (
                    #krate::MapArmBase,
                    #krate::MapArmSlot::new(
                        |mut __kp: __KP2, _i: usize| __kp.deserialize_key::<
                            #krate::MatchVals<usize, [(&'static str, usize); #n_cands]>
                        >([#( #match_cands_tokens, )*]),
                        |__vp: #krate::owned::VP<__KP2>,
                         __k: #krate::MatchVals<usize, [(&'static str, usize); #n_cands]>| async move {
                            match __k.0 {
                                #( #val_arms )*
                                _ => ::core::unreachable!(),
                            }
                        },
                    ),
                )
            }

            fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                let ((), __opt) = __outputs;
                __opt.map(|(_k, __v)| __v)
            }
        }
    }
}

/// Owned-family counterpart to `borrow::gen_enum_candidate_map_field_provider_borrow`
/// — see there for the full rationale (internally-tagged enums race every
/// non-`other` variant's own fields concurrently via `CandidateArmStack` until
/// the tag key resolves, since more than one variant's fields could plausibly
/// share the parent struct's map before the tag arrives).
///
/// Tuple variants and enums mixing tagged with `#[strede(untagged)]` variants
/// are handled the same way as the borrow family: an unconditionally
/// unsatisfiable where-clause bound rather than a hard `syn::Error`, since
/// this impl is emitted for every internally-tagged enum regardless of
/// whether it's ever used as a flatten target.
fn gen_enum_candidate_map_field_provider_owned(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> TokenStream2 {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    let has_untagged_mix = classified.iter().any(|cv| cv.untagged);

    let candidates: Vec<(usize, &ClassifiedVariant)> = classified
        .iter()
        .filter(|cv| !cv.other)
        .enumerate()
        .collect();

    let tag_cands_tokens: Vec<TokenStream2> = candidates
        .iter()
        .flat_map(|(i, cv)| {
            let mut v = vec![{
                let wn = &cv.wire_name;
                quote! { (#wn, #i) }
            }];
            for alias in &cv.aliases {
                v.push(quote! { (#alias, #i) });
            }
            v
        })
        .collect();

    // --- impl generics: __KP2: MapKeyProbeOwned ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbeOwned));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::DeserializeOwned<
                        <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            if has_untagged_mix {
                // Never satisfied - see the function doc comment above. Must
                // bind to `__KP2` (still abstract here), not a concrete type
                // like `()` - see the borrow-family comment for why.
                wc.predicates
                    .push(syn::parse_quote!(__KP2: #krate::FlattenUnsupported));
            }
            for (_, cv) in &candidates {
                match &cv.kind {
                    VariantKind::Struct(_) => {
                        let helper_name = format_ident!("__VariantOwned{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProviderOwned<__KP2>
                        ));
                    }
                    VariantKind::Newtype(ty) => {
                        wc.predicates.push(syn::parse_quote!(
                            #ty: #krate::MapFieldProviderOwned<__KP2>
                        ));
                    }
                    VariantKind::Tuple(_) => {
                        // No map-shaped tuple helper exists - never satisfiable,
                        // same rationale as `FlattenUnsupported` above.
                        let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProviderOwned<__KP2>
                        ));
                    }
                    VariantKind::Unit => {}
                }
            }
        }
    }
    let (mfp_impl_generics, _, mfp_where_clause) = mfp_impl_gen.split_for_impl();

    let arms_const_tokens: TokenStream2 = {
        let terms: Vec<TokenStream2> = candidates
            .iter()
            .map(|(_, cv)| match &cv.kind {
                VariantKind::Unit => quote! { 0usize },
                VariantKind::Newtype(ty) => {
                    quote! { <#ty as #krate::MapFieldProviderOwned<__KP2>>::ARMS }
                }
                VariantKind::Struct(_) => {
                    let helper_name = format_ident!("__VariantOwned{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::ARMS }
                }
                VariantKind::Tuple(_) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::ARMS }
                }
            })
            .collect();
        quote! { 1usize #( + #terms )* }
    };

    let candidate_pieces: Vec<TokenStream2> = candidates
        .iter()
        .map(|(i, cv)| {
            let vname = &cv.variant.ident;
            match &cv.kind {
                VariantKind::Unit => quote! {
                    #i => #krate::MapArmBase => |()| ::core::option::Option::Some(#name::#vname)
                },
                VariantKind::Newtype(ty) => quote! {
                    #i => <#ty as #krate::MapFieldProviderOwned<__KP2>>::make_arms()
                        => |__o| <#ty as #krate::MapFieldProviderOwned<__KP2>>::from_outputs(__o).map(#name::#vname)
                },
                VariantKind::Struct(fields) => {
                    let helper_name = format_ident!("__VariantOwned{}", cv.index);
                    let field_names: Vec<_> =
                        fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::from_outputs(__o)
                                .map(|__v| #name::#vname { #( #field_names: __v.#field_names, )* })
                    }
                }
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    let field_indices: Vec<syn::Index> =
                        (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::from_outputs(__o)
                                .map(|__v| #name::#vname( #( __v.#field_indices, )* ))
                    }
                }
            }
        })
        .collect();

    quote! {
        impl #mfp_impl_generics #krate::MapFieldProviderOwned<__KP2> for #name #ty_generics
            #mfp_where_clause
        {
            type Outputs = ::core::option::Option<#name #ty_generics>;
            const ARMS: usize = #arms_const_tokens;
            type WireNames = [(&'static str, usize); 1];

            fn wire_names() -> Self::WireNames {
                [(#tag_field, 0usize)]
            }

            fn make_arms() -> impl #krate::MapArmStackOwned<__KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                #krate::CandidateArmStackOwned!(
                    #krate::candidate_arms! { #( #candidate_pieces, )* },
                    #tag_field,
                    [#( #tag_cands_tokens, )*],
                    __KP2,
                    #krate::owned::VP<__KP2>
                )
            }

            fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                __outputs
            }
        }
    }
}

/// Owned-family counterpart to
/// `borrow::gen_enum_candidate_map_field_provider_untagged_borrow` - see
/// there for the full rationale. Emits `MapFieldProviderOwned<__KP2>` for a
/// purely untagged enum (`#[strede(untagged)]`, no `tag`) via
/// `NoTagCandidateArmStack`, the tag-less soft-elimination counterpart to
/// `CandidateArmStack`.
///
/// Only struct-shaped and map-shaped-newtype candidates are viable: unit
/// variants are rejected here too (unlike internally-tagged), and tuple
/// variants are rejected via the same unconditionally-unsatisfiable
/// `MapFieldProviderOwned` bound the tag-based provider already uses.
fn gen_enum_candidate_map_field_provider_untagged_owned(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> TokenStream2 {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    let has_unit = classified
        .iter()
        .any(|cv| matches!(cv.kind, VariantKind::Unit));

    let candidates: Vec<(usize, &ClassifiedVariant)> = classified
        .iter()
        .filter(|cv| !matches!(cv.kind, VariantKind::Unit))
        .enumerate()
        .collect();

    // --- impl generics: __KP2: MapKeyProbeOwned ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbeOwned));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::DeserializeOwned<
                        <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            if has_unit {
                // Never satisfied - see the function doc comment above. Must
                // bind to `__KP2` (still abstract here), not a concrete type.
                wc.predicates
                    .push(syn::parse_quote!(__KP2: #krate::FlattenUnsupported));
            }
            for (_, cv) in &candidates {
                match &cv.kind {
                    VariantKind::Struct(_) => {
                        let helper_name = format_ident!("__VariantOwned{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProviderOwned<__KP2>
                        ));
                    }
                    VariantKind::Newtype(ty) => {
                        wc.predicates.push(syn::parse_quote!(
                            #ty: #krate::MapFieldProviderOwned<__KP2>
                        ));
                    }
                    VariantKind::Tuple(_) => {
                        // No map-shaped tuple helper exists - never
                        // satisfiable, same trick as the internally-tagged
                        // provider's own tuple rejection.
                        let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProviderOwned<__KP2>
                        ));
                    }
                    VariantKind::Unit => unreachable!("Unit excluded from `candidates` above"),
                }
            }
        }
    }
    let (mfp_impl_generics, _, mfp_where_clause) = mfp_impl_gen.split_for_impl();

    let arms_const_tokens: TokenStream2 = {
        let terms: Vec<TokenStream2> = candidates
            .iter()
            .map(|(_, cv)| match &cv.kind {
                VariantKind::Newtype(ty) => {
                    quote! { <#ty as #krate::MapFieldProviderOwned<__KP2>>::ARMS }
                }
                VariantKind::Struct(_) => {
                    let helper_name = format_ident!("__VariantOwned{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::ARMS }
                }
                VariantKind::Tuple(_) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::ARMS }
                }
                VariantKind::Unit => unreachable!("Unit excluded from `candidates` above"),
            })
            .collect();
        quote! { 0usize #( + #terms )* }
    };

    let candidate_pieces: Vec<TokenStream2> = candidates
        .iter()
        .map(|(i, cv)| {
            let vname = &cv.variant.ident;
            match &cv.kind {
                VariantKind::Newtype(ty) => quote! {
                    #i => <#ty as #krate::MapFieldProviderOwned<__KP2>>::make_arms()
                        => |__o| <#ty as #krate::MapFieldProviderOwned<__KP2>>::from_outputs(__o).map(#name::#vname)
                },
                VariantKind::Struct(fields) => {
                    let helper_name = format_ident!("__VariantOwned{}", cv.index);
                    let field_names: Vec<_> =
                        fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::from_outputs(__o)
                                .map(|__v| #name::#vname { #( #field_names: __v.#field_names, )* })
                    }
                }
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    let field_indices: Vec<syn::Index> =
                        (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProviderOwned<__KP2>>::from_outputs(__o)
                                .map(|__v| #name::#vname( #( __v.#field_indices, )* ))
                    }
                }
                VariantKind::Unit => unreachable!("Unit excluded from `candidates` above"),
            }
        })
        .collect();

    // `candidate_arms!` requires at least one entry - an enum whose every
    // variant is Unit (already permanently blocked via `has_unit` above)
    // falls back to a plain `CandidateBase` so the body still type-checks.
    let candidates_expr = if candidate_pieces.is_empty() {
        quote! { #krate::CandidateBase }
    } else {
        quote! { #krate::candidate_arms! { #( #candidate_pieces, )* } }
    };

    quote! {
        impl #mfp_impl_generics #krate::MapFieldProviderOwned<__KP2> for #name #ty_generics
            #mfp_where_clause
        {
            type Outputs = ::core::option::Option<#name #ty_generics>;
            const ARMS: usize = #arms_const_tokens;
            type WireNames = [(&'static str, usize); 0];

            fn wire_names() -> Self::WireNames {
                []
            }

            fn make_arms() -> impl #krate::MapArmStackOwned<__KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                #krate::NoTagCandidateArmStack::new(#candidates_expr)
            }

            fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                __outputs
            }
        }
    }
}

/// Generates the `MapFieldProviderOwned<__KP2>` impl for an adjacently-tagged
/// enum (`#[strede(tag = "t", content = "c")]`) used as a
/// `#[strede(flatten)]` target.
///
/// Owned-family counterpart to
/// `gen_enum_candidate_map_field_provider_adjacent_borrow` (see its doc
/// comment for the full design rationale: exactly 2 fixed arms regardless of
/// variant count, content arm always races every non-unit candidate's
/// `deserialize_value`, tag/content cross-check deferred to `from_outputs`).
///
/// Unlike the borrow-family version, the content race here MUST use
/// concurrent `select_probe!(biased; ...)` racing rather than a sequential
/// try-in-order chain: this is a streaming source, and awaiting one forked
/// candidate to completion before touching the next can deadlock as soon as
/// any candidate needs a buffer refill to determine its own result (same
/// hazard documented on `gen_untagged_probe_arms_owned` above).
#[allow(clippy::too_many_arguments)]
fn gen_enum_candidate_map_field_provider_adjacent_owned(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    content_field: &str,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> TokenStream2 {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    let has_untagged_mix = classified.iter().any(|cv| cv.untagged);

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
    let tag_cands_entries: Vec<TokenStream2> = variant_candidates
        .iter()
        .map(|(wn, idx)| quote! { (#wn, #idx) })
        .collect();
    let tag_cands_count = variant_candidates.len();

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

    // --- impl generics: __KP2: MapKeyProbeOwned ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbeOwned));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::DeserializeOwned<
                        <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            if has_untagged_mix {
                wc.predicates
                    .push(syn::parse_quote!(__KP2: #krate::FlattenUnsupported));
            }
            for &(_, cv) in &nonunit_variants {
                match &cv.kind {
                    VariantKind::Struct(_) => {
                        let helper_name = format_ident!("__VariantOwned{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::DeserializeOwned<
                                <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                    VariantKind::Tuple(_) => {
                        let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::DeserializeOwned<
                                <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                    VariantKind::Newtype(ty) => {
                        wc.predicates.push(syn::parse_quote!(
                            #ty: #krate::DeserializeOwned<
                                <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                    VariantKind::Unit => {}
                }
            }
        }
    }
    let (mfp_impl_generics, _, mfp_where_clause) = mfp_impl_gen.split_for_impl();

    // --- content arm: concurrent select_probe! race over non-unit candidates ---
    let n_nonunit = nonunit_variants.len();
    let vp_idents: Vec<syn::Ident> = (0..n_nonunit)
        .map(|i| {
            if i + 1 == n_nonunit {
                format_ident!("__vp")
            } else {
                format_ident!("__vp_{}", i)
            }
        })
        .collect();
    let fork_decls: Vec<TokenStream2> = (0..n_nonunit.saturating_sub(1))
        .map(|i| {
            let ident = &vp_idents[i];
            quote! { let mut #ident = __vp.fork(); }
        })
        .collect();
    let content_race_arms: Vec<TokenStream2> = nonunit_variants
        .iter()
        .enumerate()
        .map(|(i, &(local_idx, cv))| {
            let vname = &cv.variant.ident;
            let vp_ident = &vp_idents[i];
            let (de_type, construction) = match &cv.kind {
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
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                    let field_indices: Vec<syn::Index> =
                        (0..fields.len()).map(syn::Index::from).collect();
                    (
                        quote! { #helper_name },
                        quote! { #name::#vname( #( __v.#field_indices, )* ) },
                    )
                }
                VariantKind::Unit => unreachable!(),
            };
            quote! {
                async move {
                    match #vp_ident.deserialize_value::<#de_type>(()).await? {
                        #krate::Probe::Hit((__vc, __v)) => {
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (#krate::Match, (#local_idx, #construction)))))
                        }
                        #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                    }
                }
            }
        })
        .collect();

    let unit_match_arms: Vec<TokenStream2> = unit_variants
        .iter()
        .map(|&(local_idx, cv)| {
            let vname = &cv.variant.ident;
            quote! { #local_idx => ::core::option::Option::Some(#name::#vname), }
        })
        .collect();

    // `__vp` only needs `mut` when at least one `.fork()` call is emitted
    // (two or more non-unit candidates); an all-unit enum never touches
    // `__vp` at all, so it needs an explicit `let _ =` to avoid an
    // unused-variable warning instead.
    let vp_mut = if n_nonunit >= 2 {
        quote! { mut }
    } else {
        quote! {}
    };
    let unused_vp_guard = if n_nonunit == 0 {
        quote! { let _ = &__vp; }
    } else {
        quote! {}
    };
    let content_body = if n_nonunit == 0 {
        quote! { ::core::result::Result::Ok(#krate::Probe::Miss) }
    } else {
        quote! {
            #krate::select_probe!(biased;
                #( #content_race_arms, )*
                @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
            )
        }
    };

    quote! {
        impl #mfp_impl_generics #krate::MapFieldProviderOwned<__KP2> for #name #ty_generics
            #mfp_where_clause
        {
            type Outputs = (
                (
                    (),
                    ::core::option::Option<(
                        #krate::Match,
                        #krate::MatchVals<usize, [(&'static str, usize); #tag_cands_count]>,
                    )>,
                ),
                ::core::option::Option<(#krate::Match, (usize, #name #ty_generics))>,
            );
            const ARMS: usize = 2;
            type WireNames = [(&'static str, usize); 2];

            fn wire_names() -> Self::WireNames {
                [(#tag_field, 0usize), (#content_field, 1usize)]
            }

            fn make_arms() -> impl #krate::MapArmStackOwned<__KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                (
                    (#krate::MapArmBase,
                     #krate::MapArmSlot::new(
                         |mut __kp: __KP2, _i: usize| __kp.deserialize_key::<#krate::Match>(#tag_field),
                         |__vp: #krate::owned::VP<__KP2>, __k| async move {
                             let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                 #krate::MatchVals<usize, [(&'static str, usize); #tag_cands_count]>
                             >([#( #tag_cands_entries, )*]).await);
                             ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                         },
                     )),
                    #krate::MapArmSlot::new(
                        |mut __kp: __KP2, _i: usize| __kp.deserialize_key::<#krate::Match>(#content_field),
                        move |#vp_mut __vp: #krate::owned::VP<__KP2>, __k| async move {
                            let _ = &__k;
                            #unused_vp_guard
                            #( #fork_decls )*
                            #content_body
                        },
                    ),
                )
            }

            fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                let ((_, __opt_tag), __opt_content) = __outputs;
                match (__opt_tag, __opt_content) {
                    (
                        ::core::option::Option::Some((_, #krate::MatchVals(__tag_idx, _))),
                        ::core::option::Option::None,
                    ) => match __tag_idx {
                        #( #unit_match_arms )*
                        _ => ::core::option::Option::None,
                    },
                    (
                        ::core::option::Option::Some((_, #krate::MatchVals(__tag_idx, _))),
                        ::core::option::Option::Some((_, (__content_idx, __v))),
                    ) => {
                        if __tag_idx == __content_idx {
                            ::core::option::Option::Some(__v)
                        } else {
                            ::core::option::Option::None
                        }
                    }
                    _ => ::core::option::Option::None,
                }
            }
        }
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
/// Field-kind classification shared by the segment-building helpers below —
/// mirrors `struct_::expand_owned`'s `FieldKind`/`Segment` split so a variant's
/// `#[strede(flatten)]` fields compose via `StackConcat` + `MapFieldProviderOwned`
/// instead of being treated as ordinary nested-map fields.
enum VariantFieldKindOwned<'a> {
    Skip,
    Regular { reg_idx: usize },
    Flatten { ty: &'a syn::Type },
}

enum VariantSegmentOwned<'a> {
    Regular(Vec<usize>),
    Flatten { ty: &'a syn::Type },
}

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

            // Classify each field as Skip / Regular / Flatten (declaration order).
            let field_kinds: Vec<VariantFieldKindOwned> = {
                let mut reg_idx = 0usize;
                field_names
                    .iter()
                    .zip(cf.iter())
                    .zip(field_types.iter())
                    .map(|((_n, c), ty)| {
                        if c.skip_deserializing {
                            VariantFieldKindOwned::Skip
                        } else if c.flatten == crate::common::FlattenMode::None {
                            let r = reg_idx;
                            reg_idx += 1;
                            VariantFieldKindOwned::Regular { reg_idx: r }
                        } else {
                            VariantFieldKindOwned::Flatten { ty }
                        }
                    })
                    .collect()
            };

            // Group consecutive regular fields into segments; each flatten field is
            // its own segment. Segments are joined with `StackConcat`.
            let segments: Vec<VariantSegmentOwned> = {
                let mut out: Vec<VariantSegmentOwned> = vec![];
                let mut cur_reg: Vec<usize> = vec![];
                for kind in &field_kinds {
                    match kind {
                        VariantFieldKindOwned::Skip => {}
                        VariantFieldKindOwned::Regular { reg_idx } => cur_reg.push(*reg_idx),
                        VariantFieldKindOwned::Flatten { ty } => {
                            if !cur_reg.is_empty() {
                                out.push(VariantSegmentOwned::Regular(core::mem::take(
                                    &mut cur_reg,
                                )));
                            }
                            out.push(VariantSegmentOwned::Flatten { ty });
                        }
                    }
                }
                if !cur_reg.is_empty() {
                    out.push(VariantSegmentOwned::Regular(cur_reg));
                }
                out
            };

            // Regular-only filtered views (skip_deserializing and flatten fields excluded).
            let de_classified: Vec<_> = cf
                .iter()
                .filter(|c| !c.skip_deserializing && c.flatten == crate::common::FlattenMode::None)
                .collect();
            let de_field_names: Vec<_> = field_names
                .iter()
                .zip(cf.iter())
                .filter(|(_, c)| {
                    !c.skip_deserializing && c.flatten == crate::common::FlattenMode::None
                })
                .map(|(n, _)| *n)
                .collect();
            let de_field_types: Vec<_> = field_types
                .iter()
                .zip(cf.iter())
                .filter(|(_, c)| {
                    !c.skip_deserializing && c.flatten == crate::common::FlattenMode::None
                })
                .map(|(t, _)| *t)
                .collect();

            // Flatten field idents, in declaration order.
            let flatten_field_names: Vec<_> = field_names
                .iter()
                .zip(field_kinds.iter())
                .filter(|(_, k)| matches!(k, VariantFieldKindOwned::Flatten { .. }))
                .map(|(n, _)| *n)
                .collect();

            // Per-field absolute arm offset (skip fields get an unused placeholder).
            // Generic over `__KP2` (a free `MapKeyProbeOwned` parameter) rather than
            // tied to a specific `MapAccessOwned::KeyProbe` projection, matching struct_.rs.
            let arm_offset_tokens: Vec<TokenStream2> = {
                let mut out = vec![];
                let mut terms: Vec<TokenStream2> = vec![];
                for kind in &field_kinds {
                    let cur = if terms.is_empty() {
                        quote! { 0usize }
                    } else {
                        quote! { ( #( #terms )+* ) }
                    };
                    out.push(cur);
                    match kind {
                        VariantFieldKindOwned::Skip => {}
                        VariantFieldKindOwned::Regular { .. } => terms.push(quote! { 1usize }),
                        VariantFieldKindOwned::Flatten { ty } => terms.push(quote! {
                            <#ty as #krate::MapFieldProviderOwned<__KP2>>::ARMS
                        }),
                    }
                }
                out
            };

            // Builds an arm slot for a regular field (races deserialize_key against
            // deserialize_key_by_index for positional formats), same shape as before.
            let build_arm_slot = |reg_idx: usize| -> TokenStream2 {
                let dcf = de_classified[reg_idx];
                let ft = de_field_types[reg_idx];
                let mut wire_names: Vec<&str> = vec![dcf.wire_name.as_str()];
                for alias in &dcf.aliases {
                    wire_names.push(alias.as_str());
                }
                let key_fn = if wire_names.len() == 1 {
                    let name = wire_names[0];
                    quote! {
                        |mut __kp: __KP2, __i: usize| async move {
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
                        |mut __kp: __KP2, __i: usize| async move {
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
                    |__vp: #krate::owned::VP<__KP2>, __k| async move {
                        let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ft>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                    }
                };
                quote! { #krate::MapArmSlot::new(#key_fn, #val_fn) }
            };

            // --- arm stack expression (regular segments left-nested, flatten segments
            // delegate to MapFieldProviderOwned::make_arms(), joined via StackConcat) ---
            // This is the literal body of `MapFieldProviderOwned::make_arms()` — no
            // DetectDuplicatesOwned wrapping here (that only happens in the
            // DeserializeFromMapOwned impl's own arms-building expression below).
            let make_arms_expr: TokenStream2 = {
                let mut acc: Option<TokenStream2> = None;
                for seg in &segments {
                    let piece = match seg {
                        VariantSegmentOwned::Regular(regs) => {
                            let mut t = quote! { #krate::MapArmBase };
                            for r in regs {
                                let slot = build_arm_slot(*r);
                                t = quote! { (#t, #slot) };
                            }
                            t
                        }
                        VariantSegmentOwned::Flatten { ty } => quote! {
                            <#ty as #krate::MapFieldProviderOwned<__KP2>>::make_arms()
                        },
                    };
                    acc = Some(match acc {
                        None => piece,
                        Some(prev) => quote! { #krate::StackConcat(#prev, #piece) },
                    });
                }
                acc.unwrap_or_else(|| quote! { #krate::MapArmBase })
            };

            // --- wire-names array expression, offset per segment ---
            // This is the literal body of `MapFieldProviderOwned::wire_names()`.
            let wire_names_expr: TokenStream2 = {
                let mut field_iter = field_kinds.iter().enumerate();
                let mut acc: Option<TokenStream2> = None;
                for seg in &segments {
                    let piece = match seg {
                        VariantSegmentOwned::Regular(regs) => {
                            let mut entries: Vec<TokenStream2> = vec![];
                            for _ in 0..regs.len() {
                                loop {
                                    let (i, kind) =
                                        field_iter.next().expect("regular field present");
                                    if let VariantFieldKindOwned::Regular { reg_idx } = kind {
                                        let offset = &arm_offset_tokens[i];
                                        let dcf = de_classified[*reg_idx];
                                        let primary = &dcf.wire_name;
                                        entries.push(quote! { (#primary, #offset) });
                                        for a in &dcf.aliases {
                                            entries.push(quote! { (#a, #offset) });
                                        }
                                        break;
                                    }
                                }
                            }
                            let n = entries.len();
                            quote! { [#( #entries ),*] as [(&'static str, usize); #n] }
                        }
                        VariantSegmentOwned::Flatten { ty } => loop {
                            let (i, kind) = field_iter.next().expect("flatten field present");
                            if matches!(kind, VariantFieldKindOwned::Flatten { .. }) {
                                let offset = &arm_offset_tokens[i];
                                break quote! {
                                    <#ty as #krate::MapFieldProviderOwned<__KP2>>::wire_names()
                                        .map(|(__s, __i)| (__s, __i + #offset))
                                };
                            }
                        },
                    };
                    acc = Some(match acc {
                        None => piece,
                        Some(prev) => quote! { #krate::ArrayConcat::new(#prev, #piece) },
                    });
                }
                acc.unwrap_or_else(|| quote! { [] as [(&'static str, usize); 0] })
            };

            // --- type Outputs ---
            let outputs_type_tokens: TokenStream2 = {
                if segments.is_empty() {
                    quote! { () }
                } else {
                    let mut acc: Option<TokenStream2> = None;
                    for seg in &segments {
                        let seg_out = match seg {
                            VariantSegmentOwned::Regular(regs) => {
                                let mut t = quote! { () };
                                for r in regs {
                                    let dcf = de_classified[*r];
                                    let kt = {
                                        let n = 1 + dcf.aliases.len();
                                        if n == 1 {
                                            quote! { #krate::Match }
                                        } else {
                                            quote! { #krate::MatchVals<(), [(&'static str, ()); #n]> }
                                        }
                                    };
                                    let vt = de_field_types[*r];
                                    t = quote! { (#t, ::core::option::Option<(#kt, #vt)>) };
                                }
                                t
                            }
                            VariantSegmentOwned::Flatten { ty } => quote! {
                                <#ty as #krate::MapFieldProviderOwned<__KP2>>::Outputs
                            },
                        };
                        acc = Some(match acc {
                            None => seg_out,
                            Some(prev) => quote! { (#prev, #seg_out) },
                        });
                    }
                    acc.unwrap()
                }
            };

            // --- const ARMS ---
            let arms_const_tokens: TokenStream2 = {
                let terms: Vec<TokenStream2> = field_kinds
                    .iter()
                    .filter_map(|k| match k {
                        VariantFieldKindOwned::Skip => None,
                        VariantFieldKindOwned::Regular { .. } => Some(quote! { 1usize }),
                        VariantFieldKindOwned::Flatten { ty } => Some(quote! {
                            <#ty as #krate::MapFieldProviderOwned<__KP2>>::ARMS
                        }),
                    })
                    .collect();
                if terms.is_empty() {
                    quote! { 0usize }
                } else {
                    quote! { #( #terms )+* }
                }
            };

            // --- type WireNames ---
            let wire_names_type_tokens: TokenStream2 = {
                let mut acc: Option<TokenStream2> = None;
                for seg in &segments {
                    let piece = match seg {
                        VariantSegmentOwned::Regular(regs) => {
                            let n: usize = regs
                                .iter()
                                .map(|r| 1 + de_classified[*r].aliases.len())
                                .sum();
                            quote! { [(&'static str, usize); #n] }
                        }
                        VariantSegmentOwned::Flatten { ty } => quote! {
                            <<#ty as #krate::MapFieldProviderOwned<__KP2>>::WireNames
                                as #krate::ConcatableArray>::OtherArray<(&'static str, usize)>
                        },
                    };
                    acc = Some(match acc {
                        None => piece,
                        Some(prev) => quote! {
                            #krate::ArrayConcat<(&'static str, usize), #prev, #piece>
                        },
                    });
                }
                acc.unwrap_or_else(|| quote! { [(&'static str, usize); 0] })
            };

            // --- output destructure pattern: one binding per segment ---
            let seg_out_names: Vec<syn::Ident> = (0..segments.len())
                .map(|i| format_ident!("__seg_out_{}", i))
                .collect();
            let output_pat: TokenStream2 = if seg_out_names.is_empty() {
                quote! { () }
            } else {
                let mut p: Option<TokenStream2> = None;
                for n in &seg_out_names {
                    p = Some(match p {
                        None => quote! { #n },
                        Some(prev) => quote! { (#prev, #n) },
                    });
                }
                p.unwrap()
            };

            // --- from_outputs() body: same seg_stmts/skip_stmts logic, restructured
            // to return Option<Self> instead of Result<Probe<...>, Error> ---
            let mut seg_stmts: Vec<TokenStream2> = vec![];
            {
                let mut field_iter = field_kinds.iter().enumerate();
                for (seg_i, seg) in segments.iter().enumerate() {
                    let seg_out = &seg_out_names[seg_i];
                    match seg {
                        VariantSegmentOwned::Regular(regs) => {
                            let inner_pat: TokenStream2 = {
                                let mut p = quote! { () };
                                for r in regs {
                                    let ident = format_ident!("__opt_{}", r);
                                    p = quote! { (#p, #ident) };
                                }
                                p
                            };
                            seg_stmts.push(quote! { let #inner_pat = #seg_out; });
                            for _ in 0..regs.len() {
                                loop {
                                    let (_i, kind) = field_iter.next().expect("regular");
                                    if let VariantFieldKindOwned::Regular { reg_idx } = kind {
                                        let dcf = de_classified[*reg_idx];
                                        let fname = de_field_names[*reg_idx];
                                        let opt_ident = format_ident!("__opt_{}", reg_idx);
                                        let none_branch: TokenStream2 = match &dcf.default {
                                            Some(DefaultAttr::Trait) => {
                                                quote! { ::core::default::Default::default() }
                                            }
                                            Some(DefaultAttr::Expr(expr)) => {
                                                quote! { #krate::DefaultWrapper(#expr).value() }
                                            }
                                            None => quote! {
                                                return ::core::option::Option::None
                                            },
                                        };
                                        seg_stmts.push(quote! {
                                            let #fname = match #opt_ident {
                                                ::core::option::Option::Some((_, __v)) => __v,
                                                ::core::option::Option::None => #none_branch,
                                            };
                                        });
                                        break;
                                    }
                                }
                            }
                        }
                        VariantSegmentOwned::Flatten { ty } => {
                            loop {
                                let (_i, kind) = field_iter.next().expect("flatten");
                                if matches!(kind, VariantFieldKindOwned::Flatten { .. }) {
                                    break;
                                }
                            }
                            let prior_flat = segments[..seg_i]
                                .iter()
                                .filter(|s| matches!(s, VariantSegmentOwned::Flatten { .. }))
                                .count();
                            let fname = flatten_field_names[prior_flat];
                            seg_stmts.push(quote! {
                                let #fname = match <#ty as #krate::MapFieldProviderOwned<__KP2>>
                                    ::from_outputs(#seg_out)
                                {
                                    ::core::option::Option::Some(__v) => __v,
                                    ::core::option::Option::None => return ::core::option::Option::None,
                                };
                            });
                        }
                    }
                }
            }

            // Skip-field defaults (fields never queried against the map at all).
            let skip_stmts: Vec<TokenStream2> = field_names
                .iter()
                .zip(cf.iter())
                .filter(|(_, c)| c.skip_deserializing)
                .map(|(fname, c)| {
                    let default_expr: TokenStream2 = match &c.default {
                        Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                        Some(DefaultAttr::Expr(expr)) => {
                            quote! { #krate::DefaultWrapper(#expr).value() }
                        }
                        None => unreachable!("validated in classify_fields"),
                    };
                    quote! { let #fname = #default_expr; }
                })
                .collect();

            let from_outputs_body_tokens: TokenStream2 = quote! {
                let #output_pat = __outputs;
                #( #seg_stmts )*
                #( #skip_stmts )*
                ::core::option::Option::Some(#helper_name { #( #field_names, )* })
            };

            // Value/provider bounds, per field kind (owned family has no lifetimes to track).
            // This is the where-clause for the new `MapFieldProviderOwned` impl, generic
            // over `__KP2` instead of tied to `<__M2 as MapAccessOwned>::KeyProbe`.
            let mut helper_bounds: Vec<syn::WherePredicate> = Vec::new();
            for (kind, fty) in field_kinds.iter().zip(field_types.iter()) {
                match kind {
                    VariantFieldKindOwned::Skip => {}
                    VariantFieldKindOwned::Regular { .. } => {
                        helper_bounds.push(syn::parse_quote!(
                            #fty: #krate::DeserializeOwned<
                                <#krate::owned::VP<__KP2> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                    VariantFieldKindOwned::Flatten { ty } => {
                        helper_bounds.push(syn::parse_quote!(
                            #ty: #krate::MapFieldProviderOwned<__KP2>
                        ));
                        // Generic flatten fields project through `OtherArray`; the trait
                        // doesn't propagate Copy automatically, so spell it out (no-op for
                        // concrete flatten types whose OtherArray is `[_; N]`).
                        helper_bounds.push(syn::parse_quote!(
                            <<#ty as #krate::MapFieldProviderOwned<__KP2>>::WireNames
                                as #krate::ConcatableArray>::OtherArray<(&'static str, usize)>:
                                ::core::marker::Copy
                        ));
                    }
                }
            }

            // DeserializeFromMapOwned impl where-clause shrinks to a single
            // MapFieldProviderOwned bound (mirrors struct_.rs's dfm_impl_gen/dfm_where_clause).
            let dfm_where_bound: syn::WherePredicate = syn::parse_quote!(
                #helper_name: #krate::MapFieldProviderOwned<<__M2 as #krate::MapAccessOwned>::KeyProbe>
            );

            // DFM body: build arms (DetectDuplicatesOwned), iterate, reconstruct via from_outputs.
            let dfm_arms_expr = quote! {
                #krate::DetectDuplicatesOwned!(
                    <#helper_name as #krate::MapFieldProviderOwned<<__M2 as #krate::MapAccessOwned>::KeyProbe>>::make_arms(),
                    <#helper_name as #krate::MapFieldProviderOwned<<__M2 as #krate::MapAccessOwned>::KeyProbe>>::wire_names(),
                    <__M2 as #krate::MapAccessOwned>::KeyProbe,
                    #krate::owned::VP<<__M2 as #krate::MapAccessOwned>::KeyProbe>
                )
            };

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name {
                    #( #field_names: #field_types, )*
                }

                impl<__KP2: #krate::MapKeyProbeOwned> #krate::MapFieldProviderOwned<__KP2> for #helper_name
                where
                    #( #helper_bounds, )*
                {
                    type Outputs = #outputs_type_tokens;
                    const ARMS: usize = #arms_const_tokens;
                    type WireNames = #wire_names_type_tokens;
                    fn wire_names() -> Self::WireNames {
                        use #krate::ConcatableArray as _;
                        #wire_names_expr
                    }
                    fn make_arms() -> impl #krate::MapArmStackOwned<__KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                        #make_arms_expr
                    }
                    fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                        #from_outputs_body_tokens
                    }
                }

                // Shape-specific impl — used by tagged-enum dispatch via
                // `TagAwareMapOwned` + `deserialize_from_map`.
                impl<__M2: #krate::MapAccessOwned> #krate::DeserializeFromMapOwned<__M2> for #helper_name
                where
                    #dfm_where_bound,
                {
                    type Extra = ();
                    async fn deserialize_from_map_owned(
                        __map: __M2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(<__M2 as #krate::MapAccessOwned>::MapClaim, Self)>, <__M2 as #krate::MapAccessOwned>::Error>
                    {
                        let __arms = #dfm_arms_expr;
                        match __map.iterate(__arms).await? {
                            #krate::Probe::Hit((__claim, __outputs)) => {
                                match <#helper_name as #krate::MapFieldProviderOwned<
                                    <__M2 as #krate::MapAccessOwned>::KeyProbe,
                                >>::from_outputs(__outputs)
                                {
                                    ::core::option::Option::Some(__v) => {
                                        ::core::result::Result::Ok(
                                            #krate::Probe::Hit((__claim, __v))
                                        )
                                    }
                                    ::core::option::Option::None => {
                                        ::core::result::Result::Ok(#krate::Probe::Miss)
                                    }
                                }
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
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
    let arms = gen_untagged_probe_arms_owned(name, &refs, &handle_names, krate);

    Ok(quote! {
        d.entry(|[#( #handle_names ),*]| async {
            #krate::select_probe!(biased;
                #( #arms, )*
                @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
            )
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

    // Tagged str arm (unit variants via str_chunks). Tagged variants take
    // priority over untagged ones, and `select_probe!(biased; ...)` gives
    // that to us for free by declaration order, so this arm is listed first.
    let str_arm = str_handle.as_ref().map(|h| {
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
            async move {
                match #h.deserialize_value::<#unit_key_type>(#unit_key_extra).await? {
                    #krate::Probe::Hit((__unit_claim, __k)) => {
                        let __matched = #unit_key_idx;
                        match __matched {
                            #( #unit_match_arms )*
                            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    }
                    #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }
        }
    });

    // Tagged map arm.
    let map_arm = map_handle.as_ref().map(|h| {
        let map_body = gen_owned_enum_map_body(
            name,
            classified,
            variant_candidates,
            variant_key_sentinel,
            krate,
        );
        quote! {
            async move {
                let __e = #h;
                (async move { #map_body }).await
            }
        }
    });

    // Untagged arms, in declaration order.
    let untagged_classified: Vec<_> = classified.iter().filter(|cv| cv.untagged).collect();
    let untagged_arms =
        gen_untagged_probe_arms_owned(name, &untagged_classified, &untagged_handles, krate);

    let all_arms: Vec<TokenStream2> = str_arm
        .into_iter()
        .chain(map_arm)
        .chain(untagged_arms)
        .collect();

    Ok(quote! {
        d.entry(|[#( #all_handles ),*]| async {
            #krate::select_probe!(biased;
                #( #all_arms, )*
                @miss => ::core::result::Result::Ok(#krate::Probe::Miss),
            )
        }).await
    })
}

/// Generate untagged probe chain for owned family.
/// Build one `select_probe!` arm per untagged variant (owned family).
///
/// Each variant's handle is a live forked handle sharing the same underlying
/// buffer as its siblings (CLAUDE.md's "owned family — parallel scanning and
/// deadlock hazard"): they must all be raced concurrently rather than
/// awaited one at a time, since awaiting one to completion while its
/// siblings sit untouched can deadlock as soon as any candidate needs a
/// buffer refill to determine its own result (e.g. a numeric candidate that
/// only learns it overflowed once the whole digit run has arrived). Returns
/// bare arm expressions - not yet wrapped in `select_probe!` - so callers
/// with additional (non-untagged) arms of their own can splice them in ahead
/// while still relying on `select_probe!(biased; ...)` for declaration-order
/// priority.
fn gen_untagged_probe_arms_owned(
    name: &syn::Ident,
    variants: &[&ClassifiedVariant],
    handles: &[syn::Ident],
    krate: &syn::Path,
) -> Vec<TokenStream2> {
    let mut arms = Vec::new();
    for (i, cv) in variants.iter().enumerate() {
        let handle = &handles[i];
        let vname = &cv.variant.ident;
        let arm = match &cv.kind {
            VariantKind::Unit => {
                quote! {
                    async move {
                        match #handle.deserialize_value::<()>(()).await? {
                            #krate::Probe::Hit((__c, _)) => {
                                ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname)))
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    }
                }
            }
            VariantKind::Newtype(ty) => {
                quote! {
                    async move {
                        match #handle.deserialize_value::<#ty>(()).await? {
                            #krate::Probe::Hit((__c, __v)) => {
                                ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname(__v))))
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    }
                }
            }
            VariantKind::Tuple(fields) => {
                let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                let field_indices: Vec<syn::Index> =
                    (0..fields.len()).map(syn::Index::from).collect();
                quote! {
                    async move {
                        match #handle.deserialize_value::<#helper_name>(()).await? {
                            #krate::Probe::Hit((__c, __v)) => {
                                ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname( #( __v.#field_indices, )* ))))
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    }
                }
            }
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__VariantOwned{}", cv.index);
                let field_names: Vec<_> =
                    fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                quote! {
                    async move {
                        match #handle.deserialize_value::<#helper_name>(()).await? {
                            #krate::Probe::Hit((__c, __v)) => {
                                ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname { #( #field_names: __v.#field_names, )* })))
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    }
                }
            }
        };
        arms.push(arm);
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
    orig_generics: &syn::Generics,
) -> syn::Result<TokenStream2> {
    let has_nonunit = classified
        .iter()
        .any(|cv| !matches!(cv.kind, VariantKind::Unit));

    let flatten_provider = gen_enum_candidate_map_field_provider_owned(
        name,
        classified,
        tag_field,
        krate,
        container_attrs,
        orig_generics,
    );

    let (body, helpers) = if !has_nonunit {
        let body = expand_owned_internally_tagged_unit_only(
            name,
            classified,
            tag_field,
            variant_candidates,
            variant_key_sentinel,
            krate,
        )?;
        let struct_helpers =
            gen_struct_variant_helpers_owned(classified, krate, container_attrs.rename_all);
        let tuple_helpers = gen_tuple_variant_helpers_owned(classified, krate);
        (
            body,
            quote! {
                #struct_helpers
                #tuple_helpers
                #flatten_provider
            },
        )
    } else {
        let (body, helpers) = expand_owned_internally_tagged_with_nonunit(
            name,
            classified,
            tag_field,
            variant_candidates,
            variant_key_sentinel,
            krate,
            container_attrs,
        )?;
        (
            body,
            quote! {
                #helpers
                #flatten_provider
            },
        )
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
    orig_generics: &syn::Generics,
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

    // Emitted unconditionally (regardless of unit-only vs. with-nonunit
    // above) so this enum can be used as someone else's `#[strede(flatten)]`
    // target no matter which dispatch shape its own standalone path takes -
    // mirrors internally-tagged's identical rationale.
    let flatten_provider = gen_enum_candidate_map_field_provider_adjacent_owned(
        name,
        classified,
        tag_field,
        content_field,
        krate,
        container_attrs,
        orig_generics,
    );
    let helpers = quote! { #helpers #flatten_provider };

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
