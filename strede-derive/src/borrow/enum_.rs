use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput};

use super::gen_container_from_borrow;
use crate::common::{
    ClassifiedVariant, DefaultAttr, FieldContext, VariantKind, all_field_types, borrow_lifetimes,
    classify_fields, classify_variants, field_bound_borrow, has_universal_blanket,
    insert_de_and_d_borrow, other_variant, parse_container_attrs, type_param_bound_borrow,
};

pub(super) fn expand(input: DeriveInput, krate: &syn::Path) -> syn::Result<TokenStream2> {
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

    // Build impl generics: prepend 'de, add __D, add 'de: 'a bounds for field type lifetimes.
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
            let has_untagged_any = classified.iter().any(|cv| cv.untagged);
            for ty in &field_types {
                for lt in borrow_lifetimes(ty, &None) {
                    wc.predicates.push(syn::parse_quote!('de: #lt));
                }
                // Skip for universal-blanket types to avoid impl/where-clause ambiguity.
                if !has_universal_blanket(ty) {
                    wc.predicates
                        .push(field_bound_borrow(krate, ty, FieldContext::MapValue));
                }
                // Untagged variants dispatch via `__e.deserialize_value::<T>(())` on Entry.
                if has_untagged_any && !has_universal_blanket(ty) {
                    wc.predicates.push(syn::parse_quote!(
                        #ty: #krate::Deserialize<
                            'de,
                            <__D::Entry as #krate::Entry<'de>>::SubDeserializer,
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
                    (): #krate::Deserialize<
                        'de,
                        <__D::Entry as #krate::Entry<'de>>::SubDeserializer,
                        Extra = ()
                    >
                ));
            }
            // Map iteration uses Match/Skip key probes.
            // For enums: map iteration only happens for non-unit non-untagged variants.
            let dup_n: usize = classified
                .iter()
                .filter(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit))
                .map(|cv| 1 + cv.aliases.len())
                .sum();
            let _ = dup_n; // universal Match/MatchVals/Skip impls cover the key bounds

            // Adjacent-tagged non-unit variants dispatch via
            // `__vp.deserialize_value::<HelperT>(())` on the content slot. The
            // helper's own blanket impl re-projects through a fresh sub-de,
            // pushing inner-field obligations one level deeper than the outer
            // impl's bounds can cover. Adding the helper-as-type bound here
            // turns the dispatch obligation into an assumption.
            let is_adjacent = container_attrs.tag.is_some() && container_attrs.content.is_some();
            let is_internal = container_attrs.tag.is_some() && container_attrs.content.is_none();
            if is_adjacent {
                for cv in &classified {
                    if cv.untagged {
                        continue;
                    }
                    let helper_ty: syn::Type = match &cv.kind {
                        VariantKind::Struct(_) => {
                            let id = format_ident!("__Variant{}", cv.index);
                            syn::parse_quote!(#id)
                        }
                        VariantKind::Tuple(_) => {
                            let id = format_ident!("__TupleVariant{}", cv.index);
                            syn::parse_quote!(#id)
                        }
                        VariantKind::Newtype(_) | VariantKind::Unit => continue,
                    };
                    wc.predicates.push(field_bound_borrow(
                        krate,
                        &helper_ty,
                        FieldContext::MapValue,
                    ));
                }
            }
            // Internally-tagged newtype variants dispatch via
            // `<InnerTy as DeserializeFromMap<'de, TagAwareMap<…>>>::deserialize_from_map(…)`.
            // The user-defined `InnerTy`'s DFM impl re-projects through TagAwareMap's
            // KeyProbe; its inner-field obligations (e.g. `u32: Deserialize<'de, …>`)
            // aren't in the outer's where-clause when the variant is a newtype around
            // a user struct rather than naming its fields directly. Add a HRTB-quantified
            // bound so the dispatch obligation is satisfied by assumption.
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
                            for<'__v> #ty: #krate::DeserializeFromMap<
                                'de,
                                #krate::TagAwareMap<
                                    'de, '__v,
                                    <__D::Entry as #krate::Entry<'de>>::Map,
                                    #n_cands,
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

            impl #impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize(
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
                            __se.get::<#ty>(()).await
                        }).await);
                        let (__seq_back, #acc) = #krate::or_miss!(__v.data());
                        let __seq = __seq_back;
                    }
                })
                .collect();

            // Collect 'de: 'a bounds for all field types and D3 SeqElem bounds.
            let mut helper_bounds: Vec<syn::WherePredicate> = Vec::new();
            for fty in &field_types {
                for lt in borrow_lifetimes(fty, &None) {
                    helper_bounds.push(syn::parse_quote!('de: #lt));
                }
                // D3: tuple variant helper reads via `__se.get::<T>(())` — SeqElem context on __D2.
                helper_bounds.push(syn::parse_quote!(
                    #fty: #krate::Deserialize<
                        'de,
                        <#krate::borrow::SE<'de, __D2> as #krate::SeqEntry<'de>>::SubDeserializer,
                        Extra = ()
                    >
                ));
            }

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name( #( #field_types, )* );

                impl<'de, __D2: #krate::Deserializer<'de>> #krate::Deserialize<'de, __D2> for #helper_name
                where
                    #( #helper_bounds, )*
                {
                    type Extra = ();
                    async fn deserialize(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
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
            // KP/VP are projected directly from __M2 since the helper's DeserializeFromMap
            // impl is generic over the access type, not a Deserializer.
            let arm_slots: Vec<TokenStream2> = field_names
                .iter()
                .zip(field_types.iter())
                .zip(de_classified.iter())
                .map(|((fname, fty), dcf)| {
                    let mut wnames: Vec<&str> = vec![dcf.wire_name.as_str()];
                    for a in &dcf.aliases { wnames.push(a.as_str()); }
                    let key_fn = if wnames.len() == 1 {
                        let wn = wnames[0];
                        quote! { |mut __kp: <__M2 as #krate::MapAccess<'de>>::KeyProbe| __kp.deserialize_key::<#krate::Match>(#wn) }
                    } else {
                        quote! { |mut __kp: <__M2 as #krate::MapAccess<'de>>::KeyProbe| __kp.deserialize_key::<#krate::MatchVals<(), _>>([#( (#wnames, ()), )*]) }
                    };
                    let out_name = format_ident!("__out_{}", fname);
                    let _ = out_name;
                    quote! {
                        #krate::MapArmSlot::new(
                            #key_fn,
                            |__vp: #krate::borrow::VP<'de, <__M2 as #krate::MapAccess<'de>>::KeyProbe>, __k| async move {
                                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#fty>(()).await);
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
                            move |__kp: <__M2 as #krate::MapAccess<'de>>::KeyProbe| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                            |__vp: #krate::borrow::VP<'de, <__M2 as #krate::MapAccess<'de>>::KeyProbe>| __vp.skip(),
                        )
                    }
                }
            };

            // Collect 'de: 'a bounds for all field types and the map-value bound rooted at __M2.
            let mut helper_bounds: Vec<syn::WherePredicate> = Vec::new();
            for fty in &field_types {
                for lt in borrow_lifetimes(fty, &None) {
                    helper_bounds.push(syn::parse_quote!('de: #lt));
                }
                helper_bounds.push(syn::parse_quote!(
                    #fty: #krate::Deserialize<
                        'de,
                        <#krate::borrow::VP<'de, <__M2 as #krate::MapAccess<'de>>::KeyProbe> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            // Universal Match/Skip impls cover map-key probe bounds — no explicit predicates needed.

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name {
                    #( #field_names: #field_types, )*
                }

                // Shape-specific impl — used by tagged-enum dispatch via
                // `TagAwareMap` + `deserialize_from_map`.
                impl<'de, __M2: #krate::MapAccess<'de>> #krate::DeserializeFromMap<'de, __M2> for #helper_name
                where
                    #( #helper_bounds, )*
                {
                    type Extra = ();
                    async fn deserialize_from_map(
                        __map: __M2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(<__M2 as #krate::MapAccess<'de>>::MapClaim, Self)>, <__M2 as #krate::MapAccess<'de>>::Error>
                    {
                        let __arms = #arms_expr;
                        match __map.iterate(__arms).await? {
                            #krate::Probe::Hit((__claim, #output_pat)) => {
                                #( #field_finalizers )*
                                ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #helper_name { #( #field_names, )* })))
                            }
                            #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                        }
                    }
                }

                // Universal entry point — used when this helper is the value of
                // a map key (adjacent-tagged content field, externally-tagged
                // single-key value).
                impl<'de, __D2: #krate::Deserializer<'de>> #krate::Deserialize<'de, __D2> for #helper_name
                where
                    #helper_name: #krate::DeserializeFromMap<'de, <__D2::Entry as #krate::Entry<'de>>::Map, Extra = ()>,
                {
                    type Extra = ();
                    async fn deserialize(
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
                let helper_name = format_ident!("__TupleVariant{}", cv.index);
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
                let helper_name = format_ident!("__Variant{}", cv.index);
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
                        __kp.deserialize_key::<#krate::Match>(#wn)
                    }
                }
            } else {
                quote! {
                    |mut __kp: #krate::borrow::KP<'de, __D>| {
                        __kp.deserialize_key::<#krate::MatchVals<(), _>>([#( (#wire_names, ()), )*])
                    }
                }
            };
            let val_fn = match &cv.kind {
                VariantKind::Newtype(ty) => quote! {
                    |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                        let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#ty>(()).await);
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
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#helper_name>(()).await);
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
                move |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
            )
        }};
        if has_other {
            expr = quote! { (#expr, #krate::VirtualArmSlot::new(
                |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip>(()),
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

            impl #impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize(
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
                let __m = #krate::TagAwareMap::new(
                    #fork_ident,
                    #tag_field,
                    [#( #tag_cands_entries, )*],
                    #local_idx,
                    __tag_value,
                );
                match <#de_type as #krate::DeserializeFromMap<'de, _>>::deserialize_from_map(__m, ()).await? {
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
                            |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip>(()),
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
                        |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip>(()),
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

            impl #impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize(
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
                                 __kp.deserialize_key::<#krate::Match>(#tag_field)
                             },
                             |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                                 let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                     #krate::MatchVals<usize, #tag_cands_count>
                                 >([#( #tag_cands_entries, )*]).await);
                                 ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                             },
                         )),
                        #krate::MapArmSlot::new(
                            |mut __kp: #krate::borrow::KP<'de, __D>| {
                                __kp.deserialize_key::<#krate::Match>(#content_field)
                            },
                            |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#de_type>(()).await);
                                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                            },
                        )
                    );
                    let __wn = #dup_wire_names;
                    let __dd = #krate::DetectDuplicatesOwned::new(
                        __inner_arms,
                        __wn,
                        move |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                        |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
                    );
                    (__dd, #krate::VirtualArmSlot::new(
                        |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip>(()),
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

    // Unit variant fallback: iterate the outer map looking for the tag field only.
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

    let unit_fallback = if !unit_match_arms.is_empty() || other_variant(classified).is_some() {
        quote! {
            let __unit_arms = (
                (#krate::MapArmBase,
                 #krate::MapArmSlot::new(
                     |mut __kp: #krate::borrow::KP<'de, __D>| {
                         __kp.deserialize_key::<#krate::Match>(#tag_field)
                     },
                     |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                         let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                             #krate::MatchVals<usize, #tag_cands_count>
                         >([#( #tag_cands_entries, )*]).await);
                         ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                     },
                 )),
                #krate::VirtualArmSlot::new(
                    |__kp: #krate::borrow::KP<'de, __D>| __kp.deserialize_key::<#krate::Skip>(()),
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
