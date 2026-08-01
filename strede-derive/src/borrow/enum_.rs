use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput};

use super::gen_container_from_borrow;
use crate::common::{
    ClassifiedVariant, DefaultAttr, FieldContext, VariantKind, all_field_types, borrow_lifetimes,
    classify_fields, classify_variants, field_bound_borrow, has_universal_blanket,
    insert_de_and_d_borrow, other_variant, parse_container_attrs, type_param_bound_borrow,
};

/// Insert `'de` (if absent) and `__E: EnumAccess<'de>` into `impl_gen`.
/// Used for `DeserializeFromEnum` impl emission (external/untagged paths).
fn insert_de_and_e_borrow(impl_gen: &mut syn::Generics, krate: &syn::Path) {
    let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
    if !has_de {
        impl_gen.params.insert(0, syn::parse_quote!('de));
    }
    impl_gen
        .params
        .push(syn::parse_quote!(__E: #krate::EnumAccess<'de>));
}

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
    let d_ident = format_ident!("__D");

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
                    wc.predicates.push(field_bound_borrow(
                        krate,
                        ty,
                        FieldContext::MapValue,
                        &d_ident,
                    ));
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
            // Untagged struct/tuple variants dispatch via
            // `deserialize_value::<HelperT>(())` (`gen_untagged_probe_chain_borrow` /
            // `expand_enum_with_untagged`'s `deserialize_value_by_shape`), i.e. the
            // *helper type itself* is the payload passed to `deserialize_value`, not
            // its individual field types (the `field_types` loop above pushes bounds
            // keyed on raw field types, which is correct for `Newtype(ty)` variants
            // where `ty` *is* the payload, but does nothing useful for `Struct`/`Tuple`
            // variants - their payload is `__VariantN`/`__TupleVariantN`). Missing this
            // bound left every untagged struct/tuple variant with a non-blanket field
            // type unable to compile at all (never previously covered by any test).
            for cv in &classified {
                if !cv.untagged {
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
                wc.predicates.push(syn::parse_quote!(
                    #helper_ty: #krate::Deserialize<
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
                        &d_ident,
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

    if let Some(ref tag_field) = container_attrs.tag {
        if let Some(ref content_field) = container_attrs.content {
            return expand_enum_adjacent_tagged_borrow(
                name,
                &classified,
                tag_field,
                content_field,
                krate,
                &container_attrs,
                &input.generics,
            );
        }
        return expand_enum_internally_tagged(
            name,
            &classified,
            tag_field,
            krate,
            &container_attrs,
            &input.generics,
        );
    }

    let has_tagged_unit = classified
        .iter()
        .any(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit));
    let has_tagged_nonunit = classified
        .iter()
        .any(|cv| !cv.untagged && !matches!(cv.kind, VariantKind::Unit));
    let has_untagged = classified.iter().any(|cv| cv.untagged);

    if !has_untagged {
        // Externally-tagged (no untagged variants): use the new
        // DeserializeFromEnum + Deserialize two-impl approach.
        return expand_enum_external_tagged_borrow(
            name,
            &classified,
            krate,
            &container_attrs,
            &input.generics,
        );
    }

    if !has_tagged_unit && !has_tagged_nonunit {
        let body =
            expand_enum_untagged_only(name, &classified, krate, &container_attrs, &input.generics)?;
        let tuple_variant_helpers = gen_tuple_variant_helpers_borrow(&classified, krate);
        let struct_variant_helpers =
            gen_struct_variant_helpers_borrow(&classified, krate, container_attrs.rename_all);
        // `MapFieldProvider` impl so this purely-untagged enum can be used as
        // a `#[strede(flatten)]` field's type - see
        // `gen_enum_candidate_map_field_provider_untagged_borrow`.
        let flatten_provider = gen_enum_candidate_map_field_provider_untagged_borrow(
            name,
            &classified,
            krate,
            &container_attrs,
            &input.generics,
        );
        return Ok(quote! {
            #[allow(unreachable_code)]
            const _: () = {
                use #krate::{
                    DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                    MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                    SeqAccess as _, SeqEntry as _, StrAccess as _,
                };
                #tuple_variant_helpers
                #struct_variant_helpers
                #flatten_provider
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
        });
    }
    expand_enum_with_untagged(name, &classified, krate, &container_attrs, &input.generics)
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
            let helper_d_ident = format_ident!("__D2");
            let mut helper_bounds: Vec<syn::WherePredicate> = Vec::new();
            for fty in &field_types {
                for lt in borrow_lifetimes(fty, &None) {
                    helper_bounds.push(syn::parse_quote!('de: #lt));
                }
                // D3: tuple variant helper reads via `__se.get::<T>(())` — SeqElem context on __D2.
                helper_bounds.push(field_bound_borrow(
                    krate,
                    fty,
                    FieldContext::SeqElem,
                    &helper_d_ident,
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
/// Field-kind classification shared by the segment-building helpers below —
/// mirrors `struct_::expand`'s `FieldKind`/`Segment` split so a variant's
/// `#[strede(flatten)]` fields compose via `StackConcat` + `MapFieldProvider`
/// instead of being treated as ordinary nested-map fields.
enum VariantFieldKind<'a> {
    Skip,
    Regular {
        reg_idx: usize,
    },
    Flatten {
        ty: &'a syn::Type,
        borrow: &'a Option<crate::common::BorrowAttr>,
    },
}

enum VariantSegment<'a> {
    Regular(Vec<usize>),
    Flatten {
        ty: &'a syn::Type,
        #[allow(dead_code)]
        borrow: &'a Option<crate::common::BorrowAttr>,
    },
}

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

            // Classify each field as Skip / Regular / Flatten (declaration order).
            let field_kinds: Vec<VariantFieldKind> = {
                let mut reg_idx = 0usize;
                field_names
                    .iter()
                    .zip(cf.iter())
                    .zip(field_types.iter())
                    .map(|((_n, c), ty)| {
                        if c.skip_deserializing {
                            VariantFieldKind::Skip
                        } else if c.flatten == crate::common::FlattenMode::None {
                            let r = reg_idx;
                            reg_idx += 1;
                            VariantFieldKind::Regular { reg_idx: r }
                        } else {
                            VariantFieldKind::Flatten {
                                ty,
                                borrow: &c.borrow,
                            }
                        }
                    })
                    .collect()
            };

            // Group consecutive regular fields into segments; each flatten field is
            // its own segment. Segments are joined with `StackConcat`.
            let segments: Vec<VariantSegment> = {
                let mut out: Vec<VariantSegment> = vec![];
                let mut cur_reg: Vec<usize> = vec![];
                for kind in &field_kinds {
                    match kind {
                        VariantFieldKind::Skip => {}
                        VariantFieldKind::Regular { reg_idx } => cur_reg.push(*reg_idx),
                        VariantFieldKind::Flatten { ty, borrow } => {
                            if !cur_reg.is_empty() {
                                out.push(VariantSegment::Regular(core::mem::take(&mut cur_reg)));
                            }
                            out.push(VariantSegment::Flatten { ty, borrow });
                        }
                    }
                }
                if !cur_reg.is_empty() {
                    out.push(VariantSegment::Regular(cur_reg));
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
                .filter(|(_, k)| matches!(k, VariantFieldKind::Flatten { .. }))
                .map(|(n, _)| *n)
                .collect();

            // Per-field absolute arm offset (skip fields get an unused placeholder).
            // Generic over `__KP2` (a free `MapKeyProbe<'de>` parameter) rather than
            // tied to a specific `MapAccess::KeyProbe` projection, matching struct_.rs.
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
                        VariantFieldKind::Skip => {}
                        VariantFieldKind::Regular { .. } => terms.push(quote! { 1usize }),
                        VariantFieldKind::Flatten { ty, .. } => terms.push(quote! {
                            <#ty as #krate::MapFieldProvider<'de, __KP2>>::ARMS
                        }),
                    }
                }
                out
            };

            // Builds an arm slot for a regular field (races deserialize_key against
            // deserialize_key_by_index for positional formats), same shape as before.
            let build_arm_slot = |reg_idx: usize| -> TokenStream2 {
                let dcf = de_classified[reg_idx];
                let fty = de_field_types[reg_idx];
                let mut wnames: Vec<&str> = vec![dcf.wire_name.as_str()];
                for a in &dcf.aliases {
                    wnames.push(a.as_str());
                }
                let key_fn = if wnames.len() == 1 {
                    let wn = wnames[0];
                    quote! {
                        |mut __kp: __KP2, __i: usize| async move {
                            let __kp2 = __kp.fork();
                            #krate::select_probe! {
                                __kp.deserialize_key::<#krate::Match>(#wn),
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
                                __kp.deserialize_key::<#krate::MatchVals<(), _>>([#( (#wnames, ()), )*]),
                                async move {
                                    let (__kc, ()) = #krate::hit!(__kp2.deserialize_key_by_index(__i).await);
                                    ::core::result::Result::Ok(#krate::Probe::Hit((__kc, #krate::MatchVals((), ::core::marker::PhantomData))))
                                },
                            }
                        }
                    }
                };
                quote! {
                    #krate::MapArmSlot::new(
                        #key_fn,
                        |__vp: #krate::borrow::VP<'de, __KP2>, __k| async move {
                            let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#fty>(()).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                        }
                    )
                }
            };

            // --- arm stack expression (regular segments left-nested, flatten segments
            // delegate to MapFieldProvider::make_arms(), joined via StackConcat) ---
            // This is the literal body of `MapFieldProvider::make_arms()` — no
            // DetectDuplicates wrapping here (that only happens in the
            // DeserializeFromMap impl's own arms-building expression below).
            let make_arms_expr: TokenStream2 = {
                let mut acc: Option<TokenStream2> = None;
                for seg in &segments {
                    let piece = match seg {
                        VariantSegment::Regular(regs) => {
                            let mut t = quote! { #krate::MapArmBase };
                            for r in regs {
                                let slot = build_arm_slot(*r);
                                t = quote! { (#t, #slot) };
                            }
                            t
                        }
                        VariantSegment::Flatten { ty, .. } => quote! {
                            <#ty as #krate::MapFieldProvider<'de, __KP2>>::make_arms()
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
            // This is the literal body of `MapFieldProvider::wire_names()`.
            let wire_names_expr: TokenStream2 = {
                let mut field_iter = field_kinds.iter().enumerate();
                let mut acc: Option<TokenStream2> = None;
                for seg in &segments {
                    let piece = match seg {
                        VariantSegment::Regular(regs) => {
                            let mut entries: Vec<TokenStream2> = vec![];
                            for _ in 0..regs.len() {
                                loop {
                                    let (i, kind) =
                                        field_iter.next().expect("regular field present");
                                    if let VariantFieldKind::Regular { reg_idx } = kind {
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
                        VariantSegment::Flatten { ty, .. } => loop {
                            let (i, kind) = field_iter.next().expect("flatten field present");
                            if matches!(kind, VariantFieldKind::Flatten { .. }) {
                                let offset = &arm_offset_tokens[i];
                                break quote! {
                                    <#ty as #krate::MapFieldProvider<'de, __KP2>>::wire_names()
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
                            VariantSegment::Regular(regs) => {
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
                            VariantSegment::Flatten { ty, .. } => quote! {
                                <#ty as #krate::MapFieldProvider<'de, __KP2>>::Outputs
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
                        VariantFieldKind::Skip => None,
                        VariantFieldKind::Regular { .. } => Some(quote! { 1usize }),
                        VariantFieldKind::Flatten { ty, .. } => Some(quote! {
                            <#ty as #krate::MapFieldProvider<'de, __KP2>>::ARMS
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
                        VariantSegment::Regular(regs) => {
                            let n: usize = regs
                                .iter()
                                .map(|r| 1 + de_classified[*r].aliases.len())
                                .sum();
                            quote! { [(&'static str, usize); #n] }
                        }
                        VariantSegment::Flatten { ty, .. } => quote! {
                            <<#ty as #krate::MapFieldProvider<'de, __KP2>>::WireNames
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
                        VariantSegment::Regular(regs) => {
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
                                    if let VariantFieldKind::Regular { reg_idx } = kind {
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
                        VariantSegment::Flatten { ty, .. } => {
                            loop {
                                let (_i, kind) = field_iter.next().expect("flatten");
                                if matches!(kind, VariantFieldKind::Flatten { .. }) {
                                    break;
                                }
                            }
                            let prior_flat = segments[..seg_i]
                                .iter()
                                .filter(|s| matches!(s, VariantSegment::Flatten { .. }))
                                .count();
                            let fname = flatten_field_names[prior_flat];
                            seg_stmts.push(quote! {
                                let #fname = match <#ty as #krate::MapFieldProvider<'de, __KP2>>
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

            // Collect 'de: 'a bounds and value/provider bounds, per field kind.
            // This is the where-clause for the new `MapFieldProvider` impl, generic
            // over `__KP2` instead of tied to `<__M2 as MapAccess<'de>>::KeyProbe`.
            let mut helper_bounds: Vec<syn::WherePredicate> = Vec::new();
            for (kind, fty) in field_kinds.iter().zip(field_types.iter()) {
                match kind {
                    VariantFieldKind::Skip => {}
                    VariantFieldKind::Regular { .. } => {
                        for lt in borrow_lifetimes(fty, &None) {
                            helper_bounds.push(syn::parse_quote!('de: #lt));
                        }
                        helper_bounds.push(syn::parse_quote!(
                            #fty: #krate::Deserialize<
                                'de,
                                <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                    VariantFieldKind::Flatten { ty, borrow } => {
                        for lt in borrow_lifetimes(ty, borrow) {
                            helper_bounds.push(syn::parse_quote!('de: #lt));
                        }
                        helper_bounds.push(syn::parse_quote!(
                            #ty: #krate::MapFieldProvider<'de, __KP2>
                        ));
                        // Generic flatten fields project through `OtherArray`; the trait
                        // doesn't propagate Copy automatically, so spell it out (no-op for
                        // concrete flatten types whose OtherArray is `[_; N]`).
                        helper_bounds.push(syn::parse_quote!(
                            <<#ty as #krate::MapFieldProvider<'de, __KP2>>::WireNames
                                as #krate::ConcatableArray>::OtherArray<(&'static str, usize)>:
                                ::core::marker::Copy
                        ));
                    }
                }
            }
            // Universal Match/Skip impls cover map-key probe bounds — no explicit predicates needed.

            // DeserializeFromMap impl where-clause shrinks to a single MapFieldProvider bound
            // (mirrors struct_.rs's dfm_impl_gen/dfm_where_clause construction).
            let dfm_where_bound: syn::WherePredicate = syn::parse_quote!(
                #helper_name: #krate::MapFieldProvider<'de, <__M2 as #krate::MapAccess<'de>>::KeyProbe>
            );

            // DFM body: build arms (DetectDuplicates), iterate, reconstruct via from_outputs.
            let dfm_arms_expr = quote! {
                #krate::DetectDuplicates!(
                    <#helper_name as #krate::MapFieldProvider<'de, <__M2 as #krate::MapAccess<'de>>::KeyProbe>>::make_arms(),
                    <#helper_name as #krate::MapFieldProvider<'de, <__M2 as #krate::MapAccess<'de>>::KeyProbe>>::wire_names(),
                    <__M2 as #krate::MapAccess<'de>>::KeyProbe,
                    #krate::borrow::VP<'de, <__M2 as #krate::MapAccess<'de>>::KeyProbe>
                )
            };

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name {
                    #( #field_names: #field_types, )*
                }

                impl<'de, __KP2: #krate::MapKeyProbe<'de>> #krate::MapFieldProvider<'de, __KP2> for #helper_name
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
                    fn make_arms() -> impl #krate::MapArmStack<'de, __KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                        #make_arms_expr
                    }
                    fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                        #from_outputs_body_tokens
                    }
                }

                // Shape-specific impl — used by tagged-enum dispatch via
                // `TagAwareMap` + `deserialize_from_map`.
                impl<'de, __M2: #krate::MapAccess<'de>> #krate::DeserializeFromMap<'de, __M2> for #helper_name
                where
                    #dfm_where_bound,
                {
                    type Extra = ();
                    async fn deserialize_from_map(
                        __map: __M2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(<__M2 as #krate::MapAccess<'de>>::MapClaim, Self)>, <__M2 as #krate::MapAccess<'de>>::Error>
                    {
                        let __arms = #dfm_arms_expr;
                        match __map.iterate(__arms).await? {
                            #krate::Probe::Hit((__claim, __outputs)) => {
                                match <#helper_name as #krate::MapFieldProvider<
                                    'de,
                                    <__M2 as #krate::MapAccess<'de>>::KeyProbe,
                                >>::from_outputs(__outputs)
                                {
                                    ::core::option::Option::Some(__v) => {
                                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)))
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

/// Generate `DeserializeFromEnum<'de, __E>` + `Deserialize<'de, __D>` impls for
/// externally-tagged enums (no `#[strede(tag)]` / `#[strede(untagged)]`).
///
/// The `DeserializeFromEnum` impl drives variant dispatch via `EnumAccess::iterate`
/// with an arm stack. The `Deserialize` impl delegates via `deserialize_enum_into`.
fn expand_enum_external_tagged_borrow(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> syn::Result<TokenStream2> {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    // --- Build impl generics for DeserializeFromEnum<'de, __E> ---
    let mut enum_impl_gen = orig_generics.clone();
    insert_de_and_e_borrow(&mut enum_impl_gen, krate);

    // Collect payload types for non-unit non-other tagged variants.
    let payload_types: Vec<syn::Type> = classified
        .iter()
        .filter(|cv| !cv.untagged && !cv.other && !matches!(cv.kind, VariantKind::Unit))
        .map(|cv| match &cv.kind {
            VariantKind::Newtype(ty) => syn::parse_quote!(#ty),
            VariantKind::Struct(_) => {
                let id = format_ident!("__Variant{}", cv.index);
                syn::parse_quote!(#id)
            }
            VariantKind::Tuple(_) => {
                let id = format_ident!("__TupleVariant{}", cv.index);
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
                wc.predicates.push(type_param_bound_borrow(krate, ident));
            }
            // Payload types must be Deserialize via the PayloadDeserializer.
            for pty in &payload_types {
                for lt in borrow_lifetimes(pty, &None) {
                    wc.predicates.push(syn::parse_quote!('de: #lt));
                }
                wc.predicates.push(syn::parse_quote!(
                    #pty: #krate::Deserialize<
                        'de,
                        <__E::VariantProbe as #krate::EnumVariantProbe<'de>>::PayloadDeserializer,
                        Extra = ()
                    >
                ));
            }
        }
    }
    let (enum_impl_generics, _, enum_where_clause) = enum_impl_gen.split_for_impl();

    // --- Build impl generics for Deserialize<'de, __D> ---
    let mut de_impl_gen = orig_generics.clone();
    insert_de_and_d_borrow(&mut de_impl_gen, krate);
    {
        let wc = de_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_borrow(krate, ident));
            }
            // Require that Self: DeserializeFromEnum for the Entry's Enum type.
            wc.predicates.push(syn::parse_quote!(
                #name #ty_generics: #krate::DeserializeFromEnum<
                    'de,
                    <__D::Entry as #krate::Entry<'de>>::Enum,
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
            // Add aliases - they map to the same local_idx
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
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
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
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
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
                    let helper_name = format_ident!("__Variant{}", cv.index);
                    quote! {
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
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
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
                    quote! {
                        #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
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
    let tuple_variant_helpers = gen_tuple_variant_helpers_borrow(classified, krate);
    let struct_variant_helpers =
        gen_struct_variant_helpers_borrow(classified, krate, container_attrs.rename_all);

    // `MapFieldProvider` impl so this enum can be used as a `#[strede(flatten)]`
    // field's type — see `gen_enum_map_field_provider_borrow`.
    let map_field_provider_impl =
        gen_enum_map_field_provider_borrow(name, classified, krate, container_attrs, orig_generics);

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

    let deserialize_body = match other_variant(classified) {
        Some(other_vname) => quote! {
            d.entry(|[__e1, __e2]| async {
                match __e1.deserialize_enum_into::<Self>(()).await? {
                    #krate::Probe::Hit(__v) => ::core::result::Result::Ok(#krate::Probe::Hit(__v)),
                    #krate::Probe::Miss => {
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
                DefaultValue as _, Deserialize as _, DeserializeFromEnum as _,
                Deserializer as _, Entry as _, EnumAccess as _, EnumVariantProbe as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #tuple_variant_helpers
            #struct_variant_helpers
            #map_field_provider_impl

            impl #enum_impl_generics #krate::DeserializeFromEnum<'de, __E>
                for #name #ty_generics
                #enum_where_clause
            {
                type Extra = ();
                async fn deserialize_from_enum(
                    __e: __E,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__E::Claim, Self)>, __E::Error>
                {
                    #deserialize_from_enum_body
                }
            }

            impl #de_impl_generics #krate::Deserialize<'de, __D>
                for #name #ty_generics
                #de_where_clause
            {
                type Extra = ();
                async fn deserialize(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    #deserialize_body
                }
            }
        };
    })
}

/// Emit `MapFieldProvider<'de, __KP2>` for an externally-tagged enum so it can
/// be used as a `#[strede(flatten)]` field's type.
///
/// Externally tagging only ever contributes one key/value pair (the matched
/// variant's wire name → its payload) no matter how many variants exist, so
/// this is a single arm: a `MatchVals` key race over every (non-`other`,
/// non-untagged) variant's wire name + aliases, dispatching by the matched
/// index to that variant's payload. A unit variant's payload is `()`
/// (`"VariantName": null` on the wire) — the same "no payload" convention
/// used for unit structs elsewhere in the derive. `#[strede(other)]` is not
/// supported through this arm: an `other` variant is simply never a
/// candidate, so an unmatched key just misses this arm and falls through to
/// whatever the parent struct's own unknown-field handling does.
fn gen_enum_map_field_provider_borrow(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> TokenStream2 {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    // Same candidate set as expand_enum_external_tagged_borrow's arm_slots.
    let tagged_non_other: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && !cv.other)
        .collect();
    let n_cands: usize = tagged_non_other.iter().map(|cv| 1 + cv.aliases.len()).sum();

    // --- impl generics: 'de, __KP2: MapKeyProbe<'de> ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        let has_de = mfp_impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
        if !has_de {
            mfp_impl_gen.params.insert(0, syn::parse_quote!('de));
        }
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbe<'de>));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::Deserialize<
                        'de,
                        <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
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
                    (): #krate::Deserialize<
                        'de,
                        <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            for cv in &tagged_non_other {
                let pty: Option<syn::Type> = match &cv.kind {
                    VariantKind::Newtype(ty) => Some(syn::parse_quote!(#ty)),
                    VariantKind::Struct(_) => {
                        let id = format_ident!("__Variant{}", cv.index);
                        Some(syn::parse_quote!(#id))
                    }
                    VariantKind::Tuple(_) => {
                        let id = format_ident!("__TupleVariant{}", cv.index);
                        Some(syn::parse_quote!(#id))
                    }
                    VariantKind::Unit => None,
                };
                if let Some(pty) = pty {
                    for lt in borrow_lifetimes(&pty, &None) {
                        wc.predicates.push(syn::parse_quote!('de: #lt));
                    }
                    wc.predicates.push(syn::parse_quote!(
                        #pty: #krate::Deserialize<
                            'de,
                            <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
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
                    let helper_name = format_ident!("__Variant{}", cv.index);
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
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
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

    // ---- compile-time wire-name collision detection (see strede::Fields) ---
    // An externally-tagged enum's *only* contribution when flattened is its
    // variant names (+ aliases) — the matched variant's own inner fields live
    // one map level deeper (a separate, nested map), never surfacing here.
    let variant_name_tokens: Vec<TokenStream2> = tagged_non_other
        .iter()
        .flat_map(|cv| {
            let mut v = vec![{
                let wn = &cv.wire_name;
                quote! { #wn }
            }];
            for alias in &cv.aliases {
                v.push(quote! { #alias });
            }
            v
        })
        .collect();
    let self_dup_check = if orig_generics.type_params().next().is_none() {
        quote! { const _: () = #krate::NoInternalDuplicates::<#name #ty_generics>::CHECK; }
    } else {
        quote! {}
    };
    // Fresh, plain generics (no __KP2) — NAMES is pure string literal data,
    // independent of the map-key-probe machinery the MapFieldProvider impl
    // above needs. Reusing `mfp_impl_generics` here would leave `__KP2`
    // unconstrained (E0207): it doesn't appear in `Fields` or `Self`.
    let (fields_impl_generics, _, fields_where_clause) = orig_generics.split_for_impl();
    let fields_impl_tokens = quote! {
        impl #fields_impl_generics #krate::Fields for #name #ty_generics
            #fields_where_clause
        {
            const NAMES: &'static [&'static str] = &[ #( #variant_name_tokens ),* ];
        }
        #self_dup_check
    };

    quote! {
        #fields_impl_tokens

        impl #mfp_impl_generics #krate::MapFieldProvider<'de, __KP2> for #name #ty_generics
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

            fn make_arms() -> impl #krate::MapArmStack<'de, __KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                (
                    #krate::MapArmBase,
                    #krate::MapArmSlot::new(
                        |mut __kp: __KP2, _i: usize| __kp.deserialize_key::<
                            #krate::MatchVals<usize, [(&'static str, usize); #n_cands]>
                        >([#( #match_cands_tokens, )*]),
                        |__vp: #krate::borrow::VP<'de, __KP2>,
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

/// Emit `MapFieldProvider<'de, __KP2>` for an internally-tagged enum so it can
/// be used as a `#[strede(flatten)]` field's type.
///
/// Unlike external tagging (one arm, no ambiguity), internally-tagged
/// variants may share the same map as the parent struct and the tag key
/// isn't guaranteed to arrive first, so every non-`other` variant races its
/// own fields concurrently via the `CandidateArmStack` runtime primitive
/// (`strede/src/map_arm/{mod,borrow,owned}.rs`) until the tag key resolves,
/// at which point every other candidate permanently stops racing. Struct
/// variants reuse their existing `__VariantN` helper's `MapFieldProvider`
/// impl (already ported to compose nested `#[flatten]` fields via
/// `StackConcat` — see `gen_struct_variant_helpers_borrow`) unchanged; a
/// unit variant contributes zero arms (`MapArmBase`) and resolves the
/// instant the tag selects it; a newtype variant's inner type must itself be
/// map-shaped (`MapFieldProvider`), the same restriction internally-tagged
/// enums already impose on their standalone (non-flatten) dispatch. Tuple
/// variants are rejected here: their standalone dispatch already requires a
/// map-shaped payload, and no map-shaped tuple helper exists.
///
/// `#[strede(other)]` is not supported through this arm (never a flatten
/// candidate, identical to the externally-tagged precedent above).
///
/// Tuple variants and enums mixing tagged with `#[strede(untagged)]`
/// variants are structurally unsupported (the former: no map-shaped tuple
/// helper exists; the latter: `CandidateArmStack`'s `NoTag`/untagged support
/// is deferred — TESTING_GAPS.md item #3(B-2)). Neither is rejected via a
/// hard `syn::Error` here, because this impl is emitted unconditionally for
/// *every* internally-tagged enum (mirroring the externally-tagged
/// precedent) regardless of whether it's ever actually used as a flatten
/// target — hard-erroring here would break compilation of such an enum's
/// ordinary standalone derive. Instead, both cases bake an unconditionally
/// unsatisfiable bound into this impl's where-clause (a tuple variant's
/// helper type genuinely has no `MapFieldProvider` impl; a mixed-untagged
/// enum gets `(): FlattenUnsupported`, which nothing implements). Rust does
/// not eagerly check where-clause satisfiability at impl-definition time, so
/// the enum's own derive still compiles; only an actual attempt to flatten
/// such an enum fails, with a "trait bound not satisfied" error surfacing at
/// the flattening struct's own derive-generated code — i.e. "at the point
/// flatten is applied", not at this enum's definition.
fn gen_enum_candidate_map_field_provider_borrow(
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

    // --- impl generics: 'de, __KP2: MapKeyProbe<'de> ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        let has_de = mfp_impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
        if !has_de {
            mfp_impl_gen.params.insert(0, syn::parse_quote!('de));
        }
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbe<'de>));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::Deserialize<
                        'de,
                        <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            if has_untagged_mix {
                // Never satisfied - see the function doc comment above. Must
                // bind to `__KP2` (still abstract here), not a concrete type
                // like `()` - a concrete never-implemented bound is proven
                // impossible eagerly, breaking this enum's own derive; a
                // bound on a still-generic parameter defers to the first
                // actual flatten use site instead.
                wc.predicates
                    .push(syn::parse_quote!(__KP2: #krate::FlattenUnsupported));
            }
            for (_, cv) in &candidates {
                match &cv.kind {
                    VariantKind::Struct(_) => {
                        let helper_name = format_ident!("__Variant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProvider<'de, __KP2>
                        ));
                    }
                    VariantKind::Newtype(ty) => {
                        for lt in borrow_lifetimes(ty, &None) {
                            wc.predicates.push(syn::parse_quote!('de: #lt));
                        }
                        wc.predicates.push(syn::parse_quote!(
                            #ty: #krate::MapFieldProvider<'de, __KP2>
                        ));
                    }
                    VariantKind::Tuple(_) => {
                        // No map-shaped tuple helper exists - this bound is
                        // never satisfiable, same rationale as `FlattenUnsupported`
                        // above but via the tuple helper's genuinely-absent impl.
                        let helper_name = format_ident!("__TupleVariant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProvider<'de, __KP2>
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
                    quote! { <#ty as #krate::MapFieldProvider<'de, __KP2>>::ARMS }
                }
                VariantKind::Struct(_) => {
                    let helper_name = format_ident!("__Variant{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::ARMS }
                }
                VariantKind::Tuple(_) => {
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::ARMS }
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
                    #i => <#ty as #krate::MapFieldProvider<'de, __KP2>>::make_arms()
                        => |__o| <#ty as #krate::MapFieldProvider<'de, __KP2>>::from_outputs(__o).map(#name::#vname)
                },
                VariantKind::Struct(fields) => {
                    let helper_name = format_ident!("__Variant{}", cv.index);
                    let field_names: Vec<_> =
                        fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::from_outputs(__o)
                                .map(|__v| #name::#vname { #( #field_names: __v.#field_names, )* })
                    }
                }
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
                    let field_indices: Vec<syn::Index> =
                        (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::from_outputs(__o)
                                .map(|__v| #name::#vname( #( __v.#field_indices, )* ))
                    }
                }
            }
        })
        .collect();

    // ---- compile-time wire-name collision detection (see strede::Fields) ---
    // Mirrors the existing runtime `WireNames`/`wire_names()` precedent just
    // above (`[(&'static str, usize); 1]`, tag field only): candidate variant
    // fields are *not* recursed into, exactly like `DetectDuplicates` already
    // doesn't check across candidate boundaries (candidates race the same key
    // stream and are disambiguated by the tag, not by wire-name uniqueness —
    // see the accepted declaration-order tie-break behavior documented on
    // `CandidateArmStack`). Only the tag field name itself is a genuinely
    // fixed, always-present wire name this enum contributes.
    let (fields_impl_generics, _, fields_where_clause) = orig_generics.split_for_impl();
    let fields_impl_tokens = quote! {
        impl #fields_impl_generics #krate::Fields for #name #ty_generics
            #fields_where_clause
        {
            const NAMES: &'static [&'static str] = &[ #tag_field ];
        }
    };

    quote! {
        #fields_impl_tokens

        impl #mfp_impl_generics #krate::MapFieldProvider<'de, __KP2> for #name #ty_generics
            #mfp_where_clause
        {
            type Outputs = ::core::option::Option<#name #ty_generics>;
            const ARMS: usize = #arms_const_tokens;
            type WireNames = [(&'static str, usize); 1];

            fn wire_names() -> Self::WireNames {
                [(#tag_field, 0usize)]
            }

            fn make_arms() -> impl #krate::MapArmStack<'de, __KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                #krate::CandidateArmStack!(
                    #krate::candidate_arms! { #( #candidate_pieces, )* },
                    #tag_field,
                    [#( #tag_cands_tokens, )*],
                    __KP2,
                    #krate::borrow::VP<'de, __KP2>
                )
            }

            fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                __outputs
            }
        }
    }
}

/// Emit `MapFieldProvider<'de, __KP2>` for a purely untagged enum
/// (`#[strede(untagged)]`, no `tag`) so it can be used as a
/// `#[strede(flatten)]` field's type.
///
/// Unlike internally-tagged, there is no discriminant key at all: every
/// candidate variant's own fields race directly against the parent's shared
/// key stream from round one, via the `NoTagCandidateArmStack` runtime
/// primitive (`strede/src/map_arm/{mod,borrow,owned}.rs`). A candidate is
/// permanently excluded from the race the first round some *other* live
/// candidate's arms recognize a key that this candidate's own arms do not -
/// proof this candidate can't be the real variant. See CLAUDE.md's "Untagged
/// flatten" section for the full design.
///
/// Only struct-shaped and map-shaped-newtype candidates are viable - unlike
/// internally-tagged, unit variants are *also* rejected here (a unit
/// candidate contributes zero arms, so it would be trivially "always fully
/// satisfied" from round zero with no way to select it, absent a tag).
/// Neither restriction is enforced via a hard `syn::Error`, since this impl
/// is emitted unconditionally for every purely-untagged enum regardless of
/// whether it's ever actually flattened (mirroring the tag-based provider's
/// own precedent): a unit variant anywhere pushes an unconditionally
/// unsatisfiable `__KP2: FlattenUnsupported` bound (deferring the error to
/// the first actual flatten use site, not this enum's own definition); a
/// tuple variant's helper has no `MapFieldProvider` impl at all, so its own
/// per-variant bound is unconditionally unsatisfiable the same way
/// internally-tagged's tuple rejection already works.
fn gen_enum_candidate_map_field_provider_untagged_borrow(
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

    // --- impl generics: 'de, __KP2: MapKeyProbe<'de> ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        let has_de = mfp_impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
        if !has_de {
            mfp_impl_gen.params.insert(0, syn::parse_quote!('de));
        }
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbe<'de>));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::Deserialize<
                        'de,
                        <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            if has_unit {
                // Never satisfied - see the function doc comment above. Must
                // bind to `__KP2` (still abstract here), not a concrete type
                // like `()` - see `gen_enum_candidate_map_field_provider_borrow`'s
                // identical `has_untagged_mix` handling for why.
                wc.predicates
                    .push(syn::parse_quote!(__KP2: #krate::FlattenUnsupported));
            }
            for (_, cv) in &candidates {
                match &cv.kind {
                    VariantKind::Struct(_) => {
                        let helper_name = format_ident!("__Variant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProvider<'de, __KP2>
                        ));
                    }
                    VariantKind::Newtype(ty) => {
                        for lt in borrow_lifetimes(ty, &None) {
                            wc.predicates.push(syn::parse_quote!('de: #lt));
                        }
                        wc.predicates.push(syn::parse_quote!(
                            #ty: #krate::MapFieldProvider<'de, __KP2>
                        ));
                    }
                    VariantKind::Tuple(_) => {
                        // No map-shaped tuple helper exists - never
                        // satisfiable, same trick as the internally-tagged
                        // provider's own tuple rejection.
                        let helper_name = format_ident!("__TupleVariant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::MapFieldProvider<'de, __KP2>
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
                    quote! { <#ty as #krate::MapFieldProvider<'de, __KP2>>::ARMS }
                }
                VariantKind::Struct(_) => {
                    let helper_name = format_ident!("__Variant{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::ARMS }
                }
                VariantKind::Tuple(_) => {
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
                    quote! { <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::ARMS }
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
                    #i => <#ty as #krate::MapFieldProvider<'de, __KP2>>::make_arms()
                        => |__o| <#ty as #krate::MapFieldProvider<'de, __KP2>>::from_outputs(__o).map(#name::#vname)
                },
                VariantKind::Struct(fields) => {
                    let helper_name = format_ident!("__Variant{}", cv.index);
                    let field_names: Vec<_> =
                        fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::from_outputs(__o)
                                .map(|__v| #name::#vname { #( #field_names: __v.#field_names, )* })
                    }
                }
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
                    let field_indices: Vec<syn::Index> =
                        (0..fields.len()).map(syn::Index::from).collect();
                    quote! {
                        #i => <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::make_arms()
                            => |__o| <#helper_name as #krate::MapFieldProvider<'de, __KP2>>::from_outputs(__o)
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

    // ---- compile-time wire-name collision detection (see strede::Fields) ---
    // An untagged enum has no discriminant key at all — nothing is a fixed
    // wire name here, mirroring the runtime `WireNames = [(); 0]` precedent
    // directly below (candidate fields are not recursed into, same rationale
    // as the tagged candidate providers above).
    let (fields_impl_generics, _, fields_where_clause) = orig_generics.split_for_impl();
    let fields_impl_tokens = quote! {
        impl #fields_impl_generics #krate::Fields for #name #ty_generics
            #fields_where_clause
        {
            const NAMES: &'static [&'static str] = &[];
        }
    };

    quote! {
        #fields_impl_tokens

        impl #mfp_impl_generics #krate::MapFieldProvider<'de, __KP2> for #name #ty_generics
            #mfp_where_clause
        {
            type Outputs = ::core::option::Option<#name #ty_generics>;
            const ARMS: usize = #arms_const_tokens;
            type WireNames = [(&'static str, usize); 0];

            fn wire_names() -> Self::WireNames {
                []
            }

            fn make_arms() -> impl #krate::MapArmStack<'de, __KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                #krate::NoTagCandidateArmStack::new(#candidates_expr)
            }

            fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                __outputs
            }
        }
    }
}

/// Generates the `MapFieldProvider<'de, __KP2>` impl for an adjacently-tagged
/// enum (`#[strede(tag = "t", content = "c")]`) used as a
/// `#[strede(flatten)]` target.
///
/// Unlike internally-tagged's `CandidateArmStack` (whose arm count scales
/// with every candidate's own field count, since fields are spread into the
/// shared parent key space), this contributes exactly 2 fixed arms
/// regardless of variant count: one for `tag_field`, one for `content_field`.
/// Per-candidate arms sharing the identical `content_field` key would let the
/// first-declared candidate always win the key race regardless of type (see
/// `race_keys`'s declaration-order tie-break) - the content arm instead
/// always races every non-unit candidate's `deserialize_value::<CandidateType>()`
/// against forked copies of the same value probe, sequentially trying each
/// in turn (safe in the borrow family only - the whole buffer is already
/// materialized, no streaming deadlock risk; see `gen_untagged_probe_chain_borrow`
/// for the same convention). The tag/content cross-check is deferred to
/// `from_outputs`, mirroring the standalone adjacent-tagged path's own final
/// `match (opt_tag, opt_content) { ... }` check.
#[allow(clippy::too_many_arguments)]
fn gen_enum_candidate_map_field_provider_adjacent_borrow(
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

    // --- impl generics: 'de, __KP2: MapKeyProbe<'de> ---
    let mut mfp_impl_gen = orig_generics.clone();
    {
        let has_de = mfp_impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
        if !has_de {
            mfp_impl_gen.params.insert(0, syn::parse_quote!('de));
        }
        mfp_impl_gen
            .params
            .push(syn::parse_quote!(__KP2: #krate::MapKeyProbe<'de>));
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::Deserialize<
                        'de,
                        <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            if has_untagged_mix {
                // Never satisfied - same rationale as the internally-tagged
                // provider's own `has_untagged_mix` handling (see
                // `gen_enum_candidate_map_field_provider_borrow` above and
                // CLAUDE.md's flatten section): must bind to the still-abstract
                // `__KP2`, not a concrete type, so this enum's own derive
                // still compiles and only an actual flatten attempt surfaces
                // the error.
                wc.predicates
                    .push(syn::parse_quote!(__KP2: #krate::FlattenUnsupported));
            }
            for &(_, cv) in &nonunit_variants {
                match &cv.kind {
                    VariantKind::Struct(_) => {
                        let helper_name = format_ident!("__Variant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::Deserialize<
                                'de,
                                <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                    VariantKind::Tuple(_) => {
                        let helper_name = format_ident!("__TupleVariant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            #helper_name: #krate::Deserialize<
                                'de,
                                <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                    VariantKind::Newtype(ty) => {
                        for lt in borrow_lifetimes(ty, &None) {
                            wc.predicates.push(syn::parse_quote!('de: #lt));
                        }
                        wc.predicates.push(syn::parse_quote!(
                            #ty: #krate::Deserialize<
                                'de,
                                <#krate::borrow::VP<'de, __KP2> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
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

    // --- content arm: sequential try-in-order race over non-unit candidates ---
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
                    let helper_name = format_ident!("__Variant{}", cv.index);
                    let field_names: Vec<_> =
                        fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                    (
                        quote! { #helper_name },
                        quote! { #name::#vname { #( #field_names: __v.#field_names, )* } },
                    )
                }
                VariantKind::Tuple(fields) => {
                    let helper_name = format_ident!("__TupleVariant{}", cv.index);
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
                match #vp_ident.deserialize_value::<#de_type>(()).await? {
                    #krate::Probe::Hit((__vc, __v)) => {
                        return ::core::result::Result::Ok(
                            #krate::Probe::Hit((__vc, (#krate::Match, (#local_idx, #construction))))
                        );
                    }
                    #krate::Probe::Miss => {}
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

    // ---- compile-time wire-name collision detection (see strede::Fields) ---
    // Mirrors the runtime `WireNames` precedent below (`[tag, content]`, no
    // candidate recursion) — see `gen_enum_candidate_map_field_provider_borrow`'s
    // identical rationale. `content_field` is always opaque (never merged),
    // and `tag_field` is the only genuinely fixed wire name; a self-dup check
    // is meaningful here since a misconfigured container could set the two to
    // the same string.
    let (fields_impl_generics, _, fields_where_clause) = orig_generics.split_for_impl();
    let self_dup_check = if orig_generics.type_params().next().is_none() {
        quote! { const _: () = #krate::NoInternalDuplicates::<#name #ty_generics>::CHECK; }
    } else {
        quote! {}
    };
    let fields_impl_tokens = quote! {
        impl #fields_impl_generics #krate::Fields for #name #ty_generics
            #fields_where_clause
        {
            const NAMES: &'static [&'static str] = &[ #tag_field, #content_field ];
        }
        #self_dup_check
    };

    quote! {
        #fields_impl_tokens

        impl #mfp_impl_generics #krate::MapFieldProvider<'de, __KP2> for #name #ty_generics
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

            fn make_arms() -> impl #krate::MapArmStack<'de, __KP2, Outputs = Self::Outputs, Dynamic = #krate::False> {
                (
                    (#krate::MapArmBase,
                     #krate::MapArmSlot::new(
                         |mut __kp: __KP2, _i: usize| __kp.deserialize_key::<#krate::Match>(#tag_field),
                         |__vp: #krate::borrow::VP<'de, __KP2>, __k| async move {
                             let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                 #krate::MatchVals<usize, [(&'static str, usize); #tag_cands_count]>
                             >([#( #tag_cands_entries, )*]).await);
                             ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                         },
                     )),
                    #krate::MapArmSlot::new(
                        |mut __kp: __KP2, _i: usize| __kp.deserialize_key::<#krate::Match>(#content_field),
                        move |#vp_mut __vp: #krate::borrow::VP<'de, __KP2>, __k| async move {
                            let _ = &__k;
                            #unused_vp_guard
                            #( #fork_decls )*
                            #( #content_race_arms )*
                            ::core::result::Result::Ok(#krate::Probe::Miss)
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

/// All untagged — emit two-impl pattern via `deserialize_value_by_shape`.
fn expand_enum_untagged_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    _container_attrs: &crate::common::ContainerAttrs,
    _orig_generics: &syn::Generics,
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
/// Mixed tagged + untagged — emit two-impl pattern.
///
/// All arms (tagged-unit by name, tagged-nonunit by name+payload, untagged by shape)
/// race concurrently in a single `iterate` call. Declaration order determines priority.
fn expand_enum_with_untagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> syn::Result<TokenStream2> {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

    // Collect payload types for untagged variants for where-clause bounds.
    let untagged_payload_types: Vec<syn::Type> = classified
        .iter()
        .filter(|cv| cv.untagged)
        .map(|cv| untagged_payload_type(cv))
        .collect();

    // Payload types for tagged non-unit variants.
    let tagged_nonunit_payload_types: Vec<syn::Type> = classified
        .iter()
        .filter(|cv| !cv.untagged && !cv.other && !matches!(cv.kind, VariantKind::Unit))
        .map(|cv| match &cv.kind {
            VariantKind::Newtype(ty) => syn::parse_quote!(#ty),
            VariantKind::Struct(_) => {
                let id = format_ident!("__Variant{}", cv.index);
                syn::parse_quote!(#id)
            }
            VariantKind::Tuple(_) => {
                let id = format_ident!("__TupleVariant{}", cv.index);
                syn::parse_quote!(#id)
            }
            VariantKind::Unit => unreachable!(),
        })
        .collect();

    // --- DeserializeFromEnum impl generics ---
    let mut enum_impl_gen = orig_generics.clone();
    insert_de_and_e_borrow(&mut enum_impl_gen, krate);
    {
        let wc = enum_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_borrow(krate, ident));
            }
            // Tagged non-unit payload bounds via PayloadDeserializer.
            for pty in &tagged_nonunit_payload_types {
                for lt in borrow_lifetimes(pty, &None) {
                    wc.predicates.push(syn::parse_quote!('de: #lt));
                }
                wc.predicates.push(syn::parse_quote!(
                    #pty: #krate::Deserialize<
                        'de,
                        <__E::VariantProbe as #krate::EnumVariantProbe<'de>>::PayloadDeserializer,
                        Extra = ()
                    >
                ));
            }
            // Untagged payload bounds via PayloadDeserializer.
            for pty in &untagged_payload_types {
                for lt in borrow_lifetimes(pty, &None) {
                    wc.predicates.push(syn::parse_quote!('de: #lt));
                }
                wc.predicates.push(syn::parse_quote!(
                    #pty: #krate::Deserialize<
                        'de,
                        <__E::VariantProbe as #krate::EnumVariantProbe<'de>>::PayloadDeserializer,
                        Extra = ()
                    >
                ));
            }
        }
    }
    let (enum_impl_generics, _, enum_where_clause) = enum_impl_gen.split_for_impl();

    // --- Deserialize<'de, __D> impl generics ---
    let mut de_impl_gen = orig_generics.clone();
    insert_de_and_d_borrow(&mut de_impl_gen, krate);
    {
        let wc = de_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_borrow(krate, ident));
            }
            wc.predicates.push(syn::parse_quote!(
                #name #ty_generics: #krate::DeserializeFromEnum<
                    'de,
                    <__D::Entry as #krate::Entry<'de>>::Enum,
                    Extra = ()
                >
            ));
        }
    }
    let (de_impl_generics, _, de_where_clause) = de_impl_gen.split_for_impl();

    // Build arm slots: tagged-unit, tagged-nonunit, untagged (in declaration order).
    let tagged_non_other: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && !cv.other)
        .collect();
    let untagged_variants: Vec<_> = classified.iter().filter(|cv| cv.untagged).collect();

    let mut arm_slots: Vec<TokenStream2> = Vec::new();
    let mut out_names: Vec<syn::Ident> = Vec::new();
    let mut result_arms: Vec<TokenStream2> = Vec::new();
    let mut arm_idx = 0usize;

    // Tagged arms.
    for cv in &tagged_non_other {
        let out = format_ident!("__out_ev{}", arm_idx);
        let vname = &cv.variant.ident;
        let mut candidates: Vec<(&str, usize)> = vec![(cv.wire_name.as_str(), arm_idx)];
        for alias in &cv.aliases {
            candidates.push((alias.as_str(), arm_idx));
        }
        let cands_tokens: Vec<TokenStream2> = candidates
            .iter()
            .map(|(wn, idx)| quote! { (#wn, #idx) })
            .collect();

        let cv_idx = cv.index;
        let slot = match &cv.kind {
            VariantKind::Unit => quote! {
                #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
                    let __vp2 = __vp.fork();
                    let (__claim, _) = #krate::hit!(#krate::select_probe! {
                        __vp.deserialize_unit_by_name([#( #cands_tokens, )*]),
                        __vp2.deserialize_unit_by_index(#cv_idx),
                    });
                    ::core::result::Result::Ok(#krate::Probe::Hit((__claim, ())))
                })
            },
            VariantKind::Newtype(ty) => quote! {
                #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
                    let __vp2 = __vp.fork();
                    let (__claim, _, __v) = #krate::hit!(#krate::select_probe! {
                        __vp.deserialize_payload_by_name::<#ty, _>([#( #cands_tokens, )*], ()),
                        __vp2.deserialize_payload_by_index::<#ty>(#cv_idx, ()),
                    });
                    ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)))
                })
            },
            VariantKind::Struct(_) => {
                let helper_name = format_ident!("__Variant{}", cv.index);
                quote! {
                    #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
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
                let helper_name = format_ident!("__TupleVariant{}", cv.index);
                quote! {
                    #krate::EnumArmSlot::new(|mut __vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
                        let __vp2 = __vp.fork();
                        let (__claim, _, __v) = #krate::hit!(#krate::select_probe! {
                            __vp.deserialize_payload_by_name::<#helper_name, _>([#( #cands_tokens, )*], ()),
                            __vp2.deserialize_payload_by_index::<#helper_name>(#cv_idx, ()),
                        });
                        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)))
                    })
                }
            }
        };
        let result = match &cv.kind {
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
                let fnames: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
                quote! {
                    if let ::core::option::Option::Some(__v) = #out {
                        return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname { #( #fnames: __v.#fnames, )* })));
                    }
                }
            }
            VariantKind::Tuple(fields) => {
                let fidxs: Vec<syn::Index> = (0..fields.len()).map(syn::Index::from).collect();
                quote! {
                    if let ::core::option::Option::Some(__v) = #out {
                        return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname( #( __v.#fidxs, )* ))));
                    }
                }
            }
        };
        arm_slots.push(slot);
        out_names.push(out);
        result_arms.push(result);
        arm_idx += 1;
    }

    // Untagged arms.
    for cv in &untagged_variants {
        let out = format_ident!("__out_ev{}", arm_idx);
        let vname = &cv.variant.ident;
        let pty = untagged_payload_type(cv);
        let slot = quote! {
            #krate::EnumArmSlot::new(|__vp: <__E as #krate::EnumAccess<'de>>::VariantProbe| async move {
                let (__claim, __v) = #krate::hit!(__vp.deserialize_value_by_shape::<#pty>(()).await);
                ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __v)))
            })
        };
        let result = untagged_hit_arm(name, vname, &cv.kind, &out, krate);
        arm_slots.push(slot);
        out_names.push(out);
        result_arms.push(result);
        arm_idx += 1;
    }

    let arms_expr = {
        let mut expr = quote! { #krate::EnumArmBase };
        for slot in &arm_slots {
            expr = quote! { (#expr, #slot) };
        }
        expr
    };
    let output_pat = {
        let mut pat = quote! { () };
        for out in &out_names {
            pat = quote! { (#pat, #out) };
        }
        pat
    };

    let other_arm = match other_variant(classified) {
        Some(vname) => {
            quote! { ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname))) }
        }
        None => quote! { ::core::result::Result::Ok(#krate::Probe::Miss) },
    };

    let tuple_helpers = gen_tuple_variant_helpers_borrow(classified, krate);
    let struct_helpers =
        gen_struct_variant_helpers_borrow(classified, krate, container_attrs.rename_all);

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, DeserializeFromEnum as _,
                Deserializer as _, Entry as _, EnumAccess as _, EnumVariantProbe as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #tuple_helpers
            #struct_helpers

            impl #enum_impl_generics #krate::DeserializeFromEnum<'de, __E>
                for #name #ty_generics
                #enum_where_clause
            {
                type Extra = ();
                async fn deserialize_from_enum(
                    __e: __E,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__E::Claim, Self)>, __E::Error>
                {
                    let __arms = #arms_expr;
                    match __e.iterate(__arms).await? {
                        #krate::Probe::Hit((__claim, #output_pat)) => {
                            #( #result_arms )*
                            #other_arm
                        }
                        #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                    }
                }
            }

            impl #de_impl_generics #krate::Deserialize<'de, __D>
                for #name #ty_generics
                #de_where_clause
            {
                type Extra = ();
                async fn deserialize(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    d.entry(|[__e]| async {
                        __e.deserialize_enum_into::<Self>(()).await
                    }).await
                }
            }
        };
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

/// The payload type used for an untagged variant in `deserialize_value_by_shape`.
fn untagged_payload_type(cv: &ClassifiedVariant) -> syn::Type {
    match &cv.kind {
        VariantKind::Unit => syn::parse_quote!(()),
        VariantKind::Newtype(ty) => syn::parse_quote!(#ty),
        VariantKind::Tuple(_) => {
            let id = format_ident!("__TupleVariant{}", cv.index);
            syn::parse_quote!(#id)
        }
        VariantKind::Struct(_) => {
            let id = format_ident!("__Variant{}", cv.index);
            syn::parse_quote!(#id)
        }
    }
}

/// Build the `if let Some(v) = out { return Ok(Hit((..., Name::Variant(...)))) }` arm
/// for one untagged variant's output slot.
fn untagged_hit_arm(
    name: &syn::Ident,
    vname: &syn::Ident,
    kind: &VariantKind,
    out: &syn::Ident,
    krate: &syn::Path,
) -> TokenStream2 {
    match kind {
        VariantKind::Unit => quote! {
            if let ::core::option::Option::Some(_) = #out {
                return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname)));
            }
        },
        VariantKind::Newtype(_) => quote! {
            if let ::core::option::Option::Some(__v) = #out {
                return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname(__v))));
            }
        },
        VariantKind::Struct(fields) => {
            let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
            quote! {
                if let ::core::option::Option::Some(__v) = #out {
                    return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname { #( #field_names: __v.#field_names, )* })));
                }
            }
        }
        VariantKind::Tuple(fields) => {
            let field_indices: Vec<syn::Index> = (0..fields.len()).map(syn::Index::from).collect();
            quote! {
                if let ::core::option::Option::Some(__v) = #out {
                    return ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname( #( __v.#field_indices, )* ))));
                }
            }
        }
    }
}

/// Generate `DeserializeFromEnum` + `Deserialize` impls for an internally tagged enum
/// (`#[strede(tag = "field")]`).
///
/// Supports unit variants, newtype variants, tuple variants, and struct variants.
fn expand_enum_internally_tagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> syn::Result<TokenStream2> {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

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

    // Regenerated in both branches (unconditionally, over the full
    // `classified` list) so struct/tuple helper types always exist for any
    // struct/tuple-kind variant - including an untagged non-unit variant
    // mixed alongside an otherwise all-unit tagged set, which the unit-only
    // branch below would otherwise reference in the flatten `MapFieldProvider`
    // impl without ever having defined.
    let flatten_provider = gen_enum_candidate_map_field_provider_borrow(
        name,
        classified,
        tag_field,
        krate,
        container_attrs,
        orig_generics,
    );

    let (de_call, helpers) = if !has_nonunit {
        let body = expand_borrow_internally_tagged_unit_only(
            name,
            classified,
            tag_field,
            &variant_candidates,
            krate,
        )?;
        let struct_helpers =
            gen_struct_variant_helpers_borrow(classified, krate, container_attrs.rename_all);
        let tuple_helpers = gen_tuple_variant_helpers_borrow(classified, krate);
        (
            quote! {
                d.entry(|[__e]| async {
                    let __e = #krate::hit!(__e.deserialize_map().await);
                    #body
                }).await
            },
            quote! {
                #struct_helpers
                #tuple_helpers
                #flatten_provider
            },
        )
    } else {
        let (body, helpers) = expand_borrow_internally_tagged_with_nonunit(
            name,
            classified,
            tag_field,
            &variant_candidates,
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

    // --- Build single Deserialize<'de, __D> impl ---
    let mut de_impl_gen = orig_generics.clone();
    insert_de_and_d_borrow(&mut de_impl_gen, krate);
    {
        let n_cands: usize = classified
            .iter()
            .filter(|cv| !cv.untagged)
            .map(|cv| 1 + cv.aliases.len())
            .sum();
        let wc = de_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_borrow(krate, ident));
            }
            for cv in classified {
                if cv.untagged {
                    continue;
                }
                match &cv.kind {
                    VariantKind::Newtype(ty) => {
                        wc.predicates.push(syn::parse_quote!(
                            for<'__v> #ty: #krate::DeserializeFromMap<
                                'de,
                                #krate::TagAwareMap<
                                    'de, '__v,
                                    <__D::Entry as #krate::Entry<'de>>::Map,
                                    [(&'static str, usize); #n_cands],
                                >,
                                Extra = (),
                            >
                        ));
                    }
                    VariantKind::Struct(_) => {
                        let helper_name = format_ident!("__Variant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            for<'__v> #helper_name: #krate::DeserializeFromMap<
                                'de,
                                #krate::TagAwareMap<
                                    'de, '__v,
                                    <__D::Entry as #krate::Entry<'de>>::Map,
                                    [(&'static str, usize); #n_cands],
                                >,
                                Extra = (),
                            >
                        ));
                    }
                    VariantKind::Tuple(_) => {
                        let helper_name = format_ident!("__TupleVariant{}", cv.index);
                        wc.predicates.push(syn::parse_quote!(
                            for<'__v> #helper_name: #krate::DeserializeFromMap<
                                'de,
                                #krate::TagAwareMap<
                                    'de, '__v,
                                    <__D::Entry as #krate::Entry<'de>>::Map,
                                    [(&'static str, usize); #n_cands],
                                >,
                                Extra = (),
                            >
                        ));
                    }
                    VariantKind::Unit => {}
                }
            }
        }
    }
    let (de_impl_generics, _, de_where_clause) = de_impl_gen.split_for_impl();

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #helpers

            impl #de_impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #de_where_clause {
                type Extra = ();
                async fn deserialize(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    #de_call
                }
            }
        };
    })
}

/// Unit-only internally-tagged enum body (borrow family).
///
/// Receives an already-opened map (`__e: __M: MapAccess<'de>`) and runs tag-capture iteration.
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
        {
            let mut __map = __e;
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
        }
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

    let mut all_entry_idents: Vec<syn::Ident> = Vec::new();
    let mut select_arms: Vec<TokenStream2> = Vec::new();

    for (arm_i, &(local_idx, cv)) in nonunit_variants.iter().enumerate() {
        let vname = &cv.variant.ident;
        let entry_ident = format_ident!("__e_{}", arm_i);
        all_entry_idents.push(entry_ident.clone());

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

        select_arms.push(quote! {
            async move {
                let __map = #krate::hit!(#entry_ident.deserialize_map().await);
                let __tag_cell: ::core::cell::Cell<::core::option::Option<usize>> =
                    ::core::cell::Cell::new(::core::option::Option::None);
                let __m = #krate::TagAwareMap::new(
                    __map,
                    #tag_field,
                    [#( #tag_cands_entries, )*],
                    #local_idx,
                    &__tag_cell,
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

    // Unit arm: handles all unit variants and the `other` fallback.
    let needs_unit_arm = !unit_variants.is_empty() || other_variant(classified).is_some();
    if needs_unit_arm {
        let unit_entry_ident = format_ident!("__e_unit");
        all_entry_idents.push(unit_entry_ident.clone());
        let str_match = unit_str_match_arms(name, classified, krate);
        let n_cands = tag_cands_entries.len();

        select_arms.push(quote! {
            async move {
                let mut __map = #krate::hit!(#unit_entry_ident.deserialize_map().await);
                let __tag_cell: ::core::cell::Cell<::core::option::Option<usize>> =
                    ::core::cell::Cell::new(::core::option::Option::None);
                let __tag_candidates: [(&'static str, usize); #n_cands] = [#( #tag_cands_entries, )*];
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
                        let __tag_candidates2: [(&'static str, usize); #n_cands] = [#( #tag_cands_entries, )*];
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

// ---------------------------------------------------------------------------
// Adjacent-tagged enum derive  (#[strede(tag = "t", content = "c")])
// ---------------------------------------------------------------------------

/// Generate `DeserializeFromEnum` + `Deserialize` impls for an adjacently tagged enum
/// (`#[strede(tag = "t", content = "c")]`, borrow family).
///
/// Wire format: `{"t": "VariantName", "c": <payload>}` (key order-independent).
/// Unit variants have no content field: `{"t": "VariantName"}`.
fn expand_enum_adjacent_tagged_borrow(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    content_field: &str,
    krate: &syn::Path,
    container_attrs: &crate::common::ContainerAttrs,
    orig_generics: &syn::Generics,
) -> syn::Result<TokenStream2> {
    let (_, ty_generics, _) = orig_generics.split_for_impl();

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

    let (de_call, helpers) = if !has_nonunit {
        // Unit-only: same as internally tagged (no content field needed).
        let body = expand_borrow_internally_tagged_unit_only(
            name,
            classified,
            tag_field,
            &variant_candidates,
            krate,
        )?;
        (
            quote! {
                d.entry(|[__e]| async {
                    let __e = #krate::hit!(__e.deserialize_map().await);
                    #body
                }).await
            },
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

    // Emitted unconditionally (regardless of unit-only vs. with-nonunit
    // above) so this enum can be used as someone else's `#[strede(flatten)]`
    // target no matter which dispatch shape its own standalone path takes -
    // mirrors internally-tagged's identical rationale.
    let flatten_provider = gen_enum_candidate_map_field_provider_adjacent_borrow(
        name,
        classified,
        tag_field,
        content_field,
        krate,
        container_attrs,
        orig_generics,
    );
    let helpers = quote! { #helpers #flatten_provider };

    // --- Build single Deserialize<'de, __D> impl ---
    let mut de_impl_gen = orig_generics.clone();
    insert_de_and_d_borrow(&mut de_impl_gen, krate);
    {
        let wc = de_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in orig_generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_borrow(krate, ident));
            }
            // Adjacent-tagged non-unit helpers must be Deserialize via MapValue.
            for cv in classified {
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
                    VariantKind::Newtype(ty) => {
                        syn::parse_quote!(#ty)
                    }
                    VariantKind::Unit => continue,
                };
                wc.predicates.push(syn::parse_quote!(
                    #helper_ty: #krate::Deserialize<
                        'de,
                        <#krate::borrow::VP2<'de, __D> as #krate::MapValueProbe<'de>>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
        }
    }
    let (de_impl_generics, _, de_where_clause) = de_impl_gen.split_for_impl();

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                MapAccess as _, MapKeyProbe as _, MapValueProbe as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #helpers

            impl #de_impl_generics #krate::Deserialize<'de, __D> for #name #ty_generics #de_where_clause {
                type Extra = ();
                async fn deserialize(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    #de_call
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

    let mut all_entry_idents: Vec<syn::Ident> = Vec::new();
    let mut select_arms: Vec<TokenStream2> = Vec::new();

    for (arm_i, &(local_idx, cv)) in nonunit_variants.iter().enumerate() {
        let vname = &cv.variant.ident;
        let entry_ident = format_ident!("__e_{}", arm_i);
        all_entry_idents.push(entry_ident.clone());

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

        select_arms.push(quote! {
            async move {
                let mut __map = #krate::hit!(#entry_ident.deserialize_map().await);
                let __arms = {
                    let __inner_arms = (
                        (#krate::MapArmBase,
                         #krate::MapArmSlot::new(
                             |mut __kp: #krate::borrow::KP<'de, __D>, _i: usize| {
                                 __kp.deserialize_key::<#krate::Match>(#tag_field)
                             },
                             |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                                 let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                     #krate::MatchVals<usize, [(&'static str, usize); #tag_cands_count]>
                                 >([#( #tag_cands_entries, )*]).await);
                                 ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                             },
                         )),
                        #krate::MapArmSlot::new(
                            |mut __kp: #krate::borrow::KP<'de, __D>, _i: usize| {
                                __kp.deserialize_key::<#krate::Match>(#content_field)
                            },
                            |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#de_type>(()).await);
                                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                            },
                        )
                    );
                    let __wn = #dup_wire_names;
                    let __dd = #krate::DetectDuplicates::new(
                        __inner_arms,
                        __wn,
                        move |__kp: #krate::borrow::KP<'de, __D>, _i: usize| __kp.deserialize_key::<#krate::MatchVals<usize, _>>(__wn),
                        |__vp: #krate::borrow::VP2<'de, __D>| __vp.skip(),
                    );
                    (__dd, #krate::VirtualArmSlot::new(
                        |__kp: #krate::borrow::KP<'de, __D>, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
                        |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                            use #krate::MapValueProbe as _;
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

    // Unit variant arm: iterate the map looking for the tag field only.
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
                let __unit_arms = (
                    (#krate::MapArmBase,
                     #krate::MapArmSlot::new(
                         |mut __kp: #krate::borrow::KP<'de, __D>, _i: usize| {
                             __kp.deserialize_key::<#krate::Match>(#tag_field)
                         },
                         |__vp: #krate::borrow::VP2<'de, __D>, __k| async move {
                             let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<
                                 #krate::MatchVals<usize, [(&'static str, usize); #tag_cands_count]>
                             >([#( #tag_cands_entries, )*]).await);
                             ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v))))
                         },
                     )),
                    #krate::VirtualArmSlot::new(
                        |__kp: #krate::borrow::KP<'de, __D>, _i: usize| __kp.deserialize_key::<#krate::Skip>(()),
                        |__vp: #krate::borrow::VP2<'de, __D>, _k: #krate::Skip| async move {
                            use #krate::MapValueProbe as _;
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
