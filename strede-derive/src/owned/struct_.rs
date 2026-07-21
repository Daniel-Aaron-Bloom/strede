use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

use super::gen_container_from_owned;
use crate::common::{
    DefaultAttr, FieldContext, apply_field_bound, classify_fields, field_bound_owned,
    insert_d_owned, type_param_bound_owned,
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
        let d_ident = format_ident!("__D");
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
                    |ty| field_bound_owned(krate, ty, FieldContext::Direct, &d_ident),
                );
                if let Some(ft) = &transparent_cf.from {
                    wc.predicates.push(field_bound_owned(
                        krate,
                        ft,
                        FieldContext::Direct,
                        &d_ident,
                    ));
                } else if let Some(ft) = &transparent_cf.try_from {
                    wc.predicates.push(field_bound_owned(
                        krate,
                        ft,
                        FieldContext::Direct,
                        &d_ident,
                    ));
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

    // ---- tuple struct: gather seq variables used by DeserializeFromSeqOwned impl ----
    // Named structs have acc_names=[] and dfs_impl=empty; tuple structs populate both.

    let (acc_names, dfs_impl) = if !is_named {
        let acc_names: Vec<syn::Ident> = (0..field_types.len())
            .map(|i| format_ident!("__f{}", i))
            .collect();

        // Non-skipped field types (and their classified attrs) for generic bounds.
        let de_field_types_and_cfs: Vec<_> = field_types
            .iter()
            .zip(classified_fields.iter())
            .filter(|(_, cf)| !cf.skip_deserializing)
            .collect();

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

        // Build DeserializeFromSeqOwned<__S> where-clause: bounds on __S::Elem::SubDeserializer.
        let mut dfs_gen = input.generics.clone();
        dfs_gen
            .params
            .push(syn::parse_quote!(__S: #krate::SeqAccessOwned));
        {
            let wc = dfs_gen.make_where_clause();
            if let Some(preds) = &container_attrs.bound {
                wc.predicates.extend(preds.iter().cloned());
            } else {
                for tp in input.generics.type_params() {
                    let ident = &tp.ident;
                    wc.predicates.push(syn::parse_quote!(
                        #ident: #krate::DeserializeOwned<
                            <<__S as #krate::SeqAccessOwned>::Elem as #krate::SeqEntryOwned>::SubDeserializer,
                            Extra = ()
                        >
                    ));
                }
                for (ty, cf) in &de_field_types_and_cfs {
                    let has_custom = cf.deserialize_owned_with.is_some()
                        || cf.from.is_some()
                        || cf.try_from.is_some();
                    apply_field_bound(wc, ty, &cf.bound, has_custom, |t| {
                        syn::parse_quote!(
                            #t: #krate::DeserializeOwned<
                                <<__S as #krate::SeqAccessOwned>::Elem as #krate::SeqEntryOwned>::SubDeserializer,
                                Extra = ()
                            >
                        )
                    });
                    if let Some(ft) = &cf.from {
                        wc.predicates.push(syn::parse_quote!(
                            #ft: #krate::DeserializeOwned<
                                <<__S as #krate::SeqAccessOwned>::Elem as #krate::SeqEntryOwned>::SubDeserializer,
                                Extra = ()
                            >
                        ));
                    } else if let Some(ft) = &cf.try_from {
                        wc.predicates.push(syn::parse_quote!(
                            #ft: #krate::DeserializeOwned<
                                <<__S as #krate::SeqAccessOwned>::Elem as #krate::SeqEntryOwned>::SubDeserializer,
                                Extra = ()
                            >
                        ));
                    }
                }
            }
        }
        let (dfs_impl_generics, _, dfs_where_clause) = dfs_gen.split_for_impl();

        let acc_names_tok = acc_names.clone();
        let dfs = quote! {
            impl #dfs_impl_generics #krate::DeserializeFromSeqOwned<__S> for #name #ty_generics
                #dfs_where_clause
            {
                type Extra = ();
                async fn deserialize_from_seq_owned(
                    mut __seq: __S,
                    _extra: (),
                ) -> ::core::result::Result<
                    #krate::Probe<(<__S as #krate::SeqAccessOwned>::SeqClaim, Self)>,
                    <__S as #krate::SeqAccessOwned>::Error,
                > {
                    #( #seq_reads )*
                    let __v = #krate::hit!(__seq.next::<1, _, _, ()>(|[__se]| async {
                        ::core::result::Result::Ok(#krate::Probe::Miss)
                    }).await);
                    let __claim = #krate::or_miss!(__v.done());
                    ::core::result::Result::Ok(#krate::Probe::Hit((
                        __claim,
                        #name( #( #acc_names_tok, )* ),
                    )))
                }
            }
        };

        (acc_names, dfs)
    } else {
        (vec![], quote! {})
    };

    // ---- named struct (map-based, existing codegen) -------------------------

    // For named structs use the real field ident; for tuple structs use __f0, __f1, ... (same as
    // acc_names computed above, which serve as local variable names throughout codegen).
    let field_names: Vec<syn::Ident> = if is_named {
        fields.iter().map(|f| f.ident.clone().unwrap()).collect()
    } else {
        acc_names.clone()
    };

    // Filtered views: only non-skipped fields participate in deserialization.
    // Split into regular (arm-based) and flatten fields.
    let de_field_names: Vec<_> = field_names
        .iter()
        .zip(classified_fields.iter())
        .filter(|(_, cf)| !cf.skip_deserializing && cf.flatten == crate::common::FlattenMode::None)
        .map(|(n, _)| n)
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
        .map(|((n, t), cf)| (n as &syn::Ident, *t, cf.flatten))
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

    // Walk all fields in declaration order, classifying each as Skip / Regular / Flatten.
    enum FieldKind<'a> {
        Skip,
        Regular { reg_idx: usize },
        Flatten { ty: &'a syn::Type },
    }
    let field_kinds: Vec<FieldKind> = {
        let mut reg_idx = 0usize;
        field_names
            .iter()
            .zip(classified_fields.iter())
            .zip(field_types.iter())
            .map(|((_n, cf), ty)| {
                if cf.skip_deserializing {
                    FieldKind::Skip
                } else if cf.flatten == crate::common::FlattenMode::None {
                    let r = reg_idx;
                    reg_idx += 1;
                    FieldKind::Regular { reg_idx: r }
                } else {
                    FieldKind::Flatten { ty }
                }
            })
            .collect()
    };

    enum Segment<'a> {
        Regular(Vec<usize>),
        Flatten { ty: &'a syn::Type },
    }
    let segments: Vec<Segment> = {
        let mut out: Vec<Segment> = vec![];
        let mut cur_reg: Vec<usize> = vec![];
        for kind in &field_kinds {
            match kind {
                FieldKind::Skip => {}
                FieldKind::Regular { reg_idx } => cur_reg.push(*reg_idx),
                FieldKind::Flatten { ty } => {
                    if !cur_reg.is_empty() {
                        out.push(Segment::Regular(core::mem::take(&mut cur_reg)));
                    }
                    out.push(Segment::Flatten { ty });
                }
            }
        }
        if !cur_reg.is_empty() {
            out.push(Segment::Regular(cur_reg));
        }
        out
    };

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
                FieldKind::Skip => {}
                FieldKind::Regular { .. } => terms.push(quote! { 1usize }),
                FieldKind::Flatten { ty, .. } => terms.push(quote! {
                    <#ty as #krate::MapFieldProviderOwned<__KP>>::ARMS
                }),
            }
        }
        out
    };

    let key_type_for = |cf: &crate::common::ClassifiedField| -> TokenStream2 {
        let n = 1 + cf.aliases.len();
        if n == 1 {
            quote! { #krate::Match }
        } else {
            quote! { #krate::MatchVals<(), [(&'static str, ()); #n]> }
        }
    };

    let build_arm_slot = |reg_idx: usize| -> TokenStream2 {
        let cf = de_classified[reg_idx];
        let val_type = &de_value_types[reg_idx];
        let val_conv = &de_value_conversions[reg_idx];
        let mut wire_names: Vec<&str> = vec![cf.wire_name.as_str()];
        for alias in &cf.aliases {
            wire_names.push(alias.as_str());
        }
        let key_fn = if wire_names.len() == 1 {
            let wname = wire_names[0];
            quote! {
                |mut __kp: __KP, __i: usize| async move {
                    let __kp2 = __kp.fork();
                    #krate::select_probe! {
                        __kp.deserialize_key::<#krate::Match>(#wname),
                        async move {
                            let (__kc, ()) = #krate::hit!(__kp2.deserialize_key_by_index(__i).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((__kc, #krate::Match)))
                        },
                    }
                }
            }
        } else {
            quote! {
                |mut __kp: __KP, __i: usize| async move {
                    let __kp2 = __kp.fork();
                    #krate::select_probe! {
                        __kp.deserialize_key::<#krate::MatchVals<(), _>>([#( (#wire_names, ()), )*]),
                        async move {
                            let (__kc, ()) = #krate::hit!(__kp2.deserialize_key_by_index(__i).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((
                                __kc,
                                #krate::MatchVals((), ::core::marker::PhantomData),
                            )))
                        },
                    }
                }
            }
        };
        let val_fn = quote! {
            |__vp: #krate::owned::VP<__KP>, __k| async move {
                let (__vc, __v) = #krate::hit!(__vp.deserialize_value::<#val_type>(()).await);
                ::core::result::Result::Ok(#krate::Probe::Hit((__vc, (__k, __v #val_conv))))
            }
        };
        quote! { #krate::MapArmSlot::new(#key_fn, #val_fn) }
    };

    // --- type Outputs ---
    let outputs_type_tokens: TokenStream2 = {
        if segments.is_empty() {
            quote! { () }
        } else {
            let mut acc: Option<TokenStream2> = None;
            for seg in &segments {
                let seg_out = match seg {
                    Segment::Regular(regs) => {
                        let mut t = quote! { () };
                        for r in regs {
                            let cf = de_classified[*r];
                            let kt = key_type_for(cf);
                            let vt = &de_value_types[*r];
                            t = quote! { (#t, ::core::option::Option<(#kt, #vt)>) };
                        }
                        t
                    }
                    Segment::Flatten { ty } => quote! {
                        <#ty as #krate::MapFieldProviderOwned<__KP>>::Outputs
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
                FieldKind::Skip => None,
                FieldKind::Regular { .. } => Some(quote! { 1usize }),
                FieldKind::Flatten { ty, .. } => Some(quote! {
                    <#ty as #krate::MapFieldProviderOwned<__KP>>::ARMS
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
                Segment::Regular(regs) => {
                    let n: usize = regs
                        .iter()
                        .map(|r| 1 + de_classified[*r].aliases.len())
                        .sum();
                    quote! { [(&'static str, usize); #n] }
                }
                Segment::Flatten { ty } => quote! {
                    <<#ty as #krate::MapFieldProviderOwned<__KP>>::WireNames
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

    // --- fn wire_names() body ---
    let wire_names_body_tokens: TokenStream2 = {
        let mut field_iter = field_kinds.iter().enumerate();
        let mut acc: Option<TokenStream2> = None;
        for seg in &segments {
            let piece = match seg {
                Segment::Regular(regs) => {
                    let mut entries: Vec<TokenStream2> = vec![];
                    for _ in 0..regs.len() {
                        loop {
                            let (i, kind) = field_iter.next().expect("regular field present");
                            if matches!(kind, FieldKind::Regular { .. }) {
                                let offset = &arm_offset_tokens[i];
                                let reg_idx = if let FieldKind::Regular { reg_idx } = kind {
                                    *reg_idx
                                } else {
                                    unreachable!()
                                };
                                let cf = de_classified[reg_idx];
                                let primary = &cf.wire_name;
                                entries.push(quote! { (#primary, #offset) });
                                for a in &cf.aliases {
                                    entries.push(quote! { (#a, #offset) });
                                }
                                break;
                            }
                        }
                    }
                    let n = entries.len();
                    quote! { [#( #entries ),*] as [(&'static str, usize); #n] }
                }
                Segment::Flatten { ty } => loop {
                    let (i, kind) = field_iter.next().expect("flatten field present");
                    if matches!(kind, FieldKind::Flatten { .. }) {
                        let offset = &arm_offset_tokens[i];
                        break quote! {
                            <#ty as #krate::MapFieldProviderOwned<__KP>>::wire_names()
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

    // --- fn make_arms() body ---
    let make_arms_body_tokens: TokenStream2 = {
        let mut acc: Option<TokenStream2> = None;
        for seg in &segments {
            let piece = match seg {
                Segment::Regular(regs) => {
                    let mut t = quote! { #krate::MapArmBase };
                    for r in regs {
                        let slot = build_arm_slot(*r);
                        t = quote! { (#t, #slot) };
                    }
                    t
                }
                Segment::Flatten { ty } => quote! {
                    <#ty as #krate::MapFieldProviderOwned<__KP>>::make_arms()
                },
            };
            acc = Some(match acc {
                None => piece,
                Some(prev) => quote! { #krate::StackConcat(#prev, #piece) },
            });
        }
        acc.unwrap_or_else(|| quote! { #krate::MapArmBase })
    };

    // --- fn from_outputs() body ---
    let from_outputs_body_tokens: TokenStream2 = {
        let seg_out_names: Vec<syn::Ident> = (0..segments.len())
            .map(|i| format_ident!("__seg_out_{}", i))
            .collect();
        let outer_pat: TokenStream2 = if seg_out_names.is_empty() {
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

        let mut seg_stmts: Vec<TokenStream2> = vec![];
        let mut field_iter = field_kinds.iter().enumerate();
        for (seg_i, seg) in segments.iter().enumerate() {
            let seg_out = &seg_out_names[seg_i];
            match seg {
                Segment::Regular(regs) => {
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
                            if let FieldKind::Regular { reg_idx } = kind {
                                let cf = de_classified[*reg_idx];
                                let fname = de_field_names[*reg_idx];
                                let opt_ident = format_ident!("__opt_{}", reg_idx);
                                let val_conv = &de_value_conversions[*reg_idx];
                                let none_branch: TokenStream2 = match &cf.default {
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
                                        ::core::option::Option::Some((_, __v)) => __v #val_conv,
                                        ::core::option::Option::None => #none_branch,
                                    };
                                });
                                break;
                            }
                        }
                    }
                }
                Segment::Flatten { ty } => {
                    loop {
                        let (_i, kind) = field_iter.next().expect("flatten");
                        if let FieldKind::Flatten { .. } = kind {
                            break;
                        }
                    }
                    let fname = {
                        let prior_flat = segments[..seg_i]
                            .iter()
                            .filter(|s| matches!(s, Segment::Flatten { .. }))
                            .count();
                        flatten_fields[prior_flat].0
                    };
                    seg_stmts.push(quote! {
                        let #fname = match <#ty as #krate::MapFieldProviderOwned<__KP>>
                            ::from_outputs(#seg_out)
                        {
                            ::core::option::Option::Some(__v) => __v,
                            ::core::option::Option::None => return ::core::option::Option::None,
                        };
                    });
                }
            }
        }

        let skip_stmts: Vec<TokenStream2> = field_names
            .iter()
            .zip(classified_fields.iter())
            .filter(|(_, cf)| cf.skip_deserializing)
            .map(|(fname, cf)| {
                let default_expr: TokenStream2 = match &cf.default {
                    Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                    Some(DefaultAttr::Expr(expr)) => {
                        quote! { #krate::DefaultWrapper(#expr).value() }
                    }
                    None => unreachable!("validated in classify_fields"),
                };
                quote! { let #fname = #default_expr; }
            })
            .collect();

        let construction = if is_named {
            quote! { #name { #( #field_names, )* } }
        } else {
            quote! { #name( #( #field_names, )* ) }
        };

        quote! {
            let #outer_pat = __outputs;
            #( #seg_stmts )*
            #( #skip_stmts )*
            ::core::option::Option::Some(#construction)
        }
    };

    // ------------------------------------------------------------------
    // Where-clause builders
    // ------------------------------------------------------------------

    let mut mfp_impl_gen = input.generics.clone();
    mfp_impl_gen
        .params
        .push(syn::parse_quote!(__KP: #krate::MapKeyProbeOwned));
    {
        let wc = mfp_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in input.generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(syn::parse_quote!(
                    #ident: #krate::DeserializeOwned<
                        <#krate::owned::VP<__KP> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                        Extra = ()
                    >
                ));
            }
            for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                let has_custom = cf.deserialize_owned_with.is_some()
                    || cf.from.is_some()
                    || cf.try_from.is_some();
                apply_field_bound(wc, ty, &cf.bound, has_custom, |t| {
                    syn::parse_quote!(
                        #t: #krate::DeserializeOwned<
                            <#krate::owned::VP<__KP> as #krate::MapValueProbeOwned>::ValueSubDeserializer,
                            Extra = ()
                        >
                    )
                });
            }
            for (_, flat_ty, _) in &flatten_fields {
                wc.predicates.push(syn::parse_quote!(
                    #flat_ty: #krate::MapFieldProviderOwned<__KP>
                ));
                wc.predicates.push(syn::parse_quote!(
                    <<#flat_ty as #krate::MapFieldProviderOwned<__KP>>::WireNames
                        as #krate::ConcatableArray>::OtherArray<(&'static str, usize)>:
                        ::core::marker::Copy
                ));
            }
        }
    }
    let (mfp_impl_generics, _, mfp_where_clause) = mfp_impl_gen.split_for_impl();

    // DeserializeFromMapOwned impl where-clause.
    let mut dfm_impl_gen = input.generics.clone();
    crate::common::insert_m_owned(&mut dfm_impl_gen, krate);
    {
        let wc = dfm_impl_gen.make_where_clause();
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
            wc.predicates.push(syn::parse_quote!(
                #name #ty_generics: #krate::MapFieldProviderOwned<
                    <__M as #krate::MapAccessOwned>::KeyProbe
                >
            ));
        }
    }
    let (dfm_impl_generics, _, dfm_where_clause) = dfm_impl_gen.split_for_impl();

    // DeserializeOwned impl where-clause.
    let mut de_impl_gen = input.generics.clone();
    insert_d_owned(&mut de_impl_gen, krate);
    {
        let wc = de_impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for tp in input.generics.type_params() {
                let ident = &tp.ident;
                wc.predicates.push(type_param_bound_owned(krate, ident));
            }
            wc.predicates.push(syn::parse_quote!(
                #name #ty_generics: #krate::DeserializeFromMapOwned<
                    <__D::Entry as #krate::EntryOwned>::Map,
                    Extra = ()
                >
            ));
            if !is_named {
                wc.predicates.push(syn::parse_quote!(
                    #name #ty_generics: #krate::DeserializeFromSeqOwned<
                        <__D::Entry as #krate::EntryOwned>::Seq,
                        Extra = ()
                    >
                ));
            }
        }
    }
    let (de_impl_generics, _, de_where_clause) = de_impl_gen.split_for_impl();

    let dfm_arms_expr: TokenStream2 = {
        let mut e = quote! {
            #krate::DetectDuplicatesOwned!(
                <Self as #krate::MapFieldProviderOwned<<__M as #krate::MapAccessOwned>::KeyProbe>>::make_arms(),
                <Self as #krate::MapFieldProviderOwned<<__M as #krate::MapAccessOwned>::KeyProbe>>::wire_names(),
                <__M as #krate::MapAccessOwned>::KeyProbe,
                #krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>
            )
        };
        if allow_unknown_fields {
            e = quote! {
                #krate::SkipUnknownOwned!(
                    #e,
                    <__M as #krate::MapAccessOwned>::KeyProbe,
                    #krate::owned::VP<<__M as #krate::MapAccessOwned>::KeyProbe>
                )
            };
        }
        e
    };

    let de_body = if is_named {
        quote! {
            d.entry(|[__e]| async {
                __e.deserialize_map_into::<Self>(()).await
            }).await
        }
    } else {
        quote! {
            d.entry(|[__e1, __e2]| async {
                #krate::select_probe!(biased;
                    __e1.deserialize_map_into::<Self>(()),
                    __e2.deserialize_seq_into::<Self>(()),
                )
            }).await
        }
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                ConcatableArray as _, DefaultValue as _, DeserializeFromSeqOwned as _,
                DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapAccessOwned as _, MapKeyProbeOwned as _, MapValueProbeOwned as _,
                SeqAccessOwned as _, SeqEntryOwned as _, StrAccessOwned as _,
            };

            #de_with_wrappers

            #dfs_impl

            impl #mfp_impl_generics #krate::MapFieldProviderOwned<__KP> for #name #ty_generics
                #mfp_where_clause
            {
                type Outputs = #outputs_type_tokens;
                const ARMS: usize = #arms_const_tokens;
                type WireNames = #wire_names_type_tokens;
                fn wire_names() -> Self::WireNames {
                    #wire_names_body_tokens
                }
                fn make_arms() -> impl #krate::MapArmStackOwned<__KP, Outputs = Self::Outputs, Dynamic = #krate::False> {
                    #make_arms_body_tokens
                }
                fn from_outputs(__outputs: Self::Outputs) -> ::core::option::Option<Self> {
                    #from_outputs_body_tokens
                }
            }

            impl #dfm_impl_generics #krate::DeserializeFromMapOwned<__M> for #name #ty_generics
                #dfm_where_clause
            {
                type Extra = ();
                async fn deserialize_from_map_owned(
                    __map: __M,
                    _extra: (),
                ) -> ::core::result::Result<
                    #krate::Probe<(<__M as #krate::MapAccessOwned>::MapClaim, Self)>,
                    <__M as #krate::MapAccessOwned>::Error,
                > {
                    let __arms = #dfm_arms_expr;
                    match __map.iterate(__arms).await? {
                        #krate::Probe::Hit((__claim, __outputs)) => {
                            match <Self as #krate::MapFieldProviderOwned<
                                <__M as #krate::MapAccessOwned>::KeyProbe,
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

            impl #de_impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics
                #de_where_clause
            {
                type Extra = ();
                async fn deserialize_owned(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    #de_body
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
