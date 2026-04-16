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
    let has_s = impl_gen.lifetimes().any(|l| l.lifetime.ident == "s");
    if !has_s {
        impl_gen.params.insert(0, syn::parse_quote!('s));
    }
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            wc.predicates
                .push(syn::parse_quote!(#from_ty: #krate::DeserializeOwned<'s>));
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

            impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    let (__c, __v) = #krate::hit!(<#from_ty as #krate::DeserializeOwned<'s>>::deserialize(d, ()).await);
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
                let mut impl_gen = input.generics.clone();
                let has_s = impl_gen.lifetimes().any(|l| l.lifetime.ident == "s");
                if !has_s {
                    impl_gen.params.insert(0, syn::parse_quote!('s));
                }
                let (impl_generics, _, where_clause) = impl_gen.split_for_impl();
                return Ok(quote! {
                    #[allow(unreachable_code)]
                    const _: () = {
                        use #krate::{DefaultValue as _, DeserializerOwned as _, EntryOwned as _};

                        impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                            async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
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
        let has_s = impl_gen.lifetimes().any(|l| l.lifetime.ident == "s");
        if !has_s {
            impl_gen.params.insert(0, syn::parse_quote!('s));
        }
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
                    |ty| syn::parse_quote!(#ty: #krate::DeserializeOwned<'s>),
                );
                if let Some(ft) = &transparent_cf.from {
                    wc.predicates
                        .push(syn::parse_quote!(#ft: #krate::DeserializeOwned<'s>));
                } else if let Some(ft) = &transparent_cf.try_from {
                    wc.predicates
                        .push(syn::parse_quote!(#ft: #krate::DeserializeOwned<'s>));
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
                    let (__c, __tmp) = #krate::hit!(<#from_ty as #krate::DeserializeOwned<'s>>::deserialize(d, ()).await);
                    (__c, <#transparent_ty as ::core::convert::From<#from_ty>>::from(__tmp))
                }
            }
        } else if let Some(try_from_ty) = &transparent_cf.try_from {
            quote! {
                {
                    let (__c, __tmp) = #krate::hit!(<#try_from_ty as #krate::DeserializeOwned<'s>>::deserialize(d, ()).await);
                    (__c, #krate::or_miss!(<#transparent_ty as ::core::convert::TryFrom<#try_from_ty>>::try_from(__tmp).ok()))
                }
            }
        } else {
            quote! { #krate::hit!(<#transparent_ty as #krate::DeserializeOwned<'s>>::deserialize(d, ()).await) }
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

                impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                    async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
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
        let has_s = impl_gen.lifetimes().any(|l| l.lifetime.ident == "s");
        if !has_s {
            impl_gen.params.insert(0, syn::parse_quote!('s));
        }
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
                        |t| syn::parse_quote!(#t: #krate::DeserializeOwned<'s>),
                    );
                    if let Some(ft) = &cf.from {
                        wc.predicates
                            .push(syn::parse_quote!(#ft: #krate::DeserializeOwned<'s>));
                    } else if let Some(ft) = &cf.try_from {
                        wc.predicates
                            .push(syn::parse_quote!(#ft: #krate::DeserializeOwned<'s>));
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

                impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                    async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
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
    let acc_names: Vec<_> = field_names
        .iter()
        .map(|n| format_ident!("__f_{}", n))
        .collect();
    let field_finalizers =
        gen_field_finalizers(&field_names, &acc_names, &classified_fields, krate);

    // Filtered views: only non-skipped fields participate in deserialization.
    let de_field_names: Vec<_> = field_names
        .iter()
        .zip(classified_fields.iter())
        .filter(|(_, cf)| !cf.skip_deserializing)
        .map(|(n, _)| *n)
        .collect();
    let de_field_types: Vec<_> = field_types
        .iter()
        .zip(classified_fields.iter())
        .filter(|(_, cf)| !cf.skip_deserializing)
        .map(|(t, _)| *t)
        .collect();
    let de_field_strs: Vec<_> = classified_fields
        .iter()
        .filter(|cf| !cf.skip_deserializing)
        .map(|cf| cf.wire_name.clone())
        .collect();
    let de_acc_names: Vec<_> = acc_names
        .iter()
        .zip(classified_fields.iter())
        .filter(|(_, cf)| !cf.skip_deserializing)
        .map(|(a, _)| a.clone())
        .collect();
    let de_field_count = de_field_names.len();
    let de_field_indices: Vec<_> = (0..de_field_count).collect();

    // Build candidates for the chunk matcher: primary names + aliases, each mapping
    // back to the field's index in the de_field_* arrays.
    let de_field_candidates: Vec<(String, usize)> = classified_fields
        .iter()
        .filter(|cf| !cf.skip_deserializing)
        .enumerate()
        .flat_map(|(idx, cf)| {
            let mut pairs = vec![(cf.wire_name.clone(), idx)];
            for alias in &cf.aliases {
                pairs.push((alias.clone(), idx));
            }
            pairs
        })
        .collect();

    // For deserialize_owned_with / from / try_from: compute value types and conversions.
    let de_classified: Vec<_> = classified_fields
        .iter()
        .filter(|cf| !cf.skip_deserializing)
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

    // Build impl generics: prepend 's if not already present, add DeserializeOwned<'s> bounds.
    let mut impl_gen = input.generics.clone();
    let has_s = impl_gen.lifetimes().any(|l| l.lifetime.ident == "s");
    if !has_s {
        impl_gen.params.insert(0, syn::parse_quote!('s));
    }
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
                    |t| syn::parse_quote!(#t: #krate::DeserializeOwned<'s>),
                );
            }
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();


    let known_field_arms: Vec<TokenStream2> = de_field_indices
        .iter()
        .zip(
            de_field_names
                .iter()
                .zip(de_value_types.iter().zip(de_value_conversions.iter())),
        )
        .map(|(idx, (n, (vt, vc)))| {
            let hit_val = if allow_unknown_fields {
                quote! { ::core::option::Option::Some(__Field::#n(__v #vc)) }
            } else {
                quote! { __Field::#n(__v #vc) }
            };
            quote! {
                #idx => {
                    let (__c, __v) = #krate::hit!(__ve.value::<#vt, _>(()).await);
                    ::core::result::Result::Ok(
                        #krate::Probe::Hit((__c, #hit_val))
                    )
                }
            }
        })
        .collect();

    let unknown_field_arm = if allow_unknown_fields {
        quote! {
            _ => {
                let __c = __ve.skip().await?;
                ::core::result::Result::Ok(
                    #krate::Probe::Hit((__c, ::core::option::Option::None))
                )
            }
        }
    } else {
        quote! {
            _ => {
                ::core::result::Result::Ok(
                    #krate::Probe::<(_, __Field #ty_generics)>::Miss,
                )
            }
        }
    };

    let data_arm_body = if allow_unknown_fields {
        quote! {
            if let ::core::option::Option::Some(__field) = __field {
                match __field {
                    #(
                        __Field::#de_field_names(__v) => {
                            if #de_acc_names.is_some() {
                                return ::core::result::Result::Err(
                                    <__D::Error as #krate::DeserializeError>::duplicate_field(#de_field_strs)
                                );
                            }
                            #de_acc_names = ::core::option::Option::Some(__v);
                        }
                    )*
                }
            }
        }
    } else {
        quote! {
            match __field {
                #(
                    __Field::#de_field_names(__v) => {
                        if #de_acc_names.is_some() {
                            return ::core::result::Result::Err(
                                <__D::Error as #krate::DeserializeError>::duplicate_field(#de_field_strs)
                            );
                        }
                        #de_acc_names = ::core::option::Option::Some(__v);
                    }
                )*
            }
        }
    };

    let field_key_sentinel = if allow_unknown_fields { Some(de_field_count) } else { None };
    let (field_key_type, field_key_extra, field_key_idx) =
        key_matcher_tokens(&de_field_candidates, field_key_sentinel, krate);

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapAccessOwned as _, MapKeyEntryOwned as _, MapValueEntryOwned as _,
                SeqAccessOwned as _, SeqEntryOwned as _, StrAccessOwned as _,
            };

            #de_with_wrappers

            impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    d.entry(|[__e]| async {
                        let mut __map = #krate::hit!(__e.deserialize_map().await);

                        #[allow(non_camel_case_types)]
                        enum __Field #ty_generics {
                            #( #de_field_names(#de_field_types), )*
                        }

                        #(
                            let mut #de_acc_names: ::core::option::Option<#de_field_types> =
                                ::core::option::Option::None;
                        )*

                        let __claim = loop {
                            match #krate::hit!(__map.next_kv(|[__ke]| async {
                                let (__c, _k, __field) = #krate::hit!(__ke.key(
                                    #field_key_extra,
                                    |__k: &#field_key_type, [__ve]| {
                                        let __idx = #field_key_idx;
                                        async move {
                                            match __idx {
                                                #( #known_field_arms )*
                                                #unknown_field_arm
                                            }
                                        }
                                    },
                                ).await);
                                ::core::result::Result::Ok(
                                    #krate::Probe::Hit((__c, __field)),
                                )
                            }).await) {
                                #krate::Chunk::Data((__map_back, __field)) => {
                                    #data_arm_body

                                    __map = __map_back;
                                }
                                #krate::Chunk::Done(__c) => break __c,
                            }
                        };

                        #( #field_finalizers )*

                        ::core::result::Result::Ok(
                            #krate::Probe::Hit((__claim, #name { #( #field_names, )* }))
                        )
                    }).await
                }
            }
        };
    })
}

/// Generate per-field finalization: extract from Option, applying defaults where configured.
/// Skipped fields have no accumulator and always use their default.
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
                    ::core::option::Option::Some(__v) => __v,
                    ::core::option::Option::None => #none_branch,
                };
            }
        })
        .collect()
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
                impl<'s> #krate::DeserializeOwned<'s> for #wrapper {
                    async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
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
                impl<'s> #krate::DeserializeOwned<'s> for #wrapper
                where #from_ty: #krate::DeserializeOwned<'s>
                {
                    async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__c, __v) = #krate::hit!(<#from_ty as #krate::DeserializeOwned<'s>>::deserialize(d, ()).await);
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
                impl<'s> #krate::DeserializeOwned<'s> for #wrapper
                where #try_from_ty: #krate::DeserializeOwned<'s>
                {
                    async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                        d: __D,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        let (__c, __v) = #krate::hit!(<#try_from_ty as #krate::DeserializeOwned<'s>>::deserialize(d, ()).await);
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
    let has_s = impl_gen.lifetimes().any(|l| l.lifetime.ident == "s");
    if !has_s {
        impl_gen.params.insert(0, syn::parse_quote!('s));
    }
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            for ty in &field_types {
                wc.predicates
                    .push(syn::parse_quote!(#ty: #krate::DeserializeOwned<'s>));
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
        return expand_owned_enum_internally_tagged(
            name,
            &classified,
            tag_field,
            &variant_candidates,
            variant_key_sentinel,
            krate,
            &impl_generics,
            &ty_generics,
            where_clause,
        );
    }

    let body = if !has_untagged {
        if has_tagged_unit && !has_tagged_nonunit {
            expand_owned_enum_unit_only(name, &classified, variant_key_sentinel, krate)?
        } else if !has_tagged_unit && has_tagged_nonunit {
            expand_owned_enum_map_only(name, &classified, &variant_candidates, variant_key_sentinel, krate)?
        } else {
            expand_owned_enum_mixed(name, &classified, &variant_candidates, variant_key_sentinel, krate)?
        }
    } else if !has_tagged_unit && !has_tagged_nonunit {
        expand_owned_enum_untagged_only(name, &classified, krate)?
    } else {
        expand_owned_enum_with_untagged(name, &classified, &variant_candidates, variant_key_sentinel, krate)?
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
                MapAccessOwned as _, MapKeyEntryOwned as _, MapValueEntryOwned as _,
                SeqAccessOwned as _, SeqEntryOwned as _, StrAccessOwned as _,
            };

            #tuple_variant_helpers
            #struct_variant_helpers

            impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
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
/// - `key_type`  — the type to annotate `__k` with in the closure
/// - `extra_expr` — the extra value to pass as the first arg to `.key()`
/// - `idx_access` — how to extract the matched `usize` from `__k` in the closure body
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
    let indices_lit: Vec<proc_macro2::Literal> = indices.iter().map(|i| proc_macro2::Literal::usize_suffixed(*i)).collect();
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

                impl<'s> #krate::DeserializeOwned<'s> for #helper_name
                where
                    #( #field_types: #krate::DeserializeOwned<'s>, )*
                {
                    async fn deserialize<__D2: #krate::DeserializerOwned<'s>>(
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
    let sentinel = if has_other { Some(variant_key_sentinel) } else { None };
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
    let map_body = gen_owned_enum_map_body(name, classified, variant_candidates, variant_key_sentinel, krate);
    Ok(quote! {
        d.entry(|[__e]| async {
            let mut __map = #krate::hit!(__e.deserialize_map().await);
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
    // Build candidates for known unit str variants only (no sentinel — unknown strings
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
    // No sentinel — plain MatchVals; if the token is not a matching str it returns Miss
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

    let map_body = gen_owned_enum_map_body(name, classified, variant_candidates, variant_key_sentinel, krate);

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
                        let mut __map = #krate::hit!(__e3.deserialize_map().await);
                        #map_body
                    },
                    miss => ::core::result::Result::Ok(#krate::Probe::Miss),
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
                        let mut __map = #krate::hit!(__e2.deserialize_map().await);
                        #map_body
                    },
                    miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                }
            }).await
        })
    }
}

/// Generate the body that reads a single-key map for non-unit variant dispatch (owned family).
fn gen_owned_enum_map_body(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_candidates: &[(String, usize)],
    variant_key_sentinel: usize,
    krate: &syn::Path,
) -> TokenStream2 {
    let value_dispatch_arms =
        gen_owned_value_dispatch_arms(name, classified, variant_key_sentinel, krate);
    let (variant_key_type, variant_key_extra, variant_key_idx) =
        key_matcher_tokens(variant_candidates, Some(variant_key_sentinel), krate);

    quote! {
        // Read the single key-value pair.
        let __chunk = #krate::hit!(__map.next_kv(|[__ke]| async {
            let (__c, _k, __v) = #krate::hit!(__ke.key(
                #variant_key_extra,
                |__k: &#variant_key_type, [__ve]| {
                    let __idx = #variant_key_idx;
                    async move {
                        #value_dispatch_arms
                    }
                },
            ).await);
            ::core::result::Result::Ok(#krate::Probe::Hit((__c, __v)))
        }).await);
        let (__map_back, __enum_val) = #krate::or_miss!(__chunk.data());

        // Drain the map — expect Done.
        let mut __map = __map_back;
        let __chunk = #krate::hit!(__map.next_kv::<1, _, _, ()>(|[__ke]| async {
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await);
        let __claim = #krate::or_miss!(__chunk.done());
        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __enum_val)))
    }
}

/// Generate value dispatch arms for the owned-family map key match.
fn gen_owned_value_dispatch_arms(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    _sentinel: usize,
    krate: &syn::Path,
) -> TokenStream2 {
    // Build arms using tagged-variant local indices (matching __VariantKey output).
    let tagged: Vec<_> = classified.iter().filter(|cv| !cv.untagged).collect();
    let arms: Vec<_> = tagged.iter().enumerate().filter_map(|(idx, cv)| {
        let vname = &cv.variant.ident;
        match &cv.kind {
            VariantKind::Newtype(ty) => Some(quote! {
                #idx => {
                    let (__c, __v) = #krate::hit!(__ve.value::<#ty, _>(()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname(__v))))
                }
            }),
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__VariantOwned{}", cv.index);
                let field_names: Vec<_> = fields.iter()
                    .map(|f| f.ident.as_ref().unwrap())
                    .collect();
                Some(quote! {
                    #idx => {
                        let (__c, __v) = #krate::hit!(__ve.value::<#helper_name, _>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname { #( #field_names: __v.#field_names, )* })))
                    }
                })
            }
            VariantKind::Tuple(fields) => {
                let helper_name = format_ident!("__TupleVariantOwned{}", cv.index);
                let field_indices: Vec<syn::Index> = (0..fields.len())
                    .map(syn::Index::from)
                    .collect();
                Some(quote! {
                    #idx => {
                        let (__c, __v) = #krate::hit!(__ve.value::<#helper_name, _>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname( #( __v.#field_indices, )* ))))
                    }
                })
            }
            VariantKind::Unit => None,
        }
    }).collect();

    let map_wildcard = match other_variant(classified) {
        Some(vname) => quote! {
            _ => {
                let __claim = __ve.skip().await?;
                ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name::#vname)))
            }
        },
        None => quote! {
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        },
    };
    quote! {
        match __idx {
            #( #arms )*
            #map_wildcard
        }
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
            let field_strs: Vec<_> = cf.iter().map(|f| f.wire_name.clone()).collect();
            let field_count = field_names.len();
            let field_indices: Vec<_> = (0..field_count).collect();
            let acc_names: Vec<_> = field_names
                .iter()
                .map(|n| format_ident!("__f_{}", n))
                .collect();
            let field_finalizers = gen_field_finalizers(&field_names, &acc_names, &cf, krate);
            // No sentinel: struct variant fields don't allow unknown, unknown = Miss
            let field_candidates: Vec<(String, usize)> = cf
                .iter()
                .enumerate()
                .flat_map(|(idx, f)| {
                    let mut pairs = vec![(f.wire_name.clone(), idx)];
                    for alias in &f.aliases {
                        pairs.push((alias.clone(), idx));
                    }
                    pairs
                })
                .collect();
            let (field_key_type, field_key_extra, field_key_idx) =
                key_matcher_tokens(&field_candidates, None, krate);

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name {
                    #( #field_names: #field_types, )*
                }

                impl<'s> #krate::DeserializeOwned<'s> for #helper_name
                where
                    #( #field_types: #krate::DeserializeOwned<'s>, )*
                {
                    async fn deserialize<__D2: #krate::DeserializerOwned<'s>>(
                        d: __D2,
                        _extra: (),
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.entry(|[__e]| async {
                            let mut __map = #krate::hit!(__e.deserialize_map().await);

                            #( let mut #acc_names: ::core::option::Option<#field_types> = ::core::option::Option::None; )*

                            let __claim = loop {
                                match #krate::hit!(__map.next_kv(|[__ke]| async {
                                    let (__c, _k, __fv) = #krate::hit!(__ke.key(
                                        #field_key_extra,
                                        |__k: &#field_key_type, [__ve]| {
                                            let __fidx = #field_key_idx;
                                            async move {
                                                match __fidx {
                                                    #(
                                                        #field_indices => {
                                                            let (__c, __v) = #krate::hit!(__ve.value::<#field_types, _>(()).await);
                                                            ::core::result::Result::Ok(
                                                                #krate::Probe::Hit((__c, (#field_indices, __v)))
                                                            )
                                                        }
                                                    )*
                                                    _ => ::core::result::Result::Ok(#krate::Probe::Miss),
                                                }
                                            }
                                        },
                                    ).await);
                                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, __fv)))
                                }).await) {
                                    #krate::Chunk::Data((__map_back, __fv)) => {
                                        match __fv.0 {
                                            #(
                                                #field_indices => {
                                                    if #acc_names.is_some() {
                                                        return ::core::result::Result::Err(
                                                            <__D2::Error as #krate::DeserializeError>::duplicate_field(#field_strs)
                                                        );
                                                    }
                                                    #acc_names = ::core::option::Option::Some(__fv.1);
                                                }
                                            )*
                                            _ => unreachable!(),
                                        }
                                        __map = __map_back;
                                    }
                                    #krate::Chunk::Done(__c) => break __c,
                                }
                            };

                            #( #field_finalizers )*

                            ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #helper_name { #( #field_names, )* })))
                        }).await
                    }
                }
            });
        }
    }
    tokens
}

/// All untagged — try each variant by shape (owned family).
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
        // No other variant in the with_untagged case — unknown string = Miss
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
        let map_body = gen_owned_enum_map_body(name, classified, variant_candidates, variant_key_sentinel, krate);
        quote! {
            match #h.deserialize_map().await? {
                #krate::Probe::Hit(mut __map) => {
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
fn expand_owned_enum_internally_tagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    tag_field: &str,
    variant_candidates: &[(String, usize)],
    variant_key_sentinel: usize,
    krate: &syn::Path,
    impl_generics: impl quote::ToTokens,
    ty_generics: impl quote::ToTokens,
    where_clause: impl quote::ToTokens,
) -> syn::Result<TokenStream2> {
    let has_nonunit = classified
        .iter()
        .any(|cv| !matches!(cv.kind, VariantKind::Unit));
    if has_nonunit {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[strede(tag)] with non-unit variants is not supported yet",
        ));
    }

    // __TagKey: matches tag_field → 0, anything else → sentinel(1).
    let tag_candidates = vec![(tag_field.to_string(), 0usize)];
    let (tag_key_type, tag_key_extra, tag_key_idx) =
        key_matcher_tokens(&tag_candidates, Some(1), krate);
    let (variant_key_type, variant_key_extra, variant_key_idx) =
        key_matcher_tokens(variant_candidates, Some(variant_key_sentinel), krate);

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

    let body = quote! {
        d.entry(|[__e]| async {
            let mut __map = #krate::hit!(__e.deserialize_map().await);
            let mut __matched = #variant_key_sentinel;

            let __claim = loop {
                match #krate::hit!(__map.next_kv(|[__ke]| async {
                    let (__c, _k, __ov) = #krate::hit!(__ke.key(
                        #tag_key_extra,
                        |__k: &#tag_key_type, [__ve]| {
                            let __tag_idx = #tag_key_idx;
                            async move {
                                if __tag_idx == 0usize {
                                    let (__c, __k) = #krate::hit!(
                                        __ve.value::<#variant_key_type, _>(#variant_key_extra).await
                                    );
                                    ::core::result::Result::Ok(#krate::Probe::Hit((
                                        __c,
                                        ::core::option::Option::Some(#variant_key_idx),
                                    )))
                                } else {
                                    let __c = __ve.skip().await?;
                                    ::core::result::Result::Ok(#krate::Probe::Hit((
                                        __c,
                                        ::core::option::Option::None::<usize>,
                                    )))
                                }
                            }
                        },
                    ).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, __ov)))
                }).await) {
                    #krate::Chunk::Data((__map_back, ::core::option::Option::Some(__idx))) => {
                        __matched = __idx;
                        __map = __map_back;
                    }
                    #krate::Chunk::Data((__map_back, ::core::option::Option::None)) => {
                        __map = __map_back;
                    }
                    #krate::Chunk::Done(__c) => break __c,
                }
            };

            match __matched {
                #( #unit_match_arms )*
                #unit_wildcard
            }
        }).await
    };

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, DeserializeOwned as _, DeserializerOwned as _, EntryOwned as _,
                MapAccessOwned as _, MapKeyEntryOwned as _, MapValueEntryOwned as _,
                SeqAccessOwned as _, SeqEntryOwned as _, StrAccessOwned as _,
            };

            impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
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
