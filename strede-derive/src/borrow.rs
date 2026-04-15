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
                let has_custom = transparent_cf.deserialize_with.is_some()
                    || transparent_cf.from.is_some()
                    || transparent_cf.try_from.is_some();
                apply_field_bound(
                    wc,
                    transparent_ty,
                    &transparent_cf.bound,
                    has_custom,
                    |ty| syn::parse_quote!(#ty: #krate::Deserialize<'de>),
                );
                if let Some(ft) = &transparent_cf.from {
                    wc.predicates
                        .push(syn::parse_quote!(#ft: #krate::Deserialize<'de>));
                } else if let Some(ft) = &transparent_cf.try_from {
                    wc.predicates
                        .push(syn::parse_quote!(#ft: #krate::Deserialize<'de>));
                }
            }
        }
        let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

        // Generate deserialize expression for the transparent field.
        // Each arm produces `(__claim, __v)` where __v is the final field value.
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
                for (ty, cf) in &de_field_types_and_cfs {
                    let has_custom =
                        cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                    apply_field_bound(
                        wc,
                        ty,
                        &cf.bound,
                        has_custom,
                        |t| syn::parse_quote!(#t: #krate::Deserialize<'de>),
                    );
                    if let Some(ft) = &cf.from {
                        wc.predicates
                            .push(syn::parse_quote!(#ft: #krate::Deserialize<'de>));
                    } else if let Some(ft) = &cf.try_from {
                        wc.predicates
                            .push(syn::parse_quote!(#ft: #krate::Deserialize<'de>));
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

    // For deserialize_with / from / try_from: compute value type and conversion for each field.
    let de_classified: Vec<_> = classified_fields
        .iter()
        .filter(|cf| !cf.skip_deserializing)
        .collect();
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

    // Generate wrapper types for deserialize_with / from / try_from fields.
    let de_with_wrappers = gen_deserialize_with_wrappers_borrow(
        &de_field_names,
        &de_field_types,
        &de_classified,
        krate,
    );

    // Build the impl generics: prepend 'de if not already present, then add
    // Deserialize<'de> bounds for each non-skipped field type.
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
            for (ty, cf) in de_field_types.iter().zip(de_classified.iter()) {
                let has_custom =
                    cf.deserialize_with.is_some() || cf.from.is_some() || cf.try_from.is_some();
                apply_field_bound(
                    wc,
                    ty,
                    &cf.bound,
                    has_custom,
                    |t| syn::parse_quote!(#t: #krate::Deserialize<'de>),
                );
            }
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    // When allow_unknown_fields is set, the key callback returns Option<__Field>
    // instead of __Field, and the unknown key arm calls skip() instead of returning Miss.
    let de_field_aliases: Vec<Vec<String>> = classified_fields
        .iter()
        .filter(|cf| !cf.skip_deserializing)
        .map(|cf| cf.aliases.clone())
        .collect();
    let known_field_arms: Vec<TokenStream2> = de_field_strs
        .iter()
        .zip(de_field_aliases.iter())
        .zip(
            de_field_names
                .iter()
                .zip(de_value_types.iter().zip(de_value_conversions.iter())),
        )
        .map(|((s, aliases), (n, (vt, vc)))| {
            let hit_val = if allow_unknown_fields {
                quote! { ::core::option::Option::Some(__Field::#n(__v #vc)) }
            } else {
                quote! { __Field::#n(__v #vc) }
            };
            quote! {
                #s #( | #aliases )* => {
                    let (__c, __v) = #krate::hit!(__ve.value::<#vt, _>(()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, #hit_val)))
                }
            }
        })
        .collect();

    let unknown_field_arm = if allow_unknown_fields {
        quote! {
            _ => {
                let __c = __ve.skip().await?;
                ::core::result::Result::Ok(#krate::Probe::Hit((__c, ::core::option::Option::None)))
            }
        }
    } else {
        quote! { _ => ::core::result::Result::Ok(#krate::Probe::<(_, __Field #ty_generics)>::Miss) }
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

    Ok(quote! {
        #[allow(unreachable_code)]
        const _: () = {
            use #krate::{
                DefaultValue as _, Deserialize as _, Deserializer as _, Entry as _,
                MapAccess as _, MapKeyEntry as _, MapValueEntry as _,
                SeqAccess as _, SeqEntry as _, StrAccess as _,
            };

            #de_with_wrappers

        impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
            async fn deserialize<__D: #krate::Deserializer<'de>>(
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

                    #( let mut #de_acc_names: ::core::option::Option<#de_field_types> = ::core::option::Option::None; )*

                    let __claim = loop {
                        match #krate::hit!(__map.next_kv(|[__ke]| async {
                            let (__c, _k, __field) = #krate::hit!(__ke.key((), |__k: &&'de str, [__ve]| {
                                let __k = *__k;
                                async move {
                                    match __k {
                                        #( #known_field_arms )*
                                        #unknown_field_arm
                                    }
                                }
                            }).await);
                            ::core::result::Result::Ok(#krate::Probe::Hit((__c, __field)))
                        }).await) {
                            #krate::Chunk::Data((__map_back, __field)) => {
                                #data_arm_body
                                __map = __map_back;
                            }
                            #krate::Chunk::Done(__c) => break __c,
                        }
                    };

                    #( #field_finalizers )*

                    ::core::result::Result::Ok(#krate::Probe::Hit((__claim, #name { #( #field_names, )* })))
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
                // Skipped fields always use default (validation ensures default exists).
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

    // Build impl generics: prepend 'de, add Deserialize<'de> bounds for field types.
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
            for ty in &field_types {
                wc.predicates
                    .push(syn::parse_quote!(#ty: #krate::Deserialize<'de>));
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

    let body = if !has_untagged {
        // No untagged variants — original dispatch logic.
        if has_tagged_unit && !has_tagged_nonunit {
            expand_enum_unit_only(name, &classified, krate)?
        } else if !has_tagged_unit && has_tagged_nonunit {
            expand_enum_map_only(name, &classified, krate)?
        } else {
            expand_enum_mixed(name, &classified, krate)?
        }
    } else if !has_tagged_unit && !has_tagged_nonunit {
        // All variants are untagged.
        expand_enum_untagged_only(name, &classified, krate)?
    } else {
        // Mixed tagged + untagged.
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
                MapAccess as _, MapKeyEntry as _, MapValueEntry as _,
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

/// Generate the map key match arms for tagged non-unit variants (borrow family).
fn nonunit_map_key_arms(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> TokenStream2 {
    let arms: Vec<_> = classified.iter().filter_map(|cv| {
        if cv.untagged {
            return None;
        }
        let idx = cv.index;
        let vname = &cv.variant.ident;
        let vstr = &cv.wire_name;
        let aliases = &cv.aliases;
        match &cv.kind {
            VariantKind::Newtype(ty) => Some(quote! {
                #vstr #( | #aliases )* => {
                    let (__c, __v) = #krate::hit!(__ve.value::<#ty, _>(()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname(__v))))
                }
            }),
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__Variant{}", idx);
                let field_names: Vec<_> = fields.iter()
                    .map(|f| f.ident.as_ref().unwrap())
                    .collect();
                Some(quote! {
                    #vstr #( | #aliases )* => {
                        let (__c, __v) = #krate::hit!(__ve.value::<#helper_name, _>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname { #( #field_names: __v.#field_names, )* })))
                    }
                })
            }
            VariantKind::Tuple(fields) => {
                let helper_name = format_ident!("__TupleVariant{}", idx);
                let field_indices: Vec<syn::Index> = (0..fields.len())
                    .map(syn::Index::from)
                    .collect();
                Some(quote! {
                    #vstr #( | #aliases )* => {
                        let (__c, __v) = #krate::hit!(__ve.value::<#helper_name, _>(()).await);
                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, #name::#vname( #( __v.#field_indices, )* ))))
                    }
                })
            }
            VariantKind::Unit => None,
        }
    }).collect();

    let wildcard = match other_variant(classified) {
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
        match __k {
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

                impl<'de> #krate::Deserialize<'de> for #helper_name
                where
                    #( #field_types: #krate::Deserialize<'de>, )*
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
            let field_strs: Vec<_> = cf.iter().map(|f| f.wire_name.clone()).collect();
            let field_alias_vecs: Vec<Vec<String>> = cf.iter().map(|f| f.aliases.clone()).collect();
            let acc_names: Vec<_> = field_names
                .iter()
                .map(|n| format_ident!("__f_{}", n))
                .collect();
            let field_finalizers = gen_field_finalizers(&field_names, &acc_names, &cf, krate);

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name {
                    #( #field_names: #field_types, )*
                }

                impl<'de> #krate::Deserialize<'de> for #helper_name
                where
                    #( #field_types: #krate::Deserialize<'de>, )*
                {
                    async fn deserialize<__D2: #krate::Deserializer<'de>>(
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
                                    let (__c, _k, __fv) = #krate::hit!(__ke.key((), |__k: &&'de str, [__ve]| {
                                        let __k = *__k;
                                        async move {
                                            #[allow(unreachable_patterns)]
                                            match __k {
                                                #(
                                                    #field_strs #( | #field_alias_vecs )* => {
                                                        let (__c, __v) = #krate::hit!(__ve.value::<#field_types, _>(()).await);
                                                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, (#field_strs, __v))))
                                                    }
                                                )*
                                                _ => ::core::result::Result::Ok(#krate::Probe::Miss),
                                            }
                                        }
                                    }).await);
                                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, __fv)))
                                }).await) {
                                    #krate::Chunk::Data((__map_back, __fv)) => {
                                        match __fv.0 {
                                            #(
                                                #field_strs => {
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
            let mut __map = #krate::hit!(__e.deserialize_map().await);
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
                    let mut __map = #krate::hit!(__e2.deserialize_map().await);
                    #map_body
                },
            }
        }).await
    })
}

/// All untagged — try each variant by shape.
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

/// Mixed tagged + untagged — try tagged first, then untagged fallback.
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

    // Compute total handle count and assign names.
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

    // Tagged str section.
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

    // Tagged map section.
    let map_section = if let Some(h) = &map_handle {
        let map_body = gen_enum_map_body_borrow(name, classified, krate);
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
fn gen_enum_map_body_borrow(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> TokenStream2 {
    let key_arms = nonunit_map_key_arms(name, classified, krate);

    quote! {
        // Read the single key-value pair.
        let __chunk = #krate::hit!(__map.next_kv(|[__ke]| async {
            let (__c, _k, __v) = #krate::hit!(__ke.key((), |__k: &&'de str, [__ve]| {
                let __k = *__k;
                async move {
                    #key_arms
                }
            }).await);
            ::core::result::Result::Ok(#krate::Probe::Hit((__c, __v)))
        }).await);
        let (__map_back, __enum_val) = #krate::or_miss!(__chunk.data());
        let mut __map = __map_back;

        // Drain the map — expect Done (single-key map).
        let __chunk = #krate::hit!(__map.next_kv::<1, _, _, ()>(|[__ke]| async {
            // We don't expect another key; just propagate to get Chunk::Done.
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await);
        let __claim = #krate::or_miss!(__chunk.done());
        ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __enum_val)))
    }
}
