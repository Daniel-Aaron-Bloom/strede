use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields};

use crate::common::{
    ClassifiedVariant, DefaultAttr, VariantKind, all_field_types, classify_fields,
    classify_variants, parse_container_attrs,
};

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let container_attrs = parse_container_attrs(&input.attrs)?;
    let krate = &container_attrs.crate_path;
    match &input.data {
        Data::Struct(_) => expand_owned_struct(input, krate),
        Data::Enum(_) => expand_owned_enum(input, krate),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "DeserializeOwned can only be derived for structs and enums",
        )),
    }
}

fn expand_owned_struct(input: DeriveInput,
    krate: &syn::Path) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let named_fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "DeserializeOwned can only be derived for structs with named fields",
                ));
            }
        },
        _ => unreachable!(),
    };

    let field_names: Vec<_> = named_fields
        .iter()
        .map(|f| f.ident.as_ref().unwrap())
        .collect();
    let field_types: Vec<_> = named_fields.iter().map(|f| &f.ty).collect();
    let classified_fields = classify_fields(named_fields)?;
    let acc_names: Vec<_> = field_names
        .iter()
        .map(|n| format_ident!("__f_{}", n))
        .collect();
    let field_finalizers = gen_field_finalizers(&field_names, &acc_names, &classified_fields, krate);

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

    // For deserialize_owned_with: compute value types and conversions.
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
            } else {
                let ty = &de_field_types[de_field_names.iter().position(|n| *n == *name).unwrap()];
                quote! { #ty }
            }
        })
        .collect();
    let de_value_conversions: Vec<TokenStream2> = de_classified
        .iter()
        .map(|cf| {
            if cf.deserialize_owned_with.is_some() {
                quote! { .0 }
            } else {
                quote! {}
            }
        })
        .collect();

    let de_with_wrappers =
        gen_deserialize_with_wrappers_owned(&de_field_names, &de_field_types, &de_classified, krate);

    // ty_generics: original struct type params.
    let (_, ty_generics, _) = input.generics.split_for_impl();

    // Build impl generics: prepend 's if not already present, add DeserializeOwned<'s> bounds.
    let mut impl_gen = input.generics.clone();
    let has_s = impl_gen.lifetimes().any(|l| l.lifetime.ident == "s");
    if !has_s {
        impl_gen.params.insert(0, syn::parse_quote!('s));
    }
    {
        let wc = impl_gen.make_where_clause();
        for ty in &de_field_types {
            wc.predicates
                .push(syn::parse_quote!(#ty: #krate::DeserializeOwned<'s>));
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    // sentinel value for unknown field
    let sentinel = de_field_count;

    Ok(quote! {
        const _: () = {
            #de_with_wrappers

            // Per-struct field key matcher
            struct __FieldKey(usize);

            impl<'s> #krate::DeserializeOwned<'s> for __FieldKey {
                async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                    d: __D,
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error> {
                    d.next(|[__e]| async move {
                        let mut __chunks = match __e.deserialize_str_chunks().await? {
                            #krate::Probe::Hit(__c) => __c,
                            #krate::Probe::Miss => {
                                return ::core::result::Result::Ok(#krate::Probe::Miss);
                            }
                        };

                        let mut __remaining: [::core::option::Option<&str>; #de_field_count] = [
                            #( ::core::option::Option::Some(#de_field_strs), )*
                        ];

                        let __claim = loop {
                            match __chunks.next(|__chunk| {
                                let mut __i = 0usize;
                                while __i < #de_field_count {
                                    if let ::core::option::Option::Some(__s) = __remaining[__i] {
                                        if let ::core::option::Option::Some(__rest) = __s.strip_prefix(__chunk) {
                                            __remaining[__i] =
                                                ::core::option::Option::Some(__rest);
                                        } else {
                                            __remaining[__i] = ::core::option::Option::None;
                                        }
                                    }
                                    __i += 1;
                                }
                            }).await? {
                                #krate::Chunk::Data((__c, ())) => { __chunks = __c; }
                                #krate::Chunk::Done(__c) => break __c,
                            }
                        };

                        let mut __matched = #sentinel;
                        let mut __i = 0usize;
                        while __i < #de_field_count {
                            if let ::core::option::Option::Some("") = __remaining[__i] {
                                __matched = __i;
                                break;
                            }
                            __i += 1;
                        }

                        ::core::result::Result::Ok(
                            #krate::Probe::Hit((__claim, __FieldKey(__matched)))
                        )
                    }).await
                }
            }

            impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                    d: __D,
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                where
                    __D::Error: #krate::DeserializeError,
                {
                    d.next(|[__e]| async move {
                        let mut __map = match __e.deserialize_map().await? {
                            #krate::Probe::Hit(__m) => __m,
                            #krate::Probe::Miss => {
                                return ::core::result::Result::Ok(#krate::Probe::Miss);
                            }
                        };

                        #[allow(non_camel_case_types)]
                        enum __Field #ty_generics {
                            #( #de_field_names(#de_field_types), )*
                        }

                        #(
                            let mut #de_acc_names: ::core::option::Option<#de_field_types> =
                                ::core::option::Option::None;
                        )*

                        let __claim = loop {
                            match __map.next(|[__ke]| async move {
                                let __probe = __ke.key::<__FieldKey, 1, _, _, _>(
                                    |__k, [__ve]| {
                                        let __idx = __k.0;
                                        async move {
                                            match __idx {
                                                #(
                                                    #de_field_indices => {
                                                        match __ve.value::<#de_value_types>().await? {
                                                            #krate::Probe::Hit((__c, __v)) => {
                                                                ::core::result::Result::Ok(
                                                                    #krate::Probe::Hit((
                                                                        __c,
                                                                        __Field::#de_field_names(__v #de_value_conversions),
                                                                    ))
                                                                )
                                                            }
                                                            #krate::Probe::Miss => {
                                                                ::core::result::Result::Ok(
                                                                    #krate::Probe::Miss,
                                                                )
                                                            }
                                                        }
                                                    }
                                                )*
                                                _ => {
                                                    ::core::result::Result::Ok(
                                                        #krate::Probe::Miss,
                                                    )
                                                }
                                            }
                                        }
                                    },
                                ).await?;
                                match __probe {
                                    #krate::Probe::Hit((__c, _k, __field)) => {
                                        ::core::result::Result::Ok(
                                            #krate::Probe::Hit((__c, __field)),
                                        )
                                    }
                                    #krate::Probe::Miss => {
                                        ::core::result::Result::Ok(#krate::Probe::Miss)
                                    }
                                }
                            }).await? {
                                #krate::Probe::Hit(
                                    #krate::Chunk::Data((__map_back, __field)),
                                ) => {
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
                                    __map = __map_back;
                                }
                                #krate::Probe::Hit(
                                    #krate::Chunk::Done(__c),
                                ) => break __c,
                                #krate::Probe::Miss => {
                                    return ::core::result::Result::Ok(#krate::Probe::Miss);
                                }
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
    krate: &syn::Path) -> Vec<TokenStream2> {
    field_names
        .iter()
        .zip(acc_names.iter())
        .zip(classified_fields.iter())
        .map(|((fname, acc), cf)| {
            if cf.skip_deserializing {
                let default_expr = match &cf.default {
                    Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                    Some(DefaultAttr::Path(path)) => quote! { #path() },
                    None => unreachable!("validated in classify_fields"),
                };
                return quote! { let #fname = #default_expr; };
            }
            let none_branch = match &cf.default {
                Some(DefaultAttr::Trait) => quote! { ::core::default::Default::default() },
                Some(DefaultAttr::Path(path)) => quote! { #path() },
                None => quote! { return ::core::result::Result::Ok(#krate::Probe::Miss) },
            };
            quote! {
                let #fname = match #acc {
                    ::core::option::Option::Some(__v) => __v,
                    ::core::option::Option::None => { #none_branch }
                };
            }
        })
        .collect()
}

/// Generate wrapper newtypes for fields with `deserialize_owned_with` (owned family).
fn gen_deserialize_with_wrappers_owned(
    field_names: &[&syn::Ident],
    field_types: &[&syn::Type],
    classified: &[&crate::common::ClassifiedField],
    krate: &syn::Path) -> TokenStream2 {
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
                    ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        match #path(d).await? {
                            #krate::Probe::Hit((__c, __v)) => {
                                ::core::result::Result::Ok(#krate::Probe::Hit((__c, Self(__v))))
                            }
                            #krate::Probe::Miss => {
                                ::core::result::Result::Ok(#krate::Probe::Miss)
                            }
                        }
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

fn expand_owned_enum(input: DeriveInput,
    krate: &syn::Path) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let data = match &input.data {
        Data::Enum(d) => d,
        _ => unreachable!(),
    };

    let container_attrs = parse_container_attrs(&input.attrs)?;
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
        for ty in &field_types {
            wc.predicates
                .push(syn::parse_quote!(#ty: #krate::DeserializeOwned<'s>));
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
    let tagged_variant_strs: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .map(|cv| cv.wire_name.clone())
        .collect();
    let variant_count = tagged_variant_strs.len();
    let variant_key_sentinel = variant_count;

    let variant_key_impl = if variant_count > 0 {
        gen_chunk_matcher_impl(
            &format_ident!("__VariantKey"),
            &tagged_variant_strs,
            variant_count,
            variant_key_sentinel,
            krate,
        )
    } else {
        quote! {}
    };

    let body = if !has_untagged {
        if has_tagged_unit && !has_tagged_nonunit {
            expand_owned_enum_unit_only(name, &classified, variant_key_sentinel, krate)?
        } else if !has_tagged_unit && has_tagged_nonunit {
            expand_owned_enum_map_only(name, &classified, variant_key_sentinel, krate)?
        } else {
            expand_owned_enum_mixed(name, &classified, variant_key_sentinel, krate)?
        }
    } else if !has_tagged_unit && !has_tagged_nonunit {
        expand_owned_enum_untagged_only(name, &classified, krate)?
    } else {
        expand_owned_enum_with_untagged(name, &classified, variant_key_sentinel, krate)?
    };

    // For tuple variants, generate per-variant __TupleVariantOwnedN types.
    let tuple_variant_helpers = gen_tuple_variant_helpers_owned(&classified, krate);
    // For struct variants, generate per-variant __FieldKeyN and __VariantOwnedN types.
    let struct_variant_field_keys = gen_struct_variant_field_keys(&classified, krate);
    let struct_variant_helpers = gen_struct_variant_helpers_owned(&classified, krate);

    Ok(quote! {
        const _: () = {
            struct __VariantKey(usize);
            #variant_key_impl

            #tuple_variant_helpers
            #struct_variant_field_keys
            #struct_variant_helpers

            impl #impl_generics #krate::DeserializeOwned<'s> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                    d: __D,
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

/// Generate a chunk-matching DeserializeOwned impl for a newtype(usize) key matcher.
fn gen_chunk_matcher_impl(
    type_name: &syn::Ident,
    strs: &[String],
    count: usize,
    sentinel: usize,
    krate: &syn::Path) -> TokenStream2 {
    quote! {
        impl<'s> #krate::DeserializeOwned<'s> for #type_name {
            async fn deserialize<__D: #krate::DeserializerOwned<'s>>(
                d: __D,
            ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error> {
                d.next(|[__e]| async move {
                    let mut __chunks = match __e.deserialize_str_chunks().await? {
                        #krate::Probe::Hit(__c) => __c,
                        #krate::Probe::Miss => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                    };

                    let mut __remaining: [::core::option::Option<&str>; #count] = [
                        #( ::core::option::Option::Some(#strs), )*
                    ];

                    let __claim = loop {
                        match __chunks.next(|__chunk| {
                            let mut __i = 0usize;
                            while __i < #count {
                                if let ::core::option::Option::Some(__s) = __remaining[__i] {
                                    if let ::core::option::Option::Some(__rest) = __s.strip_prefix(__chunk) {
                                        __remaining[__i] =
                                            ::core::option::Option::Some(__rest);
                                    } else {
                                        __remaining[__i] = ::core::option::Option::None;
                                    }
                                }
                                __i += 1;
                            }
                        }).await? {
                            #krate::Chunk::Data((__c, ())) => { __chunks = __c; }
                            #krate::Chunk::Done(__c) => break __c,
                        }
                    };

                    let mut __matched = #sentinel;
                    let mut __i = 0usize;
                    while __i < #count {
                        if let ::core::option::Option::Some("") = __remaining[__i] {
                            __matched = __i;
                            break;
                        }
                        __i += 1;
                    }

                    ::core::result::Result::Ok(
                        #krate::Probe::Hit((__claim, #type_name(__matched)))
                    )
                }).await
            }
        }
    }
}

/// Generate helper tuple struct definitions and DeserializeOwned impls for tuple variants (owned family).
fn gen_tuple_variant_helpers_owned(classified: &[ClassifiedVariant],
    krate: &syn::Path) -> TokenStream2 {
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
            let seq_reads: Vec<_> = acc_names.iter().zip(field_types.iter()).map(|(acc, ty)| {
                quote! {
                    let (#acc, __seq) = match __seq.next(|[__se]| async move {
                        __se.get::<#ty>().await
                    }).await? {
                        #krate::Probe::Hit(#krate::Chunk::Data((__seq_back, __v))) => (__v, __seq_back),
                        #krate::Probe::Hit(#krate::Chunk::Done(_)) => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                        #krate::Probe::Miss => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                    };
                }
            }).collect();

            tokens.extend(quote! {
                #[allow(non_camel_case_types)]
                struct #helper_name( #( #field_types, )* );

                impl<'s> #krate::DeserializeOwned<'s> for #helper_name
                where
                    #( #field_types: #krate::DeserializeOwned<'s>, )*
                {
                    async fn deserialize<__D2: #krate::DeserializerOwned<'s>>(
                        d: __D2,
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.next(|[__e]| async move {
                            let __seq = match __e.deserialize_seq().await? {
                                #krate::Probe::Hit(s) => s,
                                #krate::Probe::Miss => return ::core::result::Result::Ok(#krate::Probe::Miss),
                            };

                            #( #seq_reads )*

                            // Expect sequence exhaustion.
                            match __seq.next::<1, _, _, ()>(|[__se]| async move {
                                ::core::result::Result::Ok(#krate::Probe::Miss)
                            }).await? {
                                #krate::Probe::Hit(#krate::Chunk::Done(__claim)) => {
                                    ::core::result::Result::Ok(#krate::Probe::Hit((
                                        __claim,
                                        #helper_name( #( #acc_names, )* ),
                                    )))
                                }
                                _ => {
                                    ::core::result::Result::Ok(#krate::Probe::Miss)
                                }
                            }
                        }).await
                    }
                }
            });
        }
    }
    tokens
}

/// Generate __FieldKeyN structs and impls for each struct variant.
fn gen_struct_variant_field_keys(classified: &[ClassifiedVariant],
    krate: &syn::Path) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for cv in classified.iter() {
        if let VariantKind::Struct(fields) = &cv.kind {
            let key_name = format_ident!("__FieldKey{}", cv.index);
            let cf = match classify_fields(fields) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let field_strs: Vec<_> = cf.iter().map(|f| f.wire_name.clone()).collect();
            let field_count = field_strs.len();
            let sentinel = field_count;

            let struct_def = quote! { struct #key_name(usize); };
            let impl_block = gen_chunk_matcher_impl(&key_name, &field_strs, field_count, sentinel, krate);

            tokens.extend(struct_def);
            tokens.extend(impl_block);
        }
    }
    tokens
}

fn expand_owned_enum_unit_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    _variant_key_sentinel: usize,
    krate: &syn::Path) -> syn::Result<TokenStream2> {
    let all_variant_strs: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged)
        .map(|cv| cv.wire_name.clone())
        .collect();
    let variant_count = all_variant_strs.len();

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

    Ok(quote! {
        d.next(|[__e]| async move {
            let mut __chunks = match __e.deserialize_str_chunks().await? {
                #krate::Probe::Hit(__c) => __c,
                #krate::Probe::Miss => {
                    return ::core::result::Result::Ok(#krate::Probe::Miss);
                }
            };

            let mut __remaining: [::core::option::Option<&str>; #variant_count] = [
                #( ::core::option::Option::Some(#all_variant_strs), )*
            ];

            let __claim = loop {
                match __chunks.next(|__chunk| {
                    let mut __i = 0usize;
                    while __i < #variant_count {
                        if let ::core::option::Option::Some(__s) = __remaining[__i] {
                            if let ::core::option::Option::Some(__rest) = __s.strip_prefix(__chunk) {
                                __remaining[__i] =
                                    ::core::option::Option::Some(__rest);
                            } else {
                                __remaining[__i] = ::core::option::Option::None;
                            }
                        }
                        __i += 1;
                    }
                }).await? {
                    #krate::Chunk::Data((__c, ())) => { __chunks = __c; }
                    #krate::Chunk::Done(__c) => break __c,
                }
            };

            let mut __matched = #variant_count; // sentinel
            let mut __i = 0usize;
            while __i < #variant_count {
                if let ::core::option::Option::Some("") = __remaining[__i] {
                    __matched = __i;
                    break;
                }
                __i += 1;
            }

            match __matched {
                #( #unit_match_arms )*
                _ => ::core::result::Result::Ok(#krate::Probe::Miss),
            }
        }).await
    })
}

fn expand_owned_enum_map_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_key_sentinel: usize,
    krate: &syn::Path) -> syn::Result<TokenStream2> {
    let map_body = gen_owned_enum_map_body(name, classified, variant_key_sentinel, krate);
    Ok(quote! {
        d.next(|[__e]| async move {
            let mut __map = match __e.deserialize_map().await? {
                #krate::Probe::Hit(__m) => __m,
                #krate::Probe::Miss => {
                    return ::core::result::Result::Ok(#krate::Probe::Miss);
                }
            };
            #map_body
        }).await
    })
}

fn expand_owned_enum_mixed(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_key_sentinel: usize,
    krate: &syn::Path) -> syn::Result<TokenStream2> {
    // Build the string chunk-match for tagged unit variant names.
    let unit_variant_strs: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit))
        .map(|cv| cv.wire_name.clone())
        .collect();
    let unit_count = unit_variant_strs.len();

    // Collect tagged unit variants for match arms.
    let tagged_units: Vec<_> = classified
        .iter()
        .filter(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit))
        .collect();

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

    let map_body = gen_owned_enum_map_body(name, classified, variant_key_sentinel, krate);

    // Sequential probing: try str_chunks first (unit variants), then map.
    Ok(quote! {
        d.next(|[__e1, __e2]| async move {
            // Try string first (unit variants).
            match __e1.deserialize_str_chunks().await? {
                #krate::Probe::Hit(__chunks) => {
                    let mut __chunks = __chunks;
                    let mut __remaining: [::core::option::Option<&str>; #unit_count] = [
                        #( ::core::option::Option::Some(#unit_variant_strs), )*
                    ];

                    let __unit_claim = loop {
                        match __chunks.next(|__chunk| {
                            let mut __i = 0usize;
                            while __i < #unit_count {
                                if let ::core::option::Option::Some(__s) = __remaining[__i] {
                                    if let ::core::option::Option::Some(__rest) = __s.strip_prefix(__chunk) {
                                        __remaining[__i] =
                                            ::core::option::Option::Some(__rest);
                                    } else {
                                        __remaining[__i] = ::core::option::Option::None;
                                    }
                                }
                                __i += 1;
                            }
                        }).await? {
                            #krate::Chunk::Data((__c, ())) => { __chunks = __c; }
                            #krate::Chunk::Done(__c) => break __c,
                        }
                    };

                    let mut __matched = #unit_count; // sentinel
                    let mut __i = 0usize;
                    while __i < #unit_count {
                        if let ::core::option::Option::Some("") = __remaining[__i] {
                            __matched = __i;
                            break;
                        }
                        __i += 1;
                    }

                    match __matched {
                        #( #unit_match_arms )*
                        _ => {
                            return ::core::result::Result::Ok(#krate::Probe::Miss);
                        }
                    }
                }
                #krate::Probe::Miss => {}
            }

            // Try map (non-unit variants).
            let mut __map = match __e2.deserialize_map().await? {
                #krate::Probe::Hit(__m) => __m,
                #krate::Probe::Miss => {
                    return ::core::result::Result::Ok(#krate::Probe::Miss);
                }
            };
            #map_body
        }).await
    })
}

/// Generate the body that reads a single-key map for non-unit variant dispatch (owned family).
fn gen_owned_enum_map_body(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_key_sentinel: usize,
    krate: &syn::Path) -> TokenStream2 {
    let value_dispatch_arms = gen_owned_value_dispatch_arms(name, classified, variant_key_sentinel, krate);

    quote! {
        // Read the single key-value pair.
        let (__map_back, __enum_val) = match __map.next(|[__ke]| async move {
            let __probe = __ke.key::<__VariantKey, 1, _, _, _>(
                |__k, [__ve]| {
                    let __idx = __k.0;
                    async move {
                        #value_dispatch_arms
                    }
                },
            ).await?;
            match __probe {
                #krate::Probe::Hit((__c, _k, __v)) => {
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, __v)))
                }
                #krate::Probe::Miss => {
                    ::core::result::Result::Ok(#krate::Probe::Miss)
                }
            }
        }).await? {
            #krate::Probe::Hit(#krate::Chunk::Data(__pair)) => __pair,
            #krate::Probe::Hit(#krate::Chunk::Done(_)) => {
                return ::core::result::Result::Ok(#krate::Probe::Miss);
            }
            #krate::Probe::Miss => {
                return ::core::result::Result::Ok(#krate::Probe::Miss);
            }
        };

        // Drain the map — expect Done.
        let mut __map = __map_back;
        match __map.next::<1, _, _, ()>(|[__ke]| async move {
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await? {
            #krate::Probe::Hit(#krate::Chunk::Done(__claim)) => {
                ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __enum_val)))
            }
            _ => {
                ::core::result::Result::Ok(#krate::Probe::Miss)
            }
        }
    }
}

/// Generate value dispatch arms for the owned-family map key match.
fn gen_owned_value_dispatch_arms(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    _sentinel: usize,
    krate: &syn::Path) -> TokenStream2 {
    // Build arms using tagged-variant local indices (matching __VariantKey output).
    let tagged: Vec<_> = classified.iter().filter(|cv| !cv.untagged).collect();
    let arms: Vec<_> = tagged.iter().enumerate().filter_map(|(idx, cv)| {
        let vname = &cv.variant.ident;
        match &cv.kind {
            VariantKind::Newtype(ty) => Some(quote! {
                #idx => {
                    match __ve.value::<#ty>().await? {
                        #krate::Probe::Hit((__c, __v)) => {
                            ::core::result::Result::Ok(
                                #krate::Probe::Hit((__c, #name::#vname(__v)))
                            )
                        }
                        #krate::Probe::Miss => {
                            ::core::result::Result::Ok(#krate::Probe::Miss)
                        }
                    }
                }
            }),
            VariantKind::Struct(fields) => {
                let helper_name = format_ident!("__VariantOwned{}", cv.index);
                let field_names: Vec<_> = fields.iter()
                    .map(|f| f.ident.as_ref().unwrap())
                    .collect();
                Some(quote! {
                    #idx => {
                        match __ve.value::<#helper_name>().await? {
                            #krate::Probe::Hit((__c, __v)) => {
                                ::core::result::Result::Ok(
                                    #krate::Probe::Hit((__c, #name::#vname { #( #field_names: __v.#field_names, )* }))
                                )
                            }
                            #krate::Probe::Miss => {
                                ::core::result::Result::Ok(#krate::Probe::Miss)
                            }
                        }
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
                        match __ve.value::<#helper_name>().await? {
                            #krate::Probe::Hit((__c, __v)) => {
                                ::core::result::Result::Ok(
                                    #krate::Probe::Hit((__c, #name::#vname( #( __v.#field_indices, )* )))
                                )
                            }
                            #krate::Probe::Miss => {
                                ::core::result::Result::Ok(#krate::Probe::Miss)
                            }
                        }
                    }
                })
            }
            VariantKind::Unit => None,
        }
    }).collect();

    quote! {
        match __idx {
            #( #arms )*
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        }
    }
}

/// Generate helper struct definitions and DeserializeOwned impls for struct variants (owned family).
fn gen_struct_variant_helpers_owned(classified: &[ClassifiedVariant],
    krate: &syn::Path) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for cv in classified.iter() {
        if let VariantKind::Struct(fields) = &cv.kind {
            let helper_name = format_ident!("__VariantOwned{}", cv.index);
            let field_key_name = format_ident!("__FieldKey{}", cv.index);
            let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
            let cf = match classify_fields(fields) {
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
                    ) -> ::core::result::Result<#krate::Probe<(__D2::Claim, Self)>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.next(|[__e]| async move {
                            let mut __map = match __e.deserialize_map().await? {
                                #krate::Probe::Hit(__m) => __m,
                                #krate::Probe::Miss => {
                                    return ::core::result::Result::Ok(#krate::Probe::Miss);
                                }
                            };

                            #( let mut #acc_names: ::core::option::Option<#field_types> = ::core::option::Option::None; )*

                            let __claim = loop {
                                match __map.next(|[__ke]| async move {
                                    let __probe = __ke.key::<#field_key_name, 1, _, _, _>(
                                        |__k, [__ve]| {
                                            let __fidx = __k.0;
                                            async move {
                                                match __fidx {
                                                    #(
                                                        #field_indices => {
                                                            match __ve.value::<#field_types>().await? {
                                                                #krate::Probe::Hit((__c, __v)) => {
                                                                    ::core::result::Result::Ok(
                                                                        #krate::Probe::Hit((__c, (#field_indices, __v)))
                                                                    )
                                                                }
                                                                #krate::Probe::Miss => {
                                                                    ::core::result::Result::Ok(#krate::Probe::Miss)
                                                                }
                                                            }
                                                        }
                                                    )*
                                                    _ => ::core::result::Result::Ok(#krate::Probe::Miss),
                                                }
                                            }
                                        },
                                    ).await?;
                                    match __probe {
                                        #krate::Probe::Hit((__c, _k, __fv)) => {
                                            ::core::result::Result::Ok(#krate::Probe::Hit((__c, __fv)))
                                        }
                                        #krate::Probe::Miss => {
                                            ::core::result::Result::Ok(#krate::Probe::Miss)
                                        }
                                    }
                                }).await? {
                                    #krate::Probe::Hit(#krate::Chunk::Data((__map_back, __fv))) => {
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
                                    #krate::Probe::Hit(#krate::Chunk::Done(__c)) => break __c,
                                    #krate::Probe::Miss => {
                                        return ::core::result::Result::Ok(#krate::Probe::Miss);
                                    }
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
    krate: &syn::Path) -> syn::Result<TokenStream2> {
    let n_handles = classified.len();
    let handle_names: Vec<_> = (0..n_handles).map(|i| format_ident!("__e{}", i)).collect();

    let refs: Vec<_> = classified.iter().collect();
    let probe_chain = gen_untagged_probe_chain_owned(name, &refs, &handle_names, krate);

    Ok(quote! {
        d.next(|[#( #handle_names ),*]| async move {
            #probe_chain
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await
    })
}

/// Mixed tagged + untagged (owned family).
fn expand_owned_enum_with_untagged(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    variant_key_sentinel: usize,
    krate: &syn::Path) -> syn::Result<TokenStream2> {
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
        let unit_variant_strs: Vec<_> = classified
            .iter()
            .filter(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit))
            .map(|cv| cv.wire_name.clone())
            .collect();
        let unit_count = unit_variant_strs.len();

        let tagged_units: Vec<_> = classified
            .iter()
            .filter(|cv| !cv.untagged && matches!(cv.kind, VariantKind::Unit))
            .collect();
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
            match #h.deserialize_str_chunks().await? {
                #krate::Probe::Hit(__chunks) => {
                    let mut __chunks = __chunks;
                    let mut __remaining: [::core::option::Option<&str>; #unit_count] = [
                        #( ::core::option::Option::Some(#unit_variant_strs), )*
                    ];
                    let __unit_claim = loop {
                        match __chunks.next(|__chunk| {
                            let mut __i = 0usize;
                            while __i < #unit_count {
                                if let ::core::option::Option::Some(__s) = __remaining[__i] {
                                    if let ::core::option::Option::Some(__rest) = __s.strip_prefix(__chunk) {
                                        __remaining[__i] =
                                            ::core::option::Option::Some(__rest);
                                    } else {
                                        __remaining[__i] = ::core::option::Option::None;
                                    }
                                }
                                __i += 1;
                            }
                        }).await? {
                            #krate::Chunk::Data((__c, ())) => { __chunks = __c; }
                            #krate::Chunk::Done(__c) => break __c,
                        }
                    };
                    let mut __matched = #unit_count;
                    let mut __i = 0usize;
                    while __i < #unit_count {
                        if let ::core::option::Option::Some("") = __remaining[__i] {
                            __matched = __i;
                            break;
                        }
                        __i += 1;
                    }
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
        let map_body = gen_owned_enum_map_body(name, classified, variant_key_sentinel, krate);
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
        d.next(|[#( #all_handles ),*]| async move {
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
    krate: &syn::Path) -> TokenStream2 {
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
                    match #handle.deserialize_value::<#ty>().await? {
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
                    match #handle.deserialize_value::<#helper_name>().await? {
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
                    match #handle.deserialize_value::<#helper_name>().await? {
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
