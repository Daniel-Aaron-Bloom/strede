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
        Data::Struct(_) => expand_struct(input, krate),
        Data::Enum(_) => expand_enum(input, krate),
        _ => Err(syn::Error::new_spanned(
            &input.ident,
            "Deserialize can only be derived for structs and enums",
        )),
    }
}

fn expand_struct(input: DeriveInput, krate: &syn::Path) -> syn::Result<TokenStream2> {
    let name = &input.ident;

    let named_fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    name,
                    "Deserialize can only be derived for structs with named fields",
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

    // For deserialize_with: compute the value type and conversion for each deserialized field.
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
            } else {
                let ty = &de_field_types[de_field_names.iter().position(|n| *n == *name).unwrap()];
                quote! { #ty }
            }
        })
        .collect();
    let de_value_conversions: Vec<TokenStream2> = de_classified
        .iter()
        .map(|cf| {
            if cf.deserialize_with.is_some() {
                quote! { .0 }
            } else {
                quote! {}
            }
        })
        .collect();

    // Generate wrapper types for deserialize_with fields.
    let de_with_wrappers =
        gen_deserialize_with_wrappers_borrow(&de_field_names, &de_field_types, &de_classified, krate);

    // ty_generics: original struct type params (without the new 'de we add below).
    let (_, ty_generics, _) = input.generics.split_for_impl();

    // Build the impl generics: prepend 'de if not already present, then add
    // Deserialize<'de> bounds for each non-skipped field type.
    let mut impl_gen = input.generics.clone();
    let has_de = impl_gen.lifetimes().any(|l| l.lifetime.ident == "de");
    if !has_de {
        impl_gen.params.insert(0, syn::parse_quote!('de));
    }
    {
        let wc = impl_gen.make_where_clause();
        for ty in &de_field_types {
            wc.predicates
                .push(syn::parse_quote!(#ty: #krate::Deserialize<'de>));
        }
    }
    let (impl_generics, _, where_clause) = impl_gen.split_for_impl();

    Ok(quote! {
        const _: () = {
            #de_with_wrappers

        impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
            async fn deserialize<__D: #krate::Deserializer<'de>>(
                d: &mut __D,
            ) -> ::core::result::Result<#krate::Probe<Self>, __D::Error>
            where
                __D::Error: #krate::DeserializeError,
            {
                d.next(|[__e]| async move {
                    let mut __map = match __e.deserialize_map().await? {
                        #krate::Probe::Hit(m) => m,
                        #krate::Probe::Miss => return ::core::result::Result::Ok(#krate::Probe::Miss),
                    };

                    #[allow(non_camel_case_types)]
                    enum __Field #ty_generics {
                        #( #de_field_names(#de_field_types), )*
                    }

                    #( let mut #de_acc_names: ::core::option::Option<#de_field_types> = ::core::option::Option::None; )*

                    let __claim = loop {
                        match __map.next(|[__ke]| async move {
                            let __probe = __ke.key::<&'de str, 1, _, _, _>(|__k, [__ve]| {
                                let __k = *__k;
                                async move {
                                    match __k {
                                        #(
                                            #de_field_strs => {
                                                match __ve.value::<#de_value_types>().await? {
                                                    #krate::Probe::Hit((__c, __v)) => {
                                                        ::core::result::Result::Ok(#krate::Probe::Hit((__c, __Field::#de_field_names(__v #de_value_conversions))))
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
                            }).await?;
                            match __probe {
                                #krate::Probe::Hit((__c, _k, __field)) => ::core::result::Result::Ok(#krate::Probe::Hit((__c, __field))),
                                #krate::Probe::Miss => ::core::result::Result::Ok(#krate::Probe::Miss),
                            }
                        }).await? {
                            #krate::Probe::Hit(#krate::Chunk::Data(__field)) => {
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
                            #krate::Probe::Hit(#krate::Chunk::Done(__c)) => break __c,
                            #krate::Probe::Miss => {
                                return ::core::result::Result::Ok(#krate::Probe::Miss);
                            }
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

/// Generate wrapper newtypes for fields with `deserialize_with` (borrow family).
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
                        d: &mut __D,
                    ) -> ::core::result::Result<#krate::Probe<Self>, __D::Error>
                    where
                        __D::Error: #krate::DeserializeError,
                    {
                        match #path(d).await? {
                            #krate::Probe::Hit(__v) => {
                                ::core::result::Result::Ok(#krate::Probe::Hit(Self(__v)))
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
// Borrow-family enum derive
// ---------------------------------------------------------------------------

fn expand_enum(input: DeriveInput, krate: &syn::Path) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let data = match &input.data {
        Data::Enum(d) => d,
        _ => unreachable!(),
    };

    let container_attrs = parse_container_attrs(&input.attrs)?;
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
        for ty in &field_types {
            wc.predicates
                .push(syn::parse_quote!(#ty: #krate::Deserialize<'de>));
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
    let struct_variant_helpers = gen_struct_variant_helpers_borrow(&classified, krate);

    Ok(quote! {
        const _: () = {
            #tuple_variant_helpers
            #struct_variant_helpers

            impl #impl_generics #krate::Deserialize<'de> for #name #ty_generics #where_clause {
                async fn deserialize<__D: #krate::Deserializer<'de>>(
                    d: &mut __D,
                ) -> ::core::result::Result<#krate::Probe<Self>, __D::Error>
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
fn unit_str_match_arms(name: &syn::Ident, classified: &[ClassifiedVariant], krate: &syn::Path) -> TokenStream2 {
    let arms: Vec<_> = classified
        .iter()
        .filter_map(|cv| {
            if !cv.untagged && matches!(cv.kind, VariantKind::Unit) {
                let vname = &cv.variant.ident;
                let vstr = &cv.wire_name;
                Some(quote! {
                    #vstr => ::core::result::Result::Ok(
                        #krate::Probe::Hit((__claim, #name::#vname))
                    ),
                })
            } else {
                None
            }
        })
        .collect();

    quote! {
        match __s {
            #( #arms )*
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        }
    }
}

/// Generate the map key match arms for tagged non-unit variants (borrow family).
fn nonunit_map_key_arms(name: &syn::Ident, classified: &[ClassifiedVariant], krate: &syn::Path) -> TokenStream2 {
    let arms: Vec<_> = classified.iter().filter_map(|cv| {
        if cv.untagged {
            return None;
        }
        let idx = cv.index;
        let vname = &cv.variant.ident;
        let vstr = &cv.wire_name;
        match &cv.kind {
            VariantKind::Newtype(ty) => Some(quote! {
                #vstr => {
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
                let helper_name = format_ident!("__Variant{}", idx);
                let field_names: Vec<_> = fields.iter()
                    .map(|f| f.ident.as_ref().unwrap())
                    .collect();
                Some(quote! {
                    #vstr => {
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
                let helper_name = format_ident!("__TupleVariant{}", idx);
                let field_indices: Vec<syn::Index> = (0..fields.len())
                    .map(syn::Index::from)
                    .collect();
                Some(quote! {
                    #vstr => {
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
        match __k {
            #( #arms )*
            _ => ::core::result::Result::Ok(#krate::Probe::Miss),
        }
    }
}

/// Generate helper tuple struct definitions and Deserialize impls for tuple variants (borrow family).
fn gen_tuple_variant_helpers_borrow(classified: &[ClassifiedVariant], krate: &syn::Path) -> TokenStream2 {
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
                        let #acc = match __seq.next(|[__se]| async move {
                            __se.get::<#ty>().await
                        }).await? {
                            #krate::Probe::Hit(#krate::Chunk::Data(__v)) => __v,
                            #krate::Probe::Hit(#krate::Chunk::Done(_)) => {
                                return ::core::result::Result::Ok(#krate::Probe::Miss);
                            }
                            #krate::Probe::Miss => {
                                return ::core::result::Result::Ok(#krate::Probe::Miss);
                            }
                        };
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
                        d: &mut __D2,
                    ) -> ::core::result::Result<#krate::Probe<Self>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.next(|[__e]| async move {
                            let mut __seq = match __e.deserialize_seq().await? {
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

/// Generate helper struct definitions and Deserialize impls for struct variants (borrow family).
fn gen_struct_variant_helpers_borrow(classified: &[ClassifiedVariant], krate: &syn::Path) -> TokenStream2 {
    let mut tokens = TokenStream2::new();
    for cv in classified.iter() {
        if let VariantKind::Struct(fields) = &cv.kind {
            let helper_name = format_ident!("__Variant{}", cv.index);
            let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();
            let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();
            let cf = match classify_fields(fields) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let field_strs: Vec<_> = cf.iter().map(|f| f.wire_name.clone()).collect();
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
                        d: &mut __D2,
                    ) -> ::core::result::Result<#krate::Probe<Self>, __D2::Error>
                    where
                        __D2::Error: #krate::DeserializeError,
                    {
                        d.next(|[__e]| async move {
                            let mut __map = match __e.deserialize_map().await? {
                                #krate::Probe::Hit(m) => m,
                                #krate::Probe::Miss => return ::core::result::Result::Ok(#krate::Probe::Miss),
                            };

                            #( let mut #acc_names: ::core::option::Option<#field_types> = ::core::option::Option::None; )*

                            let __claim = loop {
                                match __map.next(|[__ke]| async move {
                                    let __probe = __ke.key::<&'de str, 1, _, _, _>(|__k, [__ve]| {
                                        let __k = *__k;
                                        async move {
                                            match __k {
                                                #(
                                                    #field_strs => {
                                                        match __ve.value::<#field_types>().await? {
                                                            #krate::Probe::Hit((__c, __v)) => {
                                                                ::core::result::Result::Ok(#krate::Probe::Hit((__c, (#field_strs, __v))))
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
                                    }).await?;
                                    match __probe {
                                        #krate::Probe::Hit((__c, _k, __fv)) => {
                                            ::core::result::Result::Ok(#krate::Probe::Hit((__c, __fv)))
                                        }
                                        #krate::Probe::Miss => {
                                            ::core::result::Result::Ok(#krate::Probe::Miss)
                                        }
                                    }
                                }).await? {
                                    #krate::Probe::Hit(#krate::Chunk::Data(__fv)) => {
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

fn expand_enum_unit_only(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let str_match = unit_str_match_arms(name, classified, krate);
    Ok(quote! {
        d.next(|[__e]| async move {
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

fn expand_enum_mixed(
    name: &syn::Ident,
    classified: &[ClassifiedVariant],
    krate: &syn::Path,
) -> syn::Result<TokenStream2> {
    let str_match = unit_str_match_arms(name, classified, krate);
    let map_body = gen_enum_map_body_borrow(name, classified, krate);
    // Use sequential probing: try string first (unit variants), then map.
    // Can't use select_probe! because the map arm body needs .await.
    Ok(quote! {
        d.next(|[__e1, __e2]| async move {
            // Try string first (unit variants).
            match __e1.deserialize_str().await? {
                #krate::Probe::Hit((__claim, __s)) => {
                    return #str_match;
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
        d.next(|[#( #handle_names ),*]| async move {
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
        d.next(|[#( #all_handles ),*]| async move {
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
                let helper_name = format_ident!("__TupleVariant{}", cv.index);
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
                let helper_name = format_ident!("__Variant{}", cv.index);
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

/// Generate the body that reads a single-key map for non-unit variant dispatch (borrow family).
fn gen_enum_map_body_borrow(name: &syn::Ident, classified: &[ClassifiedVariant], krate: &syn::Path) -> TokenStream2 {
    let key_arms = nonunit_map_key_arms(name, classified, krate);

    quote! {
        // Read the single key-value pair.
        let __enum_val = match __map.next(|[__ke]| async move {
            let __probe = __ke.key::<&'de str, 1, _, _, _>(|__k, [__ve]| {
                let __k = *__k;
                async move {
                    #key_arms
                }
            }).await?;
            match __probe {
                #krate::Probe::Hit((__c, _k, __v)) => {
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, __v)))
                }
                #krate::Probe::Miss => {
                    ::core::result::Result::Ok(#krate::Probe::Miss)
                }
            }
        }).await? {
            #krate::Probe::Hit(#krate::Chunk::Data(__v)) => __v,
            #krate::Probe::Hit(#krate::Chunk::Done(_)) => {
                // Empty map.
                return ::core::result::Result::Ok(#krate::Probe::Miss);
            }
            #krate::Probe::Miss => {
                return ::core::result::Result::Ok(#krate::Probe::Miss);
            }
        };

        // Drain the map — expect Done (single-key map).
        match __map.next::<1, _, _, ()>(|[__ke]| async move {
            // We don't expect another key; just propagate to get Chunk::Done.
            ::core::result::Result::Ok(#krate::Probe::Miss)
        }).await? {
            #krate::Probe::Hit(#krate::Chunk::Done(__claim)) => {
                ::core::result::Result::Ok(#krate::Probe::Hit((__claim, __enum_val)))
            }
            _ => {
                // More than one key in the outer map — not a valid enum encoding.
                ::core::result::Result::Ok(#krate::Probe::Miss)
            }
        }
    }
}
