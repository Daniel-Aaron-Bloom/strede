use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput};

use crate::common::{FieldContext, field_bound_owned, insert_d_owned, parse_container_attrs};

mod enum_;
mod struct_;

pub fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let container_attrs = parse_container_attrs(&input.attrs)?;
    let krate = &container_attrs.crate_path;
    match &input.data {
        Data::Struct(_) => struct_::expand_owned(input, krate, &container_attrs),
        Data::Enum(_) => enum_::expand_owned(input, krate),
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
    insert_d_owned(&mut impl_gen, krate);
    {
        let wc = impl_gen.make_where_clause();
        if let Some(preds) = &container_attrs.bound {
            wc.predicates.extend(preds.iter().cloned());
        } else {
            wc.predicates.push(field_bound_owned(
                krate,
                from_ty,
                FieldContext::Direct,
                &format_ident!("__D"),
            ));
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

            impl #impl_generics #krate::DeserializeOwned<__D> for #name #ty_generics #where_clause {
                type Extra = ();
                async fn deserialize_owned(
                    d: __D,
                    _extra: (),
                ) -> ::core::result::Result<#krate::Probe<(__D::Claim, Self)>, __D::Error>
                {
                    let (__c, __v) = #krate::hit!(<#from_ty as #krate::DeserializeOwned<__D>>::deserialize_owned(d, ()).await);
                    ::core::result::Result::Ok(#krate::Probe::Hit((__c, #convert)))
                }
            }
        };
    })
}
