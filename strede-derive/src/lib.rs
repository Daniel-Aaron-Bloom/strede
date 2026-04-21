extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod borrow;
mod common;
mod owned;

#[proc_macro_derive(Deserialize, attributes(strede))]
pub fn derive_deserialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    borrow::expand(input)
        .unwrap_or_else(|e| e.into_compile_error())
        .into()
}

#[proc_macro_derive(DeserializeOwned, attributes(strede))]
pub fn derive_deserialize_owned(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    owned::expand(input)
        .unwrap_or_else(|e| e.into_compile_error())
        .into()
}
