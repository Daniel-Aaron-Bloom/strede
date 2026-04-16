extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod borrow;
mod common;
mod declare_comms;
mod owned;
mod select_probe;

#[proc_macro]
pub fn select_probe_inner(input: TokenStream) -> TokenStream {
    select_probe::select_probe_impl(input)
}

#[proc_macro_derive(Deserialize, attributes(strede))]
pub fn derive_deserialize(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    borrow::expand(input)
        .unwrap_or_else(|e| e.into_compile_error())
        .into()
}

#[proc_macro]
pub fn declare_comms_inner(input: TokenStream) -> TokenStream {
    declare_comms::declare_comms_impl(input)
}

#[proc_macro_derive(DeserializeOwned, attributes(strede))]
pub fn derive_deserialize_owned(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    owned::expand(input)
        .unwrap_or_else(|e| e.into_compile_error())
        .into()
}
