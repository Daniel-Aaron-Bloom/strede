use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    Expr, Ident, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// One closure arm: `|ident| expr`
struct CommsArm {
    expr: Expr,
}

enum CommsBody {
    Single(CommsArm),
    Tuple(Vec<CommsArm>),
}

struct DeclareCommsInput {
    krate: syn::Path,
    binding: Ident,
    body: CommsBody,
}

fn parse_arm(input: ParseStream) -> syn::Result<CommsArm> {
    // Accept any expression — the user is responsible for it being a closure
    // or callable that accepts &FutureCommsStorage.
    let expr: Expr = input.parse()?;
    Ok(CommsArm { expr })
}

impl Parse for DeclareCommsInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Leading `@crate_path` injected by the wrapper macro.
        let _: Token![@] = input.parse()?;
        let krate: syn::Path = input.parse()?;

        let _: Token![let] = input.parse()?;
        let binding: Ident = input.parse()?;
        let _: Token![=] = input.parse()?;

        // Body is either a parenthesized tuple of closures or a single closure.
        let body = if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            let mut arms = Vec::new();
            while !content.is_empty() {
                arms.push(parse_arm(&content)?);
                let _ = content.parse::<Token![,]>();
            }
            CommsBody::Tuple(arms)
        } else {
            CommsBody::Single(parse_arm(input)?)
        };

        Ok(DeclareCommsInput { krate, binding, body })
    }
}

pub fn declare_comms_impl(input: TokenStream) -> TokenStream {
    let DeclareCommsInput { krate, binding, body } = parse_macro_input!(input as DeclareCommsInput);

    let arms: Vec<CommsArm> = match body {
        CommsBody::Single(arm) => vec![arm],
        CommsBody::Tuple(arms) => arms,
    };

    let n = arms.len();

    // Generate unique idents for each slot's storage and pinned future.
    let storage_idents: Vec<Ident> = (0..n)
        .map(|i| Ident::new(&format!("__comms_storage_{i}"), Span::call_site()))
        .collect();
    let fut_idents: Vec<Ident> = (0..n)
        .map(|i| Ident::new(&format!("__comms_fut_{i}"), Span::call_site()))
        .collect();

    let arm_exprs: Vec<&Expr> = arms.iter().map(|a| &a.expr).collect();

    // Declare all storages first (they must outlive the FutureComms values).
    let storage_decls = storage_idents.iter().map(|s| {
        quote! {
            let #s = #krate::key_facade::FutureCommsStorage::new();
        }
    });

    // Then pin each future (calling the closure with &storage).
    let fut_decls = fut_idents.iter().zip(storage_idents.iter()).zip(arm_exprs.iter()).map(
        |((f, s), expr)| {
            quote! {
                let #f = ::core::pin::pin!(Some((#expr)(&#s)));
                let #f = ::core::cell::RefCell::new(#f);
            }
        },
    );

    // Build the FutureComms values.
    let comms_exprs = storage_idents.iter().zip(fut_idents.iter()).map(|(s, f)| {
        quote! {
            #krate::key_facade::FutureComms::new_unsafe(&#s, &#f)
        }
    });

    let binding_rhs = if n == 1 {
        let ce = comms_exprs.collect::<Vec<_>>();
        quote! { #(#ce)* }
    } else {
        quote! { ( #( #comms_exprs ),* ) }
    };

    let expanded = quote! {
        #( #storage_decls )*
        #( #fut_decls )*
        let #binding = #binding_rhs;
    };

    expanded.into()
}
