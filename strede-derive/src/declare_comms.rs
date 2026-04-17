use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    Expr, Ident, Token,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

/// One closure arm: optionally `&`-prefixed expression
struct CommsArm {
    by_ref: bool,
    expr: Expr,
}

enum CommsBody {
    Single(CommsArm),
    Tuple(Vec<CommsArm>),
}

enum Binding {
    Single(Ident),
    Tuple(Vec<Ident>),
}

struct DeclareCommsInput {
    krate: syn::Path,
    binding: Binding,
    body: CommsBody,
}

fn parse_arm(input: ParseStream) -> syn::Result<CommsArm> {
    let by_ref = input.peek(Token![&]);
    if by_ref {
        let _: Token![&] = input.parse()?;
    }
    let expr: Expr = input.parse()?;
    Ok(CommsArm { by_ref, expr })
}

impl Parse for DeclareCommsInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Leading `@crate_path` injected by the wrapper macro.
        let _: Token![@] = input.parse()?;
        let krate: syn::Path = input.parse()?;

        let _: Token![let] = input.parse()?;

        // Binding is either a parenthesized tuple of idents or a single ident.
        let binding = if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            let idents = content.parse_terminated(Ident::parse, Token![,])?;
            Binding::Tuple(idents.into_iter().collect())
        } else {
            Binding::Single(input.parse()?)
        };

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

    let by_refs: Vec<bool> = arms.iter().map(|a| a.by_ref).collect();
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

    // Build the FutureComms values — each gets its own let binding so we can
    // reference or move them independently.
    let comms_idents: Vec<Ident> = (0..n)
        .map(|i| Ident::new(&format!("__comms_val_{i}"), Span::call_site()))
        .collect();

    let comms_lets = comms_idents.iter().zip(storage_idents.iter()).zip(fut_idents.iter()).map(
        |((c, s), f)| {
            quote! {
                let #c = #krate::key_facade::FutureComms::new_unsafe(&#s, &#f);
            }
        },
    );

    let comms_refs: Vec<_> = comms_idents.iter().zip(by_refs.iter()).map(|(c, &by_ref)| {
        if by_ref {
            quote! { &#c }
        } else {
            quote! { #c }
        }
    }).collect();

    let binding_stmt = match binding {
        Binding::Single(ident) => {
            let rhs = if n == 1 {
                quote! { #(#comms_refs)* }
            } else {
                quote! { ( #( #comms_refs ),* ) }
            };
            quote! { let #ident = #rhs; }
        }
        Binding::Tuple(idents) => {
            quote! { let ( #( #idents ),* ) = ( #( #comms_refs ),* ); }
        }
    };

    let expanded = quote! {
        #( #storage_decls )*
        #( #fut_decls )*
        #( #comms_lets )*
        #binding_stmt
    };

    expanded.into()
}
