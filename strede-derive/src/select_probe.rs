use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Expr, Pat, Token, parse::Parse, parse::ParseStream, parse_macro_input};

/// An arm in `select_probe!`: `pat = expr => body`
struct ProbeArm {
    pat: Pat,
    expr: Expr,
    body: Expr,
}

/// Optional final `miss => body` arm.
struct MissArm {
    body: Expr,
}

struct SelectProbeInput {
    krate: syn::Path,
    arms: Vec<ProbeArm>,
    miss: Option<MissArm>,
}

impl Parse for SelectProbeInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse leading `@crate_path` injected by the wrapper macro.
        let _: Token![@] = input.parse()?;
        let krate: syn::Path = input.parse()?;

        let mut arms = Vec::new();
        let mut miss = None;

        while !input.is_empty() {
            // Peek for `miss =>` (identifier "miss" followed by `=>`)
            if input.peek(syn::Ident) {
                let fork = input.fork();
                let ident: syn::Ident = fork.parse()?;
                if ident == "miss" && fork.peek(Token![=>]) {
                    // Consume from real stream
                    let _: syn::Ident = input.parse()?;
                    let _: Token![=>] = input.parse()?;
                    let body: Expr = input.parse()?;
                    let _ = input.parse::<Token![,]>();
                    miss = Some(MissArm { body });
                    break;
                }
            }

            // Regular arm: `pat = expr => body`
            let pat = Pat::parse_multi_with_leading_vert(input)?;
            let _: Token![=] = input.parse()?;
            let expr: Expr = input.parse()?;
            let _: Token![=>] = input.parse()?;
            // Body: either a block `{ ... }` or a single expression up to `,`
            let body: Expr = input.parse()?;
            let _ = input.parse::<Token![,]>();
            arms.push(ProbeArm { pat, expr, body });
        }

        Ok(SelectProbeInput { krate, arms, miss })
    }
}

/// Race multiple probe futures.
///
/// Each arm has the form `pat = expr => body` where `expr` is a probe future
/// (`Result<Probe<T>, E>`), `pat` is bound from the `Hit(T)` inner value, and
/// `body` is evaluated when the arm wins.
///
/// An optional final `miss => body` arm fires when every probe returned
/// `Probe::Miss`.  Without it the macro produces `Ok(Probe::Miss)`.
///
/// The macro expands to a `poll_fn` that polls all non-missed arms on each
/// wakeup (in declaration order).  `Pending` means "waiting for data";
/// `Miss` marks an arm done; first `Hit` wins; `Err` short-circuits.
pub fn select_probe_impl(input: TokenStream) -> TokenStream {
    let SelectProbeInput { krate, arms, miss } = parse_macro_input!(input as SelectProbeInput);

    let n = arms.len();

    // Generate per-arm variable names: __probe_fut_0, __probe_miss_0, …
    let fut_names: Vec<_> = (0..n).map(|i| format_ident!("__probe_fut_{}", i)).collect();
    let miss_names: Vec<_> = (0..n)
        .map(|i| format_ident!("__probe_miss_{}", i))
        .collect();

    let fut_inits: Vec<_> = arms
        .iter()
        .zip(&fut_names)
        .map(|(arm, fut)| {
            let expr = &arm.expr;
            quote! { let mut #fut = ::core::pin::pin!(#expr); }
        })
        .collect();

    let miss_inits: Vec<_> = miss_names
        .iter()
        .map(|m| {
            quote! { let mut #m = false; }
        })
        .collect();

    let poll_blocks: Vec<_> = arms
        .iter()
        .zip(&fut_names)
        .zip(&miss_names)
        .map(|((arm, fut), done)| {
            let pat = &arm.pat;
            let body = &arm.body;
            quote! {
                if !#done {
                    match ::core::future::Future::poll(
                        ::core::pin::Pin::as_mut(&mut #fut), __cx
                    ) {
                        ::core::task::Poll::Ready(::core::result::Result::Ok(
                            #krate::Probe::Hit(#pat)
                        )) => {
                            return ::core::task::Poll::Ready(#body);
                        }
                        ::core::task::Poll::Ready(::core::result::Result::Ok(
                            #krate::Probe::Miss
                        )) => {
                            #done = true;
                        }
                        ::core::task::Poll::Ready(::core::result::Result::Err(__probe_err)) => {
                            return ::core::task::Poll::Ready(
                                ::core::result::Result::Err(__probe_err)
                            );
                        }
                        ::core::task::Poll::Pending => {}
                    }
                }
            }
        })
        .collect();

    // All-miss condition: `true && __probe_miss_0 && __probe_miss_1 && …`
    let all_miss = if miss_names.is_empty() {
        quote! { true }
    } else {
        let checks = miss_names.iter().map(|m| quote! { && #m });
        quote! { true #( #checks )* }
    };

    let miss_body = match miss {
        Some(m) => {
            let b = &m.body;
            quote! { #b }
        }
        None => quote! { ::core::result::Result::Ok(#krate::Probe::Miss) },
    };

    let expanded = quote! {
        {
            #( #fut_inits )*
            #( #miss_inits )*
            ::core::future::poll_fn(|__cx| {
                #( #poll_blocks )*
                if #all_miss {
                    return ::core::task::Poll::Ready(#miss_body);
                }
                ::core::task::Poll::Pending
            }).await
        }
    };

    expanded.into()
}
