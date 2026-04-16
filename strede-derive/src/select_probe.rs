use proc_macro::TokenStream;
use quote::quote;
use syn::{Expr, Token, parse::Parse, parse::ParseStream, parse_macro_input};

/// Optional final `miss => body` arm.
struct MissArm {
    body: Expr,
}

struct SelectProbeInput {
    krate: syn::Path,
    arms: Vec<Expr>,
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

            // Arm: any expression (bare probe call, async block, etc.)
            let expr: Expr = input.parse()?;
            let _ = input.parse::<Token![,]>();
            arms.push(expr);
        }

        Ok(SelectProbeInput { krate, arms, miss })
    }
}

/// Race multiple probe futures.
///
/// Each arm is an expression that evaluates to a future returning
/// `Result<Probe<T>, E>`.  Arms can be bare probe calls
/// (`e.deserialize_i64()`) or `async move { ... }` blocks for complex
/// logic that needs `.await`.
///
/// An optional final `miss => body` arm fires when every probe returned
/// `Probe::Miss`.  Without it the macro produces `Ok(Probe::Miss)`.
///
/// The macro expands to a `poll_fn` that polls all non-missed arms on each
/// wakeup (in declaration order).  `Pending` means "waiting for data";
/// `Miss` marks an arm done; first `Hit` wins; `Err` short-circuits.
///
/// A `__probe_kill_ref: &[Cell<bool>; N]` is in scope for all arm expressions.
/// Setting `__probe_kill_ref[i].set(true)` causes arm `i` to drop its future
/// and stop polling on the next wake-up.
pub fn select_probe_impl(input: TokenStream) -> TokenStream {
    let SelectProbeInput { krate, arms, miss } = parse_macro_input!(input as SelectProbeInput);

    let n = arms.len();

    let arm_indices: Vec<syn::Index> = (0..n).map(syn::Index::from).collect();

    // Each future is stored as `pin!(Some(expr))` so its slot can be cleared to
    // `None` when killed, dropping the future in place without moving it.
    let fut_exprs: Vec<_> = arms
        .iter()
        .map(|expr| quote! { ::core::pin::pin!(::core::option::Option::Some(#expr)) })
        .collect();

    // `Cell<bool>` is not `Copy`, so we emit one `Cell::new(false)` per arm.
    let kill_cells = (0..n).map(|_| quote! { ::core::cell::Cell::new(false) });

    // Single kill-sweep block emitted once per poll: checks all arms under one
    // __probe_new_kills guard, using compile-time tuple indices throughout.
    let kill_checks: Vec<_> = arm_indices
        .iter()
        .map(|idx| {
            quote! {
                if !__probe_done[#idx] && __probe_kill_ref[#idx].get() {
                    __probe_fut.#idx.as_mut().set(None);
                    __probe_done[#idx] = true;
                }
            }
        })
        .collect();

    let kill_sweep = quote! {
        if __probe_new_kills.get() {
            #( #kill_checks )*
            __probe_new_kills.set(false);
        }
    };

    let poll_blocks: Vec<_> = arm_indices
        .iter()
        .map(|idx| {
            quote! {
                #kill_sweep
                if !__probe_done[#idx] {
                    let __inner_pin = ::core::pin::Pin::as_mut(&mut __probe_fut.#idx)
                        .as_pin_mut()
                        .unwrap_or_else(|| panic!("probe slot is Some while !done"));
                    match ::core::future::Future::poll(__inner_pin, __cx) {
                        ::core::task::Poll::Ready(::core::result::Result::Ok(
                            #krate::Probe::Hit(__probe_val)
                        )) => {
                            return ::core::task::Poll::Ready(
                                ::core::result::Result::Ok(#krate::Probe::Hit(__probe_val))
                            );
                        }
                        ::core::task::Poll::Ready(::core::result::Result::Ok(
                            #krate::Probe::Miss
                        )) => {
                            __probe_done[#idx] = true;
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

    let miss_body = match miss {
        Some(m) => {
            let b = &m.body;
            quote! { #b }
        }
        None => quote! { ::core::result::Result::Ok(#krate::Probe::Miss) },
    };

    let expanded = quote! {
        {
            // Kill flags: arm expressions can capture __probe_kill_ref (a Copy &-ref)
            // and call kill!(i) to schedule a sibling arm for dropping.
            let __probe_kill_flags = [ #( #kill_cells ),* ];
            let __probe_kill_ref = &__probe_kill_flags;
            let __probe_new_kills = ::core::cell::Cell::new(false);
            let __probe_new_kills = &__probe_new_kills;
            #[allow(unused_macros)]
            macro_rules! kill {
                ($i:literal) => {{
                    __probe_kill_ref[$i].set(true);
                    __probe_new_kills.set(true);
                }};
            }
            let mut __probe_fut = ( #( #fut_exprs ,)* );
            let mut __probe_done = [false; #n];
            ::core::future::poll_fn(|__cx| {
                #( #poll_blocks )*
                if __probe_done.iter().all(|__d| *__d) {
                    return ::core::task::Poll::Ready(#miss_body);
                }
                ::core::task::Poll::Pending
            }).await
        }
    };

    expanded.into()
}
