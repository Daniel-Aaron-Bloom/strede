//! Compile-fail coverage for compile-time `#[strede(flatten)]` wire-name
//! collision detection (`strede::Fields` / `strede::Disjoint`). Each fixture
//! under `tests/flatten_collision_fail/` is expected to fail to compile, with
//! its accompanying `.stderr` snapshot checked in per trybuild convention. If
//! a rustc upgrade shifts the panic-macro scaffolding trybuild captures
//! around our own message text, regenerate via `TRYBUILD=overwrite`.

#[test]
fn collisions_are_rejected_at_compile_time() {
    let t = trybuild::TestCases::new();
    // At least one `.pass(..)` is required so trybuild runs a real `cargo
    // build` (full codegen) instead of `cargo check` — see
    // `tests/flatten_collision_pass/generic_flatten_no_collision.rs` for why
    // that distinction matters here.
    t.pass("tests/flatten_collision_pass/*.rs");
    t.compile_fail("tests/flatten_collision_fail/*.rs");
}
