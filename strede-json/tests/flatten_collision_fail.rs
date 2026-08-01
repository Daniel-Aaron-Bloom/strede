//! Compile-fail coverage for compile-time `#[strede(flatten)]` wire-name
//! collision detection (`strede::Fields` / `strede::Disjoint`). Each fixture
//! under `tests/flatten_collision_fail/` is expected to fail to compile, with
//! its accompanying `.stderr` snapshot checked in per trybuild convention.
//!
//! The E0080 const-eval-panic diagnostic (the scaffolding trybuild captures
//! around our own message text) has changed shape across rustc versions, and
//! CI runs the test suite against nightly/beta/stable/MSRV. A single
//! committed `.stderr` snapshot can't match all of those at once, so — per
//! dtolnay's own guidance for this exact problem
//! (<https://github.com/dtolnay/trybuild/issues/167>, "run a different set of
//! ui test files on different compiler versions") — this test only runs on
//! our pinned MSRV ("1.89", matching `rust-version` in the workspace
//! `Cargo.toml` and the `check` job in `ci.yml`), the one toolchain in the CI
//! matrix that never drifts, where the snapshots below were generated.
//! Regenerate via `cargo +1.89 test` with `TRYBUILD=overwrite` after a real
//! message-text change.

#[rustversion::stable(1.89)]
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
