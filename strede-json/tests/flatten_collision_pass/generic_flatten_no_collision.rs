//! Same shape as `flatten_collision_fail/generic_flatten_collision.rs`, but
//! `T`'s own field doesn't collide with `Wrapper`'s — must compile and
//! actually parse correctly. Also present so the driving `TestCases` includes
//! at least one `.pass(..)` fixture: trybuild only runs a real `cargo build`
//! (full codegen, required for the deferred-to-monomorphization collision
//! check to fire — see `Disjoint`) when at least one fixture is registered
//! via `.pass(..)`; with only `.compile_fail(..)` entries it uses `cargo
//! check`, which skips codegen entirely and would silently miss a
//! `FlattenTier::BareParam` collision.
use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Deserialize)]
struct Wrapper<T> {
    x: u32,
    #[strede(flatten)]
    inner: T,
}

#[derive(Deserialize)]
struct NonColliding {
    y: u32,
}

fn main() {
    let de = JsonDeserializer::new(br#"{"x": 1, "y": 2}"#.as_slice());
    let result = block_on(<Wrapper<NonColliding> as strede::Deserialize<'_, _>>::deserialize(
        de,
        (),
    ))
    .unwrap();
    match result {
        Probe::Hit((_, w)) => {
            assert_eq!(w.x, 1);
            assert_eq!(w.inner.y, 2);
        }
        Probe::Miss => panic!("expected a Hit"),
    }
}
