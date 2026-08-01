//! `Wrapper<T>` is generic over its own `#[strede(flatten)]` target — its
//! wire names can't be known until `T` is concrete, so the derive defers the
//! collision check into `wire_names()`'s body (see `FlattenTier::BareParam`).
//! The struct *definition* compiles fine regardless of `T`; only an actual,
//! concretely-monomorphized *use* with a colliding `T` fails — which is why
//! this fixture must genuinely call `deserialize` (not merely name the type)
//! to force that monomorphization to happen at compile time.
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
struct Colliding {
    x: u32, // collides with Wrapper's own "x"
}

fn main() {
    let de = JsonDeserializer::new(b"{}".as_slice());
    let _ = block_on(<Wrapper<Colliding> as strede::Deserialize<'_, _>>::deserialize(
        de,
        (),
    ));
}
