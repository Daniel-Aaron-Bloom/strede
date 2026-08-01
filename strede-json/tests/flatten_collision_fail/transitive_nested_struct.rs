//! Two flatten levels deep: Outer flattens Mid, Mid flattens Inner, and
//! Outer's own field collides with Inner's field (not Mid's), proving the
//! transitively-unioned `Fields::NAMES` catches collisions beyond one hop.
use strede_derive::Deserialize;

#[derive(Deserialize)]
struct Inner {
    a: u32,
}

#[derive(Deserialize)]
struct Mid {
    b: u32,
    #[strede(flatten)]
    inner: Inner,
}

#[derive(Deserialize)]
struct Outer {
    a: u32, // collides with Inner::a, two flatten-levels away
    #[strede(flatten)]
    mid: Mid,
}

fn main() {}
