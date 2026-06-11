use strede::DeserializeOwned;

#[derive(DeserializeOwned)]
struct Inner {
    a: u32,
    b: u32,
}
// Field before AND after the flatten — exercises the before/after arm split.
#[derive(DeserializeOwned)]
struct OuterWithSuffix {
    prefix: u32,
    #[strede(flatten)]
    inner1: Inner,
    #[strede(flatten)]
    inner2: Inner,
    #[strede(flatten)]
    inner3: Inner,
    suffix: u32,
}

fn main() {}
