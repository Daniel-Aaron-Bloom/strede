use strede::DeserializeOwned;

#[derive(DeserializeOwned)]
pub struct Inner {
    pub a: u32,
    pub b: u32,
}
// Field before AND after the flatten — exercises the before/after arm split.
#[derive(DeserializeOwned)]
pub struct OuterWithSuffix {
    pub prefix: u32,
    #[strede(flatten)]
    pub inner1: Inner,
    #[strede(flatten)]
    pub inner2: Inner,
    #[strede(flatten)]
    pub inner3: Inner,
    pub suffix: u32,
}

fn main() {}
