use strede::DeserializeOwned;

#[derive(DeserializeOwned)]
pub struct InnerA {
    pub a1: u32,
    pub a2: u32,
}
#[derive(DeserializeOwned)]
pub struct InnerB {
    pub b1: u32,
    pub b2: u32,
}
#[derive(DeserializeOwned)]
pub struct InnerC {
    pub c1: u32,
    pub c2: u32,
}
// Field before AND after the flatten — exercises the before/after arm split.
#[derive(DeserializeOwned)]
pub struct OuterWithSuffix {
    pub prefix: u32,
    #[strede(flatten)]
    pub inner1: InnerA,
    #[strede(flatten)]
    pub inner2: InnerB,
    #[strede(flatten)]
    pub inner3: InnerC,
    pub suffix: u32,
}

fn main() {}
