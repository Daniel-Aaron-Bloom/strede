use std::marker::PhantomData;

// This struct borrows the first two lifetimes but not the third.
#[derive(strede::Deserialize)]
pub struct Three<'a, 'b, 'c> {
    pub a: &'a str,
    pub b: &'b str,
    pub c: PhantomData<&'c str>,
}

#[derive(strede::Deserialize)]
pub struct Example<'a, 'b, 'c> {
    #[strede(borrow = "'a+'b")]
    // Borrow 'a and 'b only, not 'c.
    pub three: Three<'a, 'b, 'c>,
}

fn main() {}
