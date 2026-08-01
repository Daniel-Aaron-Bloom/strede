//! Two fields on the same struct whose `rename`/`alias` values collide, with
//! no `#[strede(flatten)]` involved at all — caught by the self-duplicate
//! check on the struct's own `Fields::NAMES`.
use strede_derive::Deserialize;

#[derive(Deserialize)]
struct Config {
    #[strede(rename = "x")]
    a: u32,
    #[strede(alias = "x")]
    b: u32,
}

fn main() {}
