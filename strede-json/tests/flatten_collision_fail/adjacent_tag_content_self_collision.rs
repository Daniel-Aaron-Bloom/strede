//! An adjacently-tagged enum misconfigured with the same field name for both
//! `tag` and `content` — caught by the self-duplicate check on its own
//! `Fields::NAMES`, independent of any flatten usage.
use strede_derive::Deserialize;

#[derive(Deserialize)]
#[strede(tag = "t", content = "t")]
enum Message {
    Ping,
    Data(u32),
}

fn main() {}
