//! Outer struct's own field name collides with a flattened internally-tagged
//! enum's tag field literal.
use strede_derive::Deserialize;

#[derive(Deserialize)]
#[strede(tag = "kind")]
enum Message {
    Ping,
    Pong,
}

#[derive(Deserialize)]
struct Envelope {
    kind: String, // collides with Message's own tag field "kind"
    #[strede(flatten)]
    message: Message,
}

fn main() {}
