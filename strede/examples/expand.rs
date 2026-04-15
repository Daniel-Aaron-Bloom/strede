use strede::{Deserialize, DeserializeOwned};

#[derive(DeserializeOwned, Deserialize)]
pub struct Duration {
    pub secs: u64,
    pub nanos: u32,
}

fn main() {}
