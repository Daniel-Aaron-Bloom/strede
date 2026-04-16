use strede::{Deserialize, DeserializeOwned, Probe};

#[derive(DeserializeOwned, Deserialize)]
pub struct Duration {
    pub secs: u64,
    pub nanos: u32,
}

pub async fn example() -> Result<Probe<u32>, ()> {
    strede::select_probe! {
        async move { Ok::<Probe<u32>, ()>(Probe::Miss) },
        async move { Ok::<Probe<u32>, ()>(Probe::Miss) },
        miss => Ok(Probe::Miss),
    }
}

fn main() {}
