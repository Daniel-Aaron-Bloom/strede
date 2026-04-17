use strede::{Deserialize, DeserializeOwned, MapKeyEntryOwned, Probe, declare_comms, key_facade::KeyDeserializer, select_probe};

#[derive(DeserializeOwned, Deserialize)]
pub struct Duration {
    pub secs: u64,
    pub nanos: u32,
}

pub async fn example() -> Result<Probe<u32>, ()> {
    select_probe! {
        async move { Ok::<Probe<u32>, ()>(Probe::Miss) },
        async move { Ok::<Probe<u32>, ()>(Probe::Miss) },
        miss => Ok(Probe::Miss),
    }
}

pub async fn example2<'a, E>()
where E: MapKeyEntryOwned<'a>
  {
    declare_comms! {
        let (_comms_xy, _comms_ab) = (
            &|c| Duration::deserialize_owned(KeyDeserializer::<E>::new(c), ()),
            |c| Duration::deserialize_owned(KeyDeserializer::<E>::new(c), ())
        )
    }
}

#[derive(DeserializeOwned)]
pub struct Vec2 {
    x: f64,
    y: f64,
}

#[derive(DeserializeOwned)]
#[strede(tag = "type")]
pub enum TaggedEvent {
    Ping,
    Move { x: f64, y: f64 },
    Teleport(Vec2),
}

fn main() {}
