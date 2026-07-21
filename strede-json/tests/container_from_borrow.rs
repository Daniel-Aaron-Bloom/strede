//! Borrow-family `#[strede(from = "FromType")]` / `#[strede(try_from = "FromType")]`
//! fixtures (container level) — entirely replaces field-by-field deserialization.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
#[strede(from = "u8")]
struct Scale(u32);

impl From<u8> for Scale {
    fn from(v: u8) -> Self {
        Scale(v as u32 * 10)
    }
}

#[derive(Debug, PartialEq, Deserialize)]
#[strede(try_from = "i32")]
struct Port(u16);

impl TryFrom<i32> for Port {
    type Error = ();
    fn try_from(v: i32) -> Result<Self, ()> {
        u16::try_from(v).map(Port).map_err(|_| ())
    }
}

fn parse<'de, T>(input: &'de str) -> Option<T>
where
    T: strede::Deserialize<'de, JsonDeserializer<'de>, Extra = ()>,
{
    let de = JsonDeserializer::new(input.as_bytes());
    match block_on(<T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => Some(v),
        Probe::Miss => None,
    }
}

#[test]
fn container_from_hit() {
    assert_eq!(parse::<Scale>("7"), Some(Scale(70)));
}

#[test]
fn container_from_wrong_type_misses() {
    assert_eq!(parse::<Scale>(r#""7""#), None);
}

#[test]
fn container_try_from_hit() {
    assert_eq!(parse::<Port>("8080"), Some(Port(8080)));
}

#[test]
fn container_try_from_out_of_range_misses() {
    assert_eq!(parse::<Port>("-1"), None);
}
