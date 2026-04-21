/// Profiling harness for the borrow-family (JsonDeserializer) path.
/// Run via: cargo flamegraph -p strede-bench --bin profile_borrow
use strede::Probe;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(strede_derive::Deserialize, Debug)]
pub struct Point {
    pub x: i64,
    pub y: i64,
}

#[derive(strede_derive::Deserialize, Debug)]
pub struct LogEntry {
    pub id: u64,
    pub level: u32,
    pub count: u32,
    pub ratio: f64,
    pub active: bool,
}

#[inline(never)]
fn deser_point(input: &[u8]) -> Point {
    let de = JsonDeserializer::new(input);
    let result = block_on(<Point as strede::Deserialize<'_>>::deserialize(de, ())).unwrap();
    let Probe::Hit((_, v)) = result else {
        panic!("Miss")
    };
    v
}

#[inline(never)]
fn deser_log_entry(input: &[u8]) -> LogEntry {
    let de = JsonDeserializer::new(input);
    let result = block_on(<LogEntry as strede::Deserialize<'_>>::deserialize(de, ())).unwrap();
    let Probe::Hit((_, v)) = result else {
        panic!("Miss")
    };
    v
}

fn main() {
    let point_json = br#"{"x": 42, "y": -7}"#;
    let log_json = br#"{"id": 1234567890, "level": 3, "count": 99, "ratio": 0.5, "active": true}"#;

    for _ in 0..30_000_000 {
        std::hint::black_box(deser_point(std::hint::black_box(point_json)));
        std::hint::black_box(deser_log_entry(std::hint::black_box(log_json)));
    }
}
