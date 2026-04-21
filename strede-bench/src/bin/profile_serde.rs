/// Profiling harness for serde_json — baseline comparison.
/// Run via: cargo flamegraph -p strede-bench --bin profile_serde
use serde::Deserialize as SerdeDeserialize;

#[derive(SerdeDeserialize, Debug)]
pub struct Point {
    pub x: i64,
    pub y: i64,
}

#[derive(SerdeDeserialize, Debug)]
pub struct LogEntry {
    pub id: u64,
    pub level: u32,
    pub count: u32,
    pub ratio: f64,
    pub active: bool,
}

#[inline(never)]
fn deser_point(input: &[u8]) -> Point {
    serde_json::from_slice(input).unwrap()
}

#[inline(never)]
fn deser_log_entry(input: &[u8]) -> LogEntry {
    serde_json::from_slice(input).unwrap()
}

fn main() {
    let point_json = br#"{"x": 42, "y": -7}"#;
    let log_json = br#"{"id": 1234567890, "level": 3, "count": 99, "ratio": 0.5, "active": true}"#;

    for _ in 0..5_000_000 {
        std::hint::black_box(deser_point(std::hint::black_box(point_json)));
        std::hint::black_box(deser_log_entry(std::hint::black_box(log_json)));
    }
}
