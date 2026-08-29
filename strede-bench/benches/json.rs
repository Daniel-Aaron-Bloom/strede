#![recursion_limit = "512"]

use criterion::{Criterion, criterion_group, criterion_main};
use serde::Deserialize as SerdeDeserialize;
use std::hint::black_box;
use strede::Probe;
use strede::{DeserializeOwned, SharedBuf};
use strede_json::{JsonDeserializer, chunked::ChunkedJsonDeserializer};
use strede_test_util::{block_on, block_on_loop};

// ---------------------------------------------------------------------------
// Shared data types
// ---------------------------------------------------------------------------

/// A simple two-field struct — minimal map deserialization.
#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct Point {
    x: i64,
    y: i64,
}

/// A richer struct with mixed types and more fields.
#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct LogEntry {
    id: u64,
    level: u32,
    count: u32,
    ratio: f64,
    active: bool,
}

/// Nested struct — tests map-within-map deserialization.
#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct Rect {
    top_left: Point,
    bottom_right: Point,
}

/// A 5-level deep nested object for deep-nesting benchmarks.
#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct Level5 {
    value: i64,
    active: bool,
}

#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct Level4 {
    name: u32,
    inner: Level5,
}

#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct Level3 {
    count: u32,
    data: Level4,
}

#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct Level2 {
    id: u64,
    ratio: f64,
    child: Level3,
}

#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct DeepNested {
    label: u32,
    meta: LogEntry,
    region: Rect,
    tree: Level2,
}

/// A 15-field struct — matches the field count that triggered the
/// #[inline(always)] -> #[inline] build-time fix, to measure any runtime cost.
#[derive(
    Debug, PartialEq, strede_derive::Deserialize, strede_derive::DeserializeOwned, SerdeDeserialize,
)]
struct Wide15 {
    f0: u64,
    f1: u64,
    f2: u64,
    f3: u64,
    f4: u64,
    f5: u64,
    f6: u64,
    f7: u64,
    f8: u64,
    f9: u64,
    f10: u64,
    f11: u64,
    f12: u64,
    f13: u64,
    f14: u64,
}

// ---------------------------------------------------------------------------
// Benchmark inputs (static byte slices — no allocation at benchmark time)
// ---------------------------------------------------------------------------

static POINT_JSON: &[u8] = br#"{"x": 42, "y": -7}"#;
static POINT_JSON_REORDERED: &[u8] = br#"{"y": -7, "x": 42}"#;

static LOG_JSON: &[u8] =
    br#"{"id": 1234567890, "level": 3, "count": 99, "ratio": 0.5, "active": true}"#;
static LOG_JSON_REORDERED: &[u8] =
    br#"{"active": false, "ratio": 1.23456, "count": 0, "level": 1, "id": 9}"#;

static RECT_JSON: &[u8] =
    br#"{"top_left": {"x": 0, "y": 0}, "bottom_right": {"x": 1920, "y": 1080}}"#;

static WIDE15_JSON: &[u8] = br#"{"f0":0,"f1":1,"f2":2,"f3":3,"f4":4,"f5":5,"f6":6,"f7":7,"f8":8,"f9":9,"f10":10,"f11":11,"f12":12,"f13":13,"f14":14}"#;
static WIDE15_JSON_REORDERED: &[u8] = br#"{"f14":14,"f7":7,"f0":0,"f9":9,"f2":2,"f11":11,"f4":4,"f13":13,"f6":6,"f1":1,"f8":8,"f3":3,"f10":10,"f5":5,"f12":12}"#;

static DEEP_JSON: &[u8] = br#"{
    "label": 7,
    "meta": {"id": 1234567890, "level": 3, "count": 99, "ratio": 0.5, "active": true},
    "region": {"top_left": {"x": 0, "y": 0}, "bottom_right": {"x": 1920, "y": 1080}},
    "tree": {
        "id": 42,
        "ratio": 3.14,
        "child": {
            "count": 5,
            "data": {
                "name": 100,
                "inner": {
                    "value": -999,
                    "active": false
                }
            }
        }
    }
}"#;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run a strede borrow-family deserialization and assert it hit.
macro_rules! strede_borrow {
    ($T:ty, $input:expr) => {{
        let de = JsonDeserializer::new(black_box($input));
        let result = block_on(<$T as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap();
        let Probe::Hit((_, v)) = result else {
            panic!("Miss")
        };
        v
    }};
}

/// Run a strede owned-family deserialization using SharedBuf (single-chunk EOF).
macro_rules! strede_owned {
    ($T:ty, $input:expr) => {{
        let input: &[u8] = black_box($input);
        block_on_loop(SharedBuf::with_async(
            input,
            async |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                <$T as DeserializeOwned<_>>::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap()
            },
        ))
    }};
}

/// Run a serde_json deserialization.
macro_rules! serde {
    ($T:ty, $input:expr) => {{ serde_json::from_slice::<$T>(black_box($input)).unwrap() }};
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_point(c: &mut Criterion) {
    let mut g = c.benchmark_group("point");
    g.bench_function("strede/borrow/in-order", |b| {
        b.iter(|| strede_borrow!(Point, POINT_JSON))
    });
    g.bench_function("strede/borrow/reordered", |b| {
        b.iter(|| strede_borrow!(Point, POINT_JSON_REORDERED))
    });
    g.bench_function("strede/owned/in-order", |b| {
        b.iter(|| strede_owned!(Point, POINT_JSON))
    });
    g.bench_function("strede/owned/reordered", |b| {
        b.iter(|| strede_owned!(Point, POINT_JSON_REORDERED))
    });
    g.bench_function("serde_json/in-order", |b| {
        b.iter(|| serde!(Point, POINT_JSON))
    });
    g.bench_function("serde_json/reordered", |b| {
        b.iter(|| serde!(Point, POINT_JSON_REORDERED))
    });
    g.finish();
}

fn bench_log_entry(c: &mut Criterion) {
    let mut g = c.benchmark_group("log_entry");
    g.bench_function("strede/borrow/in-order", |b| {
        b.iter(|| strede_borrow!(LogEntry, LOG_JSON))
    });
    g.bench_function("strede/borrow/reordered", |b| {
        b.iter(|| strede_borrow!(LogEntry, LOG_JSON_REORDERED))
    });
    g.bench_function("strede/owned/in-order", |b| {
        b.iter(|| strede_owned!(LogEntry, LOG_JSON))
    });
    g.bench_function("strede/owned/reordered", |b| {
        b.iter(|| strede_owned!(LogEntry, LOG_JSON_REORDERED))
    });
    g.bench_function("serde_json/in-order", |b| {
        b.iter(|| serde!(LogEntry, LOG_JSON))
    });
    g.bench_function("serde_json/reordered", |b| {
        b.iter(|| serde!(LogEntry, LOG_JSON_REORDERED))
    });
    g.finish();
}

fn bench_rect(c: &mut Criterion) {
    let mut g = c.benchmark_group("rect");
    g.bench_function("strede/borrow", |b| {
        b.iter(|| strede_borrow!(Rect, RECT_JSON))
    });
    g.bench_function("strede/owned", |b| {
        b.iter(|| strede_owned!(Rect, RECT_JSON))
    });
    g.bench_function("serde_json", |b| b.iter(|| serde!(Rect, RECT_JSON)));
    g.finish();
}

fn bench_deep_nested(c: &mut Criterion) {
    let mut g = c.benchmark_group("deep_nested");
    g.bench_function("strede/borrow", |b| {
        b.iter(|| strede_borrow!(DeepNested, DEEP_JSON))
    });
    g.bench_function("strede/owned", |b| {
        b.iter(|| strede_owned!(DeepNested, DEEP_JSON))
    });
    g.bench_function("serde_json", |b| b.iter(|| serde!(DeepNested, DEEP_JSON)));
    g.finish();
}

fn bench_wide15(c: &mut Criterion) {
    let mut g = c.benchmark_group("wide15");
    g.bench_function("strede/borrow/in-order", |b| {
        b.iter(|| strede_borrow!(Wide15, WIDE15_JSON))
    });
    g.bench_function("strede/borrow/reordered", |b| {
        b.iter(|| strede_borrow!(Wide15, WIDE15_JSON_REORDERED))
    });
    g.bench_function("strede/owned/in-order", |b| {
        b.iter(|| strede_owned!(Wide15, WIDE15_JSON))
    });
    g.bench_function("strede/owned/reordered", |b| {
        b.iter(|| strede_owned!(Wide15, WIDE15_JSON_REORDERED))
    });
    g.bench_function("serde_json/in-order", |b| {
        b.iter(|| serde!(Wide15, WIDE15_JSON))
    });
    g.bench_function("serde_json/reordered", |b| {
        b.iter(|| serde!(Wide15, WIDE15_JSON_REORDERED))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_point,
    bench_log_entry,
    bench_rect,
    bench_deep_nested,
    bench_wide15
);
criterion_main!(benches);
