# strede-json

JSON deserializer backend for [strede](../README.md).

Implements both families:

- **Borrow family** (`JsonDeserializer`) — in-memory, zero-copy, single-pass over a `&[u8]`.
- **Owned family** (`chunked` module) — async streaming over a refillable byte buffer.

## Quickstart

### Borrow family — in-memory

```rust
use strede::Deserialize;
use strede_json::JsonDeserializer;

#[derive(Deserialize)]
struct Point { x: f64, y: f64 }

let json = br#"{"x": 1.0, "y": 2.5}"#;
let (_, point) = Point::deserialize(JsonDeserializer::new(json), ()).await?.unwrap();
```

`JsonDeserializer::new` borrows the source slice for lifetime `'de`. All string
and byte probes that succeed without escape sequences return `&'de str` / `&'de [u8]`
directly from the source buffer — no copies. Probes are synchronous in the sense
that they never return `Pending`; the full input is already in memory.

The root deserializer rejects trailing non-whitespace after the value. Sub-deserializers
(created internally for nested values) skip this check.

### Owned family — chunked streaming

```rust
use strede::DeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede::shared_buf::{SharedBuf, SimpleBuffer};

#[derive(DeserializeOwned)]
struct Point { x: f64, y: f64 }

// SimpleBuffer holds the current chunk; the async closure refills it.
let mut reader = some_async_reader();
let buf = SharedBuf::new(SimpleBuffer::new(), async |b: &mut SimpleBuffer| {
    b.set(reader.next_chunk().await);
});

let d = ChunkedJsonDeserializer::new(buf);
let (_, point) = Point::deserialize_owned(d, ()).await?.unwrap();
```

`ChunkedJsonDeserializer` owns a `SharedBuf<B, F>`. Each `entry` call forks a
`Handle` from the shared buffer; when the tokenizer reaches the end of the
current chunk it awaits the refill closure `F`. An empty buffer signals EOF.

The root deserializer rejects trailing non-whitespace. `ChunkedJsonSubDeserializer`
(used for nested values internally) skips this check.

## Number types

JSON numbers are surfaced via two format-specific types:

```rust
use strede_json::{NumberBorrowed, NumberOwned};

// Borrow family — zero-copy view into the source buffer.
// NumberBorrowed<'de> stores a parsed N enum by default;
// with the `arbitrary_precision` feature it stores &'de str instead.
let n: NumberBorrowed<'_> = /* ... */;
let f: Option<f64> = n.as_f64();
let i: Option<i64> = n.as_i64();
let u: Option<u64> = n.as_u64();

// Owned family — same interface, but owns its data.
// NumberOwned stores a parsed N enum by default;
// with `arbitrary_precision` it stores String instead.
let n: NumberOwned = /* ... */;
```

`NumberBorrowed` implements `Deserialize<'de>` and `NumberOwned` implements
`DeserializeOwned`, so they work as field types in derived structs.

## Value types (`alloc` feature)

With the `alloc` feature enabled, four recursive value types are available:

| type | family | description |
|---|---|---|
| `ValueBorrowed<'de>` | borrow | recursive JSON value tree, zero-copy strings |
| `ValueOwned` | owned | recursive JSON value tree, owned strings |
| `RawValueBorrowed<'de>` | borrow | opaque capture of raw JSON bytes, borrowed |
| `RawValueOwned` | owned | opaque capture of raw JSON bytes, owned |

```rust
use strede_json::{JsonDeserializer, ValueBorrowed};
use strede::Deserialize;

let json = br#"{"x": [1, true, null]}"#;
let (_, val) = ValueBorrowed::deserialize(JsonDeserializer::new(json), ()).await?.unwrap();
```

`RawValueBorrowed` / `RawValueOwned` capture the verbatim JSON bytes of a value
(including nested structure) without parsing them further. They are
JSON-format-specific types; they implement `Deserialize` and `DeserializeOwned`
only against `JsonDeserializer` / `ChunkedJsonDeserializer`, not generically.

## Primitive impls

In strede, `Deserialize<'de>` for numeric types and `bool` is the format
backend's responsibility, not core strede's. This crate provides impls for:

- `bool`
- `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
- `i8`, `i16`, `i32`, `i64`, `i128`, `isize`
- `f32`, `f64`
- `char`
- `()`

These impls are on both `JsonDeserializer` / `JsonSubDeserializer` (borrow family)
and `ChunkedJsonDeserializer` / `ChunkedJsonSubDeserializer` (owned family).

Number parsing uses the Eisel-Lemire fast path for floats; integers are parsed
directly from the wire token.

## Features

| feature | description |
|---|---|
| `alloc` | Enables `ValueBorrowed`, `ValueOwned`, `RawValueBorrowed`, `RawValueOwned` |
| `arbitrary_precision` | `NumberBorrowed` stores `&'de str`; `NumberOwned` stores `String` — raw digits, no precision loss |

## Relationship to `strede`

See the [workspace README](../README.md) for the full trait hierarchy, `select_probe!`,
derive attributes, and the owned-family deadlock rule. This crate is the canonical
reference implementation of a strede format backend.
