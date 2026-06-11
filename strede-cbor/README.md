# strede-cbor

[![crates.io](https://img.shields.io/crates/v/strede-cbor.svg?style=for-the-badge&color=fc8d62&logo=rust)](https://crates.io/crates/strede-cbor)
[![docs.rs](https://img.shields.io/badge/docs.rs-strede--cbor-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs)](https://docs.rs/strede-cbor)

CBOR (RFC 8949) format backend for the [strede](https://crates.io/crates/strede) deserialization framework.

Two deserializers are provided:

- `CborDeserializer` — in-memory, borrow family (`'de` zero-copy).
- `ChunkedCborDeserializer` — streaming, owned family (no zero-copy borrows, reads from an async byte source via `SharedBuf`).

Both work with types derived via `#[derive(strede::Deserialize)]` /
`#[derive(strede::DeserializeOwned)]` and with manually implemented trait impls.

## Quickstart

### Borrow family (in-memory)

```rust
use strede::{Deserialize, Probe};
use strede_derive::Deserialize as DeriveDeserialize;
use strede_cbor::CborDeserializer;

#[derive(Debug, PartialEq, DeriveDeserialize)]
struct Point {
    x: f64,
    y: f64,
}

// bytes contains a CBOR map: {"x": 1.5, "y": 2.5}
let bytes: &[u8] = /* ... CBOR-encoded bytes ... */;

let de = CborDeserializer::new(bytes);
let result = Point::deserialize(de, ()).await.unwrap();
if let Probe::Hit((_, point)) = result {
    println!("{point:?}");
}
```

### Owned / streaming family

```rust
use strede::{DeserializeOwned, Probe};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_cbor::chunked::ChunkedCborDeserializer;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Point {
    x: f64,
    y: f64,
}

// SharedBuf wraps an async byte source and coordinates multi-reader access.
let result = SharedBuf::with_async(
    source,
    async |buf| { /* refill callback */ },
    async |shared| {
        let de = ChunkedCborDeserializer::new(shared);
        match Point::deserialize_owned(de, ()).await.unwrap() {
            Probe::Hit((_, v)) => Some(v),
            Probe::Miss => None,
        }
    },
).await;
```

## Wire format coverage

| CBOR type | Major type | Token | Decoded as |
|---|---|---|---|
| unsigned int (0–23) | 0 | `UInt(u64)` | `u8`, `u16`, `u32`, `u64`, `f32`, `f64` |
| unsigned int (uint8/16/32/64) | 0 | `UInt(u64)` | same |
| negative int | 1 | `NegInt(u64)` | `i8`, `i16`, `i32`, `i64`, `f32`, `f64` |
| byte string (definite) | 2 | `Bstr(len)` | `&'de [u8]` / `Vec<u8>` |
| byte string (indefinite) | 2 | `BstrIndef` | single-chunk: `&'de [u8]`; multi-chunk: chunks accessor |
| text string (definite) | 3 | `Tstr(len)` | `&'de str` / `String` |
| text string (indefinite) | 3 | `TstrIndef` | single-chunk: `&'de str`; multi-chunk: chunks accessor |
| array (definite) | 4 | `Array(Some(n))` | tuple structs, sequences |
| array (indefinite) | 4 | `Array(None)` | same |
| map (definite) | 5 | `Map(Some(n))` | named structs, maps |
| map (indefinite) | 5 | `Map(None)` | same |
| semantic tag | 6 | `Tag(u64)` | stripped by default; use `CborTag<T, H>` to inspect |
| bignum (tag 2/3) | 6 | `Tag(2\|3)` + `Bstr` | `deserialize_number_chunks::<BigEndian>()` streams magnitude bytes |
| false / true | 7 | `Bool(bool)` | `bool` |
| null / undefined | 7 | `Null` / `Undefined` | `()`, `Option<T>` (None) |
| float16 | 7 | `Float16(f32)` | `f32`, `f64` |
| float32 | 7 | `Float32(f32)` | `f32`, `f64` |
| float64 | 7 | `Float64(f64)` | `f32`, `f64` |
| any (dynamic) | — | — | `CborValue` catches all types; requires `alloc` |

Integer types accept any value that fits. For example, a `uint16` value of
`200` deserializes into `u8`, `u16`, `u32`, or `u64`; a value of `300` misses
`u8` but hits `u16`. Negative integers always miss unsigned types. Float16,
float32, and float64 widen and narrow freely. Integers coerce to floats
(e.g. a small uint deserializes as `f64`).

Unrecognised format bytes return `Probe::Miss`, not an error.

## Struct and enum representations

These match the rest of the strede ecosystem and the derive macro.
All representations work with both the borrow family (`#[derive(strede::Deserialize)]`)
and the owned/streaming family (`#[derive(strede::DeserializeOwned)]`).

### Named struct — map

```
{"x": 1, "y": 2}
```

```rust
#[derive(strede::Deserialize)]
struct Point { x: u32, y: u32 }
```

Field order is irrelevant. Missing required fields return `Probe::Miss`.
Duplicate fields return `Err`. Unknown fields return `Probe::Miss` by default;
add `#[strede(allow_unknown_fields)]` to skip them silently.

### Tuple struct — array

```
[7]
```

```rust
#[derive(strede::Deserialize)]
struct Wrapper(u32);
```

Tuple structs (including newtypes) deserialize from a CBOR array whose
element count matches the field count.

### Unit struct — null or undefined

```
null (0xf6)
```

```rust
#[derive(strede::Deserialize)]
struct Unit;
```

### Externally tagged enum (default)

Unit variants are bare strings; non-unit variants are single-key maps:

```
"Ping"
{"Move": {"x": 1.0, "y": 2.0}}
```

```rust
#[derive(strede::Deserialize)]
enum Event {
    Ping,
    Move { x: f64, y: f64 },
}
```

### Internally tagged enum

```
{"type": "Move", "x": 1.0, "y": 2.0}
```

```rust
#[derive(strede::Deserialize)]
#[strede(tag = "type")]
enum Event {
    Ping,
    Move { x: f64, y: f64 },
}
```

### Adjacently tagged enum

```
{"t": "Move", "c": {"x": 1.0, "y": 2.0}}
```

```rust
#[derive(strede::Deserialize)]
#[strede(tag = "t", content = "c")]
enum Event {
    Ping,
    Move { x: f64, y: f64 },
}
```

## Semantic tags

CBOR semantic tags (major type 6) are stripped automatically before dispatching
to type probes. If you need to inspect or require a tag, use
`CborTag<T, H: TagHandler>`:

```rust
use strede_cbor::tag::{CborTag, Required};

// Only accept values wrapped in tag number 1 (epoch-based date/time).
type EpochTimestamp = CborTag<u64, Required<1>>;

let de = CborDeserializer::new(bytes);
if let Probe::Hit((_, CborTag { value, .. })) = EpochTimestamp::deserialize(de, ()).await? {
    println!("epoch seconds: {value}");
}
```

### `TagHandler` implementations

| Type | Behaviour |
|---|---|
| `Ignored` (default) | Tags are discarded; any tag number is accepted. |
| `Required<const N: u64>` | Returns `Probe::Miss` if the tag number is not `N`. |
| `Captured` | Records the tag number for later inspection via `CborTag::handler`. |
| `Accepted<const N: u64>` | Accepts tag `N` if present; also accepts no tag at all. |

## `CborValue` — dynamic type

`CborValue` is a dynamic catch-all that can hold any CBOR value. It requires
the `alloc` feature.

```rust
#[cfg(feature = "alloc")]
{
    use strede::{Deserialize, Probe};
    use strede_cbor::{CborDeserializer, CborValue};

    let de = CborDeserializer::new(bytes);
    if let Probe::Hit((_, val)) = CborValue::deserialize(de, ()).await? {
        match val {
            CborValue::Tstr(s)    => println!("text: {s}"),
            CborValue::UInt(n)    => println!("uint: {n}"),
            CborValue::Map(pairs) => println!("{} keys", pairs.len()),
            CborValue::Tag { number, value } => println!("tag {number}: {value:?}"),
            _ => {}
        }
    }
}
```

Variants:

| Variant | Holds |
|---|---|
| `Null` | — |
| `Undefined` | — |
| `Bool(bool)` | — |
| `UInt(u64)` | major type 0 |
| `Int(i64)` | major type 1, values −2⁶³ … −1 |
| `NegIntOverflow(u64)` | major type 1, values below −2⁶³ (raw additional value N; actual = −1−N) |
| `Float(f64)` | float16 / float32 / float64, all widened to f64 |
| `Bstr(Vec<u8>)` | major type 2 (definite and indefinite) |
| `Tstr(String)` | major type 3 (definite and indefinite) |
| `Array(Vec<CborValue>)` | major type 4 (definite and indefinite) |
| `Map(Vec<(CborValue, CborValue)>)` | major type 5; key order preserved |
| `Tag { number: u64, value: Box<CborValue> }` | major type 6 |

`Map` uses `Vec` pairs rather than a hash map so key order is preserved and
any key type (including non-string and non-integer keys) is supported.

## Derive attributes

All [strede derive attributes](https://crates.io/crates/strede) work
unchanged with CBOR: `rename`, `rename_all`, `alias`, `default`,
`default = "expr"`, `skip_deserializing`, `allow_unknown_fields`,
`transparent`, `flatten`, `tag`, `tag + content`,
`untagged`, `other`, `from`, `try_from`, `deserialize_with`,
`deserialize_owned_with`, `with`, `bound`, `borrow`, `crate`.

## Feature flags

| Feature | Description |
|---|---|
| `alloc` | Enables `CborValue` and heap-allocated string/bytes decoding (`String`, `Vec<u8>`). |

## Error type

`CborError` implements `strede::DeserializeError`:

| Variant | Meaning |
|---|---|
| `UnexpectedEnd` | Buffer ended before the value was complete |
| `UnexpectedByte { byte }` | Reserved or invalid format byte |
| `InvalidUtf8` | A text string payload is not valid UTF-8 |
| `ExpectedEnd` | Trailing bytes after the top-level value |
| `DuplicateField(name)` | A map key appeared more than once |
| `InvalidBreak` | Break code (0xff) outside an indefinite-length context |
| `SkipDepthExceeded` | Skipping a value exceeded the no-alloc nesting limit |

## Limitations

- **Skip nesting depth**: the skip routine uses a compact iterative stack with
  64 slots. Values nested deeper than 64 levels return
  `Err(CborError::SkipDepthExceeded)`.
- **Indefinite bstr/tstr — borrow family**: a single-chunk indefinite byte or
  text string returns a zero-copy `Probe::Hit`. Multi-chunk indefinite strings
  return `Probe::Miss` from `deserialize_bytes` / `deserialize_str`; use
  `deserialize_bytes_chunks` / `deserialize_str_chunks` to consume all chunks.

## Workspace

| crate | description |
|---|---|
| [`strede`](https://crates.io/crates/strede) | core traits (borrow + owned families), `shared_buf` |
| [`strede-json`](https://crates.io/crates/strede-json) | JSON backend (in-memory + chunked/streaming) |
| [`strede-msgpack`](https://crates.io/crates/strede-msgpack) | MessagePack backend (in-memory + chunked/streaming) |
| [`strede-cbor`](https://crates.io/crates/strede-cbor) | CBOR backend (in-memory + chunked/streaming) |
| [`strede-derive`](https://crates.io/crates/strede-derive) | proc-macro: `Deserialize`, `DeserializeOwned` |
