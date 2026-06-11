# strede-msgpack

[![crates.io](https://img.shields.io/crates/v/strede-msgpack.svg?style=for-the-badge&color=fc8d62&logo=rust)](https://crates.io/crates/strede-msgpack)
[![docs.rs](https://img.shields.io/badge/docs.rs-strede--msgpack-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs)](https://docs.rs/strede-msgpack)

MessagePack format backend for the [strede](https://crates.io/crates/strede) deserialization framework.

Two deserializers are provided:

- `MsgpackDeserializer` — in-memory, borrow family (`'de` zero-copy).
- `ChunkedMsgpackDeserializer` — streaming, owned family (no zero-copy borrows, reads from an async byte source via `SharedBuf`).

Both work with types derived via `#[derive(strede::Deserialize)]` /
`#[derive(strede::DeserializeOwned)]` and with manually implemented trait impls.

## Quickstart

### Borrow family (in-memory)

```rust
use strede::Deserialize;
use strede_derive::Deserialize as DeriveDeserialize;
use strede_msgpack::MsgpackDeserializer;
use strede::Probe;

#[derive(Debug, PartialEq, DeriveDeserialize)]
struct Point {
    x: f64,
    y: f64,
}

// Build a fixmap: {"x": 1.5, "y": 2.5}
// (in practice your bytes come from the wire)
let input: &[u8] = /* ... msgpack bytes ... */;

let de = MsgpackDeserializer::new(input);
let result = Point::deserialize(de, ()).await.unwrap();
if let Probe::Hit((_, point)) = result {
    println!("{point:?}");
}
```

### Owned / streaming family

```rust
use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_msgpack::chunked::ChunkedMsgpackDeserializer;

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
        let de = ChunkedMsgpackDeserializer::new(shared);
        match Point::deserialize_owned(de, ()).await.unwrap() {
            Probe::Hit((_, v)) => Some(v),
            Probe::Miss => None,
        }
    },
).await;
```

## Wire format coverage

| MessagePack type | Token | Decoded as |
|---|---|---|
| nil (`0xc0`) | `Nil` | `()`, `Option<T>` (None) |
| bool (`0xc2`/`0xc3`) | `Bool` | `bool` |
| positive fixint, uint8/16/32/64 | `UInt(u64)` | `u8`, `u16`, `u32`, `u64`, `f32`, `f64` |
| negative fixint, int8/16/32/64 | `Int(i64)` | `i8`, `i16`, `i32`, `i64`, `f32`, `f64` |
| float32 / float64 | `Float32` / `Float64` | `f32`, `f64` (cross-width allowed) |
| fixstr, str8/16/32 | `Str(len)` | `&'de str` / `String`, also `&'de [u8]` / `Vec<u8>` |
| bin8/16/32 | `Bin(len)` | `&'de [u8]` / `Vec<u8>`, also `&'de str` / `String` |
| fixarray, array16/32 | `Array(count)` | tuple structs, sequences |
| fixmap, map16/32 | `Map(count)` | named structs, maps |
| fixext1/2/4/8/16 | `FixExt { type_id, data, len }` | `FixExtWrapper<T>` via `DeserializeFromFixExt`; fixext4/8 with type_id = -1 also decode as `MsgpackTimestamp` |
| ext8/16/32 | `Ext { type_id, len }` | `ExtWrapper<T>` via `DeserializeFromExtBytes` / `DeserializeFromExtBytesOwned`; ext8 len = 12 with type_id = -1 also decodes as `MsgpackTimestamp` |
| any (dynamic) | — | `MsgpackValue` catches all types; requires `alloc` |

Integer types accept any value that fits. For example, a `uint32` value of
`200` deserializes into `u8`, `u16`, `u32`, or `u64`; a value of `300`
misses `u8` but hits `u16`. Negative integers always miss unsigned types.
Float32 and float64 widen and narrow freely. Integers coerce to floats
(e.g. fixint `42` deserializes as `f64`).

Unrecognised format bytes return `Probe::Miss`, not an error.

## Struct and enum representations

These match the rest of the strede ecosystem and the derive macro.

### Named struct — map

```
{"x": 1, "y": 2}
```

```rust
#[derive(strede::Deserialize)]
struct Point { x: u32, y: u32 }
```

Field order is irrelevant. Missing required fields return `Probe::Miss`.
Duplicate fields return `Err`. Unknown fields return `Probe::Miss` by
default; add `#[strede(allow_unknown_fields)]` to skip them silently.

### Tuple struct — array

```
[7]
```

```rust
#[derive(strede::Deserialize)]
struct Wrapper(u32);
```

Tuple structs (including newtypes) deserialize from a msgpack array whose
element count matches the field count.

### Unit struct — nil

```
nil (0xc0)
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

## Derive attributes

All [strede derive attributes](https://crates.io/crates/strede) work
unchanged with msgpack: `rename`, `rename_all`, `alias`, `default`,
`default = "expr"`, `skip_deserializing`, `allow_unknown_fields`,
`transparent`, `flatten`, `tag`, `tag + content`,
`untagged`, `other`, `from`, `try_from`, `deserialize_with`,
`deserialize_owned_with`, `with`, `bound`, `borrow`, `crate`.

## Ext types

MessagePack ext types carry a signed 8-bit `type_id` and a byte payload.
`strede-msgpack` exposes them through two traits and two wrapper types.

### Fixext (fixext1/2/4/8/16)

Fixext payloads are 1, 2, 4, 8, or 16 bytes and are embedded directly in
the token — no async accessor needed. Implement `DeserializeFromFixExt` and
wrap with `FixExtWrapper<T>`:

```rust
use strede::Probe;
use strede_msgpack::{DeserializeFromFixExt, FixExtWrapper, MsgpackError};

struct Timestamp { secs: u32, nsecs: u32 }

impl DeserializeFromFixExt for Timestamp {
    type Extra = ();   // or any side-channel context type

    fn deserialize_from_fixext(
        type_id: i8,
        data: &[u8],
        _extra: (),
    ) -> Result<Probe<Self>, MsgpackError> {
        if type_id != -1 { return Ok(Probe::Miss); }  // msgpack timestamp ext = -1
        if data.len() < 8 { return Ok(Probe::Miss); }
        let secs  = u32::from_be_bytes(data[0..4].try_into().unwrap());
        let nsecs = u32::from_be_bytes(data[4..8].try_into().unwrap());
        Ok(Probe::Hit(Timestamp { secs, nsecs }))
    }
}

// Borrow family:
let de = MsgpackDeserializer::new(bytes);
if let Probe::Hit((_, FixExtWrapper(ts))) =
    FixExtWrapper::<Timestamp>::deserialize(de, ()).await.unwrap()
{
    println!("{}.{}", ts.secs, ts.nsecs);
}
```

### Variable-length ext (ext8/16/32)

Variable ext payloads arrive as a byte stream. Implement
`DeserializeFromExtBytes` (borrow family) or `DeserializeFromExtBytesOwned`
(owned family) and wrap with `ExtWrapper<T>`:

```rust
use strede::{BytesAccess, Chunk, Probe};
use strede_msgpack::{DeserializeFromExtBytes, ExtWrapper, MsgpackBytesAccess, MsgpackClaim, MsgpackError};

struct MyExt(Vec<u8>);

impl<'de> DeserializeFromExtBytes<MsgpackBytesAccess<'de>> for MyExt {
    type Extra = i8;   // expected type_id

    async fn deserialize_from_ext_bytes(
        type_id: i8,
        _len: usize,
        mut bytes: MsgpackBytesAccess<'de>,
        expected: i8,
    ) -> Result<Probe<(MsgpackClaim<'de>, Self)>, MsgpackError> {
        if type_id != expected { return Ok(Probe::Miss); }
        let mut buf = Vec::new();
        loop {
            match bytes.next_bytes(|b| b.to_vec()).await? {
                Chunk::Data((next, chunk)) => { buf.extend_from_slice(&chunk); bytes = next; }
                Chunk::Done(claim)         => return Ok(Probe::Hit((claim, MyExt(buf)))),
            }
        }
    }
}
```

For the owned family, implement `DeserializeFromExtBytesOwned` with
`ChunkedMsgpackBytesAccess<'s, B, F>` instead of `MsgpackBytesAccess<'de>`.

## Built-in types

### `MsgpackTimestamp`

The msgpack spec defines a standard timestamp extension (type_id = -1) with
three binary encodings. `MsgpackTimestamp` decodes all three automatically —
no ext trait impl needed.

```rust
use strede::{Deserialize, Probe};
use strede_msgpack::{MsgpackDeserializer, MsgpackTimestamp};

let de = MsgpackDeserializer::new(bytes);
if let Probe::Hit((_, ts)) = MsgpackTimestamp::deserialize(de, ()).await? {
    println!("seconds={} nanoseconds={}", ts.seconds, ts.nanoseconds);
}
```

The three encodings:

| Encoding | Format | Range |
|---|---|---|
| Timestamp 32 | fixext4, type_id = -1 | seconds 0 – 2³²−1, nanoseconds = 0 |
| Timestamp 64 | fixext8, type_id = -1 | seconds 0 – 2³⁴−1, nanoseconds 0 – 999999999 |
| Timestamp 96 | ext8 len=12, type_id = -1 | seconds i64 (pre-epoch), nanoseconds 0 – 999999999 |

`MsgpackTimestamp` has two public fields: `seconds: i64` and `nanoseconds: u32`.
Timestamp 96 in the **owned family** requires the `alloc` feature; without it,
ext8 len=12 type_id=-1 returns `Probe::Miss`.

### `MsgpackValue`

`MsgpackValue` is a dynamic catch-all that can hold any msgpack value. It
requires the `alloc` feature.

```rust
#[cfg(feature = "alloc")]
{
    use strede::{Deserialize, Probe};
    use strede_msgpack::{MsgpackDeserializer, MsgpackValue};

    let de = MsgpackDeserializer::new(bytes);
    if let Probe::Hit((_, val)) = MsgpackValue::deserialize(de, ()).await? {
        match val {
            MsgpackValue::Str(s)            => println!("string: {s}"),
            MsgpackValue::UInt(n)           => println!("uint: {n}"),
            MsgpackValue::Map(pairs)        => println!("{} keys", pairs.len()),
            MsgpackValue::Timestamp(ts)     => println!("ts: {}s", ts.seconds),
            MsgpackValue::Ext { type_id, data } => println!("ext {type_id}: {} bytes", data.len()),
            _ => {}
        }
    }
}
```

Variants:

| Variant | Holds |
|---|---|
| `Nil` | — |
| `Bool(bool)` | — |
| `Int(i64)` | negative fixint, int8/16/32/64 |
| `UInt(u64)` | positive fixint, uint8/16/32/64 |
| `Float32(f32)` / `Float64(f64)` | — |
| `Str(String)` | fixstr, str8/16/32 |
| `Bin(Vec<u8>)` | bin8/16/32 |
| `Array(Vec<MsgpackValue>)` | fixarray, array16/32 |
| `Map(Vec<(MsgpackValue, MsgpackValue)>)` | fixmap, map16/32; key order preserved |
| `Timestamp(MsgpackTimestamp)` | ext type_id = -1 (all three encodings) |
| `Ext { type_id: i8, data: Vec<u8> }` | all other ext types |

`Map` uses `Vec` pairs rather than a hash map so key order is preserved and
any key type (including non-string keys) is supported.

## Feature flags

| Feature | Description |
|---|---|
| `alloc` | Enables `MsgpackValue` and Timestamp 96 decoding in the owned family. |

## Error type

`MsgpackError` implements `strede::DeserializeError`:

| Variant | Meaning |
|---|---|
| `UnexpectedEnd` | Buffer ended before the value was complete |
| `UnexpectedByte { byte }` | Format byte `0xc1` (reserved, never valid) |
| `InvalidUtf8` | A `Str` payload is not valid UTF-8 |
| `ExpectedEnd` | Trailing bytes after the top-level value |
| `DuplicateField(name)` | A map key appeared more than once |
| `SkipDepthExceeded` | Skipping a value exceeded the no-alloc nesting limit (never emitted with the `alloc` feature) |

## Limitations

- **Skip nesting depth**: the chunked deserializer's skip routine uses a
  compact iterative stack with 32 slots. With the `alloc` feature enabled,
  values that overflow the stack fall back to a `Box::pin`-ed recursive call,
  so arbitrarily deep nesting is handled without error. Without `alloc`,
  values nested deeper than the stack capacity return
  `Err(MsgpackError::SkipDepthExceeded)`. The borrow-family skip is
  recursion-based and is limited by the call stack instead.

## Workspace

| crate | description |
|---|---|
| [`strede`](https://crates.io/crates/strede) | core traits (borrow + owned families), `shared_buf` |
| [`strede-json`](https://crates.io/crates/strede-json) | JSON backend (in-memory + chunked/streaming) |
| [`strede-msgpack`](https://crates.io/crates/strede-msgpack) | MessagePack backend (in-memory + chunked/streaming) |
| [`strede-derive`](https://crates.io/crates/strede-derive) | proc-macro: `Deserialize`, `DeserializeOwned` |
