# strede-postcard

Postcard format backend for [strede](../strede).

Postcard is a compact binary format used in embedded/`no_std` Rust. Unlike JSON or MessagePack, it carries **no type tags** — structure is entirely schema-driven, determined by the Rust type being deserialized.

Two deserializers are provided:

- `PostcardDeserializer` — in-memory, borrow family (`'de` zero-copy).
- `ChunkedPostcardDeserializer` — streaming, owned family (no zero-copy borrows, reads from an async byte source via `SharedBuf`) — well suited to postcard's typical embedded/serial-transport use case, where bytes arrive incrementally rather than as one complete buffer.

Both work with types derived via `#[derive(strede::Deserialize)]` /
`#[derive(strede::DeserializeOwned)]` and with manually implemented trait impls.

## Quickstart

### Borrow family (in-memory)

```rust
use strede::{Deserialize, Probe};
use strede_derive::Deserialize as DeriveDeserialize;
use strede_postcard::PostcardDeserializer;

#[derive(Debug, PartialEq, DeriveDeserialize)]
struct Point {
    x: u32,
    y: u32,
}

let input: &[u8] = /* ... postcard bytes ... */;

let de = PostcardDeserializer::new(input);
if let Probe::Hit((_, point)) = Point::deserialize(de, ()).await.unwrap() {
    println!("{point:?}");
}
```

### Owned / streaming family

```rust
use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_postcard::chunked::ChunkedPostcardDeserializer;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Point {
    x: u32,
    y: u32,
}

// SharedBuf wraps an async byte source and coordinates multi-reader access.
let result = SharedBuf::with_async(
    source,
    async |buf| { /* refill callback */ },
    async |shared| {
        let de = ChunkedPostcardDeserializer::new(shared);
        match Point::deserialize_owned(de, ()).await.unwrap() {
            Probe::Hit((_, v)) => Some(v),
            Probe::Miss => None,
        }
    },
).await;
```

## Wire format coverage

| Type | Encoding |
|------|----------|
| `bool` | `0x00` = false, `0x01` = true |
| `u8`–`u64` | unsigned ULEB128 varint |
| `i8`–`i64` | zigzag + ULEB128 varint |
| `u128` | two consecutive ULEB128 varints (lo, hi) |
| `i128` | zigzag on two consecutive ULEB128 varints |
| `f32` | 4 bytes little-endian IEEE 754 |
| `f64` | 8 bytes little-endian IEEE 754 |
| `char` | ULEB128 Unicode codepoint |
| `()` / unit struct / unit variant | zero bytes |
| `&str` (borrow) / `String` (both) | ULEB128 length + raw UTF-8 bytes |
| `&[u8]` (borrow) / `Vec<u8>` (both) | ULEB128 length + raw bytes |
| `Option<T>` | `0x00` = None; `0x01` + payload = Some |
| newtype struct | same as inner type |
| tuple / tuple struct | ULEB128 element count + elements in order |
| named struct | fields in declaration order (no names on wire) |
| seq / `Vec<T>` | ULEB128 element count + elements |
| map | ULEB128 pair count + alternating key/value |
| enum | ULEB128 variant index + variant payload |

## Struct deserialization

Named structs deserialize **by field position**, not by name. There are no field names on the wire. When you `#[derive(strede::Deserialize)]` / `#[derive(strede::DeserializeOwned)]` on a named struct, the generated code calls `deserialize_key_by_index` — postcard's `MapKeyProbe` (`ChunkedPostcardMapKeyProbe` in the owned family) matches each field by its ordinal position (0, 1, 2, …) rather than by a string key. This is identical between the two families; only the underlying byte source (in-memory slice vs. async stream) differs.

Field order in the postcard wire data must match the Rust declaration order exactly.

## Enum deserialization

Enums use an index-based discriminant (ULEB128 varint). `#[derive(strede::Deserialize)]` / `#[derive(strede::DeserializeOwned)]` work directly — the generated code calls `deserialize_unit_by_index` and `deserialize_payload_by_index`, which postcard dispatches by position, identically in both families. The owned family reads the discriminant varint via an async, chunk-boundary-resumable decoder rather than a single synchronous read, but the wire semantics are unchanged.

```rust
#[derive(strede::Deserialize)]
enum Cmd {
    Ping,            // discriminant 0 — zero bytes payload
    Move { x: i16, y: i16 },  // discriminant 1 — two zigzag varints
}
```

### `#[strede(other)]`

`#[strede(other)]` works: it only ever targets a unit variant, so the fallback
never needs to skip a payload. Once every named/indexed variant has missed, the
unmatched discriminant is treated as carrying no payload at all — matching
upstream `postcard`+`serde`'s own `#[serde(other)]` convention (an unrecognized
discriminant likewise consumes nothing beyond itself there).

```rust
#[derive(strede::Deserialize)]
enum Cmd {
    Ping,
    Move { x: i16, y: i16 },
    #[strede(other)]
    Unknown,
}
```

Caveat: if the real (unrecognized) variant actually carried a payload on the
wire — e.g. data produced by a newer schema version with an added variant —
those bytes are left unconsumed. That typically surfaces as a top-level
`PostcardError::ExpectedEnd` (trailing bytes) rather than being silently
discarded.

## Limitations

### `skip()` is not supported for arbitrary values

Postcard is schema-driven: without knowing the Rust type, it's impossible to know how many bytes to skip. `Entry::skip()` returns `Err(PostcardError::CannotSkip)`.

As a consequence, `#[strede(allow_unknown_fields)]` has no effect on postcard:
struct fields are matched positionally, not by wire key, so there is no
"extra field" concept to skip in the first place — the attribute compiles but
never changes behavior. Trailing/mismatched bytes still surface as the usual
positional errors (`UnexpectedEnd`/`ExpectedEnd`).

`#[strede(flatten)]`, however, **is** supported. Flatten only needs to skip a
value when an unrecognized key shows up, and postcard has no such thing —
fields are matched exhaustively by position, so the outer and flattened
fields are simply read back-to-back in declaration order.

### Internally-tagged and adjacently-tagged enums are not supported

`#[strede(tag = "field")]` (internally tagged) and `#[strede(tag = "t", content = "c")]` (adjacently tagged) require matching a field **by name** inside a map. Postcard has no wire names — all dispatch is positional. Deserializing these representations returns `Probe::Miss` regardless of input.

This matches serde-postcard's position: `#[serde(tag)]` is listed as "WontImplement" in the postcard issue tracker. Only externally-tagged enums (the default: varint discriminant + payload) are supported.

### Owned-family enums with very many variants may overflow the stack

The owned family's enum-variant racing (`strede::EnumArmStackOwned`) recurses one nested generic future per declared variant. Somewhere between 60 and 90 variants, the per-level state carried by the owned family's futures (larger than the borrow family's, since each holds a live stream handle) causes a stack overflow querying the type's layout, in both debug and release builds. The borrow family does not have this ceiling — a 130-variant enum works fine there. This is a characteristic of the shared `strede::enum_arm` owned-family infrastructure, not specific to postcard.

## Error variants

| Variant | Meaning |
|---------|---------|
| `UnexpectedEnd` | Input truncated mid-value |
| `InvalidUtf8` | String bytes are not valid UTF-8 |
| `ExpectedEnd` | Trailing bytes after the top-level value |
| `DuplicateField(name)` | Same struct field appeared twice |
| `CannotSkip` | `skip()` called — unsupported for postcard |

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `alloc` | no | Enables `String` and `Vec<u8>` deserialization (both families) |

## Workspace

| Crate | Description |
|-------|-------------|
| [`strede`](../strede) | Core traits (borrow + owned families) |
| [`strede-derive`](../strede-derive) | `#[derive(Deserialize)]` proc-macro |
| [`strede-json`](../strede-json) | JSON backend |
| [`strede-msgpack`](../strede-msgpack) | MessagePack backend |
| [`strede-cbor`](../strede-cbor) | CBOR backend |
| `strede-postcard` | This crate |
