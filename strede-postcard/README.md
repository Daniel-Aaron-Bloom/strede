# strede-postcard

Postcard format backend for [strede](../strede).

Postcard is a compact binary format used in embedded/`no_std` Rust. Unlike JSON or MessagePack, it carries **no type tags** — structure is entirely schema-driven, determined by the Rust type being deserialized.

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
| `&str` / `String` | ULEB128 length + raw UTF-8 bytes |
| `&[u8]` / `Vec<u8>` | ULEB128 length + raw bytes |
| `Option<T>` | `0x00` = None; `0x01` + payload = Some |
| newtype struct | same as inner type |
| tuple / tuple struct | ULEB128 element count + elements in order |
| named struct | fields in declaration order (no names on wire) |
| seq / `Vec<T>` | ULEB128 element count + elements |
| map | ULEB128 pair count + alternating key/value |
| enum | ULEB128 variant index + variant payload |

## Struct deserialization

Named structs deserialize **by field position**, not by name. There are no field names on the wire. When you `#[derive(strede::Deserialize)]` on a named struct, the generated code calls `deserialize_key_by_index` — postcard's `MapKeyProbe` matches each field by its ordinal position (0, 1, 2, …) rather than by a string key.

Field order in the postcard wire data must match the Rust declaration order exactly.

## Enum deserialization

Enums use an index-based discriminant (ULEB128 varint). `#[derive(strede::Deserialize)]` works directly — the generated code calls `deserialize_unit_by_index` and `deserialize_payload_by_index`, which postcard dispatches by position.

```rust
#[derive(strede::Deserialize)]
enum Cmd {
    Ping,            // discriminant 0 — zero bytes payload
    Move { x: i16, y: i16 },  // discriminant 1 — two zigzag varints
}
```

## Limitations

### `skip()` is not supported

Postcard is schema-driven: without knowing the Rust type, it's impossible to know how many bytes to skip. `Entry::skip()` returns `Err(PostcardError::CannotSkip)`.

As a consequence:
- `#[strede(allow_unknown_fields)]` is incompatible with postcard
- `#[strede(flatten)]` is incompatible with postcard

Both rely on skipping unknown keys, which requires type-tagged data.

### Internally-tagged and adjacently-tagged enums are not supported

`#[strede(tag = "field")]` (internally tagged) and `#[strede(tag = "t", content = "c")]` (adjacently tagged) require matching a field **by name** inside a map. Postcard has no wire names — all dispatch is positional. Deserializing these representations returns `Probe::Miss` regardless of input.

This matches serde-postcard's position: `#[serde(tag)]` is listed as "WontImplement" in the postcard issue tracker. Only externally-tagged enums (the default: varint discriminant + payload) are supported.

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
| `alloc` | no | Enables `String` and `Vec<u8>` deserialization |

## Workspace

| Crate | Description |
|-------|-------------|
| [`strede`](../strede) | Core traits (borrow + owned families) |
| [`strede-derive`](../strede-derive) | `#[derive(Deserialize)]` proc-macro |
| [`strede-json`](../strede-json) | JSON backend |
| [`strede-msgpack`](../strede-msgpack) | MessagePack backend |
| [`strede-cbor`](../strede-cbor) | CBOR backend |
| `strede-postcard` | This crate |
