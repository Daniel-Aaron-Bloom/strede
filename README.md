# strede

[![github](https://img.shields.io/badge/Daniel--Aaron--Bloom%2Fstrede-8da0cb?style=for-the-badge&logo=github&label=github&labelColor=555555)](https://github.com/Daniel-Aaron-Bloom/strede)
[![crates.io](https://img.shields.io/crates/v/strede.svg?style=for-the-badge&color=fc8d62&logo=rust)](https://crates.io/crates/strede)
[![docs.rs](https://img.shields.io/badge/docs.rs-strede-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs)](https://docs.rs/strede)
[![build status](https://img.shields.io/github/actions/workflow/status/Daniel-Aaron-Bloom/strede/ci.yml?branch=main&style=for-the-badge)](https://github.com/Daniel-Aaron-Bloom/strede/actions?query=branch%3Amain)

Async, zero-alloc, pull-based deserialization for Rust.

## The name

_Strede_ is a **STREaming DEserializer** - a name deliberately close to
[serde](https://serde.rs), the library it complements. It's also an Old English
word meaning "stream" or "channel."

Streaming is the important use case externally - but internally, the architecture
is built around coroutines. _COroutine DEserializer_ just doesn't abbreviate as well.

## Quickstart

For most users, strede looks like serde: derive `Deserialize` on your types and
pass them to a format backend.

```rust
use strede::Deserialize;

#[derive(Deserialize)]
struct Config {
    host: String,
    port: u16,
}

#[derive(Deserialize)]
#[strede(tag = "type")]
enum Event {
    Ping,
    Move { x: f64, y: f64 },
}
```

Then deserialize from JSON:

```rust
use strede_json::JsonDeserializer;

let json = r#"{"host": "localhost", "port": 8080}"#;
let config = Config::deserialize(JsonDeserializer::new(json), ()).await??;
```

For streaming/owned use (reading from an async byte source), derive
`DeserializeOwned` and use `strede_json::chunked` instead. The rest of this
document covers the lower-level traits for when you need to implement
`Deserialize` manually or write a custom format backend.

## The problem with serde

serde's `Deserializer` is push-based: it calls methods on your `Visitor`,
driving the process itself. This works well for synchronous formats but
doesn't compose naturally with async streams - often requiring the whole data stream in memory, or requiring `alloc`.

## The strede approach

strede flips the model. You advance the stream through a closure and probe
entry types with futures that resolve `Ok(Probe::Hit(...))` or `Ok(Probe::Miss)`.
A `Claim` token is returned back to the deserializer as proof-of-consumption:

```rust
// Advance the stream, racing two type probes
let value = d.entry(|[e1, e2]| async {
    select_probe! {
        async move {
            let (claim, f) = hit!(e1.deserialize_f32().await);
            Ok(Probe::Hit((claim, Value::Float(f))))
        },
        async move {
            let (claim, s) = hit!(e2.deserialize_str().await);
            Ok(Probe::Hit((claim, Value::Str(s))))
        },
        @miss => Ok(Probe::Miss),
    }
}).await?;
```

Probe results:

- `Ok(Probe::Hit((Claim, T)))` - token matched; thread `Claim` back to `entry`.
- `Ok(Probe::Miss)` - type mismatch; stream was **not** consumed.
- `Err(e)` - fatal format error (malformed data, I/O failure).
- `Pending` - no data available yet (I/O backpressure only, never a type mismatch).

`select_probe!` polls all arms; `Miss` marks an arm done, first `Hit` wins,
`Err` short-circuits. This replaces the visitor pattern with no heap allocation.

## Trait overview

### Borrow family (`'de` - zero-copy)

```
Deserialize<'de>      T::deserialize(d, ()) → Result<Probe<(Claim, T)>, Error>

Deserializer<'de>     d.entry(|[e1, ..eN]| async { Ok(Probe::Hit((claim, r))) })
                        → Result<Probe<(Claim, R)>, Error>

Entry<'de>            e.deserialize_bool()         → Result<Probe<(Claim, bool)>, Error>
                      e.deserialize_u8/u16/u32/u64/u128()
                      e.deserialize_i8/i16/i32/i64/i128()
                      e.deserialize_f32/f64()
                      e.deserialize_char()
                      e.deserialize_str()          → Result<Probe<(Claim, &'de str)>, Error>
                      e.deserialize_str_chunks()   → Result<Probe<StrAccess>, Error>
                      e.deserialize_bytes()        → Result<Probe<(Claim, &'de [u8])>, Error>
                      e.deserialize_bytes_chunks() → Result<Probe<BytesAccess>, Error>
                      e.deserialize_map()          → Result<Probe<MapAccess>, Error>
                      e.deserialize_seq()          → Result<Probe<SeqAccess>, Error>
                      e.deserialize_option::<T, ()>(())  → Result<Probe<(Claim, Option<T>)>, Error>
                      e.skip()                    → Result<Claim, Error>
                      e.deserialize_value::<T, E>(e) → Result<Probe<(Claim, T)>, Error>

Chunk<Data, Done>     Data(item) | Done(claim)

StrAccess             chunks.next_str(|&str| -> R) → Result<Chunk<(Self, R), Claim>, Error>
                      chunks.fork() → Self
BytesAccess           chunks.next_bytes(|&[u8]| -> R) → Result<Chunk<(Self, R), Claim>, Error>
                      chunks.fork() → Self

MapAccess             map.iterate(arms: impl MapArmStack) → Result<Probe<(Claim, Outputs)>, Error>
                      map.fork() → Self

MapKeyProbe           kp.deserialize_key::<K, Extra>(extra) → Result<Probe<(Claim, K)>, Error>
                      kp.fork() → Self
  MapKeyClaim         kc.into_value_probe() → MapValueProbe
    MapValueProbe     vp.deserialize_value::<V, Extra>(extra) → Result<Probe<(Claim, V)>, Error>
                      vp.skip() → Result<Claim, Error>
                      vp.fork() → Self
      MapValueClaim   vc.next_key() → Result<Probe<Chunk<(MapKeyProbe, R), Claim>>, Error>

SeqAccess             seq.next(|[e]| async { Ok(Probe::Hit((claim, r))) })
                        → Result<Probe<Chunk<(Self, R), Claim>>, Error>
                      seq.fork() → Self

SeqEntry              e.get::<T, ()>(()) → Result<Probe<(Claim, T)>, Error>
                      e.skip()     → Result<Claim, Error>
```

### Owned family (`'s` - streaming/chunked, no zero-copy borrows)

Mirrors the borrow family but drops `deserialize_str` / `deserialize_bytes`;
strings/bytes must go through chunks. `StrAccessOwned::next_str` /
`BytesAccessOwned::next_bytes` take `self` + a sync closure `FnOnce(&str) -> R`
to map the short-lived borrow to an owned value.

```
DeserializeOwned       T::deserialize_owned(d, ()) → Result<Probe<(Claim, T)>, Error>
DeserializerOwned      d.entry(self, closure) → Result<Probe<(Claim, R)>, Error>
EntryOwned             (same probes minus deserialize_str/bytes, plus skip)
StrAccessOwned         chunks.next_str(self, |&str| -> R) → Result<Chunk<(Self, R), Claim>, Error>
                           chunks.fork() → Self
BytesAccessOwned       chunks.next_bytes(self, |&[u8]| -> R) → Result<Chunk<(Self, R), Claim>, Error>
                           chunks.fork() → Self
SeqAccessOwned         seq.next(self, closure) → Result<Probe<Chunk<(Self, R), Claim>>, Error>
                           seq.fork() → Self
SeqEntryOwned          e.get(self, ()) → Result<Probe<(Claim, T)>, Error>
                           e.skip(self) → Result<Claim, Error>
```

The two families are independent - no supertrait relationship, no blanket impls.

The `Claim` for maps, sequences, and streaming strings/bytes is returned via
the `Done(claim)` variant of `Chunk` - not from the initial probe.

In the borrow family, stream advancement (`entry`) takes `self` - explicitly
sequential. Type probes consume `self` - pass `N > 1` handles to `entry` to race
them with `select_probe!` without borrow conflicts.

### Owned family - parallel scanning

The owned family reads from a streaming source where data arrives
incrementally. When `entry` passes multiple handles, or when you `fork` an
accessor, the resulting readers share the same underlying buffer.

**You must drive all forked readers concurrently** - typically via
`select_probe!`. Sequentially awaiting one reader to completion before
polling another will deadlock: the first reader may block waiting for buffer
data that cannot arrive until all sibling readers have consumed the current
chunk. This is safe to do: forked readers never interfere with each other,
and every reader is automatically suspended and resumed as new data becomes
available, provided all readers are being polled.

## Deriving

```rust
#[derive(strede::Deserialize)]
struct Config {
    host: String,
    port: u16,
    timeout_ms: u64,
}

#[derive(strede::Deserialize)]
enum Message {
    Ping,                          // unit    - "Ping"
    SetRate(f64),                  // newtype - {"SetRate": 1.5}
    Move(i32, i32),                // tuple   - {"Move": [10, 20]}
    Resize { width: u32, height: u32 }, // struct  - {"Resize": {"width": 80, "height": 24}}
}
```

The derive macro generates a `Deserialize` impl that calls `d.entry()`, enters
the map, and dispatches on each string key to fill the struct fields.
Tuple structs (e.g. `struct Pair(u32, u32)`) deserialize from JSON arrays.
For enums, the default representation is **externally tagged**: unit variants
are encoded as bare strings, and non-unit variants as single-key maps where the
key is the variant name. Newtype variants map the key to the inner value
directly; tuple variants map it to a JSON array; struct variants map it to a
JSON object. Unknown variant names return `Probe::Miss`.
`#[derive(DeserializeOwned)]` generates the equivalent for the owned family,
using `deserialize_str_chunks` for streaming key matching.

### Attributes

`#[strede(rename = "wire_name")]` on a field or variant changes the wire name
used for matching without affecting the Rust identifier:

```rust
#[derive(strede::Deserialize)]
struct Config {
    #[strede(rename = "type")]
    kind: String,
}
```

`#[strede(rename_all = "convention")]` on a struct or enum converts all
field/variant names to the given case. An explicit `rename` on a field or
variant takes priority. Supported conventions: `"lowercase"`, `"UPPERCASE"`,
`"PascalCase"`, `"camelCase"`, `"snake_case"`, `"SCREAMING_SNAKE_CASE"`,
`"kebab-case"`, `"SCREAMING-KEBAB-CASE"`:

```rust
#[derive(strede::Deserialize)]
#[strede(rename_all = "camelCase")]
struct User {
    first_name: String,   // matches "firstName"
    last_name: String,    // matches "lastName"
    #[strede(rename = "id")]
    user_id: u64,         // explicit rename wins - matches "id"
}

#[derive(strede::Deserialize)]
#[strede(rename_all = "snake_case")]
enum Event {
    UserCreated,          // matches "user_created"
    OrderPlaced(u64),     // matches {"order_placed": ...}
}
```

`#[strede(alias = "alt_name")]` on a field or variant adds an additional wire
name that matches during deserialization. Can be specified multiple times and
works alongside `rename`. Cannot be used on untagged variants:

```rust
#[derive(strede::Deserialize)]
struct Config {
    #[strede(alias = "hostname", alias = "server")]
    host: String,
}
// Accepts "host", "hostname", or "server" as the key.
```

`#[strede(tag = "field")]` on an enum marks it as internally tagged: the
variant discriminant is stored as a named field _inside_ the map rather than
as the outer key. For example, `{"type": "Move", "x": 1.0, "y": 2.0}` with
`#[strede(tag = "type")]` dispatches on `"type"` and deserializes the
remaining fields as the variant's payload. `rename`, `rename_all`, and
`alias` apply to variant names as normal.

```rust
#[derive(strede::DeserializeOwned)]
#[strede(tag = "type")]
enum Event {
    Ping,                        // {"type": "Ping"}
    Move { x: f64, y: f64 },    // {"type": "Move", "x": 1.0, "y": 2.0}
    Teleport(Point),             // {"type": "Teleport", "x": 3.0, "y": 4.0}
}
```

Both families support all variant kinds. For newtype/tuple variants, the inner
type must itself deserialize from a map - the tag facade only surfaces
`deserialize_map`, so primitives inside a newtype are not supported.

`#[strede(tag = "t", content = "c")]` on an enum marks it as adjacently
tagged: the outer map has a tag field and a separate content field; the
variant payload lives entirely inside the content value. Key order is
irrelevant. Unit variants have no content field.

```rust
#[derive(strede::Deserialize, strede::DeserializeOwned)]
#[strede(tag = "t", content = "c")]
enum Event {
    Ping,                      // {"t": "Ping"}
    Move { x: f64, y: f64 },  // {"t": "Move", "c": {"x": 1.0, "y": 2.0}}
    Wrap(Point),               // {"t": "Wrap", "c": {"x": 3.0, "y": 4.0}}
}
```

Both families are supported.

`#[strede(flatten)]` on a named struct field merges that field's map keys into
the parent struct's outer map iteration - no wrapping map token. Unknown keys
not claimed by the outer struct or any flattened type are silently skipped.
Multiple flatten fields per struct are supported. Both families are supported:
the flattened type must implement `Deserialize<'de>` (borrow) or
`DeserializeOwned` (owned).

```rust
// Borrow family
#[derive(strede::Deserialize)]
struct Inner { x: f64, y: f64 }

#[derive(strede::Deserialize)]
struct Outer {
    name: String,
    #[strede(flatten)]
    pos: Inner,
}
// Deserializes: {"name": "p", "x": 1.0, "y": 2.0}
```

For structs with 3 or more flatten fields, use `#[strede(flatten(boxed))]`
instead. Deeply-nested `StackConcat` types produce large async state machines
that can overflow the stack; `flatten(boxed)` opts each continuation future
into `Box::pin` to break the chain. Any flatten field annotated `flatten(boxed)`
enables boxed mode for the entire flatten chain. Requires the `alloc` feature.

```rust
#[derive(strede::Deserialize)]
struct Color { r: u8, g: u8, b: u8 }

#[derive(strede::Deserialize)]
struct Size { w: f64, h: f64 }

#[derive(strede::Deserialize)]
struct Point { x: f64, y: f64 }

#[derive(strede::Deserialize)]
struct Shape {
    #[strede(flatten(boxed))]
    pos: Point,
    #[strede(flatten(boxed))]
    color: Color,
    #[strede(flatten(boxed))]
    size: Size,
}
// Deserializes: {"x": 0.0, "y": 0.0, "r": 0, "g": 0, "b": 0, "w": 1.0, "h": 1.0}
```

`#[strede(untagged)]` on an enum or individual variant enables shape-based
matching instead of name tags. Variants are tried in declaration order; first
`Hit` wins. Can be mixed with tagged variants - tagged paths are tried first,
untagged variants act as fallback:

```rust
#[derive(strede::Deserialize)]
enum Msg {
    Ping,                              // tagged: "Ping"
    Data { x: i32 },                   // tagged: {"Data": {"x": 1}}
    #[strede(untagged)] Raw(bool),     // untagged: tries bool directly
}
```

`#[strede(other)]` on a unit enum variant acts as a catch-all: any
unrecognized discriminant returns this variant instead of `Probe::Miss`. For
map-keyed variants the unknown key's value is skipped first. Only one `other`
variant is allowed per enum; it cannot be combined with `rename`, `alias`, or
`untagged`, and cannot coexist with untagged variants:

```rust
#[derive(strede::Deserialize)]
enum Status {
    Ok,
    Error,
    #[strede(other)]
    Unknown,
}
// "Ok" → Status::Ok, "Error" → Status::Error, "anything_else" → Status::Unknown
```

`#[strede(default)]` on a struct field uses `Default::default()` when the
field is missing. `#[strede(default = "expr")]` evaluates the expression
instead - if `expr` is a function path it is called, otherwise the value
is used directly:

```rust
#[derive(strede::Deserialize)]
struct Config {
    host: String,
    #[strede(default)]
    port: u16,                         // 0 if missing
    #[strede(default = "default_timeout")]
    timeout_ms: u64,                   // default_timeout() if missing
    #[strede(default = "3")]
    retries: u32,                      // literal 3 if missing
}
```

`#[strede(skip_deserializing)]` on a struct field excludes it from
deserialization entirely - the field always uses its default. Requires
`default` or `default = "fn"` to also be set.

`#[strede(allow_unknown_fields)]` on a struct skips unknown map keys
(consuming and discarding their values) instead of returning `Probe::Miss`:

```rust
#[derive(strede::Deserialize)]
#[strede(allow_unknown_fields)]
struct Config {
    host: String,
    port: u16,
}
// Deserializes successfully even if the JSON has extra fields.
```

`#[strede(transparent)]` on a struct with exactly one non-skipped field
makes it deserialize as that field directly, with no map or array wrapper:

```rust
#[derive(strede::Deserialize)]
#[strede(transparent)]
struct Meters(f64);
// Deserializes from a bare number like 3.14, not [3.14] or {"0": 3.14}.
```

`#[strede(deserialize_with = "path")]` on a struct field uses a custom
function instead of `T::deserialize` (borrow family).
`#[strede(deserialize_owned_with = "path")]` is the owned-family equivalent.
`#[strede(with = "module")]` is shorthand for both, using
`module::deserialize` and `module::deserialize_owned`.

`#[strede(from = "FromType")]` deserializes `FromType` and converts to the
target via `From::from`. Works at **container level** (the whole struct/enum
is produced from `FromType`) and **field level** (just that field is converted).
`#[strede(try_from = "FromType")]` is the same but uses `TryFrom::try_from`;
a failed conversion returns `Probe::Miss` rather than an error, because
conversion failures are type mismatches, not format violations. Both are
mutually exclusive with `deserialize_with` / `deserialize_owned_with` / `with`.

`#[strede(crate = "path")]` on a struct or enum overrides the default crate
path (`::strede`) used in generated code. Useful when strede is re-exported
under a different name:

```rust
#[derive(other_crate::Deserialize)]
#[strede(crate = "other_crate::strede")]
struct Foo { x: u32 }
```

`#[strede(bound = "T: MyTrait")]` overrides the where-clause predicates that
the derive macro would normally generate automatically.

At **container level** it replaces all auto-generated predicates for the
entire `impl` block (applies to both borrow and owned derives).
At **field level** it replaces the predicate for that one field; other fields
keep their auto-generated bounds. An empty string (`bound = ""`) suppresses
bounds entirely.

```rust
// Replace the auto T: Deserialize<'de> bound with a custom supertrait.
trait MyDeserialize<'de>: strede::Deserialize<'de> {}

#[derive(strede::Deserialize)]
#[strede(bound = "T: MyDeserialize<'de>")]
struct Wrapper<T> {
    inner: T,
}

// Suppress bounds on one field while keeping them on others.
#[derive(strede::Deserialize)]
struct Pair<T, U> {
    #[strede(bound = "T: Copy + strede::Deserialize<'de>")]
    first: T,
    second: U,  // auto-bound: U: Deserialize<'de>
}
```

`#[strede(borrow)]` on a struct field controls how `'de: 'lifetime` bounds
are generated for the borrow-family derive:

- **No attribute** (default): emits `'de: 'a` for lifetimes in top-level
  `&'a T`, `&'a mut T`, and `Cow<'a, T>`.
- **`#[strede(borrow)]`**: emits `'de: 'a` for _every_ lifetime in the type.
- **`#[strede(borrow = "'a + 'b")]`**: emits bounds only for the listed
  lifetimes. Accepts `+` or `,` as separators.

Generic type parameters always get `T: Deserialize<'de>` regardless:

```rust
use std::borrow::Cow;

#[derive(strede::Deserialize)]
struct Borrowed<'a> {
    name: &'a str,           // auto: 'de: 'a
    data: Cow<'a, [u8]>,     // auto: 'de: 'a
}

#[derive(strede::Deserialize)]
struct Explicit<'a, 'b, 'c> {
    #[strede(borrow = "'a + 'b")]
    inner: Custom<'a, 'b, 'c>,  // only 'de: 'a and 'de: 'b
}
```

## Utility types

**`Skip`** - `Deserialize<'de>` + `DeserializeOwned` (Extra = ()). Discards any
token unconditionally. Always `Hit`.

**`Match`** - Checks a token for an exact content match. `Extra` is the expected
value:

| impl                         | family | token  |
| ---------------------------- | ------ | ------ |
| `Deserialize<'de, &'a str>`  | borrow | string |
| `Deserialize<'de, &'a [u8]>` | borrow | bytes  |
| `DeserializeOwned<&'a str>`  | owned  | string |
| `DeserializeOwned<&'a [u8]>` | owned  | bytes  |

Returns `Hit(Match)` when content equals `extra`, `Miss` otherwise (stream not
advanced). Use with `deserialize_value` inside `select_probe!` for string-tag
dispatch:

```rust
d.entry(|[e1, e2]| select_probe! {
    e1.deserialize_value::<Match, &str>("ok"),
    e2.deserialize_value::<Match, &str>("err"),
    @miss => Ok(Probe::Miss),
})
```

The borrow-family str/bytes impls race the zero-copy probe against the chunked
fallback (N=2 entry handles) so escaped strings are handled without a separate
`d.entry` call.

**`MatchVals<T>`** - generalises `Match` to return a caller-supplied `T` on a
content match. The same four family × token impls as `Match`. `Extra` is an
array of `(&str, T)` / `(&[u8], T)` pairs, or plain `[&str; N]` / `[&[u8]; N]`
for `MatchVals<usize>` index-only dispatch. `T` must be `Copy`:

```rust
// Return the matched enum variant
e.deserialize_value::<MatchVals<MyEnum>, _>([("a", MyEnum::A), ("b", MyEnum::B)]).await

// Return the matched index
e.deserialize_value::<MatchVals<usize>, _>(["foo", "bar"]).await
```

`Match` is a thin wrapper that delegates to `MatchVals`.

**`UnwrapOrElse<T>`** - wraps `T: Deserialize<'de, Extra>` with an async
fallback. `Extra` is `(F, InnerExtra)` where `F: AsyncFnOnce() -> T`. Arm 1
tries `T::deserialize`; if it misses, arm 2 calls `skip()` to consume the entry
and then calls the fallback. The stream is always advanced exactly once:

```rust
e.deserialize_value::<UnwrapOrElse<MyType>, _>((async || MyType::default(), ())).await
```

The derive macro uses `UnwrapOrElse<MatchVals<usize>>` with a sentinel fallback
so unknown map keys produce a sentinel index while still consuming the key entry.

## Performance

strede is currently **~10–15x slower than serde_json** in the borrow family and **~20–35x slower** in the owned family on equivalent JSON
deserialization workloads. Benchmarks (Criterion, Apple M-series):

| benchmark   | strede/borrow | strede/owned | serde_json | borrow ratio | owned ratio |
| ----------- | ------------- | ------------ | ---------- | -----------: | ----------: |
| point       | ~400 ns       | ~850 ns      | ~30 ns     |         ~13× |        ~30× |
| log_entry   | ~1.0 µs       | ~3 µs        | ~90 ns     |         ~11× |        ~33× |
| rect        | ~1.1 µs       | ~2.5 µs      | ~100 ns    |         ~11× |        ~25× |
| deep_nested | ~4.4 µs       | ~11 µs       | ~500 ns    |          ~9× |        ~22× |

The gap is structural. serde's visitor pattern compiles to a tight
synchronous dispatch loop. strede's coroutine model produces nested async
state machines - one per field, per nested type, per arm of every
`select_probe!`. Each layer adds state-machine overhead at compile time and movement cost at
runtime, i.e. the cost of Rust
storing each nested future into its parent state machine on every `.await`.

Language-level [in-place initialization](https://rust-lang.github.io/rust-project-goals/2025h2/in-place-initialization.html)
would allow futures to be constructed directly into their parent state
machine slot rather than moved there, which may significantly reduce this
overhead.

**Compile times and recursion limits.** Deeply-nested async state machines also
stress the compiler's recursion limit. If you encounter `reached the recursion
limit` errors, add this to the crate:

```rust
#![recursion_limit = "256"]
```

This is the nested future types produced by coroutine deserialization,
not an artifact of macro expansion.

## Status

Early development - core traits stable with both borrow and owned families.
JSON backend implemented: in-memory borrow-family deserializer (`JsonDeserializer`)
and chunked/streaming owned-family deserializer (`strede-json::chunked`).

## Workspace

| crate           | description                                                    |
| --------------- | -------------------------------------------------------------- |
| `strede`        | core traits (borrow + owned families), `shared_buf` module     |
| `strede-json`   | JSON deserializer backend (in-memory + chunked/streaming)      |
| `strede-derive` | proc-macro: `Deserialize`, `DeserializeOwned`, `select_probe!` |
