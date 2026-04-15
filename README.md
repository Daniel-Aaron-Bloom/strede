# strede

Async, zero-alloc, pull-based deserialization for Rust.

## The problem with serde

serde's `Deserializer` is push-based: it calls methods on your `Visitor`,
driving the process itself. This works well for synchronous formats but
doesn't compose naturally with async streams — and it requires allocating
a visitor object to dispatch on unknown types.

## The strede approach

strede flips the model. You advance the stream through a closure and probe
entry types with futures that resolve `Ok(Probe::Hit(...))` or `Ok(Probe::Miss)`.
A `Claim` token threads proof-of-consumption back to the deserializer:

```rust
// Advance the stream, racing two type probes
let value = d.next(|[e1, e2]| async {
    select_probe! {
        async move {
            let (claim, f) = hit!(e1.deserialize_f32().await);
            Ok(Probe::Hit((claim, Value::Float(f))))
        },
        async move {
            let (claim, s) = hit!(e2.deserialize_str().await);
            Ok(Probe::Hit((claim, Value::Str(s))))
        },
        miss => Ok(Probe::Miss),
    }
}).await?;
```

Probe results:

- `Ok(Probe::Hit((Claim, T)))` — token matched; thread `Claim` back to `next`.
- `Ok(Probe::Miss)` — type mismatch; stream was **not** consumed.
- `Err(e)` — fatal format error (malformed data, I/O failure).
- `Pending` — no data available yet (I/O backpressure only, never a type mismatch).

`select_probe!` polls all arms; `Miss` marks an arm done, first `Hit` wins,
`Err` short-circuits. This replaces the visitor pattern with no heap allocation.

## Trait overview

### Borrow family (`'de` — zero-copy)

```
Deserialize<'de>      T::deserialize(&mut d, ()) → Result<Probe<T>, Error>

Deserializer<'de>     d.next(|[e1, ..eN]| async { Ok(Probe::Hit((claim, r))) })
                        → Result<Probe<R>, Error>

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

StrAccess             chunks.next() → Result<Chunk<&str, Claim>, Error>
                      chunks.fork() → Self
BytesAccess           chunks.next() → Result<Chunk<&[u8], Claim>, Error>
                      chunks.fork() → Self

MapAccess             map.next(|[ke]| async { Ok(Probe::Hit((claim, r))) })
                        → Result<Probe<Chunk<R, Claim>>, Error>
                      map.fork() → Self

MapKeyEntry           ke.key::<K, N, ..>(|&k, [ve; N]| async { Ok(Probe::Hit((claim, r))) })
                        → Result<Probe<(Claim, K, R)>, Error>

MapValueEntry         ve.value::<V, ()>(()) → Result<Probe<(Claim, V)>, Error>
                      ve.skip()     → Result<Claim, Error>

SeqAccess             seq.next(|[e]| async { Ok(Probe::Hit((claim, r))) })
                        → Result<Probe<Chunk<R, Claim>>, Error>
                      seq.fork() → Self

SeqEntry              e.get::<T, ()>(()) → Result<Probe<(Claim, T)>, Error>
                      e.skip()     → Result<Claim, Error>
```

### Owned family (`'s` — streaming/chunked, no zero-copy borrows)

Mirrors the borrow family but `next` takes `self` by value. Drops
`deserialize_str` / `deserialize_bytes`; strings/bytes must go through chunks.
`StrAccessOwned::next` / `BytesAccessOwned::next` take `self` + a sync closure
`FnOnce(&str) -> R` to map the short-lived borrow to an owned value.

```
DeserializeOwned<'s>       T::deserialize(d, ()) → Result<Probe<(Claim, T)>, Error>
DeserializerOwned<'s>      d.next(self, closure) → Result<Probe<(Claim, R)>, Error>
EntryOwned<'s>             (same probes minus deserialize_str/bytes, plus skip)
StrAccessOwned<'s>         chunks.next(self, |&str| -> R) → Result<Chunk<(Self, R), Claim>, Error>
                           chunks.fork() → Self
BytesAccessOwned<'s>       chunks.next(self, |&[u8]| -> R) → Result<Chunk<(Self, R), Claim>, Error>
                           chunks.fork() → Self
MapAccessOwned<'s>         map.next(self, closure) → Result<Probe<Chunk<(Self, R), Claim>>, Error>
                           map.fork() → Self
MapKeyEntryOwned<'s>       ke.key(self, closure) → Result<Probe<(Claim, K, R)>, Error>
MapValueEntryOwned<'s>     ve.value(self, ()) → Result<Probe<(Claim, V)>, Error>
                           ve.skip(self) → Result<Claim, Error>
SeqAccessOwned<'s>         seq.next(self, closure) → Result<Probe<Chunk<(Self, R), Claim>>, Error>
                           seq.fork() → Self
SeqEntryOwned<'s>          e.get(self, ()) → Result<Probe<(Claim, T)>, Error>
                           e.skip(self) → Result<Claim, Error>
```

The two families are independent — no supertrait relationship, no blanket impls.

The `Claim` for maps, sequences, and streaming strings/bytes is returned via
the `Done(claim)` variant of `Chunk` — not from the initial probe.

In the borrow family, stream advancement (`next`) takes `&mut self` — explicitly
sequential. Type probes consume `self` — pass `N > 1` handles to `next` to race
them with `select_probe!` without borrow conflicts.

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
    Ping,                          // unit    — "Ping"
    SetRate(f64),                  // newtype — {"SetRate": 1.5}
    Move(i32, i32),                // tuple   — {"Move": [10, 20]}
    Resize { width: u32, height: u32 }, // struct  — {"Resize": {"width": 80, "height": 24}}
}
```

The derive macro generates a `Deserialize` impl that calls `d.next()`, enters
the map, and dispatches on each string key to fill the struct fields.
Tuple structs (e.g. `struct Pair(u32, u32)`) deserialize from JSON arrays.
For enums, unit variants are encoded as bare strings, and non-unit variants as
single-key maps. Newtype variants map the key to the inner value directly;
tuple variants map it to a JSON array; struct variants map it to a JSON object.
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
    user_id: u64,         // explicit rename wins — matches "id"
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

`#[strede(untagged)]` on an enum or individual variant enables shape-based
matching instead of name tags. Variants are tried in declaration order; first
`Hit` wins. Can be mixed with tagged variants — tagged paths are tried first,
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
instead — if `expr` is a function path it is called, otherwise the value
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
deserialization entirely — the field always uses its default. Requires
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

## Utility types

`strede` ships two zero-sized marker types:

**`Skip`** — `Deserialize<'de>` + `DeserializeOwned<'s>` (Extra = ()). Discards any
token unconditionally. Always `Hit`.

**`Match`** — Checks a token for an exact content match. `Extra` is the expected
value:

| impl                             | family | token  |
| -------------------------------- | ------ | ------ |
| `Deserialize<'de, &'a str>`      | borrow | string |
| `Deserialize<'de, &'a [u8]>`     | borrow | bytes  |
| `DeserializeOwned<'s, &'a str>`  | owned  | string |
| `DeserializeOwned<'s, &'a [u8]>` | owned  | bytes  |

Returns `Hit(Match)` when content equals `extra`, `Miss` otherwise (stream not
advanced). Use with `deserialize_value` inside `select_probe!` for string-tag
dispatch:

```rust
d.next(|[e1, e2]| select_probe! {
    e1.deserialize_value::<Match, &str>("ok"),
    e2.deserialize_value::<Match, &str>("err"),
    miss => Ok(Probe::Miss),
})
```

The borrow-family str/bytes impls race the zero-copy probe against the chunked
fallback (N=2 entry handles) so escaped strings are handled without a separate
`d.next` call.

## Status

Early development — core traits stable with both borrow and owned families.
JSON backend implemented: in-memory borrow-family deserializer (`JsonDeserializer`)
and chunked/streaming owned-family deserializer (`strede-json::chunked`).
`deserialize_str` is zero-copy but returns `Miss` for strings containing escape sequences;
`deserialize_str_chunks` handles all strings including escaped ones. `strede-derive` provides
`#[derive(Deserialize)]`, `#[derive(DeserializeOwned)]`, and `select_probe!`.

## Workspace

| crate           | description                                                    |
| --------------- | -------------------------------------------------------------- |
| `strede`        | core traits (borrow + owned families), `shared_buf` module     |
| `strede-json`   | JSON deserializer backend (in-memory + chunked/streaming)      |
| `strede-derive` | proc-macro: `Deserialize`, `DeserializeOwned`, `select_probe!` |
