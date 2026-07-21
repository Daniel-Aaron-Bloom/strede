# strede — development notes

## What this is

A serde alternative built around async streaming and zero-alloc deserialization.
The core insight: separate _stream advancement_ (which must be sequential) from
_type probing_ (which benefits from concurrency).

## Workspace layout

- `strede/` — core library: traits (borrow + owned families), `shared_buf` module, no allocations, no format-specific code
- `strede-derive/` — proc-macro crate: `#[derive(Deserialize)]`, `#[derive(DeserializeOwned)]`
- `strede-json/` — JSON backend: in-memory borrow-family deserializer (`JsonDeserializer`) and chunked/streaming owned-family deserializer (`chunked` module)
- `strede-test-util/` — shared test helpers used across crates

## Key design decisions

### `entry` closure pattern and `Claim`

`Deserializer::entry` takes `self` and a closure `F: FnMut([Entry; N]) -> Fut`.
The closure receives `N` owned entry handles for the same slot and must return
`Ok(Probe::Hit((Claim, R)))`, `Ok(Probe::Miss)`, or `Err(e)`. `entry` verifies
the `Claim`, advances the stream, and returns `Ok(Probe::Hit((Claim, R)))` or
`Ok(Probe::Miss)`. The same closure pattern applies to `SeqAccess::next`.

Pass `N > 1` to race multiple probe arms via `select_probe!` without borrow
conflicts. Handles dropped without resolving do not advance the stream.

All probe methods (`deserialize_*`, `get`, `value`) consume `self` — they are not
`&self`.

### Probe future semantics — `Probe<T>`

Probe methods return `Result<Probe<T>, E>`:

- `Ok(Probe::Hit((Claim, T)))` — token matched; thread `Claim` back to `entry`.
- `Ok(Probe::Miss)` — type mismatch; stream was **not** consumed.
  This guarantee is unconditional: returning `Probe::Miss` leaves the stream
  unadvanced even if user code has partially consumed bytes through a
  `BytesAccess` or `StrAccess` handle. All reads through accessors are
  provisional until a `Claim` resolves back to `entry`.
- `Err(e)` — fatal format error (malformed data, I/O failure).
- `Pending` — no data available yet (I/O backpressure only).

`Pending` means **only** "waiting for I/O" — never a type mismatch. Type
mismatches return `Ok(Probe::Miss)` so that `select_probe!` can immediately stop
polling an arm that missed, rather than having to keep it live until data arrives.

The `hit!` macro is `?` for probe results: it propagates `Err` and `Probe::Miss`
from the enclosing function and unwraps `Probe::Hit(v)` to `v`.

### `select_probe!`

A declarative macro (`macro_rules!`) defined in `strede/src/probe.rs` and re-exported from `strede`. Syntax:

```rust
select_probe! {
    expr,                 // any future returning Result<Probe<T>, E>
    async move { ... },   // async block for arms that need .await
    @miss => body,         // optional: fires when all arms returned Miss
}
```

Each arm is an arbitrary expression evaluating to a future that returns
`Result<Probe<T>, E>`. Bare probe calls work for simple forwarding; use
`async move { ... }` blocks (with `hit!` inside) when the arm body needs
`.await` or value transformation.

Expands to a `poll_fn` that pins all futures and polls each non-missed arm
on every wakeup. First `Hit` in declaration order wins. `Miss` arms are marked
done and skipped on future polls. `Err` short-circuits immediately.

Two additional forms:

- `select_probe!(biased; ...)` — when multiple arms are simultaneously ready,
  the earliest arm in declaration order wins instead of an unspecified one.
- Inside any arm, `kill!(i)` schedules arm `i` for cancellation on the next
  poll. Useful when the calling arm has enough information to know a sibling
  can never win (e.g. the zero-copy `deserialize_str` succeeded, so the chunked
  fallback arm can be dropped before the next `.await`).

### `async fn` in traits — `Send` lint suppression

`#![allow(async_fn_in_trait)]` is set crate-wide. This means the trait does
not constrain returned futures to be `Send`. Implementors can still be `Send`;
they just aren't required to be by the trait. Locking it in would be a
breaking change to relax later.

### Zero-alloc

No `Box<dyn Future>`, no `dyn Deserializer`. All associated types are concrete
and resolved at compile time. Implementations may use interior mutability for
waker dispatch but should not heap-allocate as part of normal operation.

### `deserialize_str` vs `deserialize_str_chunks`

`Entry::deserialize_str` returns `&'de str` — a zero-copy borrow into the
source buffer. It returns `Ok(Probe::Miss)` (not `Err`) when the string
contains escape sequences, because the unescaped content cannot be
represented as a contiguous borrow without allocation.
`Entry::deserialize_str_chunks` handles all strings including escapes.
Callers that want zero-copy-when-possible race both via `select_probe!`.
This is an inherent limitation of zero-copy borrowing, not a missing feature.

## Trait hierarchy

### Borrow family (`'de` — zero-copy)

```
Deserialize<'de, Extra = ()>  — types that can pull themselves from a Deserializer;
                                Extra is side-channel context (see below)
DeserializeFromMap<'de, M>    — types that deserialize from an already-opened MapAccess
DeserializeFromSeq<'de, S>    — types that deserialize from an already-opened SeqAccess
DeserializeFromEnum<'de, E>   — types that deserialize from an already-opened EnumAccess

Deserializer<'de>     — stream handle; entry(self, closure) → Result<Probe<(Claim, R)>, Error>
  Entry<'de>          — one item slot; deserialize_* probes → Result<Probe<(Claim, T)>, Error>
                        deserialize_str(self) → Result<Probe<(Claim, &'de str)>, Error>
                        deserialize_str_chunks(self) → Result<Probe<StrAccess>, Error>
                        deserialize_bytes(self) → Result<Probe<(Claim, &'de [u8])>, Error>
                        deserialize_bytes_chunks(self) → Result<Probe<BytesAccess>, Error>
                        deserialize_number_chunks(self) → Result<Probe<NumberAccess>, Error>
                        deserialize_map(self) → Result<Probe<MapAccess>, Error>
                        deserialize_seq(self) → Result<Probe<SeqAccess>, Error>
                        deserialize_map_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                        deserialize_seq_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                        deserialize_value::<T, Extra>(self, extra) → Result<Probe<(Claim, T)>, Error>
                        deserialize_option::<T, Extra>(self, extra) → Result<Probe<(Claim, Option<T>)>, Error>
                        deserialize_enum(self) → Result<Probe<EnumAccess>, Error>
                        deserialize_enum_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                        fork(&mut self) → Self
                        skip(self) → Result<Claim, Error>
                        skip_other(self) → Result<Claim, Error>
    StrAccess         — streaming string chunks; next_str(self, |&str| -> R) → Result<Chunk<(Self, R), Claim>, Error>
    BytesAccess       — streaming byte chunks;   next_bytes(self, |&[u8]| -> R) → Result<Chunk<(Self, R), Claim>, Error>
    NumberAccess      — streaming number chunks; next_number_chunk(self, |&str| -> R) → Result<Chunk<(Self, R), Claim>, Error>
    MapAccess<'de>    — in-progress map; iterate(self, arms: impl MapArmStack) → Result<Probe<(Claim, Outputs)>, Error>
      MapKeyProbe<'de>   — key probe handle; deserialize_key::<K>(self, extra) → Result<Probe<(KeyClaim, K)>, Error>
                                             deserialize_key_by_index(self, expected: usize) → Result<Probe<(KeyClaim, ())>, Error>
                                             fork(&mut self) → Self
        MapKeyClaim<'de> — after key matched; into_value_probe(self) → Result<MapValueProbe, Error>
          MapValueProbe<'de> — value probe handle; deserialize_value::<V>(self, extra) → Result<Probe<(ValueClaim, V)>, Error>
                                                   deserialize_map_into::<V>(self, extra) → Result<Probe<(ValueClaim, V)>, Error>
                                                   deserialize_seq_into::<V>(self, extra) → Result<Probe<(ValueClaim, V)>, Error>
                                                   fork(&mut self) → Self
                                                   skip(self) → Result<ValueClaim, Error>
            MapValueClaim<'de> — after value consumed; next_key(self, ..) → Result<NextKey<KeyProbe, MapClaim>, Error>
    SeqAccess<'de>    — in-progress seq; next(self, closure) → Result<Probe<Chunk<(Self, R), Claim>>, Error>
      SeqEntry        — one element slot; get::<T, Extra>(self, extra) → Result<Probe<(Claim, T)>, Error>
                                          get_map_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                                          get_seq_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                                          fork(&mut self) → Self
                                          skip(self) → Result<Claim, Error>
    EnumAccess<'de>   — in-progress enum (externally-tagged); iterate(self, arms: impl EnumArmStack) → Result<Probe<(Claim, Outputs)>, Error>
      EnumVariantProbe<'de> — variant probe; all methods default to Ok(Probe::Miss)
                              deserialize_unit_by_name(self, candidates) → Result<Probe<(Claim, usize)>, Error>
                              deserialize_payload_by_name::<T>(self, candidates, extra) → Result<Probe<(Claim, usize, T)>, Error>
                              deserialize_unit_by_index(self, idx) → Result<Probe<(Claim, usize)>, Error>
                              deserialize_payload_by_index::<T>(self, idx, extra) → Result<Probe<(Claim, usize, T)>, Error>
                              fork(&mut self) → Self

Probe<T>              — Hit(T) | Miss  (type-dispatch result, not an error)
Chunk<Data, Done>     — Data(item) | Done(claim)  (used by streaming accessors)
NextKey<KeyProbe, MapClaim> — Entry(KeyProbe) | Done(MapClaim)  (map iteration result)
```

### Owned family (`'s` — streaming/chunked, no zero-copy borrows)

Mirrors the borrow family but drops `deserialize_str` / `deserialize_bytes`
(strings/bytes must go through chunks). `StrAccessOwned::next_str` /
`BytesAccessOwned::next_bytes` take `self` + a sync closure `FnOnce(&str) -> R`
so the short-lived borrow can be mapped to an owned value.

```
DeserializeOwned<Extra = ()>     — types that can pull themselves from a DeserializerOwned;
                                       Extra is side-channel context (see below)
DeserializeFromMapOwned<M>       — types that deserialize from an already-opened MapAccessOwned
DeserializeFromSeqOwned<S>       — types that deserialize from an already-opened SeqAccessOwned

DeserializerOwned      — entry(self, closure) → Result<Probe<(Claim, R)>, Error>
  EntryOwned           — deserialize_* probes (no deserialize_str/bytes)
                             deserialize_str_chunks(self) → Result<Probe<StrAccessOwned>, Error>
                             deserialize_bytes_chunks(self) → Result<Probe<BytesAccessOwned>, Error>
                             deserialize_number_chunks(self) → Result<Probe<NumberAccessOwned>, Error>
                             deserialize_map(self) → Result<Probe<MapAccessOwned>, Error>
                             deserialize_seq(self) → Result<Probe<SeqAccessOwned>, Error>
                             deserialize_map_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                             deserialize_seq_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                             deserialize_value::<T, Extra>(self, extra) → Result<Probe<(Claim, T)>, Error>
                             deserialize_option::<T, Extra>(self, extra) → Result<Probe<(Claim, Option<T>)>, Error>
                             deserialize_enum(self) → Result<Probe<EnumAccessOwned>, Error>
                             deserialize_enum_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                             fork(&mut self) → Self
                             skip(self) → Result<Claim, Error>
                             skip_other(self) → Result<Claim, Error>
    StrAccessOwned     — next_str(self, |&str| -> R) → Result<Chunk<(Self, R), Claim>, Error>
    BytesAccessOwned   — next_bytes(self, |&[u8]| -> R) → Result<Chunk<(Self, R), Claim>, Error>
    NumberAccessOwned  — next_number_chunk(self, |&str| -> R) → Result<Chunk<(Self, R), Claim>, Error>
    MapAccessOwned     — in-progress map (owned); iterate(self, arms: impl MapArmStackOwned) → Result<Probe<(Claim, Outputs)>, Error>
      MapKeyProbeOwned   — key probe; deserialize_key::<K>(self, extra) → Result<Probe<(KeyClaim, K)>, Error>
                               deserialize_key_by_index(self, expected: usize) → Result<Probe<(KeyClaim, ())>, Error>
                               fork(&mut self) → Self
        MapKeyClaimOwned — after key matched; into_value_probe(self) → Result<MapValueProbeOwned, Error>
          MapValueProbeOwned — value probe; deserialize_value::<V>(self, extra) → Result<Probe<(ValueClaim, V)>, Error>
                                   fork(&mut self) → Self
                                   skip(self) → Result<ValueClaim, Error>
            MapValueClaimOwned — after value consumed; next_key(self, ..) → Result<NextKey<KeyProbeOwned, MapClaim>, Error>
    SeqAccessOwned     — next(self, closure) → Result<Probe<Chunk<(Self, R), Claim>>, Error>
      SeqEntryOwned        — get::<T, Extra>(self, extra) → Result<Probe<(Claim, T)>, Error>
                               get_map_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                               get_seq_into::<T>(self, extra) → Result<Probe<(Claim, T)>, Error>
                               fork(&mut self) → Self
                               skip(self) → Result<Claim, Error>
    EnumAccessOwned      — in-progress enum (owned); iterate(self, arms: impl EnumArmStackOwned) → Result<Probe<(Claim, Outputs)>, Error>
      EnumVariantProbeOwned — variant probe; all methods default to Ok(Probe::Miss)
                              deserialize_unit_by_name(self, candidates) → Result<Probe<(Claim, usize)>, Error>
                              deserialize_payload_by_name::<T>(self, candidates, extra) → Result<Probe<(Claim, usize, T)>, Error>
                              deserialize_unit_by_index(self, idx) → Result<Probe<(Claim, usize)>, Error>
                              deserialize_payload_by_index::<T>(self, idx, extra) → Result<Probe<(Claim, usize, T)>, Error>
                              fork(&mut self) → Self
```

The two families are independent — no supertrait relationship, no blanket impls.
A format implements whichever family (or both) it can support.

### Owned family — parallel scanning and deadlock hazard

The owned family reads from a streaming source where data arrives
incrementally. Entry handles and probe items (`Entry`, `SeqEntry`,
`MapKeyProbe`, `MapValueProbe`, `EnumVariantProbe`, and their Owned
counterparts) share the same underlying buffer and advance through it
cooperatively via `fork`.

**For callers:** you must not await one forked handle to completion and then
decide what to do with another — that will deadlock. The first handle may
block waiting for more data to arrive, but the buffer cannot advance until
_all_ sibling handles have consumed the current chunk. Instead, race all
handles concurrently (e.g. via `select_probe!`). This is safe: forked
handles never interfere with each other, and every reader is polled and
paused as new data becomes available, as long as all of them are making
forward progress together.

**For implementors:** your `fork` implementation must ensure that the
underlying buffer does not advance past data that any live handle still
needs to read. Every forked reader must be independently resumable: when
new data arrives, all suspended readers must be woken and given the
opportunity to process it. You must never require one reader to finish
before another can make progress — doing so creates a circular dependency
that deadlocks the single-threaded executor. The `shared_buf` module
provides a reference implementation of this contract.

### `fork` — probe items vs. structural accessors

`fork` is present only on probe items, not on structural accessors. The two
categories:

**Probe items have `fork`:** `Entry`, `SeqEntry`, `MapKeyProbe`,
`MapValueProbe`, `EnumVariantProbe` (and their Owned counterparts). These are
individual slot handles at the _probe layer_ — the layer where racing multiple
type interpretations of the same slot is the core mechanism. `fork` gives
callers an independent view of the same position to hand to `select_probe!`.
The derive macro exploits this via `d.entry(|[e1, ..., eN]|)`, which delivers
N independent handles directly without an explicit `fork` call.

**Structural accessors do not have `fork`:** `MapAccess`, `SeqAccess`,
`EnumAccess`, `StrAccess`, `BytesAccess`, `NumberAccess` (and their Owned
counterparts). Once a structural accessor is open, format crates may wrap it
in adapter types driven by struct layout attributes (e.g. `TagAwareMap` for
`#[strede(tag)]`). These adapters
are single-use state machines that capture closures and cell references and
cannot be meaningfully duplicated. Requiring `fork` on structural accessors
would force every such adapter to implement it with no safe way to do so.
Any concurrency inside a structural access — for example, racing arms over a
map's key stream — is handled internally by the `map_arm` / `enum_arm`
infrastructure, which does not need to fork the accessor itself.

### Shared utilities

- `Probe<T>`, `Chunk<Data, Done>`, `hit!`, `or_miss!`, `select_probe!`, `DeserializeError` —
  shared between both families.
- `shared_buf` module (`SharedBuf`, `Handle`, `Buffer`) — zero-alloc, no_std
  async multi-reader buffer primitive used by the chunked JSON deserializer.

`or_miss!` is the Option-flavoured counterpart to `hit!`: `or_miss!(opt)` returns
`Probe::Miss` when `opt` is `None` and unwraps to the inner value otherwise.
Used in `try_from` impls and any code that needs to convert an `Option` into a
miss signal.

`Entry::deserialize_map`, `deserialize_seq`, `deserialize_str_chunks`,
`deserialize_bytes_chunks`, and `deserialize_number_chunks` return
`Result<Probe<Access>, Error>` — the `Claim` emerges from the accessor's
`Done` variant when the collection or string/number is exhausted.

`Never<'a, Claim, Error>` — the uninhabited bottom type. Implements every
trait in both families. Used as associated types (e.g. `type StrChunks =
Never<…>`) on entry/accessor impls that never produce those accessor kinds,
so the trait obligation is satisfied without dead code.

`Entry::skip_other` / `EntryOwned::skip_other` — consumes the current token as
the fallback for an externally-tagged enum's `#[strede(other)]` catch-all,
after every named/indexed variant has missed. Defaults to forwarding to
`skip`, which is correct for self-describing formats (JSON, MessagePack,
CBOR) where the unmatched value's shape can be discovered and discarded
generically. Schema-driven formats that cannot implement `skip` at all (e.g.
postcard, see `strede-postcard`) can still override `skip_other`: since
`other` is validated to target only a unit variant, the unmatched
discriminant can be treated as carrying no payload, with no need to know its
shape. This mirrors upstream `postcard`+`serde`'s own `#[serde(other)]`
convention. The caveat carries over too: if the real (unrecognized) variant
actually had a payload, those bytes are left unconsumed.

### `map_arm` module — map iteration infrastructure

`strede/src/map_arm/` contains the data structures and pin-projection helpers
used by derive-generated map iteration. The module is split into `mod.rs`
(shared types), `borrow.rs` (`MapArmStack` and borrow-family types), and
`owned.rs` (`MapArmStackOwned` and owned-family types).

**Arm-stack building blocks** (used directly in derive-generated code):

- `MapArmBase` — empty base of an arm stack. Left-nested with `+` via `Add<MapArm<S>>`.
- `MapArm<S>` — newtype that wraps one arm slot for the `+` operator.
- `MapArmSlot<K, V, KeyFn, ValFn>` — one concrete field slot. Holds the key-race closure
  `KeyFn: FnMut(KP, usize) -> KeyFut`, the value-dispatch closure `ValFn`, and the current
  `ArmState<K, V>`. The `usize` passed to `KeyFn` is the arm's global positional index
  (computed from `arm_base` at `init_race` time) — named-only arms ignore it, while arms
  that also support positional access call `kp.deserialize_key_by_index(i)` and race it via
  `select_probe!`.
- `ArmState<K, V>` — `Empty | Key(K) | Done(K, V)`. Tracks per-slot progress
  through a map iteration round.
- `NextKey<KeyProbe, MapClaim>` — returned by the value-claim's `next_key` to
  either yield `Entry(key_probe)` (more KV pairs) or `Done(claim)` (map exhausted).
- `VirtualArmSlot<K, KeyFn, ValFn>` — like `MapArmSlot` but never satisfied and
  produces no output. Used by skip-unknown, dup-detect, and tag-inject wrappers.
- `StackConcat<A, B>` — concatenates two arm stacks so that all arms from both
  are polled concurrently. Arm indices from `A` are `0..A::SIZE`; arm indices from
  `B` are offset by `A::SIZE`. Outputs are `(A::Outputs, B::Outputs)`. Emitted by
  the derive macro to splice a flatten field's `make_arms()` substack into the
  outer struct's arm stack.

**`MapArmStack` / `MapArmStackOwned` — key methods:**

`race_keys(kp)` — drives one round of key racing. Calls `init_race(kp, 0)` to create
per-arm key futures (passing each arm its global positional index), then polls all
futures via `poll_race_one` in a flat loop. Returns `Probe::Hit((arm_index, key_claim))`
for the first winning arm, or `Probe::Miss` when all arms miss. Wrapper stacks override
this to inject virtual arms before delegating to the inner `init_race`/`poll_race_one`
path. `StackConcat` routes to the correct sub-stack with the appropriate index offset.

`dispatch_value(arm_index, vp)` — converts the winning arm's key claim to a value probe
and polls its value callback via `init_dispatch`/`poll_dispatch`.

**`deserialize_key_by_index` in derive-generated key closures:**

Every field arm's key closure generated by the derive (both borrow and owned families,
both struct and enum helper derives) races `deserialize_key::<Match>(wire_name)` against
`deserialize_key_by_index(arm_index)` via `select_probe!`. Whichever hits first wins.
This means:

- **Name-based formats** (JSON, CBOR, MessagePack maps): `deserialize_key` hits on the
  wire string; `deserialize_key_by_index` returns `Miss` (default impl).
- **Positional formats** (postcard): `deserialize_key_by_index` hits because the format
  implements it to match when `current_position == expected`; `deserialize_key` returns
  `Miss` (no wire names).

Virtual arms (dup-detect, skip-unknown, tag-inject) use only `deserialize_key` and do
not call `deserialize_key_by_index` — they are infrastructure, not user fields, and have
no meaningful positional identity.

**Wrapper stacks** (generated around the base arm stack):

- `DetectDuplicates<S, const M, KeyFn, SkipFn>` — wraps any arm stack and
  returns an error when a wire key that already matched an arm appears a second
  time. `wire_names` maps each known key to its arm index for O(1) lookup.
- `TagInjectingStack<'v, S, const N, TagKeyFn, TagValFn>` — prepends a
  virtual tag arm (index 0) to the inner stack. When the tag key matches, it
  captures the matched variant index into a `Cell<Option<usize>>` and then
  checks that index against `expected_variant` at iteration end.

**Macros** (re-exported from `strede`):

- `map_arms! { key_fn => val_fn, … }` — builds a left-nested arm tuple from a
  flat list of `key_closure => value_closure` pairs. Key closures receive `(KP, usize)`
  where the `usize` is the arm's global positional index. Equivalent to writing
  `MapArmBase + MapArm(MapArmSlot::new(k0, v0)) + …`.
- `map_outputs!(pat0, pat1, …)` — destructures the left-nested output tuple
  produced by `MapArmStack::take_outputs` / `MapArmStackOwned::take_outputs`
  into flat named bindings. Expands to the nested pattern `(((), pat0), pat1)`.
- `SkipUnknown!(arms, KP, VP)` / `SkipUnknownOwned!(arms, KP, VP)` — prepend a
  virtual arm that matches any key and skips its value. Used by the
  `allow_unknown_fields` codegen. `SkipUnknownOwned!` also accepts a 1-arg
  form `SkipUnknownOwned!(arms)` where KP/VP types are inferred from context.
- `DetectDuplicates!(arms, KP, VP, M, wire_names)` / `DetectDuplicatesOwned!(…)` —
  wrap an arm stack with duplicate-key detection.
- `TagInjectingStack!(…)` / `TagInjectingStackOwned!(…)` — wrap an arm stack with
  tag injection for internally-tagged enums.

The pin-projection helpers (`SlotRaceState`, `SlotDispatchState`, `ConcatRaceState`,
`ConcatDispatchState`, `WrapperRaceState`, `WrapperDispatchState`,
`TagRaceState`, `TagDispatchState`) are public but derive-macro-facing only.

### `enum_arm` module — enum variant dispatch infrastructure

`strede/src/enum_arm/` mirrors `map_arm/` for enum variant identification. The module is split into `mod.rs` (shared types), `borrow.rs` (`EnumArmStack` trait + impls), and `owned.rs` (`EnumArmStackOwned` trait + impls).

**Key difference from map arm infrastructure**: Each arm simultaneously identifies the variant AND deserializes the payload in one call — there is no separate "race" + "dispatch" phase. This is required for interleaved formats (e.g. msgpack arrays) where discriminant and payload cannot be split.

**Arm-stack building blocks** (used directly in derive-generated code):

- `EnumArmBase` — empty base of an arm stack. Left-nested with `+` via `Add<EnumArm<S>>`.
- `EnumArm<S>` — newtype wrapper that wraps one arm slot for the `+` operator.
- `EnumArmSlot<Out, ArmFn>` — one variant arm slot. Holds `ArmFn: FnMut(VP) -> Future<Output = Result<Probe<(VP::Claim, Out)>, VP::Error>>` and an `EnumArmState<Out>`. The closure receives a forked `EnumVariantProbe`, calls one of the `deserialize_unit_by_name` / `deserialize_payload_by_name` / `deserialize_unit_by_index` / `deserialize_payload_by_index` methods, and returns `Probe<(Claim, Out)>` directly.
- `EnumArmState<Out>` — `Empty | Done(Out)`. Tracks per-slot progress (no `Key` state needed since discriminant+payload are combined).

**`EnumArmStack<'de, VP>` / `EnumArmStackOwned<VP>` — key method:**

`race(vp)` — drives the iteration. Forks `vp` into each arm via `init_race`, then polls all arm futures concurrently via `poll_race_one`. First `Probe::Hit((idx, claim))` wins. If all arms miss, returns `Probe::Miss`. `take_outputs()` extracts the left-nested `((..., Option<Out0>), Option<Out1>)` tuple after the race completes.

**`EnumVariantProbe<'de>` / `EnumVariantProbeOwned`** (implemented by formats):

```
EnumVariantProbe<'de>:
  type Error: DeserializeError
  type Claim
  type PayloadDeserializer: Deserializer<'de, Claim = Self::Claim, Error = Self::Error>

  fork(&mut self) → Self
  deserialize_unit_by_name<const N>(self, candidates: [(&'static str, usize); N])
      → Result<Probe<(Claim, usize)>, Error>
  deserialize_payload_by_name<T: Deserialize<'de, D>, const N>(self, candidates, extra)
      → Result<Probe<(Claim, usize, T)>, Error>
  deserialize_unit_by_index(self, expected_idx: usize)
      → Result<Probe<(Claim, usize)>, Error>
  deserialize_payload_by_index<T: Deserialize<'de, D>>(self, expected_idx, extra)
      → Result<Probe<(Claim, usize, T)>, Error>
```

All methods have default impls returning `Ok(Probe::Miss)` so formats only implement the methods they support (e.g. JSON only implements name-based methods).

**`EnumAccess<'de>` / `EnumAccessOwned`** (implemented by formats):

```
EnumAccess<'de>:
  type Error: DeserializeError
  type Claim
  type VariantProbe: EnumVariantProbe<'de, Claim = Self::Claim, Error = Self::Error>

  fork(&mut self) → Self
  iterate<Arms: EnumArmStack<'de, Self::VariantProbe>>(self, arms)
      → Result<Probe<(Self::Claim, Arms::Outputs)>, Self::Error>
```

**`DeserializeFromEnum<'de, E: EnumAccess<'de>>`** (implemented by derived enums):

```
DeserializeFromEnum<'de, E>:
  type Extra: Copy
  async fn deserialize_from_enum(e: E, extra: Self::Extra)
      → Result<Probe<(E::Claim, Self)>, E::Error>
```

**`Entry<'de>` / `EntryOwned` extensions:**

```
type Enum: EnumAccess<'de, Claim = Self::Claim, Error = Self::Error>
async fn deserialize_enum(self) → Result<Probe<Self::Enum>, Self::Error>
async fn deserialize_enum_into<T: DeserializeFromEnum<'de, Self::Enum>>(self, extra: T::Extra)
    → Result<Probe<(Self::Claim, T)>, Self::Error>
```

**`enum_arms!` macro** (re-exported from `strede`):

```rust
let arms = enum_arms! {
    |vp| vp.deserialize_unit_by_name([("Foo", 0)]),
    |vp| vp.deserialize_payload_by_name::<Bar, _>([("Bar", 1)], ()),
};
```

Builds a left-nested `(EnumArmBase, EnumArmSlot(...))` stack. Equivalent to `EnumArmBase + EnumArm(EnumArmSlot::new(arm0)) + EnumArm(EnumArmSlot::new(arm1)) + ...`.

**Derive-generated code pattern** (external enums only):

```rust
// DeserializeFromEnum<'de, __E> impl — owns the arm stack:
impl<'de, __E: EnumAccess<'de>> DeserializeFromEnum<'de, __E> for MyEnum
where Self: DeserializeFromEnum<'de, __E>
{
    type Extra = ();
    async fn deserialize_from_enum(e: __E, _extra: ()) -> Result<Probe<(__E::Claim, Self)>, __E::Error> {
        let mut arms = EnumArmBase
            + EnumArm(EnumArmSlot::new(|__vp| async { /* unit: deserialize_unit_by_name */ }))
            + EnumArm(EnumArmSlot::new(|__vp| async { /* payload: deserialize_payload_by_name */ }));
        let (claim, outputs) = hit!(e.iterate(&mut arms).await);
        let (((), out0), out1) = arms.take_outputs();
        if let Some(v) = out0 { return Ok(Probe::Hit((claim, MyEnum::Foo(v)))); }
        if let Some(()) = out1 { return Ok(Probe::Hit((claim, MyEnum::Bar))); }
        Ok(Probe::Miss)
    }
}

// Deserialize<'de, __D> impl — delegates via deserialize_enum_into:
impl<'de, __D: Deserializer<'de>> Deserialize<'de, __D> for MyEnum
where <__D::Entry as Entry<'de>>::Enum: EnumAccess<'de>, Self: DeserializeFromEnum<'de, <__D::Entry as Entry<'de>>::Enum>
{
    async fn deserialize(d: __D, _extra: ()) -> Result<Probe<(__D::Claim, Self)>, __D::Error> {
        d.entry(|[__e]| async { __e.deserialize_enum_into::<Self>(()).await }).await
    }
}
```

Formats that don't yet support `EnumAccess` set `type Enum = Never<...>` on their `Entry` impl; `deserialize_enum` returns `Probe::Miss`.

### Tag adapter — `TagAwareMap`

`TagAwareMap` lives in `strede/src/impls/tag_flatten.rs` and is re-exported from
`strede`. It is generated by enum derives; callers do not construct it directly.

**`TagAwareMap<'de, 'v, M, const N>`** (and `TagAwareMapOwned<'v, M, const N>`)
wraps an already-opened `MapAccess` / `MapAccessOwned` to inject a tag arm into
the inner type's `iterate` call. It prepends a virtual arm that matches the tag
key and captures the matched variant index into a `Cell<Option<usize>>`, then
post-validates that index against `expected_variant` once iteration is done. Used
by internally-tagged and adjacently-tagged enum derives.

### Flatten — `MapFieldProvider` / `MapFieldProviderOwned`

Flatten fields are handled at codegen time, not via a runtime adapter chain.
Every `#[derive(Deserialize)]` / `#[derive(DeserializeOwned)]` on a named
struct emits a `MapFieldProvider<'de, __KP>` / `MapFieldProviderOwned<__KP>`
impl that exposes the struct's arm stack via `make_arms()`, its wire-name
table via `wire_names()`, and an `Outputs → Self` reconstruction via
`from_outputs()`. The parent struct's MFP impl composes flatten children via
`StackConcat(outer_arms, <Child as MFP>::make_arms())` and shifts the child's
wire-name indices via `.map(|(s, i)| (s, i + offset))`. The result is one
`iterate` call against a unified arm stack — no continuation chain, no
intermediate cells, no boxed-future workaround.

### `deserialize_option` — null-or-T probe

`Entry::deserialize_option::<T, Extra>(self, extra)` returns:

- `Ok(Probe::Hit((claim, None)))` — null token.
- `Ok(Probe::Hit((claim, Some(v))))` — non-null token matching `T::deserialize`.
- `Ok(Probe::Miss)` — token matches neither null nor T.

It can be raced in `select_probe!` — it implicitly covers "null or T's type."
Internally it creates a sub-deserializer with the non-null token pre-loaded and
delegates to `T::deserialize(sub, extra)`. The `Deserialize<'de, Extra> for Option<T>`
blanket impl calls `d.entry(|[e]| e.deserialize_option::<T, Extra>(extra))` and
forwards `Extra` transparently — `Option<MyType>: Deserialize<'de, MyCtx>` when
`MyType: Deserialize<'de, MyCtx>`.

### `deserialize_value` — untagged support

`Entry::deserialize_value::<T, Extra>(self, extra)` delegates to `T::deserialize`
by creating a sub-deserializer with the current token pre-loaded, forwarding `extra`.
Returns `Hit((claim, value))` if T matched, `Miss` if T missed. Used by untagged
newtype/tuple/struct variants and available for general use. Exists on both
`Entry<'de>` and `EntryOwned`.

### `Extra` — side-channel context

`Deserialize<'de, Extra = ()>` (and `DeserializeOwned<Extra = ()>`) carry an
`Extra` type parameter that is threaded into `deserialize` and passed at the call
sites of `deserialize_value`, `deserialize_option`, `get`, `value`, and `key`.

All built-in and derived impls use `Extra = ()`. Custom types that need
caller-supplied context declare their own `Extra`:

```rust
impl<'de> Deserialize<'de, MyCtx> for MyType {
    async fn deserialize<D: Deserializer<'de>>(d: D, ctx: MyCtx) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let (c, inner) = hit!(e.deserialize_value::<Inner, MyCtx>(ctx).await);
            Ok(Probe::Hit((c, MyType { inner })))
        }).await
    }
}
```

`Extra` is a method-level type param on the entry methods — callers use turbofish
`get::<T, _>(extra)` to specify `T` explicitly while letting Rust infer `Extra`
from the argument. Derived impls pass `()` at every call site.

### `Skip`, `Match`, `MatchVals`, and `UnwrapOrElse` — utility deserializer types

`Skip` implements `Deserialize<'de>` and `DeserializeOwned` (both with `Extra = ()`).
It calls `Entry::skip()` unconditionally and always returns `Probe::Hit(Skip)` on
well-formed input.

`Match` implements two combinations (string tokens only):

- `Deserialize<'de, &'static str>` — borrow family, string token
- `DeserializeOwned<&'static str>` — owned family, string token

Returns `Probe::Hit(Match)` when the token's content equals `extra`, `Probe::Miss`
otherwise (wrong type or wrong content, stream not advanced).

The borrow-family impl races `deserialize_str` (zero-copy fast path) and
`deserialize_str_chunks` (escape-sequence fallback) via `select_probe!` with N=2
entry handles. The owned-family impl uses a single `deserialize_str_chunks` arm
(no zero-copy path available).

Typical use — discriminated enum dispatch:

```rust
d.entry(|[e1, e2]| select_probe! {
    e1.deserialize_value::<Match, &str>("ok"),
    e2.deserialize_value::<Match, &str>("err"),
    @miss => Ok(Probe::Miss),
})
```

`MatchVals<T, const N>(pub T, pub PhantomData<[(); N]>)` generalises `Match` to
return a caller-supplied value on a content match. `Extra` is `[(&'static str, T); N]`
— an array of `(candidate, value)` pairs. `T` must be `Copy`. Returns
`Probe::Hit(MatchVals(value, PhantomData))` on a match, `Probe::Miss` if no
candidate matches or the token is not a string. Two combinations are implemented
(string tokens only, matching `Match`):

- `Deserialize<'de, [(&'static str, T); N]>` — borrow family
- `DeserializeOwned<[(&'static str, T); N]>` — owned family

`Match` and `MatchVals` have independent implementations; `Match` does not
delegate to `MatchVals`.

```rust
// Return the matched T directly
e.deserialize_value::<MatchVals<MyEnum, 2>, _>([("a", MyEnum::A), ("b", MyEnum::B)]).await

// Return the matched index
e.deserialize_value::<MatchVals<usize, 2>, _>([("foo", 0usize), ("bar", 1usize)]).await
```

`UnwrapOrElse<T, F>(pub T, pub PhantomData<fn() -> F>)` wraps `T: Deserialize<'de, Extra>` with an async fallback.
`Extra` is `(F, InnerExtra)` where `F: AsyncFnOnce() -> T`. Uses a 2-handle
`select_probe!` internally: arm 1 tries `T::deserialize`; if it misses, arm 2
calls `skip()` to consume the entry and then calls the fallback. This guarantees
the stream is always advanced exactly once regardless of whether T matched.

```rust
// Deserialize T, or call async fallback if T misses (entry still consumed)
e.deserialize_value::<UnwrapOrElse<MyType>, _>((async || MyType::default(), ())).await
```

Primary use in the derive macro: `UnwrapOrElse<MatchVals<usize>>` with
`async || MatchVals(sentinel)` as the fallback, so unknown map keys produce a
sentinel index while still consuming the key entry.

### Derive attributes

`#[strede(rename = "wire_name")]` — on a struct field or enum variant,
changes the wire name used for key/tag matching. The Rust identifier is
unchanged.

`#[strede(rename_all = "convention")]` — on a struct or enum (container level).
Converts all field/variant names to the specified case convention. An explicit
`#[strede(rename)]` on an individual field or variant takes priority over
`rename_all`. Aliases are never transformed — they are always literal wire names.
Supported values: `"lowercase"`, `"UPPERCASE"`, `"PascalCase"`, `"camelCase"`,
`"snake_case"`, `"SCREAMING_SNAKE_CASE"`, `"kebab-case"`, `"SCREAMING-KEBAB-CASE"`.
Implemented via `convert_case` v0.11. Applied in `strede-derive/src/common.rs`
`wire_name()` — runs before classification so both borrow and owned derives see
the converted names. Does not apply to unnamed tuple-struct fields (they use
their positional index as the wire key regardless).

`#[strede(alias = "alt_name")]` — on a struct field or enum variant,
adds an additional wire name that matches during deserialization. Can be
specified multiple times. Works alongside `rename`. Cannot be used on
untagged variants.

`#[strede(transparent)]` — on a struct with exactly one non-skipped field.
The struct deserializes as its inner field directly (no map/seq wrapper).
Works on both named structs and tuple structs (newtypes). Skipped fields
use their defaults.

`#[strede(allow_unknown_fields)]` — on a struct. Unknown map keys are
skipped (value consumed and discarded) instead of causing `Probe::Miss`.
Required fields that are absent still return `Miss`; duplicate fields
still return `Err`.

**Default enum representation — externally tagged.** When no tagging attribute
is present on an enum, the representation is externally tagged: unit variants
match a bare string (`"Ping"`), and non-unit variants match a single-key map
whose key is the variant name (`{"Move": ...}`). Newtype variants map the key
to the inner value directly; tuple variants map it to a JSON array; struct
variants map it to a JSON object. Unknown variant names return `Probe::Miss`.

`#[strede(tag = "field")]` — on an enum. Internally tagged: the variant
discriminant is a field _inside_ the map, e.g. `{"type": "Move", "x": 1.0}`.
Both families support all variant kinds (unit, newtype, tuple, struct). For
newtype/tuple variants, the inner type must itself deserialize from a map (the
tag facade only surfaces `deserialize_map`; primitives inside a newtype are not
supported). `rename`, `rename_all`, and `alias` apply to variant names as
normal. Cannot be combined with `untagged` or `other`.

`#[strede(tag = "t", content = "c")]` — on an enum. Adjacently tagged: the
outer map has exactly two relevant keys — the tag field and the content field
(plus any unknowns, which are skipped). The variant payload lives entirely
inside the content value, e.g. `{"t": "Move", "c": {"x": 1.0, "y": 2.0}}`.
Key order is irrelevant. Unit variants have no content field: `{"t": "Ping"}`.
Both families are supported. Requires `tag` to be set.
`rename`, `rename_all`, and `alias` apply to variant names as normal.

`#[strede(flatten)]` — on a named struct field. The field's map keys are merged
into the parent struct's outer map iteration — no wrapping map token. The
flattened type must implement `Deserialize<'de>` (borrow family) or
`DeserializeOwned` (owned family). Multiple flatten fields per struct are
supported. Unknown keys (not claimed by the outer struct or any flattened type)
are silently skipped. Both families are supported. Cannot be combined with
`rename`, `alias`, `default`, `skip_deserializing`, `deserialize_with`,
`deserialize_owned_with`, `with`, `from`, or `try_from`.

`#[strede(untagged)]` — on an enum or individual variant. Variants marked
untagged are deserialized by shape (trying each in declaration order) rather
than by name tag. Can be mixed: tagged variants are tried first via str/map
key dispatch, untagged variants act as sequential fallback. Unit variants
match null; newtype/tuple/struct variants use `deserialize_value`.

`#[strede(other)]` — on a unit enum variant. Acts as catch-all: when no
tagged variant matches the discriminant, this variant is returned instead of
`Probe::Miss`. The derive consumes the unmatched value via
`Entry::skip_other`/`EntryOwned::skip_other` (default: forwards to `skip`)
before returning the `other` variant — see `skip_other` below. Restrictions:
unit variants only; at most one per enum; cannot combine with `rename`,
`alias`, or `untagged`; cannot coexist with `#[strede(untagged)]` variants on
the same enum.

`#[strede(default)]` — on a struct field. If the field is missing from the
data, calls `Default::default()` instead of returning `Probe::Miss`.
`#[strede(default = "expr")]` evaluates the expression instead. If `expr`
is a function path it is called; if it is a value expression (e.g. `42`,
`String::new()`, `vec![]`) it is used as-is. Implemented via a
`DefaultWrapper` type whose inherent method resolves `FnOnce` paths and
whose trait fallback returns plain values.

`#[strede(skip_deserializing)]` — on a struct field. The field is never
read from the data (its key is treated as unknown) and always uses its
default value. Requires `default` or `default = "fn"` to also be set.

`#[strede(deserialize_with = "path")]` — on a struct field. Uses a custom
function instead of `T::deserialize` for the borrow family. The function
must have the same signature as `Deserialize::deserialize`.

`#[strede(deserialize_owned_with = "path")]` — same for the owned family,
matching `DeserializeOwned::deserialize_owned` signature.

`#[strede(with = "module")]` — shorthand that sets both: uses
`module::deserialize` for borrow and `module::deserialize_owned` for owned.

`#[strede(crate = "path")]` — on a struct or enum (container level). Overrides
the default crate path (`::strede`) used in generated code. Useful when strede
is re-exported under a different name (e.g. a wrapper crate that re-exports
`strede` as `my_crate::strede`).

`#[strede(bound = "T: MyTrait")]` — on a struct or enum (container level),
or on a struct field (field level).

- **Container level**: replaces _all_ auto-generated where-clause predicates
  in the outer `impl` block with the provided predicates. Applies to both
  borrow and owned derives. An empty string (`bound = ""`) suppresses all
  auto-generated bounds.
- **Field level**: replaces the predicate that would be emitted for that
  specific field's type only. Other fields keep their auto-generated bounds.
  An empty string suppresses the bound for that field only.

Container-level takes priority: when `bound` is set on the container,
field-level `bound` annotations are ignored.

The predicate string is parsed as a comma-separated list of where-clause
predicates and may reference any in-scope type parameters or lifetimes
(`'de` for the borrow family, `'s` for owned). Multiple predicates are
separated by commas, just as in a `where` clause.

`#[strede(borrow)]` — on a struct field (borrow family only). Controls how
`'de: 'lifetime` bounds are generated for that field. Three forms:

- **No attribute** (default): the derive emits `'de: 'a` for each lifetime
  that appears directly in a `&'a T`, `&'a mut T`, or `Cow<'a, T>` at the
  top level of the field type. Lifetimes buried inside other generics are
  not included.
- **`#[strede(borrow)]`**: emits `'de: 'a` for _every_ lifetime found
  anywhere in the field type, including those inside nested generics.
- **`#[strede(borrow = "'a + 'b")]`**: emits `'de: 'a` only for the
  explicitly listed lifetimes. Accepts `+` or `,` as separators.

In all cases, generic type parameters on the struct/enum always get a
`T: Deserialize<'de>` bound (like `Clone` does). The `borrow` attribute only
controls which lifetime outlives bounds are emitted; it does not suppress or
replace the type-parameter bounds.

```rust
use std::marker::PhantomData;

#[derive(strede::Deserialize)]
struct Three<'a, 'b, 'c> {
    a: &'a str,               // auto: 'de: 'a
    b: &'b str,               // auto: 'de: 'b
    c: PhantomData<&'c str>,  // auto: nothing (not a top-level ref/Cow)
}

#[derive(strede::Deserialize)]
struct Example<'a, 'b, 'c> {
    #[strede(borrow = "'a + 'b")]
    // Only 'de: 'a and 'de: 'b, not 'de: 'c.
    three: Three<'a, 'b, 'c>,
}
```

`#[strede(from = "FromType")]` — on a struct, enum, or field. Deserializes
`FromType` and converts to `Self` (container) or `FieldType` (field) via
`From::from`. The generated bound is `FromType: Deserialize<'de>` (borrow) or
`FromType: DeserializeOwned` (owned); no bound is emitted for `Self` /
`FieldType`. Container-level use generates an impl that entirely replaces
the normal field-by-field deserialization.

`#[strede(try_from = "FromType")]` — same as `from`, but uses `TryFrom::try_from`.
A failed conversion returns `Probe::Miss` (via `or_miss!`) rather than an error —
conversion failures are type mismatches, not format violations.

Both attributes are mutually exclusive with each other and with
`deserialize_with` / `deserialize_owned_with` / `with` on the same item.
