# strede-bincode

Bincode format backend for [`strede`](https://github.com/Daniel-Aaron-Bloom/strede).

Bincode is schema-driven (no type tags on the wire) like `strede-postcard`,
but unlike postcard its wire encoding is itself configurable. This crate
supports the real `bincode` crate's wire matrix as a compile-time generic
parameter:

- **Endianness**: little or big
- **Int encoding**: fixed-width (`Fixint`, matches bincode 1.x / bincode2's
  `legacy()`) or variable-width (`Varint`, matches bincode2's `standard()`
  default)

```rust
use strede_bincode::{BincodeDeserializer, Standard, Legacy};

// bincode2's standard() config: little-endian, varint
let d = BincodeDeserializer::<Standard>::new(bytes);

// bincode 1.x / bincode2's legacy() config: little-endian, fixed-width
let d = BincodeDeserializer::<Legacy>::new(bytes);
```

See the crate's module docs for known limitations shared with
`strede-postcard` (schema-driven formats cannot implement `skip()`, so
`allow_unknown_fields`/`flatten` are unsupported).
