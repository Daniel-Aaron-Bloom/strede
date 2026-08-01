//! Regression test: skipping a JSON number that ends exactly at true EOF
//! (no trailing terminator byte anywhere in the buffer) must succeed.
//!
//! A JSON number is the only value type with no closing delimiter, so
//! `NumberAccess::next_chunk` returning `Err(UnexpectedEnd)` on an empty
//! buffer is ambiguous in general - it means "refill and see" for a
//! genuinely streamed source. But `strede-json::full` is fully in-memory:
//! there is no refill, so an empty buffer at a terminal number state is
//! always the real end of input, never "more digits pending". `skip_value`
//! (and the `arbitrary_precision`-only `NumberBorrowed` impl, both in
//! `full.rs`) used to loop `next_chunk` until it returned `None`, which
//! only happens when an actual terminator byte is found - a bare top-level
//! number with nothing after it has no such byte, so the loop propagated
//! `UnexpectedEnd` instead of recognizing the terminal state. Discovered
//! while adding behavioral coverage for TESTING_GAPS.md item #11
//! (`arbitrary_precision` without `alloc`), whose unit tests happen to feed
//! bare numbers as the entire input.

use strede::borrow::Deserialize;
use strede::{Probe, Skip};
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

fn skips(input: &str) -> bool {
    let d = JsonDeserializer::new(input.as_bytes());
    matches!(
        block_on(<Skip as Deserialize<'_, _>>::deserialize(d, ())),
        Ok(Probe::Hit(_))
    )
}

#[test]
fn skip_bare_integer_at_eof() {
    assert!(skips("42"));
    assert!(skips("0"));
    assert!(skips("-7"));
}

#[test]
fn skip_bare_float_at_eof() {
    assert!(skips("3.14"));
    assert!(skips("1e10"));
    assert!(skips("-2.5e-3"));
}

#[test]
fn skip_number_followed_by_terminator_still_works() {
    // Sanity check: the common case (terminator byte present) was never broken.
    assert!(skips("42 "));
}
