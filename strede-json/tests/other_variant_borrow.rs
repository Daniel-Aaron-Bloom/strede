//! Borrow-family `#[strede(other)]` fixtures.
//!
//! Exercises both arms of `match_entry_str_against`'s zero-copy-vs-chunked
//! fallback (see `strede/src/impls/string_enum.rs`): a plain string hits the
//! zero-copy `deserialize_str` path, while a string containing an escape
//! sequence forces the `deserialize_str_chunks` fallback. Both must reach the
//! `#[strede(other)]` catch-all correctly when the (decoded) name is unknown.

use strede::Probe;
use strede_derive::Deserialize;
use strede_json::JsonDeserializer;
use strede_test_util::block_on;

#[derive(Debug, PartialEq, Deserialize)]
enum Color {
    Red,
    Green,
    #[strede(other)]
    Other,
}

fn parse(input: &str) -> Color {
    let de = JsonDeserializer::new(input.as_bytes());
    match block_on(<Color as strede::Deserialize<'_, _>>::deserialize(de, ())).unwrap() {
        Probe::Hit((_, v)) => v,
        Probe::Miss => panic!("Miss"),
    }
}

#[test]
fn known_variant_zero_copy() {
    assert_eq!(parse("\"Red\""), Color::Red);
    assert_eq!(parse("\"Green\""), Color::Green);
}

#[test]
fn known_variant_escaped() {
    // "Red" decodes to "Red" but cannot be a zero-copy slice, forcing
    // the deserialize_str_chunks fallback.
    assert_eq!(parse("\"R\\u0065d\""), Color::Red);
    assert_eq!(parse("\"Gr\\u0065\\u0065n\""), Color::Green);
}

#[test]
fn unknown_falls_to_other_zero_copy() {
    assert_eq!(parse("\"Blue\""), Color::Other);
    assert_eq!(parse("\"Yellow\""), Color::Other);
}

#[test]
fn unknown_falls_to_other_escaped() {
    // "Blue" decodes to "Blue", still unknown, still forced through the
    // chunked fallback since deserialize_str misses on the escape.
    assert_eq!(parse("\"Bl\\u0075e\""), Color::Other);
    assert_eq!(parse("\"\\u0059ellow\""), Color::Other);
}
