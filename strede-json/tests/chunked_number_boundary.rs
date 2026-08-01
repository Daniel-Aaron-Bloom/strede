//! Regression coverage for a bug in owned-family chunked JSON number parsing:
//! `NumberAccess::next_chunk` (`strede-json/src/token/number_access.rs`)
//! treated "no bytes left in the *currently delivered* chunk, and the digit
//! state so far is a valid stopping point" as unconditional proof the number
//! was complete. It had no way to distinguish that from "the stream paused
//! mid-number, more digits are coming on the next refill" - so whenever a
//! chunk boundary fell between two digits, the first digit was finalized as
//! the whole number and the rest were left unconsumed, surfacing later as
//! `UnexpectedByte { byte: 0 }` (the map iterator expecting `,`/`}` and
//! finding a stray digit instead).
//!
//! This was originally mis-attributed (see TESTING_GAPS.md) to the
//! internally-tagged-enum + nested-flatten-field combination, because the
//! only repro on hand happened to use multi-digit numbers alongside that
//! shape. Isolation showed the enum/tag/flatten machinery is irrelevant: it
//! reproduces with a bare single-field plain struct under any chunked
//! delivery (any chunk size >= 1) whenever the field's value is more than
//! one digit. Single-digit numbers always happened to "work" by lucky
//! coincidence (the wrong early guess and the right answer agree), which is
//! why no existing test caught this.
//!
//! Fixed by making `next_chunk` always signal `UnexpectedEnd` on an empty
//! chunk (never guessing "done" on its own) and adding
//! `NumberAccess::is_terminal()` so callers can make the "is this actually
//! the end" call only once a real refill attempt confirms no more bytes
//! exist anywhere in the stream.

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_derive::DeserializeOwned as DeriveDeserializeOwned;
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop_bounded;

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Single {
    radius: u32,
}

#[derive(Debug, PartialEq, DeriveDeserializeOwned)]
struct Pair {
    note: u32,
    radius: u32,
}

macro_rules! parse_chunked {
    ($ty:ty, $input:expr, $chunk_size:expr) => {{
        let input: &[u8] = $input;
        let chunk_size: usize = $chunk_size;
        let pos = ::core::cell::Cell::new(chunk_size.min(input.len()));
        block_on_loop_bounded(
            SharedBuf::with_async(
                &input[..chunk_size.min(input.len())],
                async |buf: &mut &[u8]| {
                    let start = pos.get();
                    let end = (start + chunk_size).min(input.len());
                    pos.set(end);
                    *buf = &input[start..end];
                },
                async |shared| {
                    let de = ChunkedJsonDeserializer::new(shared);
                    match <$ty as DeserializeOwned<_>>::deserialize_owned(de, ()).await {
                        Ok(Probe::Hit((_, v))) => Ok(Some(v)),
                        Ok(Probe::Miss) => Ok(None),
                        Err(e) => Err(format!("{e:?}")),
                    }
                },
            ),
            50_000,
        )
    }};
}

#[test]
fn multidigit_int_single_field_struct() {
    let input: &[u8] = br#"{"radius": 55}"#;
    for chunk_size in 1..=5 {
        assert_eq!(
            parse_chunked!(Single, input, chunk_size),
            Ok(Some(Single { radius: 55 })),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn multidigit_int_second_field() {
    // The boundary-sensitive field isn't first, so this also exercises the
    // map-key race continuing correctly after the earlier field is done.
    let input: &[u8] = br#"{"note": 1, "radius": 55}"#;
    for chunk_size in 1..=5 {
        assert_eq!(
            parse_chunked!(Pair, input, chunk_size),
            Ok(Some(Pair {
                note: 1,
                radius: 55
            })),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bare_multidigit_number_document() {
    // Nothing follows the number at all - true end-of-stream immediately
    // after a terminal digit state, exercising the `is_terminal()` path.
    let input: &[u8] = br#"1234"#;
    for chunk_size in 1..=4 {
        assert_eq!(
            parse_chunked!(u32, input, chunk_size),
            Ok(Some(1234u32)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn bare_single_digit_number_document() {
    let input: &[u8] = br#"7"#;
    assert_eq!(parse_chunked!(u32, input, 1), Ok(Some(7u32)));
}

#[test]
fn multidigit_float_document() {
    let input: &[u8] = br#"8.24691"#;
    for chunk_size in 1..=4 {
        assert_eq!(
            parse_chunked!(f64, input, chunk_size),
            Ok(Some(8.24691f64)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn multidigit_exponent_document() {
    let input: &[u8] = br#"1.5e10"#;
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_chunked!(f64, input, chunk_size),
            Ok(Some(1.5e10f64)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn multidigit_negative_int_document() {
    let input: &[u8] = br#"-4321"#;
    for chunk_size in 1..=3 {
        assert_eq!(
            parse_chunked!(i32, input, chunk_size),
            Ok(Some(-4321i32)),
            "chunk_size={chunk_size}"
        );
    }
}

#[test]
fn truncated_number_at_true_eof_is_still_an_error() {
    // Genuinely malformed: a trailing `.` with no fractional digits, and
    // nothing after it. Must not be silently accepted just because the
    // stream ended - `is_terminal()` correctly reports this state as
    // non-terminal (AfterDot), so it stays an error.
    let input: &[u8] = br#"5."#;
    for chunk_size in 1..=2 {
        let result = parse_chunked!(f64, input, chunk_size);
        assert!(
            result.is_err(),
            "chunk_size={chunk_size}: expected error, got {:?}",
            result
        );
    }
}

#[test]
fn truncated_after_minus_at_true_eof_is_still_an_error() {
    let input: &[u8] = br#"-"#;
    let result = parse_chunked!(i32, input, 1);
    assert!(result.is_err(), "expected error, got {:?}", result);
}
