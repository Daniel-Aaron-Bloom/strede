//! Chunk-boundary/truncation fuzzing for JSON string escape sequences
//! (TESTING_GAPS.md item #4).
//!
//! Every escape-decoding state in `strede-json/src/token/str_access.rs`'s
//! `PartialEscape` state machine (`Backslash`, `Unicode { digits, value }`,
//! `HighSurrogate`, `HighSurrogateBackslash`, `HighSurrogateDigits`) already
//! has a hand-picked unit test exercising one specific split point. This
//! file instead sweeps *every* chunk size from 1 up to the full input
//! length (mirroring `strede-json/src/number/decimal_seq.rs`'s own
//! `parse_chunked`/`for cs in 1..=s.len()` sweep, and
//! `strede-json/tests/chunked_number_boundary.rs`'s integration-level
//! version of the same discipline) so that every possible split point is
//! exercised at the owned-family/`ChunkedJsonDeserializer` level, not just
//! the handful of split points a hand-picked unit test happens to pick.
//! `chunk_size=1` alone already forces a suspend/resume boundary after
//! every single byte, so the sweep's cumulative coverage includes every
//! individual split point at least once.

#![cfg(feature = "alloc")]

use strede::{DeserializeOwned, Probe, SharedBuf};
use strede_json::chunked::ChunkedJsonDeserializer;
use strede_test_util::block_on_loop_bounded;

macro_rules! parse_chunked {
    ($input:expr, $chunk_size:expr) => {{
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
                    match <String as DeserializeOwned<_>>::deserialize_owned(de, ()).await {
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

/// Sweep every chunk size from 1 to `input.len()`, asserting each one
/// decodes to exactly `expected`.
fn assert_sweep_matches(input: &[u8], expected: &str) {
    for chunk_size in 1..=input.len() {
        assert_eq!(
            parse_chunked!(input, chunk_size),
            Ok(Some(expected.to_string())),
            "chunk_size={chunk_size}"
        );
    }
}

/// Sweep every chunk size from 1 to `input.len()`, asserting each one
/// errors (never panics, never silently accepts truncated/malformed input).
fn assert_sweep_errs(input: &[u8]) {
    for chunk_size in 1..=input.len() {
        let result = parse_chunked!(input, chunk_size);
        assert!(
            result.is_err(),
            "chunk_size={chunk_size}: expected error, got {result:?}"
        );
    }
}

// -----------------------------------------------------------------------
// Well-formed strings
// -----------------------------------------------------------------------

#[test]
fn ascii_no_escapes() {
    assert_sweep_matches(br#""hello world""#, "hello world");
}

#[test]
fn common_single_char_escapes() {
    // \n \t \" \\ \/ \r \b \f
    assert_sweep_matches(
        br#""a\nb\t\"c\\d\/e\r\bf\fg""#,
        "a\nb\t\"c\\d/e\r\x08f\x0Cg",
    );
}

#[test]
fn unicode_escape_non_surrogate() {
    // café naïve, but every non-ASCII char written as a literal ASCII
    // `\uXXXX` escape in the wire bytes (not raw UTF-8), so every chunk
    // size exercises the `Unicode { digits, value }` partial-escape state.
    let src = format!("\"{}{}", "caf\\u00e9 na\\u00efve", "\"");
    assert_sweep_matches(src.as_bytes(), "café naïve");
}

#[test]
fn surrogate_pair_emoji() {
    // U+1F600 (a grinning-face emoji) as a literal ASCII surrogate-pair
    // escape - exercises `HighSurrogate`/`HighSurrogateBackslash`/
    // `HighSurrogateDigits` partial-escape states across the sweep.
    let src = format!("\"{}{}", "hi \\ud83d\\ude00 bye", "\"");
    assert_sweep_matches(src.as_bytes(), "hi \u{1F600} bye");
}

#[test]
fn surrogate_pair_min_code_point() {
    // U+10000, the first supplementary code point.
    let src = format!("\"{}{}", "\\ud800\\udc00", "\"");
    assert_sweep_matches(src.as_bytes(), "\u{10000}");
}

#[test]
fn surrogate_pair_max_code_point() {
    // U+10FFFF, the last valid Unicode code point.
    let src = format!("\"{}{}", "\\udbff\\udfff", "\"");
    assert_sweep_matches(src.as_bytes(), "\u{10FFFF}");
}

#[test]
fn mixed_escapes_heavy() {
    // Every escape kind back-to-back, to maximize the odds that a given
    // chunk size's split points land inside more than one distinct
    // `PartialEscape` state across a single sweep run.
    let src = format!(
        "\"{}{}",
        "\\n\\t\\r\\\\\\\"\\/\\b\\fA\\u00e9\\ud834\\udd1e end", "\""
    );
    assert_sweep_matches(src.as_bytes(), "\n\t\r\\\"/\x08\x0CA\u{e9}\u{1D11E} end");
}

#[test]
fn raw_multibyte_utf8_unescaped_split_every_byte() {
    // Unescaped (raw) multi-byte UTF-8 content - not an escape sequence at
    // all, but still a chunk-boundary hazard: `é`/`ö` are 2-byte UTF-8
    // sequences, and a chunk boundary can fall between their two bytes.
    // Distinct code path from `PartialEscape` (that state machine only
    // exists for `\`-escapes); this exercises the scan loop's own
    // `core::str::from_utf8` fallback in
    // `strede-json/src/token/str_access.rs::StrAccess::next_chunk`. Built
    // from a regular (non-byte) string literal since byte-string literals
    // must be ASCII.
    let input: &[u8] = "\"héllo wörld\"".as_bytes();
    assert_sweep_matches(input, "héllo wörld");
}

// -----------------------------------------------------------------------
// Truncated/malformed input at true EOF — must error, never panic or hang
// -----------------------------------------------------------------------

#[test]
fn truncated_no_closing_quote() {
    assert_sweep_errs(b"\"hel");
}

#[test]
fn truncated_mid_unicode_escape_digits() {
    // Only 2 of 4 hex digits present, then true EOF.
    assert_sweep_errs(b"\"\\u12");
}

#[test]
fn truncated_right_after_high_surrogate() {
    // High surrogate escape completes; nothing follows at all.
    assert_sweep_errs(b"\"\\ud83d");
}

#[test]
fn truncated_mid_low_surrogate_backslash() {
    // High surrogate done, low surrogate's `\` consumed, then true EOF.
    assert_sweep_errs(b"\"\\ud83d\\");
}

#[test]
fn truncated_mid_low_surrogate_digits() {
    // Only 3 of 4 low-surrogate hex digits present, then true EOF.
    assert_sweep_errs(b"\"\\ud83d\\ude0");
}

#[test]
fn truncated_lone_backslash_at_eof() {
    assert_sweep_errs(b"\"\\");
}

#[test]
fn lone_low_surrogate_is_always_invalid() {
    // Not a truncation - a genuinely malformed escape - but still must
    // error identically regardless of chunk size, not just at chunk_size
    // equal to the full input length.
    assert_sweep_errs(br#""\udc00""#);
}
