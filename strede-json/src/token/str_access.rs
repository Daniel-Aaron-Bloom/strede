use crate::JsonError;

/// One chunk yielded by [`StrAccess::next_chunk`].
///
/// Unescaped runs are zero-copy source slices (`Slice`); escape sequences
/// decode to a single Unicode scalar value (`Char`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StrChunk<'a> {
    /// Zero-copy slice of unescaped source bytes.
    Slice(&'a str),
    /// A single decoded escape sequence (e.g. `\n` → `'\n'`, `\uXXXX` → scalar).
    Char(char),
}

/// State saved when an escape sequence is split across buffer boundaries.
///
/// - `Backslash`: the `\` was the last byte in the buffer; the escape character
///   (`n`, `"`, `u`, ...) arrives in the next buffer.
/// - `Unicode { digits, value }`: `\u` plus `digits` hex digits were available;
///   `4 - digits` more hex digits are expected in the next buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PartialEscape {
    Backslash,
    Unicode {
        digits: u8,
        value: u16,
    },
    /// Completed `\uD800`–`\uDBFF`; waiting for `\` of the low surrogate.
    HighSurrogate(u16),
    /// Saw `\` after a high surrogate; waiting for `u` + 4 hex digits.
    HighSurrogateBackslash(u16),
    /// Collecting hex digits for the low surrogate half.
    HighSurrogateDigits {
        high: u16,
        digits: u8,
        value: u16,
    },
}

/// Decode a single ASCII hex digit.
fn parse_hex_digit(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some((b - b'0') as u32),
        b'a'..=b'f' => Some((b - b'a' + 10) as u32),
        b'A'..=b'F' => Some((b - b'A' + 10) as u32),
        _ => None,
    }
}

/// Handle for reading chunks of the current JSON string.
///
/// Obtained from [`Token::Str`]. Call [`next_chunk`](Self::next_chunk)
/// repeatedly until it returns `Ok(None)`, which signals that the closing `"` was
/// consumed. After that, call [`Tokenizer::new`] to continue parsing the stream.
///
/// When a buffer boundary falls inside an escape sequence, `next_chunk` saves the
/// partial state in `self.partial` and returns
/// `Ok(Some(StrChunk::Slice("")))` - an empty slice that lets the caller feed the
/// next buffer without ending or erroring the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StrAccess {
    /// Non-`None` when a buffer boundary split an escape sequence mid-way.
    partial: Option<PartialEscape>,
}

impl StrAccess {
    pub(super) fn start() -> Self {
        Self { partial: None }
    }
}

impl StrAccess {
    /// Called when a `\` has just been consumed and we need to decode the rest
    /// of the escape sequence.  May suspend (returning an empty slice) if `src`
    /// is exhausted before the escape is complete.
    fn handle_escape<'a>(&mut self, src: &mut &'a [u8]) -> Result<Option<StrChunk<'a>>, JsonError> {
        if src.is_empty() {
            self.partial = Some(PartialEscape::Backslash);
            return Ok(Some(StrChunk::Slice("")));
        }
        let c = match src[0] {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\x08',
            b'f' => '\x0C',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                *src = &src[1..]; // consume 'u'
                return self.consume_unicode_digits(0, 0, src);
            }
            _ => return Err(JsonError::InvalidEscape),
        };
        *src = &src[1..]; // consume the single-byte escape char
        self.partial = None;
        Ok(Some(StrChunk::Char(c)))
    }

    /// Read up to `4 - digits_so_far` hex digits from `src`, accumulating into
    /// `value_so_far`.  Returns `Char(c)` when all 4 digits are collected, or
    /// suspends with an empty slice when the buffer is exhausted first.
    fn consume_unicode_digits<'a>(
        &mut self,
        mut digits: u8,
        mut value: u16,
        src: &mut &'a [u8],
    ) -> Result<Option<StrChunk<'a>>, JsonError> {
        while digits < 4 && !src.is_empty() {
            let d = parse_hex_digit(src[0]).ok_or(JsonError::InvalidEscape)?;
            value = ((value as u32) << 4 | d) as u16;
            digits += 1;
            *src = &src[1..];
        }
        if digits < 4 {
            self.partial = Some(PartialEscape::Unicode { digits, value });
            return Ok(Some(StrChunk::Slice("")));
        }
        // High surrogate: expect a following \uDC00-\uDFFF.
        if (0xD800..=0xDBFF).contains(&value) {
            self.partial = Some(PartialEscape::HighSurrogate(value));
            return Ok(Some(StrChunk::Slice("")));
        }
        // Lone low surrogate: always invalid.
        if (0xDC00..=0xDFFF).contains(&value) {
            return Err(JsonError::InvalidEscape);
        }
        let c = char::from_u32(value as u32).ok_or(JsonError::InvalidEscape)?;
        self.partial = None;
        Ok(Some(StrChunk::Char(c)))
    }

    /// Called after completing a high surrogate (`\uD800`–`\uDBFF`).
    /// Expects `\` as the very next byte, then delegates to
    /// [`handle_low_surrogate_escape`](Self::handle_low_surrogate_escape).
    fn handle_high_surrogate<'a>(
        &mut self,
        high: u16,
        src: &mut &'a [u8],
    ) -> Result<Option<StrChunk<'a>>, JsonError> {
        if src.is_empty() {
            self.partial = Some(PartialEscape::HighSurrogate(high));
            return Ok(Some(StrChunk::Slice("")));
        }
        match src[0] {
            b'\\' => {
                *src = &src[1..];
                self.handle_low_surrogate_escape(high, src)
            }
            _ => Err(JsonError::InvalidEscape),
        }
    }

    /// After the `\` following a high surrogate has been consumed.
    /// Expects `u` as the next byte, then collects 4 hex digits for the low surrogate.
    fn handle_low_surrogate_escape<'a>(
        &mut self,
        high: u16,
        src: &mut &'a [u8],
    ) -> Result<Option<StrChunk<'a>>, JsonError> {
        if src.is_empty() {
            self.partial = Some(PartialEscape::HighSurrogateBackslash(high));
            return Ok(Some(StrChunk::Slice("")));
        }
        match src[0] {
            b'u' => {
                *src = &src[1..];
                self.consume_low_surrogate_digits(high, 0, 0, src)
            }
            _ => Err(JsonError::InvalidEscape),
        }
    }

    /// Reads up to `4 - digits` hex digits for the low surrogate half of a surrogate pair.
    /// Suspends with `HighSurrogateDigits` when the buffer is exhausted before all 4 digits
    /// are seen.  On completion, validates that the value is in `0xDC00`–`0xDFFF` and
    /// combines it with `high` into the final scalar.
    fn consume_low_surrogate_digits<'a>(
        &mut self,
        high: u16,
        mut digits: u8,
        mut value: u16,
        src: &mut &'a [u8],
    ) -> Result<Option<StrChunk<'a>>, JsonError> {
        while digits < 4 && !src.is_empty() {
            let d = parse_hex_digit(src[0]).ok_or(JsonError::InvalidEscape)?;
            value = ((value as u32) << 4 | d) as u16;
            digits += 1;
            *src = &src[1..];
        }
        if digits < 4 {
            self.partial = Some(PartialEscape::HighSurrogateDigits {
                high,
                digits,
                value,
            });
            return Ok(Some(StrChunk::Slice("")));
        }
        if !(0xDC00..=0xDFFF).contains(&value) {
            return Err(JsonError::InvalidEscape);
        }
        let codepoint = 0x10000u32 + ((high as u32 - 0xD800) << 10) + (value as u32 - 0xDC00);
        // SAFETY: codepoint is guaranteed valid by the surrogate range checks above.
        let c = char::from_u32(codepoint).unwrap();
        self.partial = None;
        Ok(Some(StrChunk::Char(c)))
    }

    /// Yield the next chunk of the current JSON string.
    ///
    /// - `Ok(Some(Slice(s)))` - unescaped content (zero-copy); `s` may be empty
    ///   when an escape sequence was split across a buffer boundary (call again
    ///   with the next buffer).
    /// - `Ok(Some(Char(c)))` - a fully-decoded escape sequence.
    /// - `Ok(None)` - closing `"` was just consumed. Use [`Tokenizer::new`] to
    ///   continue parsing the stream.
    /// - `Err(InvalidUtf8)` - encoding error.
    /// - `Err(UnexpectedEnd)` - stream ended inside a string with no closing `"`.
    pub(crate) fn next_chunk<'a>(
        &mut self,
        src: &mut &'a [u8],
    ) -> Result<Option<StrChunk<'a>>, JsonError> {
        // Resume a partial escape saved from a previous buffer split.
        if self.partial.is_some() {
            if src.is_empty() {
                // Still no data; suspend again.
                return Ok(Some(StrChunk::Slice("")));
            }
            return match self.partial.take().unwrap() {
                PartialEscape::Backslash => self.handle_escape(src),
                PartialEscape::Unicode { digits, value } => {
                    self.consume_unicode_digits(digits, value, src)
                }
                PartialEscape::HighSurrogate(high) => self.handle_high_surrogate(high, src),
                PartialEscape::HighSurrogateBackslash(high) => {
                    self.handle_low_surrogate_escape(high, src)
                }
                PartialEscape::HighSurrogateDigits {
                    high,
                    digits,
                    value,
                } => self.consume_low_surrogate_digits(high, digits, value, src),
            };
        }

        if src.is_empty() {
            return Err(JsonError::UnexpectedEnd);
        }

        let mut cur = 0;

        while cur < src.len() {
            match src[cur] {
                b'\\' => {
                    // Emit any unescaped content before the backslash first.
                    if cur > 0 {
                        let s = core::str::from_utf8(&src[..cur])
                            .map_err(|_| JsonError::InvalidUtf8)?;
                        *src = &src[cur..]; // leave `\` at src[0] for next call
                        return Ok(Some(StrChunk::Slice(s)));
                    }
                    // cur == 0: positioned at `\`; consume it and decode.
                    *src = &src[1..];
                    return self.handle_escape(src);
                }
                b'"' => {
                    if cur > 0 {
                        // Content before the closing quote: emit the chunk but
                        // leave `"` in the buffer so the next call returns None.
                        let s = core::str::from_utf8(&src[..cur])
                            .map_err(|_| JsonError::InvalidUtf8)?;
                        *src = &src[cur..]; // `"` stays at src[0]
                        return Ok(Some(StrChunk::Slice(s)));
                    } else {
                        // `"` is the first byte: closing quote with no preceding
                        // content; consume it and signal done.
                        *src = &src[1..];
                        return Ok(None);
                    }
                }
                b => {
                    if b < 0x20 {
                        return Err(JsonError::UnexpectedByte { byte: b });
                    }
                    cur += 1;
                }
            }
        }

        let s = core::str::from_utf8(src).map_err(|_| JsonError::InvalidUtf8)?;
        *src = &src[src.len()..];
        Ok(Some(StrChunk::Slice(s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonError;

    /// Consume a JSON string via `StrAccess`, collecting all chunks into `buf`.
    ///
    /// `src` must start at the first content byte (opening `"` already consumed),
    /// and must include the closing `"`.
    fn collect_string<'buf>(
        src: &mut &[u8],
        buf: &'buf mut [u8; 64],
    ) -> Result<&'buf str, JsonError> {
        let mut access = StrAccess::start();
        let mut len = 0;
        loop {
            let mut tmp = [0u8; 4];
            match access.next_chunk(src)? {
                Some(StrChunk::Slice(s)) => {
                    let b = s.as_bytes();
                    buf[len..len + b.len()].copy_from_slice(b);
                    len += b.len();
                }
                Some(StrChunk::Char(c)) => {
                    let b = c.encode_utf8(&mut tmp).as_bytes();
                    buf[len..len + b.len()].copy_from_slice(b);
                    len += b.len();
                }
                None => return Ok(core::str::from_utf8(&buf[..len]).unwrap()),
            }
        }
    }

    // --- basic strings ---

    #[test]
    fn string_simple() -> Result<(), JsonError> {
        let mut src: &[u8] = b"hello\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "hello");
        Ok(())
    }

    #[test]
    fn string_empty() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "");
        Ok(())
    }

    #[test]
    fn streaming_string_split_across_buffers() -> Result<(), JsonError> {
        let mut src1: &[u8] = b"hel";
        let mut access = StrAccess::start();
        let c1 = access.next_chunk(&mut src1)?.expect("chunk");
        assert_eq!(c1, StrChunk::Slice("hel"));
        // src1 exhausted; feed second buffer
        let mut src2: &[u8] = b"lo\"";
        let c2 = access.next_chunk(&mut src2)?.expect("chunk");
        assert_eq!(c2, StrChunk::Slice("lo"));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn err_string_truncated_without_closing_quote() -> Result<(), JsonError> {
        let mut src: &[u8] = b"hello";
        let mut access = StrAccess::start();
        // drain any chunks until error
        loop {
            match access.next_chunk(&mut src) {
                Ok(Some(_)) => {}
                Err(JsonError::UnexpectedEnd) => return Ok(()),
                other => panic!("expected UnexpectedEnd, got {other:?}"),
            }
        }
    }

    // --- escape sequences ---

    #[test]
    fn escape_quote() -> Result<(), JsonError> {
        let mut src: &[u8] = b"say \\\"hi\\\"\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "say \"hi\"");
        Ok(())
    }

    #[test]
    fn escape_backslash() -> Result<(), JsonError> {
        let mut src: &[u8] = b"foo\\\\bar\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "foo\\bar");
        Ok(())
    }

    #[test]
    fn escape_common() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\\n\\t\\r\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "\n\t\r");
        Ok(())
    }

    #[test]
    fn escape_unicode_ascii() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\\u0041\""; // U+0041 = 'A'
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "A");
        Ok(())
    }

    #[test]
    fn escape_unicode_two_byte() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\\u00e9\""; // U+00E9 = 'é' (2 UTF-8 bytes)
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "é");
        Ok(())
    }

    #[test]
    fn err_invalid_escape() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\\q\"";
        let mut access = StrAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidEscape
        );
        Ok(())
    }

    #[test]
    fn err_invalid_unicode_hex() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\\uXXXX\"";
        let mut access = StrAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidEscape
        );
        Ok(())
    }

    // --- raw control characters (JSON §7: must be escaped) ---

    #[test]
    fn err_raw_null_byte() {
        // U+0000 must be written as \u0000, not as a literal null byte.
        let mut src: &[u8] = b"\x00\"";
        let mut access = StrAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::UnexpectedByte { byte: 0x00 },
        );
    }

    #[test]
    fn err_raw_newline() {
        // U+000A (LF) must be written as \n, not as a raw newline.
        let mut src: &[u8] = b"\x0A\"";
        let mut access = StrAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::UnexpectedByte { byte: 0x0A },
        );
    }

    #[test]
    fn err_raw_tab() {
        // U+0009 (HT) must be written as \t, not as a raw tab.
        let mut src: &[u8] = b"\x09\"";
        let mut access = StrAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::UnexpectedByte { byte: 0x09 },
        );
    }

    #[test]
    fn err_raw_control_mid_string() {
        // Control character appearing after valid content is also rejected.
        // The scan loop errors immediately upon hitting the control byte, even
        // if there was valid content before it.
        let mut src: &[u8] = b"hello\x01world\"";
        let mut access = StrAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::UnexpectedByte { byte: 0x01 },
        );
    }

    // --- surrogate pairs ---

    #[test]
    fn surrogate_pair_basic() -> Result<(), JsonError> {
        // \uD834\uDD1E encodes U+1D11E (𝄞).
        let mut src: &[u8] = b"\\uD834\\uDD1E\"";
        let mut access = StrAccess::start();
        // High surrogate suspends with an empty slice.
        assert_eq!(access.next_chunk(&mut src)?, Some(StrChunk::Slice("")));
        // Low surrogate completes the pair.
        assert_eq!(access.next_chunk(&mut src)?, Some(StrChunk::Char('𝄞')));
        assert_eq!(access.next_chunk(&mut src)?, None);
        Ok(())
    }

    #[test]
    fn surrogate_pair_collect_string() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\\uD834\\uDD1E\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "𝄞");
        Ok(())
    }

    #[test]
    fn surrogate_pair_min() -> Result<(), JsonError> {
        // \uD800\uDC00 = U+10000, the first supplementary code point.
        let mut src: &[u8] = b"\\uD800\\uDC00\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "\u{10000}");
        Ok(())
    }

    #[test]
    fn surrogate_pair_max() -> Result<(), JsonError> {
        // \uDBFF\uDFFF = U+10FFFF, the last valid Unicode code point.
        let mut src: &[u8] = b"\\uDBFF\\uDFFF\"";
        let mut buf = [0u8; 64];
        assert_eq!(collect_string(&mut src, &mut buf)?, "\u{10FFFF}");
        Ok(())
    }

    #[test]
    fn err_lone_high_surrogate() -> Result<(), JsonError> {
        // High surrogate followed immediately by closing quote - no low surrogate.
        let mut src: &[u8] = b"\\uD834\"";
        let mut access = StrAccess::start();
        // First call suspends after the high surrogate (empty slice).
        assert_eq!(access.next_chunk(&mut src)?, Some(StrChunk::Slice("")));
        // Next byte is `"`, not `\` - invalid.
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidEscape
        );
        Ok(())
    }

    #[test]
    fn err_lone_low_surrogate() -> Result<(), JsonError> {
        // Low surrogate without a preceding high surrogate.
        let mut src: &[u8] = b"\\uDD1E\"";
        let mut access = StrAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidEscape
        );
        Ok(())
    }

    #[test]
    fn err_high_then_non_low_surrogate_u() -> Result<(), JsonError> {
        // \uD834 followed by \u0041 ('A') - second escape is not a low surrogate.
        let mut src: &[u8] = b"\\uD834\\u0041\"";
        let mut access = StrAccess::start();
        assert_eq!(access.next_chunk(&mut src)?, Some(StrChunk::Slice(""))); // suspends at high
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidEscape
        );
        Ok(())
    }

    #[test]
    fn err_high_then_non_backslash() -> Result<(), JsonError> {
        // \uD834 followed by a regular character, not `\`.
        let mut src: &[u8] = b"\\uD834abc\"";
        let mut access = StrAccess::start();
        assert_eq!(access.next_chunk(&mut src)?, Some(StrChunk::Slice(""))); // suspends at high
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidEscape
        );
        Ok(())
    }

    // --- surrogate pair split across buffer boundaries ---

    #[test]
    fn split_surrogate_between_pairs() -> Result<(), JsonError> {
        // Buffer boundary falls between the two \uXXXX escapes.
        let mut access = StrAccess::start();
        let mut src1: &[u8] = b"\\uD834";
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice(""))); // high done, suspended
        let mut src2: &[u8] = b"\\uDD1E\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('𝄞')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_surrogate_mid_low_backslash() -> Result<(), JsonError> {
        // Buffer ends at the `\` that starts the low surrogate escape.
        let mut access = StrAccess::start();
        let mut src1: &[u8] = b"\\uD834\\";
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice(""))); // high done
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice(""))); // `\` saved
        let mut src2: &[u8] = b"uDD1E\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('𝄞')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_surrogate_mid_low_u() -> Result<(), JsonError> {
        // Buffer ends at the `u` that follows the `\` of the low surrogate.
        let mut access = StrAccess::start();
        let mut src1: &[u8] = b"\\uD834\\u";
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice(""))); // high done
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice(""))); // `\u` saved
        let mut src2: &[u8] = b"DD1E\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('𝄞')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_surrogate_mid_low_digits() -> Result<(), JsonError> {
        // Buffer ends after 2 hex digits of the low surrogate.
        let mut access = StrAccess::start();
        let mut src1: &[u8] = b"\\uD834\\uDD";
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice(""))); // high done
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice(""))); // 2 low digits saved
        let mut src2: &[u8] = b"1E\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('𝄞')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_surrogate_tiny_buffers() -> Result<(), JsonError> {
        // Each byte of `\uD834\uDD1E"` in its own buffer.
        // Sequence: `\` `u` `D` `8` `3` `4` `\` `u` `D` `D` `1` `E` `"`
        let mut access = StrAccess::start();
        macro_rules! feed {
            ($b:expr) => {{
                let mut s: &[u8] = $b;
                access.next_chunk(&mut s)
            }};
        }
        assert_eq!(feed!(b"\\")?, Some(StrChunk::Slice(""))); // Backslash
        assert_eq!(feed!(b"u")?, Some(StrChunk::Slice(""))); // Unicode{0,0}
        assert_eq!(feed!(b"D")?, Some(StrChunk::Slice(""))); // digit 1
        assert_eq!(feed!(b"8")?, Some(StrChunk::Slice(""))); // digit 2
        assert_eq!(feed!(b"3")?, Some(StrChunk::Slice(""))); // digit 3
        assert_eq!(feed!(b"4")?, Some(StrChunk::Slice(""))); // digit 4 → HighSurrogate(0xD834)
        assert_eq!(feed!(b"\\")?, Some(StrChunk::Slice(""))); // HighSurrogateBackslash
        assert_eq!(feed!(b"u")?, Some(StrChunk::Slice(""))); // HighSurrogateDigits{0,0}
        assert_eq!(feed!(b"D")?, Some(StrChunk::Slice(""))); // digit 1
        assert_eq!(feed!(b"D")?, Some(StrChunk::Slice(""))); // digit 2
        assert_eq!(feed!(b"1")?, Some(StrChunk::Slice(""))); // digit 3
        assert_eq!(feed!(b"E")?, Some(StrChunk::Char('𝄞'))); // digit 4 → complete
        assert_eq!(feed!(b"\"")?, None);
        Ok(())
    }

    // --- split escape sequences across buffer boundaries ---

    #[test]
    fn split_single_char_escape_across_buffers() -> Result<(), JsonError> {
        // First buffer ends with `\`; the escape char `n` arrives in the next buffer.
        let mut buf = [0u8; 64];
        let mut src1: &[u8] = b"hello\\";
        let mut access = StrAccess::start();
        // Emits "hello" from the unescaped prefix ...
        assert_eq!(
            access.next_chunk(&mut src1)?,
            Some(StrChunk::Slice("hello"))
        );
        // ... then suspends at `\` with an empty slice (partial state saved).
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice("")));
        // Feed second buffer: `\n` completes, then closing quote.
        let mut src2: &[u8] = b"n\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('\n')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        // Verify the full decoded string via collect_string on a single buffer.
        let mut full: &[u8] = b"hello\\n\"";
        assert_eq!(collect_string(&mut full, &mut buf)?, "hello\n");
        Ok(())
    }

    #[test]
    fn split_unicode_escape_after_backslash() -> Result<(), JsonError> {
        // First buffer ends with `\`; the entire `uXXXX` arrives in the next buffer.
        // \u00e9 = U+00E9 = 'é'
        let mut src1: &[u8] = b"caf\\";
        let mut access = StrAccess::start();
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice("caf")));
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice("")));
        let mut src2: &[u8] = b"u00e9\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('é')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_unicode_escape_after_u() -> Result<(), JsonError> {
        // First buffer ends with `\u`; the 4 hex digits arrive in the next buffer.
        // \u0041 = 'A'
        let mut src1: &[u8] = b"\\u";
        let mut access = StrAccess::start();
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice("")));
        let mut src2: &[u8] = b"0041\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('A')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_unicode_escape_1_digit() -> Result<(), JsonError> {
        // \u0041 = 'A'; buffer boundary after `\u0`
        let mut src1: &[u8] = b"\\u0";
        let mut access = StrAccess::start();
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice("")));
        let mut src2: &[u8] = b"041\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('A')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_unicode_escape_2_digits() -> Result<(), JsonError> {
        // \u0041 = 'A'; buffer boundary after `\u00`
        let mut src1: &[u8] = b"\\u00";
        let mut access = StrAccess::start();
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice("")));
        let mut src2: &[u8] = b"41\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('A')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_unicode_escape_3_digits() -> Result<(), JsonError> {
        // Buffer ends after 3 of 4 hex digits; the final digit arrives in the next buffer.
        // \u0041 = U+0041 = 'A'
        let mut src1: &[u8] = b"\\u004";
        let mut access = StrAccess::start();
        // 3 digits consumed; suspends with an empty slice.
        assert_eq!(access.next_chunk(&mut src1)?, Some(StrChunk::Slice("")));
        let mut src2: &[u8] = b"1\"";
        assert_eq!(access.next_chunk(&mut src2)?, Some(StrChunk::Char('A')));
        assert_eq!(access.next_chunk(&mut src2)?, None);
        Ok(())
    }

    #[test]
    fn split_unicode_escape_tiny_buffers() -> Result<(), JsonError> {
        // \u00e9 = U+00E9 = 'é', each byte of the escape in its own buffer.
        // Sequence: `\` | `u` | `0` | `0` | `e` | `9` | `"`
        let mut access = StrAccess::start();

        let mut s: &[u8] = b"\\";
        assert_eq!(access.next_chunk(&mut s)?, Some(StrChunk::Slice(""))); // `\` → Backslash
        let mut s: &[u8] = b"u";
        assert_eq!(access.next_chunk(&mut s)?, Some(StrChunk::Slice(""))); // `u` → Unicode{0,0}
        let mut s: &[u8] = b"0";
        assert_eq!(access.next_chunk(&mut s)?, Some(StrChunk::Slice(""))); // digit 1
        let mut s: &[u8] = b"0";
        assert_eq!(access.next_chunk(&mut s)?, Some(StrChunk::Slice(""))); // digit 2
        let mut s: &[u8] = b"e";
        assert_eq!(access.next_chunk(&mut s)?, Some(StrChunk::Slice(""))); // digit 3
        let mut s: &[u8] = b"9";
        assert_eq!(access.next_chunk(&mut s)?, Some(StrChunk::Char('é'))); // digit 4 → decoded
        let mut s: &[u8] = b"\"";
        assert_eq!(access.next_chunk(&mut s)?, None);
        Ok(())
    }
}
