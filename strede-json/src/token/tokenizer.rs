use super::{NumberAccess, StrAccess, Token};
use crate::JsonError;

/// The JSON token types that don't carry string/number data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimpleToken {
    /// An indicator that we've got a partial literal and need a new chunk.
    ///
    /// The next call to [`Tokenizer::next_token`] (on the embedded tokenizer)
    /// will be [`Self::Null`] or [`Self::Bool`].
    PartialLiteral,
    Null,
    Bool(bool),
    ArrayStart,
    ArrayEnd,
    ObjectStart,
    ObjectEnd,
    Comma,
    Colon,
}

/// Internal state tracking for cross-buffer literal parsing.
/// Positioned inside a literal (`true`, `false`, `null`).
/// Contains the remaining bytes to match and the token to emit when complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PartialLiteral(&'static [u8], SimpleToken);

/// JSON tokenizer. Produces [`Token`]s from a byte slice.
///
/// `next_token` takes `self` by value; the new tokenizer state is embedded in
/// the returned [`Token::Simple`], or (for strings/numbers) a fresh
/// [`Tokenizer::new`] is available once the access handle is exhausted.
///
/// Not `Copy` — clone explicitly when you need to branch.
#[derive(Debug, Clone)]
pub(crate) struct Tokenizer {
    mode: Option<PartialLiteral>,
}

impl Tokenizer {
    pub(crate) fn new() -> Self {
        Self { mode: None }
    }

    pub(crate) fn next_token(mut self, src: &mut &[u8]) -> Result<Token, JsonError> {
        if let Some(PartialLiteral(lit, simple_tok)) = self.mode {
            if src.is_empty() {
                return Ok(Token::NoTokens(self));
            }
            self.mode = None;
            let s = self.scan_literal(src, lit, simple_tok)?;
            return Ok(Token::Simple(s, self));
        }

        self.skip_whitespace(src);

        if src.is_empty() {
            return Ok(Token::NoTokens(self));
        }

        let tok = match src[0] {
            b'{' => {
                *src = &src[1..];
                SimpleToken::ObjectStart
            }
            b'}' => {
                *src = &src[1..];
                SimpleToken::ObjectEnd
            }
            b'[' => {
                *src = &src[1..];
                SimpleToken::ArrayStart
            }
            b']' => {
                *src = &src[1..];
                SimpleToken::ArrayEnd
            }
            b',' => {
                *src = &src[1..];
                SimpleToken::Comma
            }
            b':' => {
                *src = &src[1..];
                SimpleToken::Colon
            }
            b'"' => {
                *src = &src[1..];
                return Ok(Token::Str(StrAccess::start()));
            }
            b't' => self.scan_literal(src, b"true", SimpleToken::Bool(true))?,
            b'f' => self.scan_literal(src, b"false", SimpleToken::Bool(false))?,
            b'n' => self.scan_literal(src, b"null", SimpleToken::Null)?,
            b if b == b'-' || b.is_ascii_digit() => {
                // Do NOT consume src[0]; next_chunk will see it in Start state.
                return Ok(Token::Number(NumberAccess::start()));
            }
            b => return Err(JsonError::UnexpectedByte { byte: b }),
        };

        Ok(Token::Simple(tok, self))
    }

    fn skip_whitespace(&mut self, src: &mut &[u8]) {
        while !src.is_empty() && matches!(src[0], b' ' | b'\t' | b'\n' | b'\r') {
            *src = &src[1..];
        }
    }

    fn scan_literal(
        &mut self,
        src: &mut &[u8],
        mut lit: &'static [u8],
        tok: SimpleToken,
    ) -> Result<SimpleToken, JsonError> {
        while !src.is_empty() && !lit.is_empty() {
            if src[0] != lit[0] {
                return Err(JsonError::UnexpectedByte { byte: src[0] });
            }
            *src = &src[1..];
            lit = &lit[1..];
        }
        if !lit.is_empty() {
            self.mode = Some(PartialLiteral(lit, tok));
            Ok(SimpleToken::PartialLiteral)
        } else {
            Ok(tok)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonError;

    // --- structural tokens ---

    #[test]
    fn structural_tokens() {
        let inputs: &[(&[u8], SimpleToken)] = &[
            (b"{", SimpleToken::ObjectStart),
            (b"}", SimpleToken::ObjectEnd),
            (b"[", SimpleToken::ArrayStart),
            (b"]", SimpleToken::ArrayEnd),
            (b",", SimpleToken::Comma),
            (b":", SimpleToken::Colon),
        ];
        for (input, expected) in inputs {
            let tok = Tokenizer::new();
            let mut src: &[u8] = input;
            assert!(
                matches!(tok.next_token(&mut src).unwrap(), Token::Simple(s, _) if s == *expected),
                "input: {:?}",
                input,
            );
        }
    }

    // --- whitespace ---

    #[test]
    fn whitespace_skipped_before_token() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"  \t\n42,";
        assert!(matches!(
            tok.next_token(&mut src).unwrap(),
            Token::Number(_)
        ));
    }

    // --- literals ---

    #[test]
    fn literal_null() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"null,";
        assert!(matches!(
            tok.next_token(&mut src).unwrap(),
            Token::Simple(SimpleToken::Null, _)
        ));
    }

    #[test]
    fn literal_true() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"true,";
        assert!(matches!(
            tok.next_token(&mut src).unwrap(),
            Token::Simple(SimpleToken::Bool(true), _)
        ));
    }

    #[test]
    fn literal_false() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"false,";
        assert!(matches!(
            tok.next_token(&mut src).unwrap(),
            Token::Simple(SimpleToken::Bool(false), _)
        ));
    }

    #[test]
    fn err_bad_byte_in_literal() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"nulx";
        assert_eq!(
            tok.next_token(&mut src).unwrap_err(),
            JsonError::UnexpectedByte { byte: b'x' }
        );
    }

    #[test]
    fn streaming_literal_split_across_buffers() -> Result<(), JsonError> {
        let tok = Tokenizer::new();
        let mut src1: &[u8] = b"tru";
        let partial_tok = match tok.next_token(&mut src1)? {
            Token::Simple(SimpleToken::PartialLiteral, t) => t,
            other => panic!("expected PartialLiteral token, got {other:?}"),
        };
        let mut src2: &[u8] = b"e,";
        assert!(matches!(
            partial_tok.next_token(&mut src2)?,
            Token::Simple(SimpleToken::Bool(true), _)
        ));
        Ok(())
    }

    #[test]
    fn streaming_literal_suspends_on_empty_buffer_resume() -> Result<(), JsonError> {
        // Split "false" as: "fal" | "" | "se,"
        // The empty middle buffer must return NoTokens (handing back the tokenizer
        // with state preserved) rather than hard-erroring.
        let tok = Tokenizer::new();
        let mut src1: &[u8] = b"fal";
        let partial_tok = match tok.next_token(&mut src1)? {
            Token::Simple(SimpleToken::PartialLiteral, t) => t,
            other => panic!("expected PartialLiteral after first chunk, got {other:?}"),
        };
        // Empty buffer: must suspend via NoTokens, not error.
        let mut empty: &[u8] = b"";
        let partial_tok = match partial_tok.next_token(&mut empty)? {
            Token::NoTokens(t) => t,
            other => panic!("expected NoTokens on empty buffer, got {other:?}"),
        };
        // Final chunk completes the literal.
        let mut src3: &[u8] = b"se,";
        assert!(matches!(
            partial_tok.next_token(&mut src3)?,
            Token::Simple(SimpleToken::Bool(false), _)
        ));
        Ok(())
    }

    // --- empty input ---

    #[test]
    fn empty_input_returns_no_tokens() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"";
        assert!(matches!(
            tok.next_token(&mut src).unwrap(),
            Token::NoTokens(_)
        ));
    }

    // --- whitespace before non-number tokens ---

    #[test]
    fn whitespace_skipped_before_string() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"  \t\"hello\"";
        assert!(matches!(tok.next_token(&mut src).unwrap(), Token::Str(_)));
    }

    #[test]
    fn whitespace_skipped_before_structural() {
        let inputs: &[(&[u8], SimpleToken)] = &[
            (b"  {", SimpleToken::ObjectStart),
            (b"\t}", SimpleToken::ObjectEnd),
            (b"\n[", SimpleToken::ArrayStart),
            (b" ]", SimpleToken::ArrayEnd),
            (b" ,", SimpleToken::Comma),
            (b" :", SimpleToken::Colon),
        ];
        for (input, expected) in inputs {
            let tok = Tokenizer::new();
            let mut src: &[u8] = input;
            assert!(
                matches!(tok.next_token(&mut src).unwrap(), Token::Simple(s, _) if s == *expected),
                "input: {:?}",
                input,
            );
        }
    }

    #[test]
    fn whitespace_skipped_before_literal() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"  null,";
        assert!(matches!(
            tok.next_token(&mut src).unwrap(),
            Token::Simple(SimpleToken::Null, _)
        ));
    }

    // --- control character rejection ---

    #[test]
    fn err_control_characters() {
        // Control characters that are not JSON whitespace must be rejected.
        // JSON whitespace is: 0x20 SP, 0x09 HT, 0x0A LF, 0x0D CR.
        let control_chars: &[u8] = &[0x01, 0x02, 0x08, 0x0B, 0x0E, 0x1F];
        for &byte in control_chars {
            let tok = Tokenizer::new();
            let mut src = &[byte][..];
            assert_eq!(
                tok.next_token(&mut src).unwrap_err(),
                JsonError::UnexpectedByte { byte },
                "byte 0x{byte:02X} should be rejected",
            );
        }
    }

    // --- unknown byte ---

    #[test]
    fn err_unknown_byte() {
        let tok = Tokenizer::new();
        let mut src: &[u8] = b"@";
        assert_eq!(
            tok.next_token(&mut src).unwrap_err(),
            JsonError::UnexpectedByte { byte: b'@' }
        );
    }
}
