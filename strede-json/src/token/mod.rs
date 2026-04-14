mod number_access;
mod str_access;
mod tokenizer;

pub(crate) use number_access::NumberAccess;
pub(crate) use str_access::{StrAccess, StrChunk};
pub(crate) use tokenizer::{SimpleToken, Tokenizer};

/// A single JSON token returned by [`Tokenizer::next_token`].
///
/// Str and Number variants carry the access handle directly as the continuation
/// state; the caller holds them and calls `next_chunk` to read content.
/// Simple variants embed the next [`Tokenizer`] so it can be threaded forward.
#[derive(Debug, Clone)]
pub(crate) enum Token {
    /// A JSON string value has started; use [`StrAccess::next_chunk`] to read it.
    Str(StrAccess),
    /// A JSON number has started; use [`NumberAccess::next_chunk`] to read it.
    Number(NumberAccess),
    /// Any other token (structural, literal, or partial-literal). Carries the
    /// next tokenizer state so the caller can continue parsing.
    Simple(SimpleToken, Tokenizer),
    /// The buffer was exhausted without producing a token (e.g. whitespace-only
    /// input). The tokenizer state is returned so the caller can feed the next
    /// buffer and continue.
    NoTokens(Tokenizer),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonError;

    /// Drain a `StrAccess`, asserting each chunk matches the expected sequence.
    /// `expected` is a list of `Ok(Some(...))` / `Ok(None)` patterns; a final
    /// `Ok(None)` is asserted automatically after the last element.
    fn assert_str_chunks(
        mut acc: StrAccess,
        src: &mut &[u8],
        expected: &[StrChunk<'_>],
    ) -> Result<(), JsonError> {
        for exp in expected {
            let chunk = acc.next_chunk(src)?.expect("expected chunk, got None");
            assert_eq!(&chunk, exp);
        }
        assert_eq!(acc.next_chunk(src)?, None, "expected end of string");
        Ok(())
    }

    /// Drain a `NumberAccess`, asserting each chunk matches the expected sequence.
    fn assert_number_chunks(
        mut acc: NumberAccess,
        src: &mut &[u8],
        expected: &[&str],
    ) -> Result<(), JsonError> {
        for &exp in expected {
            let chunk = acc.next_chunk(src)?.expect("expected chunk, got None");
            assert_eq!(chunk, exp);
        }
        assert_eq!(acc.next_chunk(src)?, None, "expected end of number");
        Ok(())
    }

    // --- Tokenizer → StrAccess handoff ---

    #[test]
    fn string_token_then_resume() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\"hello\",";
        let acc = match Tokenizer::new().next_token(&mut src)? {
            Token::Str(a) => a,
            other => panic!("expected Str, got {other:?}"),
        };
        assert_str_chunks(acc, &mut src, &[StrChunk::Slice("hello")])?;
        assert!(matches!(
            Tokenizer::new().next_token(&mut src)?,
            Token::Simple(SimpleToken::Comma, _)
        ));
        Ok(())
    }

    #[test]
    fn string_token_with_escape() -> Result<(), JsonError> {
        let mut src: &[u8] = b"\"hi\\nworld\"}";
        let acc = match Tokenizer::new().next_token(&mut src)? {
            Token::Str(a) => a,
            other => panic!("expected Str, got {other:?}"),
        };
        // "hi" is emitted, then the \n escape, then "world"
        assert_str_chunks(
            acc,
            &mut src,
            &[
                StrChunk::Slice("hi"),
                StrChunk::Char('\n'),
                StrChunk::Slice("world"),
            ],
        )?;
        Ok(())
    }

    // --- Tokenizer → NumberAccess handoff ---

    #[test]
    fn number_token_then_resume() -> Result<(), JsonError> {
        let mut src: &[u8] = b"42}";
        let acc = match Tokenizer::new().next_token(&mut src)? {
            Token::Number(a) => a,
            other => panic!("expected Number, got {other:?}"),
        };
        assert_number_chunks(acc, &mut src, &["42"])?;
        assert!(matches!(
            Tokenizer::new().next_token(&mut src)?,
            Token::Simple(SimpleToken::ObjectEnd, _)
        ));
        Ok(())
    }

    #[test]
    fn float_token_then_resume() -> Result<(), JsonError> {
        let mut src: &[u8] = b"-3.14]";
        let acc = match Tokenizer::new().next_token(&mut src)? {
            Token::Number(a) => a,
            other => panic!("expected Number, got {other:?}"),
        };
        assert_number_chunks(acc, &mut src, &["-3.14"])?;
        assert!(matches!(
            Tokenizer::new().next_token(&mut src)?,
            Token::Simple(SimpleToken::ArrayEnd, _)
        ));
        Ok(())
    }

    // --- full key-value pipeline ---

    #[test]
    fn object_key_value_pipeline() -> Result<(), JsonError> {
        // {"key":42}
        let mut src: &[u8] = b"{\"key\":42}";

        let tok = match Tokenizer::new().next_token(&mut src)? {
            Token::Simple(SimpleToken::ObjectStart, t) => t,
            other => panic!("expected ObjectStart, got {other:?}"),
        };
        let str_acc = match tok.next_token(&mut src)? {
            Token::Str(a) => a,
            other => panic!("expected Str, got {other:?}"),
        };
        assert_str_chunks(str_acc, &mut src, &[StrChunk::Slice("key")])?;

        let tok = match Tokenizer::new().next_token(&mut src)? {
            Token::Simple(SimpleToken::Colon, t) => t,
            other => panic!("expected Colon, got {other:?}"),
        };
        let num_acc = match tok.next_token(&mut src)? {
            Token::Number(a) => a,
            other => panic!("expected Number, got {other:?}"),
        };
        assert_number_chunks(num_acc, &mut src, &["42"])?;

        assert!(matches!(
            Tokenizer::new().next_token(&mut src)?,
            Token::Simple(SimpleToken::ObjectEnd, _)
        ));
        Ok(())
    }
}
