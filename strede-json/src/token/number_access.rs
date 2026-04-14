use crate::JsonError;

/// Tracks position within the JSON number grammar.
///
/// Terminal states (valid to end on): `Zero`, `IntDigits`, `FracDigits`, `ExpDigits`.
/// Non-terminal states (stream truncated → `UnexpectedEnd`): all others except `Finished`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberState {
    /// First byte not yet consumed; `src[0]` is `-` or a digit.
    Start,
    /// Consumed `-`; must see a digit next.
    AfterMinus,
    /// Consumed leading `0`; no more digits allowed before `.` or `e`.
    Zero,
    /// Consuming non-zero integer digits.
    IntDigits,
    /// Consumed `.`; must see at least one digit.
    AfterDot,
    /// Consuming fractional digits.
    FracDigits,
    /// Consumed `e`/`E`; must see a sign or digit.
    AfterE,
    /// Consumed sign after `e`; must see a digit.
    AfterESign,
    /// Consuming exponent digits.
    ExpDigits,
    /// All bytes of this number have been emitted; next call returns `None`.
    Finished,
}

impl NumberState {
    fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Zero | Self::IntDigits | Self::FracDigits | Self::ExpDigits
        )
    }
}

/// Handle for reading chunks of the current JSON number.
///
/// Obtained from [`Token::Number`]. Call [`next_chunk`](Self::next_chunk)
/// repeatedly until it returns `Ok(None)`, which signals that the number is
/// complete. After that, use [`Tokenizer::new`] to continue parsing the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NumberAccess {
    state: NumberState,
}

impl NumberAccess {
    pub(super) fn start() -> Self {
        Self {
            state: NumberState::Start,
        }
    }
}

impl NumberAccess {
    /// Yield the next chunk of the current JSON number.
    ///
    /// - `Ok(Some(chunk))` — more number content; zero-copy slice of the source.
    /// - `Ok(None)` — number finished. Use [`Tokenizer::new`] to continue.
    /// - `Err(InvalidNumber)` — grammar violation.
    /// - `Err(UnexpectedEnd)` — stream ended inside an incomplete number.
    pub(crate) fn next_chunk<'a>(
        &mut self,
        src: &mut &'a [u8],
    ) -> Result<Option<&'a str>, JsonError> {
        // All bytes already emitted; signal done.
        if self.state == NumberState::Finished {
            return Ok(None);
        }

        // Buffer empty — terminal states end the number cleanly; non-terminal
        // states mean the stream was truncated mid-number.
        if src.is_empty() {
            return if self.state.is_terminal() {
                Ok(None)
            } else {
                Err(JsonError::UnexpectedEnd)
            };
        }

        let mut cur_state = self.state;
        let mut cur = 0;

        loop {
            if cur >= src.len() {
                // Buffer exhausted mid-scan.
                break;
            }

            let byte = src[cur];

            let next_state: Option<NumberState> = match cur_state {
                NumberState::Start => match byte {
                    b'-' => Some(NumberState::AfterMinus),
                    b'0' => Some(NumberState::Zero),
                    b'1'..=b'9' => Some(NumberState::IntDigits),
                    _ => return Err(JsonError::InvalidNumber),
                },
                NumberState::AfterMinus => match byte {
                    b'0' => Some(NumberState::Zero),
                    b'1'..=b'9' => Some(NumberState::IntDigits),
                    _ => return Err(JsonError::InvalidNumber),
                },
                NumberState::Zero => match byte {
                    b'0'..=b'9' => return Err(JsonError::InvalidNumber), // leading zero
                    b'.' => Some(NumberState::AfterDot),
                    b'e' | b'E' => Some(NumberState::AfterE),
                    _ => None, // terminator
                },
                NumberState::IntDigits => match byte {
                    b'0'..=b'9' => Some(NumberState::IntDigits),
                    b'.' => Some(NumberState::AfterDot),
                    b'e' | b'E' => Some(NumberState::AfterE),
                    _ => None, // terminator
                },
                NumberState::AfterDot => match byte {
                    b'0'..=b'9' => Some(NumberState::FracDigits),
                    _ => return Err(JsonError::InvalidNumber),
                },
                NumberState::FracDigits => match byte {
                    b'0'..=b'9' => Some(NumberState::FracDigits),
                    b'e' | b'E' => Some(NumberState::AfterE),
                    _ => None, // terminator
                },
                NumberState::AfterE => match byte {
                    b'+' | b'-' => Some(NumberState::AfterESign),
                    b'0'..=b'9' => Some(NumberState::ExpDigits),
                    _ => return Err(JsonError::InvalidNumber),
                },
                NumberState::AfterESign => match byte {
                    b'0'..=b'9' => Some(NumberState::ExpDigits),
                    _ => return Err(JsonError::InvalidNumber),
                },
                NumberState::ExpDigits => match byte {
                    b'0'..=b'9' => Some(NumberState::ExpDigits),
                    _ => None, // terminator
                },
                NumberState::Finished => unreachable!(),
            };

            match next_state {
                Some(ns) => {
                    cur_state = ns;
                    cur += 1;
                }
                None => {
                    // Terminating byte found — do NOT consume it.
                    if cur == 0 {
                        // Already in a terminal state; signal end immediately.
                        return Ok(None);
                    }
                    let chunk =
                        core::str::from_utf8(&src[..cur]).map_err(|_| JsonError::InvalidNumber)?;
                    *src = &src[cur..]; // terminator stays at src[0]
                    self.state = NumberState::Finished;
                    return Ok(Some(chunk));
                }
            }
        }

        // Buffer exhausted — cur == src.len(), cur > 0.
        let chunk = core::str::from_utf8(&src[..cur]).map_err(|_| JsonError::InvalidNumber)?;
        *src = &src[cur..];

        // Keep cur_state so the next buffer can continue the number.
        self.state = cur_state;

        Ok(Some(chunk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JsonError;

    /// Parse a complete in-memory number and return the single chunk.
    fn single_number_chunk(input: &[u8]) -> Result<&str, JsonError> {
        let mut src = input;
        let mut access = NumberAccess::start();
        let chunk = access.next_chunk(&mut src)?.expect("expected number chunk");
        assert_eq!(access.next_chunk(&mut src)?, None, "expected number end");
        Ok(chunk)
    }

    // --- valid integers ---

    #[test]
    fn integer_zero() {
        assert_eq!(single_number_chunk(b"0}").unwrap(), "0");
    }

    #[test]
    fn integer_negative_zero() {
        assert_eq!(single_number_chunk(b"-0}").unwrap(), "-0");
    }

    #[test]
    fn integer_positive() {
        assert_eq!(single_number_chunk(b"42,").unwrap(), "42");
    }

    #[test]
    fn integer_negative() {
        assert_eq!(single_number_chunk(b"-42]").unwrap(), "-42");
    }

    // --- valid fractions ---

    #[test]
    fn fraction_simple() {
        assert_eq!(single_number_chunk(b"3.14}").unwrap(), "3.14");
    }

    #[test]
    fn fraction_negative() {
        assert_eq!(single_number_chunk(b"-2.5,").unwrap(), "-2.5");
    }

    // --- valid exponents ---

    #[test]
    fn exponent_simple() {
        assert_eq!(single_number_chunk(b"1e10}").unwrap(), "1e10");
    }

    #[test]
    fn exponent_upper_e() {
        assert_eq!(single_number_chunk(b"1E10}").unwrap(), "1E10");
    }

    #[test]
    fn exponent_negative() {
        assert_eq!(single_number_chunk(b"2.5E-3,").unwrap(), "2.5E-3");
    }

    #[test]
    fn exponent_explicit_plus() {
        assert_eq!(single_number_chunk(b"-3e+2]").unwrap(), "-3e+2");
    }

    // --- grammar violations ---

    #[test]
    fn err_leading_zero() -> Result<(), JsonError> {
        let mut src: &[u8] = b"01";
        let mut access = NumberAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidNumber
        );
        Ok(())
    }

    #[test]
    fn err_dot_without_frac_digits() -> Result<(), JsonError> {
        let mut src: &[u8] = b"1.}";
        let mut access = NumberAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidNumber
        );
        Ok(())
    }

    #[test]
    fn err_e_without_digits() -> Result<(), JsonError> {
        let mut src: &[u8] = b"1e}";
        let mut access = NumberAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidNumber
        );
        Ok(())
    }

    #[test]
    fn err_e_sign_without_digits() -> Result<(), JsonError> {
        let mut src: &[u8] = b"1e+}";
        let mut access = NumberAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidNumber
        );
        Ok(())
    }

    #[test]
    fn err_double_minus() -> Result<(), JsonError> {
        let mut src: &[u8] = b"--3";
        let mut access = NumberAccess::start();
        assert_eq!(
            access.next_chunk(&mut src).unwrap_err(),
            JsonError::InvalidNumber
        );
        Ok(())
    }

    // --- streaming: number split across two buffers ---

    #[test]
    fn streaming_split_integer() -> Result<(), JsonError> {
        // "12" then "34}" — the number 1234
        let mut src1: &[u8] = b"12";
        let mut access = NumberAccess::start();
        let c1 = access.next_chunk(&mut src1)?.expect("chunk");

        let mut src2: &[u8] = b"34}";
        let c2 = access.next_chunk(&mut src2)?.expect("chunk");
        assert_eq!(access.next_chunk(&mut src2)?, None);

        // c1 + c2 must equal "1234"
        let mut buf = [0u8; 4];
        let b1 = c1.as_bytes();
        let b2 = c2.as_bytes();
        buf[..b1.len()].copy_from_slice(b1);
        buf[b1.len()..b1.len() + b2.len()].copy_from_slice(b2);
        assert_eq!(&buf[..b1.len() + b2.len()], b"1234");

        Ok(())
    }

    #[test]
    fn streaming_split_exponent() -> Result<(), JsonError> {
        // "1e" (buffer exhausted in AfterE) then "10}" — yields "1e10"
        let mut src1: &[u8] = b"1e";
        let mut access = NumberAccess::start();
        let c1 = access.next_chunk(&mut src1)?.expect("chunk");

        let mut src2: &[u8] = b"10}";
        let c2 = access.next_chunk(&mut src2)?.expect("chunk");
        assert_eq!(access.next_chunk(&mut src2)?, None);

        assert_eq!(c1, "1e");
        assert_eq!(c2, "10");

        Ok(())
    }

    #[test]
    fn streaming_truncated_after_minus_is_error() -> Result<(), JsonError> {
        let mut src1: &[u8] = b"-";
        let mut access = NumberAccess::start();
        // buffer exhausted after `-`, state=AfterMinus; returns the chunk
        assert_eq!(access.next_chunk(&mut src1)?, Some("-"));
        // empty buffer with non-terminal state → error
        let mut empty: &[u8] = b"";
        assert_eq!(
            access.next_chunk(&mut empty).unwrap_err(),
            JsonError::UnexpectedEnd
        );

        Ok(())
    }

    // --- number at exact buffer boundary (terminal state, no terminator byte) ---

    #[test]
    fn number_at_eof_terminal_state() -> Result<(), JsonError> {
        let mut src1: &[u8] = b"42";
        let mut access = NumberAccess::start();
        assert_eq!(access.next_chunk(&mut src1)?, Some("42"));
        let mut empty: &[u8] = b"";
        assert_eq!(access.next_chunk(&mut empty)?, None);
        Ok(())
    }
}
