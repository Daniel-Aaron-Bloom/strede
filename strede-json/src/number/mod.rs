//! `Number` types for JSON deserialization.
//!
//! Two distinct types because the borrow and owned families have fundamentally
//! different storage characteristics:
//!
//! - [`NumberBorrowed`] (borrow family): backed by [`N`] parsed from a
//!   zero-copy `&'de str`. With `arbitrary_precision`, stores `&'de str`
//!   directly. The `Deserialize` impl for `arbitrary_precision` lives in
//!   `full.rs` as a format-specific impl on `JsonSubDeserializer`.
//! - [`NumberOwned`] (owned family): backed by [`N`] computed via a streaming
//!   state machine across chunks (no intermediate buffer). With
//!   `arbitrary_precision` + `alloc`, stores `alloc::string::String`.

#[cfg(not(feature = "arbitrary_precision"))]
use core::marker::PhantomData;

#[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
use strede::Chunk;
#[cfg(not(feature = "arbitrary_precision"))]
use strede::borrow::{Deserialize, Deserializer, Entry};
use strede::owned::{DeserializeOwned, DeserializerOwned, EntryOwned, NumberAccessOwned};
#[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
use strede::select_probe;
use strede::{Probe, hit};

#[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
extern crate alloc;

#[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
mod decimal_seq;
#[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
mod parser;

// ===========================================================================
// Internal discriminator
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum N {
    PosInt(u64),
    NegInt(i64),
    Float(f64),
}

// ===========================================================================
// parse_n: borrow-side classifier for a complete number string.
// ===========================================================================

fn parse_n(s: &str) -> Option<N> {
    let bytes = s.as_bytes();
    let is_float = bytes.iter().any(|&b| b == b'.' || b == b'e' || b == b'E');
    if is_float {
        let f = s.parse::<f64>().ok()?;
        if !f.is_finite() {
            return None;
        }
        Some(N::Float(f))
    } else if bytes.first() == Some(&b'-') {
        Some(N::NegInt(s.parse::<i64>().ok()?))
    } else {
        Some(N::PosInt(s.parse::<u64>().ok()?))
    }
}

// ===========================================================================
// Public types
// ===========================================================================

/// A JSON number from the borrow family.
#[cfg(not(feature = "arbitrary_precision"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumberBorrowed<'de> {
    n: N,
    _de: PhantomData<&'de ()>,
}

#[cfg(feature = "arbitrary_precision")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumberBorrowed<'de> {
    pub(crate) raw: &'de str,
}

impl<'de> NumberBorrowed<'de> {
    /// Returns the number as a `u64` if it can be represented exactly.
    pub fn as_u64(&self) -> Option<u64> {
        #[cfg(not(feature = "arbitrary_precision"))]
        {
            n_as_u64(self.n)
        }
        #[cfg(feature = "arbitrary_precision")]
        {
            parse_n(self.raw).and_then(n_as_u64)
        }
    }

    /// Returns the number as an `i64` if it can be represented exactly.
    pub fn as_i64(&self) -> Option<i64> {
        #[cfg(not(feature = "arbitrary_precision"))]
        {
            n_as_i64(self.n)
        }
        #[cfg(feature = "arbitrary_precision")]
        {
            parse_n(self.raw).and_then(n_as_i64)
        }
    }

    /// Returns the number as an `f64` (lossy for integers >2^53).
    pub fn as_f64(&self) -> Option<f64> {
        #[cfg(not(feature = "arbitrary_precision"))]
        {
            Some(n_as_f64(self.n))
        }
        #[cfg(feature = "arbitrary_precision")]
        {
            parse_n(self.raw).map(n_as_f64)
        }
    }

    /// Returns the raw JSON number string.
    #[cfg(feature = "arbitrary_precision")]
    pub fn as_str(&self) -> &'de str {
        self.raw
    }
}

/// A JSON number from the owned (streaming) family.
#[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NumberOwned {
    n: N,
}

#[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumberOwned {
    raw: alloc::string::String,
}

impl NumberOwned {
    pub fn as_u64(&self) -> Option<u64> {
        #[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
        {
            n_as_u64(self.n)
        }
        #[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
        {
            parse_n(&self.raw).and_then(n_as_u64)
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        #[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
        {
            n_as_i64(self.n)
        }
        #[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
        {
            parse_n(&self.raw).and_then(n_as_i64)
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        #[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
        {
            Some(n_as_f64(self.n))
        }
        #[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
        {
            parse_n(&self.raw).map(n_as_f64)
        }
    }

    #[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

// Shared accessor logic on N
#[inline]
fn n_as_u64(n: N) -> Option<u64> {
    match n {
        N::PosInt(u) => Some(u),
        N::NegInt(_) => None,
        N::Float(f) => {
            if f >= 0.0 && f.is_finite() && f == f.trunc() && f < (u64::MAX as f64) + 1.0 {
                Some(f as u64)
            } else {
                None
            }
        }
    }
}

#[inline]
fn n_as_i64(n: N) -> Option<i64> {
    match n {
        N::PosInt(u) => i64::try_from(u).ok(),
        N::NegInt(i) => Some(i),
        N::Float(f) => {
            if f.is_finite() && f == f.trunc() && f >= (i64::MIN as f64) && f < -(i64::MIN as f64) {
                Some(f as i64)
            } else {
                None
            }
        }
    }
}

#[inline]
fn n_as_f64(n: N) -> f64 {
    match n {
        N::PosInt(u) => u as f64,
        N::NegInt(i) => i as f64,
        N::Float(f) => f,
    }
}

// ===========================================================================
// Deserialize / DeserializeOwned impls
// ===========================================================================

#[cfg(not(feature = "arbitrary_precision"))]
impl<'de, D: Deserializer<'de>> Deserialize<'de, D> for NumberBorrowed<'de> {
    type Extra = ();
    async fn deserialize(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        use strede::Chunk;
        use strede::borrow::NumberAccess;
        d.entry(|[e]| async {
            let chunks = hit!(e.deserialize_number_chunks().await);
            // For in-memory JSON the number is always a single chunk; parse it
            // directly. If a future borrow-family format streams numbers in
            // multiple chunks this will need the parser-state-machine treatment.
            match chunks.next_number_chunk(parse_n).await? {
                Chunk::Data((rest, Some(n))) => match rest.next_number_chunk(|_| ()).await? {
                    Chunk::Done(claim) => Ok(Probe::Hit((
                        claim,
                        NumberBorrowed {
                            n,
                            _de: PhantomData,
                        },
                    ))),
                    Chunk::Data(_) => Ok(Probe::Miss),
                },
                Chunk::Data((_, None)) => Ok(Probe::Miss),
                Chunk::Done(_) => Ok(Probe::Miss),
            }
        })
        .await
    }
}

impl<D: DeserializerOwned> DeserializeOwned<D> for NumberOwned {
    type Extra = ();
    async fn deserialize_owned(d: D, _: ()) -> Result<Probe<(D::Claim, Self)>, D::Error> {
        d.entry(|[e]| async {
            let chunks = hit!(e.deserialize_number_chunks().await);

            #[cfg(not(all(feature = "arbitrary_precision", feature = "alloc")))]
            {
                let mut c1 = chunks;
                let c2 = c1.fork();
                let c3 = c1.fork();
                let c4 = c1.fork();
                select_probe! {
                    async move {
                        let (claim, u) = hit!(parser::parse_int_pos(c1).await);
                        Ok(Probe::Hit((claim, NumberOwned { n: N::PosInt(u) })))
                    },
                    async move {
                        let (claim, i) = hit!(parser::parse_int_neg(c2).await);
                        Ok(Probe::Hit((claim, NumberOwned { n: N::NegInt(i) })))
                    },
                    async move {
                        let (claim, fp, neg) = hit!(parser::parse_float_fast(c3).await);
                        let f = parser::biased_fp_to_float(fp, neg);
                        if !f.is_finite() {
                            return Ok(Probe::Miss);
                        }
                        Ok(Probe::Hit((claim, NumberOwned { n: N::Float(f) })))
                    },
                    async move {
                        let (claim, fp, neg) = hit!(parser::parse_float_slow(c4).await);
                        let f = parser::biased_fp_to_float(fp, neg);
                        if !f.is_finite() {
                            return Ok(Probe::Miss);
                        }
                        Ok(Probe::Hit((claim, NumberOwned { n: N::Float(f) })))
                    },
                }
            }
            #[cfg(all(feature = "arbitrary_precision", feature = "alloc"))]
            {
                let mut chunks = chunks;
                let mut raw = alloc::string::String::new();
                let claim = loop {
                    match chunks.next_number_chunk(|s| raw.push_str(s)).await? {
                        Chunk::Data((next, ())) => chunks = next,
                        Chunk::Done(claim) => break claim,
                    }
                };
                Ok(Probe::Hit((claim, NumberOwned { raw })))
            }
        })
        .await
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(all(test, not(all(feature = "arbitrary_precision", feature = "alloc"))))]
mod tests {
    use super::*;
    extern crate std;
    use strede_test_util::block_on;

    use crate::full::JsonDeserializer;

    fn deser_borrowed(input: &str) -> Result<NumberBorrowed<'_>, crate::JsonError> {
        let d = JsonDeserializer::new(input.as_bytes());
        let res = block_on(<NumberBorrowed as strede::Deserialize<'_, _>>::deserialize(
            d,
            (),
        ));
        match res {
            Ok(Probe::Hit((_, n))) => Ok(n),
            Ok(Probe::Miss) => Err(crate::JsonError::InvalidNumber),
            Err(e) => Err(e),
        }
    }

    #[test]
    fn parse_n_u64_basic() {
        assert_eq!(parse_n("0"), Some(N::PosInt(0)));
        assert_eq!(parse_n("42"), Some(N::PosInt(42)));
        assert_eq!(parse_n("18446744073709551615"), Some(N::PosInt(u64::MAX)));
    }

    #[test]
    fn parse_n_u64_overflow() {
        assert_eq!(parse_n("18446744073709551616"), None);
        assert_eq!(parse_n("99999999999999999999"), None);
    }

    #[test]
    fn parse_n_i64_negative() {
        assert_eq!(parse_n("-1"), Some(N::NegInt(-1)));
        assert_eq!(parse_n("-7"), Some(N::NegInt(-7)));
        assert_eq!(parse_n("-9223372036854775808"), Some(N::NegInt(i64::MIN)));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn parse_n_float_basic() {
        match parse_n("3.14").unwrap() {
            N::Float(f) => assert!((f - 3.14).abs() < 1e-12),
            _ => panic!(),
        }
        match parse_n("1.0").unwrap() {
            N::Float(f) => assert_eq!(f, 1.0),
            _ => panic!(),
        }
        match parse_n("0.0").unwrap() {
            N::Float(f) => assert_eq!(f, 0.0),
            _ => panic!(),
        }
    }

    #[test]
    fn parse_n_float_exponent() {
        match parse_n("1e10").unwrap() {
            N::Float(f) => assert_eq!(f, 1e10),
            _ => panic!(),
        }
        match parse_n("1.5e-3").unwrap() {
            N::Float(f) => assert!((f - 0.0015).abs() < 1e-15),
            _ => panic!(),
        }
        match parse_n("-2.5E2").unwrap() {
            N::Float(f) => assert_eq!(f, -250.0),
            _ => panic!(),
        }
    }

    #[test]
    fn parse_n_many_digits() {
        // 1.0000000000000000000000000000 should still be 1.0
        match parse_n("1.0000000000000000000000000000").unwrap() {
            N::Float(f) => assert_eq!(f, 1.0),
            _ => panic!(),
        }
    }

    #[test]
    fn deserialize_borrowed_int() {
        let n = deser_borrowed("42").unwrap();
        assert_eq!(n.as_u64(), Some(42));
        assert_eq!(n.as_i64(), Some(42));
        assert_eq!(n.as_f64(), Some(42.0));
    }

    #[test]
    fn deserialize_borrowed_neg() {
        let n = deser_borrowed("-7").unwrap();
        assert_eq!(n.as_u64(), None);
        assert_eq!(n.as_i64(), Some(-7));
        assert_eq!(n.as_f64(), Some(-7.0));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn deserialize_borrowed_float() {
        let n = deser_borrowed("3.14").unwrap();
        assert_eq!(n.as_u64(), None);
        assert_eq!(n.as_i64(), None);
        assert!((n.as_f64().unwrap() - 3.14).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn parse_n_lemire_hard() {
        // A few values that exercise the Eisel-Lemire path beyond the fast path.
        let cases = [
            ("1.7976931348623157e308", f64::MAX),
            ("2.2250738585072014e-308", f64::MIN_POSITIVE),
            ("5e-324", 5e-324_f64),
            ("1e23", 1e23_f64),
            ("9.999999999999999e22", 9.999999999999999e22_f64),
        ];
        for (s, expected) in cases {
            match parse_n(s).unwrap() {
                N::Float(f) => assert_eq!(f, expected, "parsing {s}"),
                _ => panic!("expected float for {s}"),
            }
        }
    }

    #[test]
    fn streaming_owned_int() {
        use crate::chunked::ChunkedJsonDeserializer;
        use strede::shared_buf::SharedBuf;
        use strede_test_util::block_on;

        let v = block_on(SharedBuf::with_async(
            b"42" as &[u8],
            async move |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                let (_, n) = NumberOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap();
                n
            },
        ));
        assert_eq!(v.as_u64(), Some(42));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn streaming_owned_float() {
        use crate::chunked::ChunkedJsonDeserializer;
        use strede::shared_buf::SharedBuf;
        use strede_test_util::block_on;

        let v = block_on(SharedBuf::with_async(
            b"3.14" as &[u8],
            async move |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                let (_, n) = NumberOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap();
                n
            },
        ));
        assert!((v.as_f64().unwrap() - 3.14).abs() < 1e-12);
    }

    #[test]
    fn streaming_owned_negative() {
        use crate::chunked::ChunkedJsonDeserializer;
        use strede::shared_buf::SharedBuf;
        use strede_test_util::block_on;

        let v = block_on(SharedBuf::with_async(
            b"-9223372036854775808" as &[u8],
            async move |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                let (_, n) = NumberOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap();
                n
            },
        ));
        assert_eq!(v.as_i64(), Some(i64::MIN));
    }

    #[test]
    fn streaming_owned_many_digits() {
        // 1.0000000000000000000000000000 spans buffers but should still equal 1.0
        use crate::chunked::ChunkedJsonDeserializer;
        use strede::shared_buf::SharedBuf;
        use strede_test_util::block_on;

        let v = block_on(SharedBuf::with_async(
            b"1.0000000000000000000000000000" as &[u8],
            async move |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                let (_, n) = NumberOwned::deserialize_owned(de, ())
                    .await
                    .unwrap()
                    .unwrap();
                n
            },
        ));
        assert_eq!(v.as_f64(), Some(1.0));
    }

    // -----------------------------------------------------------------------
    // Fuzz helpers
    // -----------------------------------------------------------------------

    /// Minimal LCG for reproducible pseudo-random inputs without pulling in
    /// external crates. Constants from Knuth (MMIX).
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }

        fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + self.next() % (hi - lo)
        }
    }

    /// Drive `NumberOwned::deserialize_owned` against a `ChunkedJsonDeserializer`
    /// fed from a static byte slice. Returns `None` on `Probe::Miss`.
    fn streaming_parse(s: &str) -> Option<N> {
        use crate::chunked::ChunkedJsonDeserializer;
        use strede::shared_buf::SharedBuf;

        let bytes = s.as_bytes().to_owned();
        block_on(SharedBuf::with_async(
            bytes.as_slice(),
            async move |buf: &mut &[u8]| {
                *buf = &[];
            },
            async |shared| {
                let de = ChunkedJsonDeserializer::new(shared);
                match NumberOwned::deserialize_owned(de, ()).await.unwrap() {
                    Probe::Hit((_, n)) => Some(n.n),
                    Probe::Miss => None,
                }
            },
        ))
    }

    /// Generate a random JSON number string with a mix of shapes:
    /// plain integers, negatives, floats, and floats with large exponents.
    fn random_number(rng: &mut Lcg) -> std::string::String {
        let neg = rng.next().is_multiple_of(4);
        let big = rng.next().is_multiple_of(3);

        let int_digits = rng.next_range(1, if big { 25 } else { 10 });
        let mut int_part = std::string::String::new();
        for i in 0..int_digits {
            let d = if i == 0 {
                rng.next_range(1, 10)
            } else {
                rng.next_range(0, 10)
            };
            int_part.push((b'0' + d as u8) as char);
        }

        let frac = if rng.next().is_multiple_of(2) {
            let frac_digits = rng.next_range(1, if big { 35 } else { 10 });
            let mut s = std::string::String::from(".");
            for _ in 0..frac_digits {
                s.push((b'0' + rng.next_range(0, 10) as u8) as char);
            }
            s
        } else {
            std::string::String::new()
        };

        let exp = if rng.next().is_multiple_of(2) {
            let limit = if big { 400 } else { 30 };
            let e = rng.next_range(0, limit * 2) as i32 - limit as i32;
            std::format!("e{e}")
        } else {
            std::string::String::new()
        };

        let sign = if neg { "-" } else { "" };
        std::format!("{sign}{int_part}{frac}{exp}")
    }

    // -----------------------------------------------------------------------
    // Fuzz test: streaming path vs. parse_n reference
    // -----------------------------------------------------------------------

    #[test]
    fn fuzz_streaming_vs_parse_n_vs_str_parse() {
        let mut rng = Lcg(0xdeadbeef_cafef00d);

        let mut mismatches = 0u32;
        let mut failures: Vec<std::string::String> = std::vec::Vec::new();
        let iterations = 100_000;

        for _ in 0..iterations {
            let s = random_number(&mut rng);

            let from_parse_n = parse_n(&s);
            let from_streaming = streaming_parse(&s);

            if from_parse_n.is_some() != from_streaming.is_some() {
                eprintln!("\n=== success mismatch: {s:?} ===");
                eprintln!("  parse_n={from_parse_n:?} streaming={from_streaming:?}");
            }
            assert_eq!(
                from_parse_n.is_some(),
                from_streaming.is_some(),
                "success mismatch for {s:?}: parse_n={from_parse_n:?} streaming={from_streaming:?}",
            );

            let Some(pn) = from_parse_n else { continue };
            let Some(st) = from_streaming else { continue };

            let is_float_pn = matches!(pn, N::Float(_));
            let is_float_st = matches!(st, N::Float(_));
            assert_eq!(
                is_float_pn, is_float_st,
                "variant mismatch for {s:?}: parse_n={pn:?} streaming={st:?}",
            );

            match (pn, st) {
                (N::Float(a), N::Float(b)) => {
                    if a.to_bits() != b.to_bits() {
                        eprintln!("\n=== internal mismatch: {s:?}: parse_n={a} streaming={b} ===",);
                    }
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "float bit mismatch for {s:?}: parse_n={a} streaming={b}",
                    );

                    let oracle: f64 = s.parse().unwrap_or(f64::NAN);
                    if oracle.is_finite() && a.to_bits() != oracle.to_bits() {
                        mismatches += 1;
                        failures.push(s.clone());
                        eprintln!(
                            "oracle mismatch for {s:?}: ours={a} ({:#018x}) oracle={oracle} ({:#018x})",
                            a.to_bits(),
                            oracle.to_bits(),
                        );
                    }
                }
                (N::PosInt(a), N::PosInt(b)) => assert_eq!(a, b, "u64 mismatch for {s:?}"),
                (N::NegInt(a), N::NegInt(b)) => assert_eq!(a, b, "i64 mismatch for {s:?}"),
                _ => panic!("variant mismatch (should have been caught above) for {s:?}"),
            }
        }

        if !failures.is_empty() {
            eprintln!(
                "\n=== {} failing cases (paste into targeted regression test) ===",
                failures.len(),
            );
            for f in &failures {
                eprintln!("    {f:?},");
            }
        }
        assert_eq!(
            mismatches, 0,
            "{mismatches} oracle mismatches found over {iterations} random inputs (see stderr for details)",
        );
    }

    // -----------------------------------------------------------------------
    // Targeted regression cases
    // -----------------------------------------------------------------------

    #[test]
    fn many_digits_targeted() {
        let cases: &[&str] = &[
            // 20+ integer digits
            "12345678901234567890",
            "99999999999999999999",
            // Many integer digits + fraction
            "1234567890123456789.5",
            "12345678901234567890.0",
            // Leading fractional zeros
            "0.00000000000000000001",
            "0.000000000000000000012345678901234567890",
            // Large positive exponent
            "1.23456789012345678901e100",
            "9.9999999999999999999e307",
            // Large negative exponent
            "1.23456789012345678901e-100",
            "1.0000000000000000000000000001e-300",
            // Negative with many digits
            "-1234567890123456789012345.0",
            "-0.000000000000000000012345678901234567890",
            // Near subnormal boundary
            "5e-324",
            "1e-323",
            // Near f64::MAX
            "1.7976931348623157e308",
        ];

        for s in cases {
            let from_parse_n = parse_n(s);
            let from_streaming = streaming_parse(s);

            assert_eq!(
                from_parse_n.is_some(),
                from_streaming.is_some(),
                "success mismatch for {s:?}",
            );

            match (from_parse_n, from_streaming) {
                (Some(N::Float(a)), Some(N::Float(b))) => {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "float mismatch for {s:?}: parse_n={a} streaming={b}",
                    );
                    let oracle: f64 = s.parse().unwrap_or(f64::NAN);
                    if oracle.is_finite() {
                        assert_eq!(
                            a.to_bits(),
                            oracle.to_bits(),
                            "oracle mismatch for {s:?}: ours={a} oracle={oracle}",
                        );
                    }
                }
                (Some(N::PosInt(a)), Some(N::PosInt(b))) => {
                    assert_eq!(a, b, "u64 mismatch for {s:?}")
                }
                (Some(N::NegInt(a)), Some(N::NegInt(b))) => {
                    assert_eq!(a, b, "i64 mismatch for {s:?}")
                }
                (None, None) => {}
                other => panic!("variant mismatch for {s:?}: {other:?}"),
            }
        }
    }
}
