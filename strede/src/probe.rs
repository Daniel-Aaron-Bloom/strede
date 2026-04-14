/// Result of a type-dispatch probe.
///
/// Returned inside `Result<Probe<T>, E>` by all [`Entry`](crate::Entry) probe methods and
/// [`Deserialize::deserialize`](crate::Deserialize::deserialize).
///
/// - `Hit(value)` — token matched; `value` is `(Claim, T)` or an access type.
/// - `Miss` — token did not match this probe's expected type; stream was **not**
///   consumed. Other probes racing via [`select_probe!`](crate::select_probe) can still win.
///
/// Fatal format errors are the outer `Err(e)` and propagate normally with `?`.
/// `Pending` on a probe future means *only* "no data available yet."
#[derive(Debug, PartialEq)]
pub enum Probe<T> {
    Hit(T),
    Miss,
}

impl<T> Probe<T> {
    /// Unwrap a `Hit`, panicking on `Miss`.
    pub fn unwrap(self) -> T {
        match self {
            Probe::Hit(v) => v,
            Probe::Miss => panic!("called Probe::unwrap() on a Miss value"),
        }
    }

    /// Convert to `Result`, treating `Miss` as an error via the provided closure.
    pub fn require<E>(self, on_miss: impl FnOnce() -> E) -> Result<T, E> {
        match self {
            Probe::Hit(v) => Ok(v),
            Probe::Miss => Err(on_miss()),
        }
    }

    pub fn is_miss(&self) -> bool {
        matches!(self, Probe::Miss)
    }
    pub fn is_hit(&self) -> bool {
        matches!(self, Probe::Hit(_))
    }
}

/// Propagates `Err` and `Ok(Probe::Miss)` from the enclosing function;
/// unwraps `Ok(Probe::Hit(v))` to `v`.
///
/// The enclosing function must return `Result<Probe<_>, E>`.
///
/// # Example
/// ```rust,ignore
/// let mut map = hit!(e.deserialize_map().await);
/// let (claim, val) = hit!(e.deserialize_bool().await);
/// ```
#[macro_export]
macro_rules! hit {
    ($e:expr) => {
        match $e? {
            ::strede::Probe::Hit(v) => v,
            ::strede::Probe::Miss => return ::core::result::Result::Ok(::strede::Probe::Miss),
        }
    };
}

/// Yielded by streaming accessors ([`StrAccess::next`](crate::StrAccess::next),
/// [`BytesAccess::next`](crate::BytesAccess::next),
/// [`MapAccess::next`](crate::MapAccess::next),
/// [`SeqAccess::next`](crate::SeqAccess::next)).
///
/// - `Data(item)` — another item of data.
/// - `Done(claim)` — the stream is exhausted; thread `claim` back to the
///   outer [`Deserializer::next`](crate::Deserializer::next) as proof-of-consumption.
pub enum Chunk<Data, Done> {
    Data(Data),
    Done(Done),
}
