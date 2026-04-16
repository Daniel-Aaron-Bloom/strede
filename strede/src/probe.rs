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

    /// Map the inner value of a `Hit`, leaving `Miss` unchanged.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Probe<U> {
        match self {
            Probe::Hit(v) => Probe::Hit(f(v)),
            Probe::Miss => Probe::Miss,
        }
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
            $crate::Probe::Hit(v) => v,
            $crate::Probe::Miss => return ::core::result::Result::Ok($crate::Probe::Miss),
        }
    };
}

/// Unwraps an `Option`, returning `Ok(Probe::Miss)` from the enclosing
/// function on `None`.
///
/// # Example
/// ```rust,ignore
/// let v = or_miss!(some_option);
/// ```
#[macro_export]
macro_rules! or_miss {
    ($e:expr) => {
        match $e {
            ::core::option::Option::Some(v) => v,
            ::core::option::Option::None => return ::core::result::Result::Ok($crate::Probe::Miss),
        }
    };
}

/// Race multiple probe futures, returning the first `Hit`.
///
/// Each arm is an expression that evaluates to a future returning
/// `Result<Probe<T>, E>`. Arms can be bare probe calls
/// (`e.deserialize_i64()`) or `async move { … }` blocks for logic that
/// needs `.await` or value transformation.
///
/// ```rust,ignore
/// let val = select_probe! {
///     e1.deserialize_bool(),
///     async move { hit!(e2.deserialize_i64().await); Ok(Probe::Hit((c, v as bool))) },
///     miss => Ok(Probe::Miss),
/// }
/// .await?;
/// ```
///
/// # Behavior
///
/// - Arms are polled in declaration order on every wake-up.
/// - `Pending` means "waiting for I/O" — the arm stays live.
/// - `Ok(Probe::Miss)` marks an arm done; it is skipped on future polls.
/// - `Ok(Probe::Hit(_))` — first hit in declaration order wins.
/// - `Err(e)` short-circuits immediately.
/// - When all arms have returned `Miss`, the optional `miss => body` arm fires;
///   without it the macro produces `Ok(Probe::Miss)`.
///
/// # Killing sibling arms
///
/// A `kill!(i)` macro is in scope inside every arm expression. Calling it
/// schedules arm `i` for cancellation: its future is dropped in place on the
/// next poll, before any arm is polled that iteration. This is useful when one
/// arm has already seen enough to know a sibling can never win.
///
/// There is no point calling `kill!(i)` immediately before returning — the
/// sibling only dies on the *next* poll, which never comes once this arm
/// returns a `Hit`. `kill!` is only meaningful when the calling arm still has
/// work to do (e.g. an `.await` point) after ruling out the sibling.
///
/// ```rust,ignore
/// select_probe! {
///     async move {
///         let (c, s) = hit!(e1.deserialize_str().await);
///         kill!(1); // chunks fallback can't win now; drop it before we await again
///         let owned = some_async_transform(s).await;
///         Ok(Probe::Hit((c, owned)))
///     },
///     async move { /* chunks fallback */ },
/// }
/// ```
#[macro_export]
macro_rules! select_probe {
    ($($tt:tt)*) => {
        $crate::select_probe_inner!(@$crate $($tt)*)
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

impl<Data, Done> Chunk<Data, Done> {
    pub fn data(self) -> Option<Data> {
        match self {
            Self::Data(d) => Some(d),
            _ => None,
        }
    }
    pub fn done(self) -> Option<Done> {
        match self {
            Self::Done(d) => Some(d),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use core::future::Future;
    use core::pin::Pin;
    use core::sync::atomic::{AtomicBool, Ordering};
    use core::task::{Context, Poll};

    fn poll_once<F: Future>(f: Pin<&mut F>) -> Poll<F::Output> {
        let w = strede_test_util::noop_waker();
        let mut cx = Context::from_waker(&w);
        f.poll(&mut cx)
    }

    #[test]
    fn kill_drops_future() {
        static DROPPED: AtomicBool = AtomicBool::new(false);

        struct SetOnDrop;
        impl Drop for SetOnDrop {
            fn drop(&mut self) {
                DROPPED.store(true, Ordering::SeqCst);
            }
        }

        let _guard = SetOnDrop;
        // arm 0: kills arm 1, returns Miss
        // arm 1: holds a SetOnDrop; once killed its future should be dropped
        // arm 2: checks DROPPED is true, returns Hit
        let fut = async {
            crate::select_probe! {
                async move {
                    kill!(1);
                    Ok(Probe::Miss)
                },
                async move {
                    let _guard = _guard;
                    core::future::pending::<Result<Probe<u32>, ()>>().await
                },
                async move {
                    assert!(DROPPED.load(Ordering::SeqCst), "arm 1 was not dropped");
                    Ok(Probe::Hit(42u32))
                },
            }
        };
        let mut fut = core::pin::pin!(fut);
        // Two polls: first lets all arms run once (arm 0 kills arm 1, arm 2 sees the drop)
        loop {
            match poll_once(fut.as_mut()) {
                Poll::Ready(result) => {
                    assert_eq!(result, Ok(Probe::Hit(42u32)));
                    break;
                }
                Poll::Pending => {}
            }
        }
    }
}
