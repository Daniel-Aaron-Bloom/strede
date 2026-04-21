use core::{
    cell::Cell,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use pin_project::pin_project;

/// Result of a type-dispatch probe.
///
/// Returned inside `Result<Probe<T>, E>` by all [`Entry`](crate::Entry) probe methods and
/// [`Deserialize::deserialize`](crate::Deserialize::deserialize).
///
/// - `Hit(value)` - token matched; `value` is `(Claim, T)` or an access type.
/// - `Miss` - token did not match this probe's expected type; stream was **not**
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
/// (`e.deserialize_i64()`) or `async move { ... }` blocks for logic that
/// needs `.await` or value transformation.
///
/// ```rust,ignore
/// let val = select_probe! {
///     e1.deserialize_bool(),
///     async move { hit!(e2.deserialize_i64().await); Ok(Probe::Hit((c, v as bool))) },
///     @miss => Ok(Probe::Miss),
/// }
/// .await?;
/// ```
///
/// # Behavior
///
/// - Arms are polled in declaration order on every wake-up.
/// - `Pending` means "waiting for I/O" - the arm stays live.
/// - `Ok(Probe::Miss)` marks an arm done; it is skipped on future polls.
/// - `Ok(Probe::Hit(_))` - first hit in declaration order wins.
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
/// There is no point calling `kill!(i)` immediately before returning - the
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
/// Builds the arm list via `+` on `SelectProbeBase`.
/// `__select_probe_arms!(a, b, c)` expands to
/// `SelectProbeBase + ProbeArm(a) + ProbeArm(b) + ProbeArm(c)`.
#[doc(hidden)]
#[macro_export]
macro_rules! __select_probe_arms {
    ($($arm:expr),+) => {
        $crate::probe::SelectProbeBase $(+ $crate::probe::ProbeArm($arm))+
    };
}

#[macro_export]
macro_rules! select_probe {
    (biased; $($arm:expr),* $(, @miss => $miss:expr)? $(,)?) => {$crate::__select_probe_inner!(true; $($arm),* $(, @miss => $miss)?)};
    ($($arm:expr),* $(, @miss => $miss:expr)? $(,)?) => {$crate::__select_probe_inner!(false; $($arm),* $(, @miss => $miss)?)};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __select_probe_inner {
    ($bias:literal; $($arm:expr),* $(, @miss => $miss:expr)?) => {{
        #[allow(unused_imports)]
        use $crate::probe::MissFallback as _;
        let __probe_kills = <$crate::probe::KillManager<_>>::new();
        let __probe_kills = &__probe_kills;
        #[allow(unused_macros)]
        macro_rules! kill {
            ($i:literal) => { __probe_kills.mark($i) };
        }
        let _miss = ::core::result::Result::Ok($crate::Probe::Miss);
        let _miss2: *const _ = &_miss;
        $(let _miss = $miss;)?
        let miss = || $crate::probe::MissWrapper(_miss).into_future();
        $crate::probe::align_types(_miss2, &miss);
        let __probe_futs = $crate::__select_probe_arms!($($arm),+);
        $crate::probe::select_probe($bias, __probe_kills, __probe_futs, miss).await
    }};
}

#[doc(hidden)]
pub struct MissWrapper<T>(pub T);

// Inherent method for Future types (takes priority)
impl<T: Future> MissWrapper<T> {
    pub fn into_future(self) -> T {
        self.0
    }
}

/// Trait method for non-Future types (fallback)
#[doc(hidden)]
pub trait MissFallback {
    type Fut: Future;
    fn into_future(self) -> Self::Fut;
}
impl<T, E> MissFallback for MissWrapper<Result<Probe<T>, E>> {
    type Fut = core::future::Ready<Result<Probe<T>, E>>;
    fn into_future(self) -> Self::Fut {
        core::future::ready(self.0)
    }
}

#[doc(hidden)]
pub trait SelectProbeKillFlag {
    const SIZE: usize;
    fn new() -> Self;
    fn is_marked(&self, i: usize) -> bool;
    fn mark(&self, i: usize);
}

impl SelectProbeKillFlag for () {
    const SIZE: usize = 0;
    fn new() -> Self {}
    fn is_marked(&self, _i: usize) -> bool {
        unreachable!()
    }
    fn mark(&self, _i: usize) {
        unreachable!()
    }
}

impl<T: SelectProbeKillFlag> SelectProbeKillFlag for (T, Cell<bool>) {
    const SIZE: usize = T::SIZE + 1;
    fn new() -> Self {
        (T::new(), Cell::new(false))
    }
    fn is_marked(&self, i: usize) -> bool {
        debug_assert!(i < Self::SIZE);
        if i == Self::SIZE - 1 {
            self.1.get()
        } else {
            self.0.is_marked(i)
        }
    }
    fn mark(&self, i: usize) {
        debug_assert!(i < Self::SIZE);
        if i == Self::SIZE - 1 {
            self.1.set(true);
        } else {
            self.0.mark(i)
        }
    }
}

#[doc(hidden)]
pub struct KillManager<T: SelectProbeKillFlag> {
    flags: T,
    new_kills: Cell<bool>,
}

impl<T: SelectProbeKillFlag> Default for KillManager<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SelectProbeKillFlag> KillManager<T> {
    pub fn new() -> Self {
        Self {
            flags: T::new(),
            new_kills: Cell::new(false),
        }
    }
    pub fn mark(&self, i: usize) {
        self.flags.mark(i);
        self.new_kills.set(true);
    }
    fn take_new_kills(&self) -> bool {
        self.new_kills.replace(false)
    }
}

/// Wrapper that marks a future as one arm of a [`select_probe!`](crate::select_probe) call.
///
/// Used with `+` on [`SelectProbeBase`] to build the arm list without recursive macros:
/// `SelectProbeBase + ProbeArm(f1) + ProbeArm(f2) + ...`
#[doc(hidden)]
pub struct ProbeArm<F>(pub F);

/// Marker type used as the seed of the `ProbeArm` addition chain.
#[doc(hidden)]
pub struct SelectProbeBase;

// SelectProbeBase + ProbeArm(f) → SP1
impl<F> core::ops::Add<ProbeArm<F>> for SelectProbeBase {
    type Output = SP1<F>;
    fn add(self, rhs: ProbeArm<F>) -> SP1<F> {
        SP1(Some(rhs.0))
    }
}

/// Flat accumulator: 1 arm.
#[doc(hidden)]
#[pin_project]
pub struct SP1<A>(#[pin] Option<A>);

/// Flat accumulator: 2 arms.
#[doc(hidden)]
#[pin_project]
pub struct SP2<A, B>(#[pin] Option<A>, #[pin] Option<B>);

/// Flat accumulator: 3 arms.
#[doc(hidden)]
#[pin_project]
pub struct SP3<A, B, C>(#[pin] Option<A>, #[pin] Option<B>, #[pin] Option<C>);

/// Flat accumulator: 4 arms.
#[doc(hidden)]
#[pin_project]
pub struct SP4<A, B, C, D>(
    #[pin] Option<A>,
    #[pin] Option<B>,
    #[pin] Option<C>,
    #[pin] Option<D>,
);

// SP1 + ProbeArm → SP2
impl<A, B> core::ops::Add<ProbeArm<B>> for SP1<A> {
    type Output = SP2<A, B>;
    fn add(self, rhs: ProbeArm<B>) -> SP2<A, B> {
        SP2(self.0, Some(rhs.0))
    }
}

// SP2 + ProbeArm → SP3
impl<A, B, C> core::ops::Add<ProbeArm<C>> for SP2<A, B> {
    type Output = SP3<A, B, C>;
    fn add(self, rhs: ProbeArm<C>) -> SP3<A, B, C> {
        SP3(self.0, self.1, Some(rhs.0))
    }
}

// SP3 + ProbeArm → SP4
impl<A, B, C, D> core::ops::Add<ProbeArm<D>> for SP3<A, B, C> {
    type Output = SP4<A, B, C, D>;
    fn add(self, rhs: ProbeArm<D>) -> SP4<A, B, C, D> {
        SP4(self.0, self.1, self.2, Some(rhs.0))
    }
}

// SP4 + ProbeArm → overflow into left-nested SelectProbeSlot chain
impl<A, B, C, D, E> core::ops::Add<ProbeArm<E>> for SP4<A, B, C, D> {
    type Output = SelectProbeSlot<SP4<A, B, C, D>, E>;
    fn add(self, rhs: ProbeArm<E>) -> SelectProbeSlot<SP4<A, B, C, D>, E> {
        SelectProbeSlot {
            rest: self,
            fut: Some(rhs.0),
        }
    }
}

// SelectProbeSlot<L, F> + ProbeArm → keep left-nesting
impl<L, F, G> core::ops::Add<ProbeArm<G>> for SelectProbeSlot<L, F> {
    type Output = SelectProbeSlot<SelectProbeSlot<L, F>, G>;
    fn add(self, rhs: ProbeArm<G>) -> SelectProbeSlot<SelectProbeSlot<L, F>, G> {
        SelectProbeSlot {
            rest: self,
            fut: Some(rhs.0),
        }
    }
}

/// One slot in the overflow left-nested chain (5+ arms after the flat SP4 base).
#[doc(hidden)]
#[pin_project]
pub struct SelectProbeSlot<Rest, F> {
    #[pin]
    rest: Rest,
    #[pin]
    fut: Option<F>,
}

/// Identity conversion - SP1–SP4 and SelectProbeSlot are already in slot form.
#[doc(hidden)]
pub trait IntoSelectProbeSlots {
    type Slots;
    fn into_slots(self) -> Self::Slots;
}

impl<A> IntoSelectProbeSlots for SP1<A> {
    type Slots = Self;
    fn into_slots(self) -> Self {
        self
    }
}
impl<A, B> IntoSelectProbeSlots for SP2<A, B> {
    type Slots = Self;
    fn into_slots(self) -> Self {
        self
    }
}
impl<A, B, C> IntoSelectProbeSlots for SP3<A, B, C> {
    type Slots = Self;
    fn into_slots(self) -> Self {
        self
    }
}
impl<A, B, C, D> IntoSelectProbeSlots for SP4<A, B, C, D> {
    type Slots = Self;
    fn into_slots(self) -> Self {
        self
    }
}
impl<L: IntoSelectProbeSlots, F> IntoSelectProbeSlots for SelectProbeSlot<L, F> {
    type Slots = SelectProbeSlot<L::Slots, F>;
    fn into_slots(self) -> Self::Slots {
        SelectProbeSlot {
            rest: self.rest.into_slots(),
            fut: self.fut,
        }
    }
}

#[doc(hidden)]
pub trait SelectProbeFutures<T, E> {
    const SIZE: usize;
    type KF: SelectProbeKillFlag;
    type DF: SelectProbeDoneFlag;
    fn process_kills(self: Pin<&mut Self>, kill_flags: &Self::KF, done: &mut Self::DF);
    fn poll_one(
        self: Pin<&mut Self>,
        i: usize,
        done: &mut Self::DF,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<T>, E>>;
}

// Helper: poll a pinned Option<F> slot, updating the done flag.
// Returns Poll::Ready(Some(result)) on hit/err, Poll::Pending otherwise.
// Sets *done = true on Miss.
#[inline(always)]
fn poll_slot<T, E, F: Future<Output = Result<Probe<T>, E>>>(
    slot: Pin<&mut Option<F>>,
    done: &mut bool,
    cx: &mut Context<'_>,
) -> Poll<Option<Result<Probe<T>, E>>> {
    if *done {
        return Poll::Pending;
    }
    match slot.as_pin_mut() {
        None => {
            *done = true;
            Poll::Pending
        }
        Some(fut) => match Future::poll(fut, cx) {
            Poll::Ready(Ok(Probe::Hit(v))) => Poll::Ready(Some(Ok(Probe::Hit(v)))),
            Poll::Ready(Ok(Probe::Miss)) => {
                *done = true;
                Poll::Pending
            }
            Poll::Ready(Err(e)) => Poll::Ready(Some(Err(e))),
            Poll::Pending => Poll::Pending,
        },
    }
}

/// Kill flags for flat SP1–SP4: one `Cell<bool>` per arm.
#[doc(hidden)]
pub struct FlatKillFlags<const N: usize>([Cell<bool>; N]);
impl<const N: usize> FlatKillFlags<N> {
    fn new() -> Self {
        Self(core::array::from_fn(|_| Cell::new(false)))
    }
}
impl<const N: usize> SelectProbeKillFlag for FlatKillFlags<N> {
    const SIZE: usize = N;
    fn new() -> Self {
        Self::new()
    }
    fn is_marked(&self, i: usize) -> bool {
        self.0[i].get()
    }
    fn mark(&self, i: usize) {
        self.0[i].set(true);
    }
}

/// Done flags for flat SP1–SP4: one `bool` per arm.
#[doc(hidden)]
pub struct FlatDoneFlags<const N: usize>([bool; N]);
impl<const N: usize> FlatDoneFlags<N> {
    fn new() -> Self {
        Self([false; N])
    }
}
impl<const N: usize> SelectProbeDoneFlag for FlatDoneFlags<N> {
    const SIZE: usize = N;
    fn new() -> Self {
        Self::new()
    }
    fn all_done(&self) -> bool {
        self.0.iter().all(|&d| d)
    }
}

// Flat SP1 impl
impl<T, E, A: Future<Output = Result<Probe<T>, E>>> SelectProbeFutures<T, E> for SP1<A> {
    const SIZE: usize = 1;
    type KF = FlatKillFlags<1>;
    type DF = FlatDoneFlags<1>;
    fn process_kills(self: Pin<&mut Self>, kill_flags: &Self::KF, done: &mut Self::DF) {
        if !done.0[0] && kill_flags.0[0].get() {
            self.project().0.set(None);
            done.0[0] = true;
        }
    }
    fn poll_one(
        self: Pin<&mut Self>,
        i: usize,
        done: &mut Self::DF,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<T>, E>> {
        debug_assert_eq!(i, 0);
        match poll_slot(self.project().0, &mut done.0[0], cx) {
            Poll::Ready(Some(r)) => Poll::Ready(r),
            _ => Poll::Pending,
        }
    }
}

// Flat SP2 impl
impl<T, E, A: Future<Output = Result<Probe<T>, E>>, B: Future<Output = Result<Probe<T>, E>>>
    SelectProbeFutures<T, E> for SP2<A, B>
{
    const SIZE: usize = 2;
    type KF = FlatKillFlags<2>;
    type DF = FlatDoneFlags<2>;
    fn process_kills(self: Pin<&mut Self>, kill_flags: &Self::KF, done: &mut Self::DF) {
        let mut p = self.project();
        if !done.0[0] && kill_flags.0[0].get() {
            p.0.set(None);
            done.0[0] = true;
        }
        if !done.0[1] && kill_flags.0[1].get() {
            p.1.set(None);
            done.0[1] = true;
        }
    }
    fn poll_one(
        self: Pin<&mut Self>,
        i: usize,
        done: &mut Self::DF,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<T>, E>> {
        debug_assert!(i < 2);
        let p = self.project();
        match i {
            0 => match poll_slot(p.0, &mut done.0[0], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
            _ => match poll_slot(p.1, &mut done.0[1], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
        }
    }
}

// Flat SP3 impl
impl<
    T,
    E,
    A: Future<Output = Result<Probe<T>, E>>,
    B: Future<Output = Result<Probe<T>, E>>,
    C: Future<Output = Result<Probe<T>, E>>,
> SelectProbeFutures<T, E> for SP3<A, B, C>
{
    const SIZE: usize = 3;
    type KF = FlatKillFlags<3>;
    type DF = FlatDoneFlags<3>;
    fn process_kills(self: Pin<&mut Self>, kill_flags: &Self::KF, done: &mut Self::DF) {
        let mut p = self.project();
        if !done.0[0] && kill_flags.0[0].get() {
            p.0.set(None);
            done.0[0] = true;
        }
        if !done.0[1] && kill_flags.0[1].get() {
            p.1.set(None);
            done.0[1] = true;
        }
        if !done.0[2] && kill_flags.0[2].get() {
            p.2.set(None);
            done.0[2] = true;
        }
    }
    fn poll_one(
        self: Pin<&mut Self>,
        i: usize,
        done: &mut Self::DF,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<T>, E>> {
        debug_assert!(i < 3);
        let p = self.project();
        match i {
            0 => match poll_slot(p.0, &mut done.0[0], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
            1 => match poll_slot(p.1, &mut done.0[1], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
            _ => match poll_slot(p.2, &mut done.0[2], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
        }
    }
}

// Flat SP4 impl
impl<
    T,
    E,
    A: Future<Output = Result<Probe<T>, E>>,
    B: Future<Output = Result<Probe<T>, E>>,
    C: Future<Output = Result<Probe<T>, E>>,
    D: Future<Output = Result<Probe<T>, E>>,
> SelectProbeFutures<T, E> for SP4<A, B, C, D>
{
    const SIZE: usize = 4;
    type KF = FlatKillFlags<4>;
    type DF = FlatDoneFlags<4>;
    fn process_kills(self: Pin<&mut Self>, kill_flags: &Self::KF, done: &mut Self::DF) {
        let mut p = self.project();
        if !done.0[0] && kill_flags.0[0].get() {
            p.0.set(None);
            done.0[0] = true;
        }
        if !done.0[1] && kill_flags.0[1].get() {
            p.1.set(None);
            done.0[1] = true;
        }
        if !done.0[2] && kill_flags.0[2].get() {
            p.2.set(None);
            done.0[2] = true;
        }
        if !done.0[3] && kill_flags.0[3].get() {
            p.3.set(None);
            done.0[3] = true;
        }
    }
    fn poll_one(
        self: Pin<&mut Self>,
        i: usize,
        done: &mut Self::DF,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<T>, E>> {
        debug_assert!(i < 4);
        let p = self.project();
        match i {
            0 => match poll_slot(p.0, &mut done.0[0], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
            1 => match poll_slot(p.1, &mut done.0[1], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
            2 => match poll_slot(p.2, &mut done.0[2], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
            _ => match poll_slot(p.3, &mut done.0[3], cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            },
        }
    }
}

// Overflow: SelectProbeSlot left-nesting for 5+ arms (Rest must be SP4 or another SelectProbeSlot)
impl<T, E, F: Future<Output = Result<Probe<T>, E>>, Rest: SelectProbeFutures<T, E>>
    SelectProbeFutures<T, E> for SelectProbeSlot<Rest, F>
{
    const SIZE: usize = Rest::SIZE + 1;
    type KF = (Rest::KF, Cell<bool>);
    type DF = (Rest::DF, bool);
    fn process_kills(self: Pin<&mut Self>, kill_flags: &Self::KF, done: &mut Self::DF) {
        let mut this = self.project();
        this.rest.process_kills(&kill_flags.0, &mut done.0);
        if !done.1 && kill_flags.1.get() {
            this.fut.set(None);
            done.1 = true;
        }
    }
    fn poll_one(
        self: Pin<&mut Self>,
        i: usize,
        done: &mut Self::DF,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Probe<T>, E>> {
        debug_assert!(i < Self::SIZE);
        if i == Self::SIZE - 1 {
            match poll_slot(self.project().fut, &mut done.1, cx) {
                Poll::Ready(Some(r)) => Poll::Ready(r),
                _ => Poll::Pending,
            }
        } else {
            self.project().rest.poll_one(i, &mut done.0, cx)
        }
    }
}

#[doc(hidden)]
pub trait SelectProbeDoneFlag {
    const SIZE: usize;
    fn new() -> Self;
    fn all_done(&self) -> bool;
}

impl SelectProbeDoneFlag for () {
    const SIZE: usize = 0;
    fn new() -> Self {}
    fn all_done(&self) -> bool {
        true
    }
}

impl<T: SelectProbeDoneFlag> SelectProbeDoneFlag for (T, bool) {
    const SIZE: usize = T::SIZE + 1;
    fn new() -> Self {
        (T::new(), false)
    }
    fn all_done(&self) -> bool {
        self.1 && self.0.all_done()
    }
}

#[doc(hidden)]
pub const fn align_types<T, E, M, Fut>(_: *const Result<Probe<T>, E>, _: *const M)
where
    M: FnOnce() -> Fut,
    Fut: Future<Output = Result<Probe<T>, E>>,
{
}

/// Runs select_probe polling. Takes bare futures via `IntoSelectProbeSlots`,
/// pins them internally.
pub async fn select_probe<T, E, Raw, M, Fut>(
    biased: bool,
    kills: &KillManager<<Raw::Slots as SelectProbeFutures<T, E>>::KF>,
    futures: Raw,
    miss: M,
) -> Result<Probe<T>, E>
where
    Raw: IntoSelectProbeSlots,
    Raw::Slots: SelectProbeFutures<T, E>,
    M: FnOnce() -> Fut,
    Fut: Future<Output = Result<Probe<T>, E>>,
{
    let mut slots = core::pin::pin!(futures.into_slots());
    let mut done = <Raw::Slots as SelectProbeFutures<T, E>>::DF::new();
    let v = core::future::poll_fn(move |cx| {
        for i in 0..<Raw::Slots as SelectProbeFutures<T, E>>::SIZE {
            if kills.take_new_kills() {
                slots.as_mut().process_kills(&kills.flags, &mut done);
            }
            match slots.as_mut().poll_one(i, &mut done, cx) {
                Poll::Ready(Ok(Probe::Hit(val))) if biased => {
                    // Re-poll arms 0..i to see if a higher-priority arm is
                    // also ready. If so, prefer it over arm i.
                    for j in 0..i {
                        if kills.take_new_kills() {
                            slots.as_mut().process_kills(&kills.flags, &mut done);
                        }
                        match slots.as_mut().poll_one(j, &mut done, cx) {
                            Poll::Ready(Ok(Probe::Hit(earlier_val))) => {
                                drop(val);
                                return Poll::Ready(Some(Ok(Probe::Hit(earlier_val))));
                            }
                            Poll::Ready(Ok(Probe::Miss)) => {}
                            Poll::Ready(Err(e)) => {
                                drop(val);
                                return Poll::Ready(Some(Err(e)));
                            }
                            Poll::Pending => {}
                        }
                    }
                    return Poll::Ready(Some(Ok(Probe::Hit(val))));
                }
                Poll::Ready(result) => return Poll::Ready(Some(result)),
                Poll::Pending => {}
            }
        }
        if done.all_done() {
            return Poll::Ready(None);
        }
        Poll::Pending
    })
    .await;
    if let Some(v) = v {
        return v;
    }
    miss().await
}

/// Yielded by streaming accessors ([`StrAccess::next_str`](crate::StrAccess::next_str),
/// [`BytesAccess::next_bytes`](crate::BytesAccess::next_bytes),
/// [`MapAccess::next_kv`](crate::MapAccess::next_kv),
/// [`SeqAccess::next`](crate::SeqAccess::next)).
///
/// - `Data(item)` - another item of data.
/// - `Done(claim)` - the stream is exhausted; thread `claim` back to the
///   outer [`Deserializer::entry`](crate::Deserializer::entry) as proof-of-consumption.
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

    pub fn map_data<Data2>(self, f: impl FnOnce(Data) -> Data2) -> Chunk<Data2, Done> {
        match self {
            Self::Data(d) => Chunk::Data(f(d)),
            Self::Done(d) => Chunk::Done(d),
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
