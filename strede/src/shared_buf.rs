//! Shared buffer barrier — zero-alloc, no_std async multi-reader primitive.
//!
//! A [`SharedBuf`] holds a raw pointer to an externally-owned buffer slice
//! and a loader closure that produces successive buffers.  Any number of
//! [`Handle`]s grant read access to the current slice.  When every handle
//! calls [`Handle::next`], the last one synchronously calls the stored loader
//! to install the next buffer and wake the others; all waiting handles then
//! receive new [`Handle`]s automatically.
//!
//! # Safety model
//!
//! The buffer pointer is untracked by the type system across cycles.  The
//! loader closure carries the `'buf` lifetime, which ensures every slice it
//! returns is valid for at least as long as any derived [`Handle`] can exist.
//! The `with` and `with_async` constructors enforce this by bounding the
//! closure's scope to the [`SharedBuf`]'s stack lifetime.
//!
//! Forking is serialized by the borrow checker, not by runtime checks:
//! [`SharedBuf::fork`] takes `&mut self`, which is unobtainable while any
//! [`Handle`] or pending wait future is alive (both hold `&'s SharedBuf`).
//! [`Handle::fork`] takes `&mut self` on a [`Handle`], which can only exist
//! when `remaining > 0`.  Together these make the only callable `fork` paths
//! exactly the ones in which the load state is `Idle` — no waiter or
//! drop-triggered `NeedsLoad`/`Loading` state can race with `fork`.  To keep
//! that invariant, the last `WaitFuture` to drop in the `NeedsLoad` phase
//! reverts the state to `Idle`: with no waiters left, the signal that a load
//! is needed is stale, and the current buffer is still valid.

use core::{
    cell::{Cell, UnsafeCell},
    future::Future,
    marker::{PhantomData, PhantomPinned},
    mem,
    pin::Pin,
    ptr,
    task::{Context, Poll, Waker},
};

// ---------------------------------------------------------------------------
// Buffer trait
// ---------------------------------------------------------------------------

/// A type that can expose itself as a `&[u8]` slice.  `SharedBuf` owns a `B`
/// and hands out `&[u8]` views of it via [`Handle::buf`]; the loader closure
/// mutates it in place on each cycle.
pub trait Buffer {
    fn as_slice(&self) -> &[u8];
}

impl<const N: usize> Buffer for [u8; N] {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        &self[..]
    }
}

impl Buffer for &[u8] {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        self
    }
}

impl Buffer for &mut [u8] {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        self
    }
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct Inner {
    /// Head of the intrusive waiter list.
    waiters: *mut WaiterNode,
}

impl Inner {
    const fn new() -> Self {
        Self {
            waiters: ptr::null_mut(),
        }
    }
}

// ---------------------------------------------------------------------------
// WaiterNode — intrusive doubly-linked list node, lives inside WaitFuture
// ---------------------------------------------------------------------------

struct WaiterNode {
    waker: Option<Waker>,
    /// Points to the `next` field of the preceding node, or to `Inner::waiters`.
    prev_next: *mut *mut WaiterNode,
    next: *mut WaiterNode,
}

impl WaiterNode {
    const fn new() -> Self {
        Self {
            waker: None,
            prev_next: ptr::null_mut(),
            next: ptr::null_mut(),
        }
    }

    /// Insert `node` at the head of the list rooted at `*head`.
    ///
    /// # Safety
    /// `node` and `head` must be valid, non-aliased pointers.
    unsafe fn insert(node: *mut WaiterNode, head: *mut *mut WaiterNode) {
        unsafe {
            (*node).prev_next = head;
            (*node).next = *head;
            if !(*head).is_null() {
                (**head).prev_next = ptr::addr_of_mut!((*node).next);
            }
            *head = node;
        }
    }

    /// Remove `node` from whatever list it is currently in.
    ///
    /// # Safety
    /// `node` must be currently inserted (prev_next non-null).
    unsafe fn remove(node: *mut WaiterNode) {
        unsafe {
            *(*node).prev_next = (*node).next;
            if !(*node).next.is_null() {
                (*(*node).next).prev_next = (*node).prev_next;
            }
            (*node).prev_next = ptr::null_mut();
            (*node).next = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// LoadState
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LoadState<F> {
    /// Current buffer is valid; no load needed or in progress.  Holds the
    /// loader closure.
    Idle(F),
    /// All handles dropped without next(); a waiter must trigger the load.
    /// Still holds the loader closure.
    NeedsLoad(F),
    /// A waiter (or last-handle next()) is currently awaiting the loader
    /// future.  The loader has been moved out onto that task's stack; other
    /// waiters that wake up must not advance until this transitions back
    /// to Idle.
    Loading,
}

// ---------------------------------------------------------------------------
// LoaderGuard — drop guard that reinstates the loader on cancellation
// ---------------------------------------------------------------------------

/// Holds the loader while it is checked out of `load_state`.  On drop,
/// puts it back as `NeedsLoad` so the buffer is never permanently wedged.
struct LoaderGuard<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: &'s SharedBuf<B, F>,
    loader: Option<F>,
    was_needs_load: bool,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Drop for LoaderGuard<'s, B, F> {
    fn drop(&mut self) {
        if let Some(loader) = self.loader.take() {
            if self.was_needs_load {
                self.shared.load_state.set(LoadState::NeedsLoad(loader));
                // The waiter that claimed NeedsLoad was cancelled mid-load.
                // Hand the baton to another queued waiter so it can drive
                // the load; otherwise the remaining waiters would sit on
                // stale wakers forever.  If no waiter remains, the last
                // WaitFuture's drop (prev_next == null, LoadState::NeedsLoad
                // arm) will revert the state to Idle.
                self.shared.wake_one();
            } else {
                self.shared.load_state.set(LoadState::Idle(loader));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SharedBuf
// ---------------------------------------------------------------------------

/// Shared coordination state.  Stack-allocated; all derived types hold a
/// `&'s SharedBuf` borrow, so Rust prevents it from being moved or dropped
/// while any handle exists.
///
/// `B` is the owned buffer type (implementing [`Buffer`]).  The loader
/// mutates it in place on each cycle.  `F` is the loader's type.
pub struct SharedBuf<B: Buffer, F: AsyncFnMut(&mut B)> {
    /// Current buffer.  Mutated in place (via the loader closure) each time a
    /// load fires.  Wrapped in `UnsafeCell` so `Handle::buf` can hand out
    /// `&[u8]` views while the cell otherwise allows mutation under the
    /// single-threaded (`!Sync`) discipline.
    buf: UnsafeCell<B>,
    /// Number of active [`Handle`]s that have not yet called `next`.
    remaining: Cell<usize>,
    /// Total live handles (both [`Handle`]s and internal wait futures).
    total: Cell<usize>,
    /// Monotonically incremented each time a new buffer is installed.
    epoch: Cell<usize>,
    /// Monotonically incremented each time a new handle is added.
    handle_id: Cell<usize>,
    /// Tracks the loading lifecycle for the current cycle, and owns the
    /// loader closure when not actively running.
    load_state: Cell<LoadState<F>>,
    inner: UnsafeCell<Inner>,
}

// SAFETY: single-threaded use enforced by !Sync.
unsafe impl<B: Send + Buffer, F: Send + AsyncFnMut(&mut B)> Send for SharedBuf<B, F> {}

impl<B: Buffer, F: AsyncFnMut(&mut B)> SharedBuf<B, F> {
    /// Run `f` with a shared buffer state initialized with `initial`.  The
    /// loader fires whenever the last handle calls [`Handle::next`], mutating
    /// the buffer in place.
    pub fn with<R>(initial: B, loader: F, f: impl FnOnce(&mut SharedBuf<B, F>) -> R) -> R {
        let mut shared = SharedBuf {
            buf: UnsafeCell::new(initial),
            remaining: Cell::new(0),
            total: Cell::new(0),
            epoch: Cell::new(1),
            handle_id: Cell::new(0),
            load_state: Cell::new(LoadState::Idle(loader)),
            inner: UnsafeCell::new(Inner::new()),
        };
        f(&mut shared)
    }

    /// Async counterpart of [`SharedBuf::with`].
    ///
    /// `shared` and the loader live inside the pinned async state machine,
    /// so all buffer pointers and waiter node addresses are stable for the
    /// entire poll sequence.
    pub async fn with_async<R>(
        initial: B,
        loader: F,
        f: impl AsyncFnOnce(&mut SharedBuf<B, F>) -> R,
    ) -> R {
        let mut shared = SharedBuf {
            buf: UnsafeCell::new(initial),
            remaining: Cell::new(0),
            total: Cell::new(0),
            epoch: Cell::new(1),
            handle_id: Cell::new(0),
            load_state: Cell::new(LoadState::Idle(loader)),
            inner: UnsafeCell::new(Inner::new()),
        };
        f(&mut shared).await
    }

    /// Wake exactly one waiter and remove it from the list.  Returns `true` if
    /// a waiter was woken, `false` if the list was empty.
    ///
    /// The `&mut Inner` borrow is scoped out before `w.wake()` runs: a
    /// reentrant waker may call back into `SharedBuf` methods that re-borrow
    /// `Inner`, and two live `&mut Inner` at once would be UB.
    #[inline]
    fn wake_one(&self) -> bool {
        let node = {
            // SAFETY: single-threaded; borrow ends before any wake() runs.
            let inner = unsafe { &mut *self.inner.get() };
            inner.waiters
        };
        if node.is_null() {
            return false;
        }
        // SAFETY: node is a valid WaiterNode inserted by WaitFuture::poll.
        unsafe {
            WaiterNode::remove(node);
            if let Some(w) = (*node).waker.take() {
                w.wake();
            }
        }
        true
    }

    /// Reset `remaining` to `total` and wake all waiters.  Called after a
    /// fresh buffer has been installed.
    ///
    /// Clears `prev_next` and `next` on each node so that `WaitFuture::drop`
    /// and the re-poll path can detect that the node is no longer in any list.
    ///
    /// The `&mut Inner` borrow is scoped out before any `w.wake()` runs: a
    /// reentrant waker may call back into `SharedBuf` methods that re-borrow
    /// `Inner`, and two live `&mut Inner` at once would be UB.
    fn wake_all(&self) {
        self.remaining.set(self.total.get());
        // Replace the head of the list with null. The head node still needs a `prev_next`, so set that to our temp
        let mut node = {
            // SAFETY: single-threaded; borrow ends before any wake() runs.
            let inner = unsafe { &mut *self.inner.get() };
            mem::replace(&mut inner.waiters, ptr::null_mut())
        };
        if !node.is_null() {
            unsafe { (*node).prev_next = &mut node };
        }

        while !node.is_null() {
            let old = node;
            // SAFETY: node is a valid WaiterNode inserted by WaitFuture::poll.
            unsafe { WaiterNode::remove(node) }
            if let Some(w) = unsafe { (*old).waker.take() } {
                w.wake();
            }
        }
    }

    unsafe fn add_waiter<'s>(&self, f: &mut WaitFuture<'s, B, F>) {
        unsafe {
            // SAFETY: single-threaded; all &mut borrows are non-overlapping in time.
            let inner = &mut *self.inner.get();

            WaiterNode::insert(ptr::addr_of_mut!(f.node), ptr::addr_of_mut!(inner.waiters));
        }
    }

    /// Current load phase, ignoring the loader payload.
    #[inline]
    fn phase(&self) -> LoadState<()> {
        // This gets compiled down to a single CMP instruction, so there really isn't a window
        // where the intermediary state is visible.
        let state = self.load_state.replace(LoadState::Loading);
        let phase = match state {
            LoadState::Idle(_) => LoadState::Idle(()),
            LoadState::NeedsLoad(_) => LoadState::NeedsLoad(()),
            LoadState::Loading => LoadState::Loading,
        };
        self.load_state.set(state);
        phase
    }

    /// Take the loader out of `load_state`, returning an RAII guard that
    /// puts it back on drop.  Panics if the state is already `Loading`.
    fn checkout_loader(&self) -> LoaderGuard<'_, B, F> {
        let prev = self.load_state.replace(LoadState::Loading);
        let was_needs_load = matches!(&prev, LoadState::NeedsLoad(_));
        let loader = match prev {
            LoadState::Idle(f) | LoadState::NeedsLoad(f) => f,
            LoadState::Loading => unreachable!("loader checked out twice"),
        };
        LoaderGuard {
            shared: self,
            loader: Some(loader),
            was_needs_load,
        }
    }

    /// Take the loader out, run it to completion, then put it back as `Idle`.
    /// If the future is cancelled mid-await, the drop guard reinstates the
    /// loader so the state never gets stuck in `Loading`.
    ///
    /// The loader mutates `self.buf` in place via an `&mut B` reborrowed from
    /// the `UnsafeCell`.  This is sound because loads only run when
    /// `remaining == 0` — no live `Handle` exists, so no concurrent
    /// `Handle::buf()` can alias the `&mut`.
    async fn run_loader(&self) {
        let mut guard = self.checkout_loader();
        // SAFETY: remaining == 0 here (enforced by callers), so no Handle is
        // alive that could alias this &mut B via Handle::buf().  WaitFutures
        // never touch self.buf.
        let b = unsafe { &mut *self.buf.get() };
        (guard.loader.as_mut().unwrap())(b).await;
        self.load_state
            .set(LoadState::Idle(guard.loader.take().unwrap()));
        core::mem::forget(guard);
    }

    /// Fork a new active [`Handle`].
    ///
    /// Increments both `remaining` and `total`.  Takes `&mut self` so this
    /// can only be called when no other [`Handle`] or pending wait future
    /// exists (both borrow `SharedBuf` shared); that condition guarantees
    /// the load state is `Idle`.  Use [`Handle::fork`] to create siblings
    /// while another handle is alive.
    pub fn fork(&mut self) -> Handle<'_, B, F> {
        self.fork_helper()
    }

    fn fork_helper(&self) -> Handle<'_, B, F> {
        assert_eq!(self.phase(), LoadState::Idle(()));
        self.remaining.set(self.remaining.get() + 1);
        self.total.set(self.total.get() + 1);
        let id = self.handle_id.get();
        self.handle_id.set(id + 1);
        Handle { shared: self, id }
    }
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// An active handle with access to the current buffer.
///
/// Dropping a `Handle` without calling [`next`] is treated as if `next` were
/// called (it decrements `remaining`), but the handle is permanently removed
/// from the pool — `total` decrements too.
///
/// [`next`]: Handle::next
pub struct Handle<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: &'s SharedBuf<B, F>,
    id: usize,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Handle<'s, B, F> {
    /// Borrow the current buffer slice.
    pub fn buf(&self) -> &[u8] {
        // SAFETY: while this Handle is alive, no load can run (loads require
        // remaining == 0 → no Handle), so no &mut B into self.shared.buf is
        // live.  The returned &[u8] is tied to &self.
        unsafe { (*self.shared.buf.get()).as_slice() }
    }

    /// Fork a new active sibling [`Handle`].
    ///
    /// Increments both `remaining` and `total`.  Returns a `Handle<'s, ...>`
    /// (not bound to `&mut self`'s lifetime), so siblings are independent and
    /// can outlive each other.  Takes `&mut self` to serialize the call
    /// against other operations on `self`; because a live `Handle` keeps
    /// `remaining > 0`, the load state is necessarily `Idle` here.
    pub fn fork(&mut self) -> Handle<'s, B, F> {
        self.shared.fork_helper()
    }

    /// Release this handle's claim on the current buffer and advance to the
    /// next one.
    ///
    /// The last handle to call `next` synchronously invokes the stored loader
    /// to obtain the next buffer, wakes all waiting handles, and returns
    /// immediately.  All other handles suspend until the last one fires.
    pub async fn next(self) -> Handle<'s, B, F> {
        let shared = self.shared;
        let id = self.id;

        // Suppress the Drop impl — we handle counting manually below.
        mem::forget(self);

        let remaining = shared.remaining.get() - 1;
        shared.remaining.set(remaining);
        struct TotalGuard<'a>(&'a Cell<usize>);
        impl<'a> Drop for TotalGuard<'a> {
            fn drop(&mut self) {
                let count = self.0.get();
                debug_assert_ne!(count, 0);
                self.0.set(count.saturating_sub(1));
            }
        }
        let guard = TotalGuard(&shared.total);

        let handle = if remaining == 0 {
            // Last handle: call the loader, install the new buffer, wake all
            // waiters.  We are the only caller right now (remaining just hit
            // 0, no other Handle is active).
            assert_eq!(shared.phase(), LoadState::Idle(()));
            shared.run_loader().await;
            shared.epoch.set(shared.epoch.get() + 1);
            shared.wake_all();
            Handle { shared, id }
        } else {
            // Not the last handle: create an internal WaitFuture and suspend
            // until the last handle calls the loader and wakes us.
            //
            // WaitFuture is !Unpin (PhantomPinned), which makes this async
            // fn's state machine !Unpin.  Executors pin the state machine
            // before polling, giving the embedded WaiterNode a stable address
            // throughout.
            let shared = WaitFuture {
                shared,
                node: WaiterNode::new(),
                epoch: shared.epoch.get(),
                waiting: false,
                _pin: PhantomPinned,
                _lifetime: PhantomData,
            }
            .await;
            // If we were woken because all handles dropped (rather than because
            // a sibling next() loaded), the state is NeedsLoad and we must
            // drive the load ourselves.  Claim the transition here so siblings
            // woken later see Loading/Idle and don't try to re-load.
            if shared.phase() == LoadState::NeedsLoad(()) {
                shared.run_loader().await;
                shared.epoch.set(shared.epoch.get() + 1);
                // We were the single waiter woken by Handle::drop's wake_one;
                // now that the load is done, wake the rest.
                shared.wake_all();
            }
            Handle { shared, id }
        };

        mem::forget(guard);
        handle
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Drop for Handle<'s, B, F> {
    fn drop(&mut self) {
        // Treat drop-without-next as if next() was called.
        let remaining = self.shared.remaining.get();
        debug_assert_ne!(remaining, 0);
        let remaining = remaining.saturating_sub(1);
        self.shared.remaining.set(remaining);

        let total = self.shared.total.get();
        debug_assert_ne!(total, 0);
        self.shared.total.set(total.saturating_sub(1));
        if remaining == 0 && self.shared.total.get() > 0 {
            // WaitFutures are still alive.  Flag that the first one to
            // re-poll must call the loader before returning its Handle.
            // Wake only ONE — the loader takes time (async) and we don't
            // want other waiters to race ahead and observe a stale buffer
            // before the load completes.  After the load, wake_all is
            // called from Handle::next.
            // Transition Idle(F) → NeedsLoad(F), preserving the loader.
            let loader = match self.shared.load_state.replace(LoadState::Loading) {
                LoadState::Idle(f) | LoadState::NeedsLoad(f) => f,
                _ => unreachable!("drop-triggered while loading, shouldn't be possible"),
            };
            self.shared.load_state.set(LoadState::NeedsLoad(loader));
            self.shared.wake_one();
        }
    }
}

// ---------------------------------------------------------------------------
// WaitFuture (private)
// ---------------------------------------------------------------------------

/// Internal future embedded in the state machine of [`Handle::next`] for
/// non-last handles.  Resolves to a new [`Handle`] once the last handle
/// calls the loader.
///
/// Must be pinned before polling (it embeds a [`WaiterNode`] whose address
/// is placed in the shared linked list).  The `PhantomPinned` marker
/// propagates `!Unpin` to the outer `async fn next` state machine.
struct WaitFuture<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: &'s SharedBuf<B, F>,
    node: WaiterNode,
    /// `Inner::epoch` at the time this future was created.  If `inner.epoch`
    /// exceeds this on first poll, the loader has already fired.
    epoch: usize,
    waiting: bool,
    _pin: PhantomPinned,
    _lifetime: PhantomData<&'s ()>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Future for WaitFuture<'s, B, F> {
    type Output = &'s SharedBuf<B, F>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: we never move out of the pinned data.
        let this = unsafe { self.get_unchecked_mut() };

        if this.waiting {
            // Re-poll. wake_one/wake_all clear prev_next before firing wakers,
            // so prev_next == null means we were genuinely woken; non-null
            // means a spurious re-poll while still in the list — refresh the
            // waker and stay Pending.
            if !this.node.prev_next.is_null() {
                this.node.waker = Some(cx.waker().clone());
                return Poll::Pending;
            }
            this.waiting = false;
            return Poll::Ready(this.shared);
        }

        // First poll: check whether the loader has already fired since this
        // WaitFuture was created, or a drop-triggered needs-load is pending.
        let phase = this.shared.phase();
        if this.shared.epoch.get() > this.epoch || phase == LoadState::NeedsLoad(()) {
            return Poll::Ready(this.shared);
        }

        // Insert into the waiter list and store the waker.
        // SAFETY: this.node has a stable address because self is pinned.
        unsafe {
            this.node.waker = Some(cx.waker().clone());
            this.shared.add_waiter(this);
        }
        this.waiting = true;
        Poll::Pending
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Drop for WaitFuture<'s, B, F> {
    fn drop(&mut self) {
        if !self.waiting {
            // Everything went good, nothing to cleanup
            return;
        }
        if self.node.prev_next.is_null() {
            // Was woken (prev_next cleared) but dropped before being re-polled.
            match self.shared.phase() {
                LoadState::NeedsLoad(()) => {
                    // We were the wake_one'd waiter for a drop-triggered load
                    // but never re-polled to claim it.  Hand the baton to
                    // another waiter; if none remains, revert to Idle — the
                    // "need a load" signal was only meaningful while someone
                    // was waiting, and the current buffer is still valid.
                    if !self.shared.wake_one() {
                        let loader = match self.shared.load_state.replace(LoadState::Loading) {
                            LoadState::NeedsLoad(f) => f,
                            _ => unreachable!("phase was NeedsLoad a moment ago"),
                        };
                        self.shared.load_state.set(LoadState::Idle(loader));
                    }
                }
                LoadState::Loading => {
                    // Should be unreachable: the only waiter that can observe
                    // LoadState::Loading here is the one actively driving the load,
                    // and that waiter's WaitFuture has `resolved = true,
                    // waiting = false` (set when poll returned Ready to the
                    // outer Handle::next), so its Drop exits via the early
                    // `if !self.waiting` return above.  Any other WaitFuture
                    // only has its prev_next cleared by wake_all, which runs
                    // *after* the load completes (phase already back to Idle).
                    debug_assert!(
                        false,
                        "WaitFuture::drop: Phase::Loading should be unreachable"
                    );
                }
                LoadState::Idle(()) => {
                    // Normal next()-driven wakeup: the new cycle's remaining
                    // already counts us; decrement it now since we aren't actually
                    // remaining and no Handle exists yet
                    let remaining = self.shared.remaining.get();
                    debug_assert_ne!(remaining, 0);
                    self.shared.remaining.set(remaining.saturating_sub(1));
                }
            }
        } else {
            // Still in the list (not yet woken): remove to avoid a dangling
            // pointer.  We were not yet counted in the new cycle's remaining.
            // SAFETY: node is currently inserted (prev_next non-null).
            unsafe { WaiterNode::remove(ptr::addr_of_mut!(self.node)) };
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Poll a pinned future once with the noop waker.
    fn poll_once<F: Future>(f: Pin<&mut F>) -> Poll<F::Output> {
        let w = strede_test_util::noop_waker();
        let mut cx = Context::from_waker(&w);
        f.poll(&mut cx)
    }

    fn yield_now() -> YieldNow {
        YieldNow { yielded: false }
    }

    struct YieldNow {
        yielded: bool,
    }

    impl Future for YieldNow {
        type Output = ();

        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.yielded {
                Poll::Ready(())
            } else {
                self.yielded = true;
                Poll::Pending
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn rewrite() {
        // Loader is called once per last-handle next() call, rewriting the
        // single `[u8; 5]` in place on each invocation.
        let values: &[&[u8; 5]] = &[b"world", b"xxxxx"];
        let mut i = 0usize;
        SharedBuf::with(
            *b"hello",
            async |b: &mut [u8; 5]| {
                *b = *values[i];
                i += 1;
            },
            |shared| {
                let h = shared.fork();
                assert_eq!(h.buf(), b"hello");

                // Sole handle → last → rewrites buf in place to b"world".
                let mut fut = core::pin::pin!(h.next());
                let h2 = match poll_once(fut.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("expected Ready for sole handle"),
                };
                assert_eq!(h2.buf(), b"world");

                // Advance once more → rewrites to b"xxxxx".
                let mut fut = core::pin::pin!(h2.next());
                let h3 = match poll_once(fut.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("expected Ready"),
                };
                assert_eq!(h3.buf(), b"xxxxx");
            },
        );
    }

    #[test]
    fn n1_single_cycle() {
        // Loader is called once by `with` (→ b"hello"), then once per
        // last-handle next() call (→ b"world!!", then → b"x").
        let bufs: &[&[u8]] = &[b"hello", b"world!!", b"x"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let h = shared.fork();
                assert_eq!(h.buf(), b"hello");

                // Sole handle → last → loads b"world!!" immediately.
                let mut fut = core::pin::pin!(h.next());
                let h2 = match poll_once(fut.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("expected Ready for sole handle"),
                };
                assert_eq!(h2.buf(), b"world!!");

                // Advance once more → loads b"x".
                let mut fut = core::pin::pin!(h2.next());
                let h3 = match poll_once(fut.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("expected Ready"),
                };
                assert_eq!(h3.buf(), b"x");
            },
        );
    }

    #[test]
    fn n2_second_handle_is_loader() {
        let bufs: &[&[u8]] = &[b"abc", b"defghi"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let mut h0 = shared.fork();
                let h1 = h0.fork();

                // h0 calls next first — not the last (remaining 2→1) → Pending.
                let mut fut0 = core::pin::pin!(h0.next());
                match poll_once(fut0.as_mut()) {
                    Poll::Pending => {}
                    Poll::Ready(_) => panic!("h0 should pend"),
                }

                // h1 calls next — last (remaining 1→0) → loads b"defghi", wakes h0.
                let mut fut1 = core::pin::pin!(h1.next());
                let h1_new = match poll_once(fut1.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("h1 should be Ready"),
                };
                assert_eq!(h1_new.buf(), b"defghi");

                // fut0 resolves on the next poll.
                let h0_new = match poll_once(fut0.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("fut0 should be Ready after load"),
                };
                assert_eq!(h0_new.buf(), b"defghi");
            },
        );
    }

    #[test]
    fn n3_round_trip_non_uniform_sizes() {
        let bufs: &[&[u8]] = &[b"one", b"two-but-longer"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let mut h0 = shared.fork();
                let mut h1 = h0.fork();
                let h2 = h1.fork();

                // Pin all three next() futures.
                let mut fut0 = core::pin::pin!(h0.next());
                let mut fut1 = core::pin::pin!(h1.next());
                let mut fut2 = core::pin::pin!(h2.next());

                // First two pend; the last one fires the loader.
                assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));
                assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));
                let h_loader_new = match poll_once(fut2.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("h2 should be Ready (last)"),
                };
                assert_eq!(h_loader_new.buf(), b"two-but-longer");

                // fut0 and fut1 resolve.
                let h0_new = match poll_once(fut0.as_mut()) {
                    Poll::Ready(h) => h,
                    _ => panic!("fut0 should be Ready"),
                };
                let h1_new = match poll_once(fut1.as_mut()) {
                    Poll::Ready(h) => h,
                    _ => panic!("fut1 should be Ready"),
                };
                assert_eq!(h0_new.buf(), b"two-but-longer");
                assert_eq!(h1_new.buf(), b"two-but-longer");
            },
        );
    }

    #[test]
    fn wait_future_drop_reduces_total() {
        let bufs: &[&[u8]] = &[b"data", b"next"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let mut h0 = shared.fork(); // total=1, remaining=1
                let h1 = h0.fork(); // total=2, remaining=2

                // Poll h0's future once → Pending (WaiterNode registered).
                // Then drop the future in a block so WaitFuture::drop fires.
                {
                    let mut fut0 = core::pin::pin!(h0.next());
                    match poll_once(fut0.as_mut()) {
                        Poll::Pending => {}
                        Poll::Ready(_) => panic!(),
                    }
                    // fut0 drops here → WaitFuture::drop removes node, total → 1.
                }

                // h1 should now be the sole remaining handle.
                let mut fut1 = core::pin::pin!(h1.next());
                let h1_new = match poll_once(fut1.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("h1 should be loader after fut0 dropped"),
                };
                assert_eq!(h1_new.buf(), b"next");
            },
        );
    }

    #[test]
    fn handle_drop_without_next_still_reaches_loader() {
        let bufs: &[&[u8]] = &[b"drop", b"ok"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let mut h0 = shared.fork(); // total=2, remaining=2
                let h1 = h0.fork();

                // Drop h0 without calling next — Drop decrements remaining.
                drop(h0);

                // h1 is now the last (remaining=1 → 0 on next()).
                let mut fut1 = core::pin::pin!(h1.next());
                let h = match poll_once(fut1.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("should be loader"),
                };
                assert_eq!(h.buf(), b"ok");
            },
        );
    }

    /// All Handles are dropped while WaitFutures are pending.  The first
    /// WaitFuture to re-poll must call the loader and return the new buffer.
    #[test]
    fn all_handles_dropped_with_waiters_pending() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let mut h0 = shared.fork(); // total=1, remaining=1
                let h1 = h0.fork(); // total=2, remaining=2

                // h0 calls next → remaining=1, WaitFuture registered.
                let mut fut0 = core::pin::pin!(h0.next());
                assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));

                // Drop h1 without calling next → remaining=0, needs_load set,
                // wake_all fires fut0's waker.
                drop(h1);

                // fut0 re-polls: sees needs_load, calls loader, returns Ready.
                let h0_new = match poll_once(fut0.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("fut0 should load and resolve"),
                };
                assert_eq!(h0_new.buf(), b"loaded");

                // The new Handle is in a consistent state: remaining=total=1.
                // Verify by reading via the handle's shared ref — going through
                // `shared` directly here would conflict with h0_new's borrow.
                assert_eq!(h0_new.shared.total.get(), 1);
                assert_eq!(h0_new.shared.remaining.get(), 1);
            },
        );
    }

    /// Cancelling `Handle::next` while the loader future is mid-await must not
    /// leave `load_state` stuck in `Loading` (which would lose the loader and
    /// permanently wedge the buffer).
    #[test]
    fn cancel_during_loader_does_not_wedge_state() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                yield_now().await;
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let h = shared.fork(); // sole handle

                // Poll once: last handle path, loader starts, hits yield_now,
                // returns Pending. State is now Loading.
                {
                    let mut fut = core::pin::pin!(h.next());
                    assert!(matches!(poll_once(fut.as_mut()), Poll::Pending));
                    // Drop the future here, mid-load.
                }

                // After cancellation, the buffer must still be usable: forking
                // a new handle should succeed (phase == Idle).
                let h2 = shared.fork();
                assert_eq!(h2.buf(), b"initial");

                // And driving it through next() should still work.
                let mut fut = core::pin::pin!(h2.next());
                let h3 = match poll_once(fut.as_mut()) {
                    Poll::Pending => match poll_once(fut.as_mut()) {
                        Poll::Ready(h) => h,
                        Poll::Pending => panic!("loader stuck after cancel"),
                    },
                    Poll::Ready(h) => h,
                };
                assert_eq!(h3.buf(), b"loaded");
            },
        );
    }

    /// Two waiters pending; handle dropped. Both get woken. Second one polls
    /// first and must not see the new buffer until the first one has loaded.
    #[test]
    fn two_waiters_second_polls_first_gets_correct_buf() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                yield_now().await;
                *b = bufs[i];
            },
            |shared| {
                let mut h0 = shared.fork();
                let mut h1 = h0.fork();
                let h2 = h1.fork();

                // h0 and h1 both call next → both pend.
                let mut fut0 = core::pin::pin!(h0.next());
                let mut fut1 = core::pin::pin!(h1.next());
                assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));
                assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                // Drop h2 without next → remaining=0, NeedsLoad set, wake_one
                // wakes only the head of the list (fut1).
                drop(h2);

                // fut1 was woken; polling it claims NeedsLoad and starts the
                // loader.  The loader yields once, so fut1 returns Pending.
                assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                // fut0 was NOT woken.  Polling it spuriously must NOT advance
                // — the load is still in progress, so it must stay Pending.
                assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));

                // Drive fut1 to completion: yield_now resolves, loader returns
                // "loaded", buffer installed, wake_all fires fut0.
                let h1_new = match poll_once(fut1.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("fut1 should be ready after load"),
                };
                assert_eq!(h1_new.buf(), b"loaded");

                // fut0 now sees the new epoch and resolves with the new buffer.
                let h0_new = match poll_once(fut0.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("fut0 should be ready after wake_all"),
                };
                assert_eq!(h0_new.buf(), b"loaded");
            },
        );
    }

    /// Regression for #1: all waiters are dropped after `NeedsLoad` was set,
    /// leaving `load_state = NeedsLoad` with no handles and no waiters.  A
    /// subsequent `fork()` should succeed (the current buffer is still valid
    /// and usable), not panic the `phase == Idle` assert in `fork_helper`.
    #[test]
    fn fork_after_all_waiters_drop_during_needs_load() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                {
                    let mut h0 = shared.fork();
                    let mut h1 = h0.fork();
                    let h2 = h1.fork();

                    // h0 and h1 park as waiters.  After these, list =
                    // [fut1, fut0] (inserted at head), remaining = 1, total = 3.
                    //
                    // We nest so every future/handle is dropped before the
                    // post-condition fork.  `drop(pinned_ref)` does not drop
                    // the underlying future (Pin wraps a &mut), so we rely on
                    // scope end instead.
                    {
                        let mut fut0 = core::pin::pin!(h0.next());
                        let mut fut1 = core::pin::pin!(h1.next());
                        assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));
                        assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                        // Drop h2 (the sole remaining Handle) without next().
                        // This flips load_state → NeedsLoad and wake_ones fut1.
                        drop(h2);

                        // Leaving this scope drops fut1 then fut0 (reverse
                        // declaration order).  fut1's drop sees NeedsLoad and
                        // wake_one's fut0.  fut0's drop sees NeedsLoad, calls
                        // wake_one (no-op, list empty).  Both decrement total.
                    }
                }

                // At this point: no handles, no waiters, load_state =
                // NeedsLoad(f).  Accounting invariant: total = remaining = 0.
                assert_eq!(shared.total.get(), 0, "total leaked waiter slots");
                assert_eq!(shared.remaining.get(), 0);

                // Now fork a fresh handle.  The existing buffer is still
                // perfectly valid (bufs[0] = b"initial"), so this should
                // return a Handle pointing at it.  Before the fix, this panics
                // because fork_helper asserts phase == Idle but phase is
                // NeedsLoad.
                let h = shared.fork();
                assert_eq!(h.buf(), b"initial");
            },
        );
    }

    /// Regression: a waiter is woken for `NeedsLoad`, re-polls and enters
    /// `run_loader`, but is cancelled mid-load.  `LoaderGuard` restores
    /// `NeedsLoad`, but the *baton* needs to be passed to another queued
    /// waiter — otherwise remaining waiters deadlock (their stored waker is
    /// never fired again), and if they too are cancelled the state is stuck
    /// in `NeedsLoad` so a subsequent `fork()` panics.
    #[test]
    fn cancelled_waiter_during_needs_load_hands_off_to_next_waiter() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded"];
        let mut i = 0usize;
        SharedBuf::with(
            bufs[0],
            async |b: &mut &[u8]| {
                yield_now().await;
                i += 1;
                *b = bufs[i];
            },
            |shared| {
                let mut h0 = shared.fork();
                let mut h1 = h0.fork();
                let h2 = h1.fork();

                // Park fut0 first, then fut1 → list = [fut1, fut0] (fut1 at head).
                let mut fut0 = core::pin::pin!(h0.next());
                assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));

                {
                    let mut fut1 = core::pin::pin!(h1.next());
                    assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                    // Drop h2 → state = NeedsLoad, wake_one wakes fut1 (head).
                    drop(h2);

                    // fut1 claims NeedsLoad, enters run_loader; loader yields.
                    assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                    // End of scope: fut1 dropped mid-load.  LoaderGuard
                    // reinstates NeedsLoad.  Without the fix, fut0's waker is
                    // never fired — the baton is lost.
                }

                // fut0 must now be able to claim the pending NeedsLoad and
                // complete the load itself.  Pre-fix, this sits Pending forever.
                // Poll once: first poll should see phase=NeedsLoad and return
                // Ready(shared) immediately (either because we were wake_one'd,
                // prev_next cleared → Ready, OR because first-poll path checks
                // phase == NeedsLoad — but we already polled once, so we're on
                // the re-poll branch that needs prev_next == null).
                // Bounded to avoid hanging on regression — the loader only
                // yields once, so completion needs at most 2 polls here.
                let mut h0_new = None;
                for _ in 0..8 {
                    if let Poll::Ready(h) = poll_once(fut0.as_mut()) {
                        h0_new = Some(h);
                        break;
                    }
                }
                let h0_new =
                    h0_new.expect("fut0 deadlocked: baton not handed off after fut1 cancel");
                assert_eq!(h0_new.buf(), b"loaded");
            },
        );
    }
}
