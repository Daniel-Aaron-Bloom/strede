//! Shared buffer barrier - zero-alloc, no_std async multi-reader primitive.
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
//! exactly the ones in which the load state is `Idle` - no waiter or
//! drop-triggered `Loading` state can race with `fork`.  The `remaining == 0`
//! condition signals that a load is needed; once all waiters drop, `remaining`
//! is already 0 and the current buffer is still valid.

use core::cell::RefCell;
use core::{
    cell::{Cell, UnsafeCell},
    future::Future,
    mem,
    pin::Pin,
    task::{Context, Poll, Waker},
};
use pin_list::{InitializedNode, NodeData, PinList, id::Unchecked};
use pin_project::{pin_project, pinned_drop};

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
// pin-list type bundle
// ---------------------------------------------------------------------------

type WaiterTypes =
    dyn pin_list::Types<Id = Unchecked, Protected = Waker, Removed = (), Unprotected = usize>;

// ---------------------------------------------------------------------------
// List - safe Cell wrapper around PinList
// ---------------------------------------------------------------------------

// WakerLists was here - fields inlined into SharedBufData.

// ---------------------------------------------------------------------------
// LoadState
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LoadState<F> {
    /// Current buffer is valid; no load needed or in progress.  Holds the
    /// loader closure.  When `remaining == 0`, a waiter must trigger a load
    /// before returning its handle.
    Idle(F),
    /// A waiter (or last-handle next()) is currently awaiting the loader
    /// future.  The loader has been moved out onto that task's stack; other
    /// waiters that wake up must not advance until this transitions back
    /// to Idle.
    Loading,
}

// ---------------------------------------------------------------------------
// LoaderGuard - drop guard that reinstates the loader on cancellation
// ---------------------------------------------------------------------------

/// Holds the loader while it is checked out of `load_state`.  On drop,
/// puts it back as `Idle` so the buffer is never permanently wedged.
/// If `remaining == 0` (the load was triggered by all handles dropping),
/// wakes one waiter to hand off the baton.
struct LoaderGuard<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: &'s SharedBufData<B, F>,
    loader: Option<F>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Drop for LoaderGuard<'s, B, F> {
    fn drop(&mut self) {
        if let Some(loader) = self.loader.take() {
            self.shared.load_state.set(LoadState::Idle(loader));
            if self.shared.remaining.get() == 0 {
                // The waiter driving the load was cancelled mid-load.
                // Hand the baton to another queued waiter so it can drive
                // the load; otherwise the remaining waiters would sit on
                // stale wakers forever.
                self.shared.wake_one();
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
struct SharedBufData<B: Buffer, F: AsyncFnMut(&mut B)> {
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
    /// Active waiter list.  Wrapped in `RefCell` for interior mutability.
    active: RefCell<PinList<WaiterTypes>>,
    /// Cleared list, swapped with `active` during wake_all.
    clear: RefCell<PinList<WaiterTypes>>,
}
/// Shared coordination state.  Stack-allocated; all derived types hold a
/// `&'s SharedBuf` borrow, so Rust prevents it from being moved or dropped
/// while any handle exists.
///
/// `B` is the owned buffer type (implementing [`Buffer`]).  The loader
/// mutates it in place on each cycle.  `F` is the loader's type.
pub struct SharedBuf<'s, B: Buffer, F: AsyncFnMut(&mut B)>(&'s SharedBufData<B, F>);
impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Clone for SharedBuf<'s, B, F> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Copy for SharedBuf<'s, B, F> {}

// SAFETY: single-threaded use enforced by !Sync.
unsafe impl<B: Send + Buffer, F: Send + AsyncFnMut(&mut B)> Send for SharedBufData<B, F> {}

impl<B: Buffer, F: AsyncFnMut(&mut B)> SharedBufData<B, F> {
    fn new(initial: B, loader: F) -> Self {
        Self {
            buf: UnsafeCell::new(initial),
            remaining: Cell::new(0),
            total: Cell::new(0),
            epoch: Cell::new(1),
            handle_id: Cell::new(0),
            load_state: Cell::new(LoadState::Idle(loader)),
            // SAFETY: Unchecked::new requires that the returned ID is never
            // compared against IDs from a different call.  We ensure this by
            // keeping each PinList private to a single SharedBuf instance.
            active: RefCell::new(PinList::new(unsafe { Unchecked::new() })),
            clear: RefCell::new(PinList::new(unsafe { Unchecked::new() })),
        }
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> SharedBuf<'s, B, F> {
    /// Run `f` with a shared buffer state initialized with `initial`.  The
    /// loader fires whenever the last handle calls [`Handle::next`], mutating
    /// the buffer in place.
    pub fn with<R>(initial: B, loader: F, f: impl FnOnce(SharedBuf<'_, B, F>) -> R) -> R {
        let shared = SharedBufData::new(initial, loader);
        f(SharedBuf(&shared))
    }

    /// Async counterpart of [`SharedBufRef::with`].
    ///
    /// `shared` and the loader live inside the pinned async state machine,
    /// so all buffer pointers and waiter node addresses are stable for the
    /// entire poll sequence.
    pub async fn with_async<R>(
        initial: B,
        loader: F,
        f: impl AsyncFnOnce(SharedBuf<'_, B, F>) -> R,
    ) -> R {
        let shared = SharedBufData::new(initial, loader);
        f(SharedBuf(&shared)).await
    }

    /// Fork a new active [`Handle`].
    ///
    /// Increments both `remaining` and `total`.  Takes `&mut self` so this
    /// can only be called when no other [`Handle`] or pending wait future
    /// exists (both borrow `SharedBuf` shared); that condition guarantees
    /// the load state is `Idle`.  Use [`Handle::fork`] to create siblings
    /// while another handle is alive.
    pub fn fork(self) -> Handle<'s, B, F> {
        self.0.fork_helper()
    }

    /// Access the inner `SharedBuf` reference (e.g. for assertions in tests).
    #[cfg(test)]
    fn inner(&self) -> &SharedBufData<B, F> {
        self.0
    }
}

impl<B: Buffer, F: AsyncFnMut(&mut B)> SharedBufData<B, F> {
    /// Wake exactly one waiter and remove it from the list.  Returns `true` if
    /// a waiter was woken, `false` if the list was empty.
    #[inline]
    fn wake_one(&self) -> bool {
        let Some(waker) = self.remove_active_front() else {
            return false;
        };
        waker.wake();
        true
    }

    /// Run the loader, bump the epoch, reset `remaining` to `total`, and
    /// wake all waiters.
    async fn load_and_wake(&self) {
        self.run_loader().await;
        self.epoch.set(self.epoch.get() + 1);
        // The list is swapped out so waker.wake() cannot re-enter and append to the list
        self.active.swap(&self.clear);
        self.remaining.set(self.total.get());
        // Drain wakers -
        while let Some(waker) = self.remove_clear_front() {
            waker.wake();
        }
    }

    fn needs_load(&self) -> bool {
        self.remaining.get() == 0
    }

    fn remove_active_front(&self) -> Option<Waker> {
        self.active
            .borrow_mut()
            .cursor_front_mut()
            .remove_current(())
            .ok()
    }
    fn remove_clear_front(&self) -> Option<Waker> {
        self.clear
            .borrow_mut()
            .cursor_front_mut()
            .remove_current(())
            .ok()
    }

    fn reset_node<'a, 'b: 'c, 'c>(
        &'a self,
        initialized: Pin<&'b mut InitializedNode<'c, WaiterTypes>>,
    ) -> (NodeData<WaiterTypes>, usize) {
        if *initialized.unprotected() == self.epoch.get() {
            initialized.reset(&mut *self.active.borrow_mut())
        } else {
            initialized.reset(&mut *self.clear.borrow_mut())
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
            LoadState::Loading => LoadState::Loading,
        };
        self.load_state.set(state);
        phase
    }

    /// Take the loader out of `load_state`, returning an RAII guard that
    /// puts it back on drop.  Panics if the state is already `Loading`.
    fn checkout_loader(&self) -> LoaderGuard<'_, B, F> {
        let prev = self.load_state.replace(LoadState::Loading);
        let loader = match prev {
            LoadState::Idle(f) => f,
            LoadState::Loading => unreachable!("loader checked out twice"),
        };
        LoaderGuard {
            shared: self,
            loader: Some(loader),
        }
    }

    /// Take the loader out, run it to completion, then put it back as `Idle`.
    /// If the future is cancelled mid-await, the drop guard reinstates the
    /// loader so the state never gets stuck in `Loading`.
    ///
    /// The loader mutates `self.buf` in place via an `&mut B` reborrowed from
    /// the `UnsafeCell`.  This is sound because loads only run when
    /// `remaining == 0` - no live `Handle` exists, so no concurrent
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

    fn fork_helper(&self) -> Handle<'_, B, F> {
        debug_assert_eq!(self.phase(), LoadState::Idle(()));
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
/// from the pool - `total` decrements too.
///
/// [`next`]: Handle::next
pub struct Handle<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: &'s SharedBufData<B, F>,
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

        // Suppress the Drop impl - we handle counting manually below.
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
            debug_assert_eq!(shared.phase(), LoadState::Idle(()));
            shared.load_and_wake().await;
            Handle { shared, id }
        } else {
            // Not the last handle: create an internal WaitFuture and suspend
            // until the last handle calls the loader and wakes us.
            let shared = WaitFuture::wait(shared).await;
            // If we were woken because all handles dropped (rather than because
            // a sibling next() loaded), remaining is 0 and we must drive the
            // load ourselves.
            if shared.remaining.get() == 0 {
                debug_assert_eq!(shared.phase(), LoadState::Idle(()));
                shared.load_and_wake().await;
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
            // WaitFutures are still alive.  The first one to re-poll will
            // see remaining == 0 and drive the load.  Wake only ONE - the
            // loader takes time (async) and we don't want other waiters to
            // race ahead and observe a stale buffer before the load
            // completes.  After the load, wake_all is called from
            // Handle::next.
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
/// Uses `pin_project` to safely project through to the pinned `pin_list::Node`.
#[pin_project(PinnedDrop)]
struct WaitFuture<'s, B: Buffer, F: AsyncFnMut(&mut B)> {
    shared: &'s SharedBufData<B, F>,
    #[pin]
    node: pin_list::Node<WaiterTypes>,
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> WaitFuture<'s, B, F> {
    async fn wait(shared: &'s SharedBufData<B, F>) -> &'s SharedBufData<B, F> {
        WaitFuture {
            shared,
            node: pin_list::Node::new(),
        }
        .await
    }
}

impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> Future for WaitFuture<'s, B, F> {
    type Output = &'s SharedBufData<B, F>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        if !this.node.is_initial() {
            // Re-poll path.  The node is initialized (linked or removed).
            // If it was removed by wake_one/wake_all, we were genuinely woken.
            // If still linked, this is a spurious re-poll - refresh the waker.
            let initialized = this
                .node
                .initialized_mut()
                .expect("node must be initialized when waiting");
            let list = &mut *this.shared.active.borrow_mut();
            return match initialized.take_removed(list) {
                Ok((_removed, _epoch)) => {
                    // Was woken (node removed from list by wake_one/wake_all).
                    Poll::Ready(*this.shared)
                }
                Err(still_linked) => {
                    // Spurious re-poll; refresh the waker.
                    *still_linked.protected_mut(list).unwrap() = cx.waker().clone();
                    Poll::Pending
                }
            };
        }

        // First poll: check whether the loader has already fired since this
        // WaitFuture was created, or a drop-triggered needs-load is pending.
        let epoch = this.shared.epoch.get();
        if this.shared.remaining.get() == 0 {
            return Poll::Ready(*this.shared);
        }

        // Insert into the waiter list and store the waker.
        // push_front = head insertion = LIFO order, matching original behavior.
        let waker = cx.waker().clone();
        this.shared
            .active
            .borrow_mut()
            .push_front(this.node, waker, epoch);
        Poll::Pending
    }
}

#[pinned_drop]
impl<'s, B: Buffer, F: AsyncFnMut(&mut B)> PinnedDrop for WaitFuture<'s, B, F> {
    fn drop(self: Pin<&mut Self>) {
        let this = self.project();

        if this.node.is_initial() {
            // Finished normally or never polled - nothing to clean up.
            return;
        }

        // The node is initialized.  Determine whether it was removed (woken)
        // or is still linked (not yet woken).
        let initialized = this
            .node
            .initialized_mut()
            .expect("node must be initialized when waiting");

        match this.shared.reset_node(initialized) {
            // Were in the list, but never woken: `reset` unlinked us.
            // We've never been "alive", so not yet counted in the new epoch's remaining.
            (pin_list::NodeData::Linked(_waker), epoch) if epoch == this.shared.epoch.get() => {}
            // Were in the list, epoch has switch so we were about to be woken up,
            // but we dropped for some reason before our time. We were counted in remaining since
            // we were in the list at wake time. Decrement remaining since no Handle will be created.
            (pin_list::NodeData::Linked(_waker), _epoch) => {
                let remaining = this.shared.remaining.get();
                debug_assert_ne!(remaining, 0);
                this.shared.remaining.set(remaining.saturating_sub(1));
            }
            // Removed from list by wake-one
            (pin_list::NodeData::Removed(()), _epoch) if this.shared.needs_load() => {
                debug_assert_eq!(_epoch, this.shared.epoch.get());
                // We were responsible for loading the next epoch, but we're gonna die
                // before we do that. Hand the baton to another waiter. We're still in
                // the old epoch, so no need to decrement remaining.
                this.shared.wake_one();
            }
            // Removed from list by a wake-all
            (pin_list::NodeData::Removed(()), _epoch) => {
                debug_assert_ne!(
                    this.shared.phase(),
                    LoadState::Loading,
                    "WaitFuture::drop: Phase::Loading should be unreachable"
                );
                // We're counted in the new epoch; decrement it now since
                // we're dying and no Handle will ever be created.
                let remaining = this.shared.remaining.get();
                debug_assert_ne!(remaining, 0);
                this.shared.remaining.set(remaining.saturating_sub(1));
            }
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

                // h0 calls next first - not the last (remaining 2→1) → Pending.
                let mut fut0 = core::pin::pin!(h0.next());
                match poll_once(fut0.as_mut()) {
                    Poll::Pending => {}
                    Poll::Ready(_) => panic!("h0 should pend"),
                }

                // h1 calls next - last (remaining 1→0) → loads b"defghi", wakes h0.
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

                // Drop h0 without calling next - Drop decrements remaining.
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
                // Verify by reading via the handle's shared ref - going through
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

                // Drop h2 without next → remaining=0, wake_one wakes only
                // the head of the list (fut1).
                drop(h2);

                // fut1 was woken; polling it sees remaining==0 and starts the
                // loader.  The loader yields once, so fut1 returns Pending.
                assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                // fut0 was NOT woken.  Polling it spuriously must NOT advance
                // - the load is still in progress, so it must stay Pending.
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

    /// Regression for #1: all waiters are dropped after all handles dropped
    /// (remaining == 0).  A subsequent `fork()` should succeed (the current
    /// buffer is still valid and usable), not panic.
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
                        // remaining hits 0, wake_one wakes fut1.
                        drop(h2);

                        // Leaving this scope drops fut1 then fut0 (reverse
                        // declaration order).  fut1's drop sees remaining==0
                        // and wake_one's fut0.  fut0's drop sees remaining==0,
                        // calls wake_one (no-op, list empty).  Both decrement
                        // total.
                    }
                }

                // At this point: no handles, no waiters, load_state =
                // Idle(f).  Accounting invariant: total = remaining = 0.
                assert_eq!(shared.inner().total.get(), 0, "total leaked waiter slots");
                assert_eq!(shared.inner().remaining.get(), 0);

                // Now fork a fresh handle.  The existing buffer is still
                // perfectly valid (bufs[0] = b"initial"), so this should
                // return a Handle pointing at it.
                let h = shared.fork();
                assert_eq!(h.buf(), b"initial");
            },
        );
    }

    /// Regression: a waiter is woken because remaining==0, re-polls and enters
    /// `run_loader`, but is cancelled mid-load.  `LoaderGuard` restores `Idle`
    /// and hands the baton to another queued waiter - otherwise remaining
    /// waiters deadlock (their stored waker is never fired again).
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

                    // Drop h2 → remaining=0, wake_one wakes fut1 (head).
                    drop(h2);

                    // fut1 sees remaining==0, enters run_loader; loader yields.
                    assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                    // End of scope: fut1 dropped mid-load.  LoaderGuard
                    // reinstates Idle and wake_one's fut0.
                }

                // fut0 must now be able to see remaining==0 and complete the
                // load itself.  Bounded to avoid hanging on regression - the
                // loader only yields once, so completion needs at most 2 polls.
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

    /// Dropping a WaitFuture whose node was inserted in a previous epoch
    /// (i.e. it was woken by wake_all but never re-polled) must reset against
    /// the `clear` list, not `active`.  This exercises the else branch of
    /// `reset_node` where the node's epoch differs from the current epoch.
    #[test]
    fn drop_stale_epoch_waiter_resets_against_clear_list() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded"];
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

                // fut0 and fut1 pend; both inserted into active list with epoch=1.
                // List after both: [fut1, fut0] (push_front).
                let mut fut0 = core::pin::pin!(h0.next());
                let mut fut1 = core::pin::pin!(h1.next());
                assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));
                assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                // fut2 is the last handle (remaining hits 0): loads, bumps
                // epoch to 2, swaps active↔clear, wakes fut1 and fut0.
                let mut fut2 = core::pin::pin!(h2.next());
                let _h2_new = match poll_once(fut2.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("h2 should be Ready (last handle)"),
                };

                // Poll fut1 - it was woken (removed from list), returns Ready.
                let _h1_new = match poll_once(fut1.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("fut1 should be Ready after wake"),
                };

                // Drop fut0 without re-polling.  Its node has epoch=1 but
                // current epoch is 2, so reset_node must use the clear list.
                // (banana - Removed, needs_load false)
                #[allow(clippy::drop_non_drop)]
                drop(fut0);
            },
        );
    }

    /// Same stale-epoch scenario as above, but all sibling handles are dropped
    /// after the load so that `remaining == 0` when fut0 is dropped.  This
    /// exercises the mango (Removed + needs_load) branch on the clear list.
    #[test]
    fn drop_stale_epoch_waiter_needs_load_hands_off_baton() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded", b"third"];
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

                let mut fut0 = core::pin::pin!(h0.next());
                let mut fut1 = core::pin::pin!(h1.next());
                assert!(matches!(poll_once(fut0.as_mut()), Poll::Pending));
                assert!(matches!(poll_once(fut1.as_mut()), Poll::Pending));

                // h2 is last → loads, epoch 1→2, wakes fut1 and fut0.
                let mut fut2 = core::pin::pin!(h2.next());
                let h2_new = match poll_once(fut2.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("h2 should be Ready"),
                };

                // Poll fut1 → Ready (node removed).
                let h1_new = match poll_once(fut1.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("fut1 should be Ready"),
                };

                // Drop both returned handles → remaining hits 0.
                drop(h1_new);
                drop(h2_new);

                // Drop fut0: stale epoch, Removed, needs_load → mango branch.
                // Must wake_one (no-op here since no other waiters).
                #[allow(clippy::drop_non_drop)]
                drop(fut0);
            },
        );
    }

    /// Re-entrant wake during `load_and_wake`'s drain of the clear list.
    /// fut1 is at the head (pushed last via push_front).  Its re-entrant
    /// waker synchronously polls fut1 to Ready, then drops fut0 - which is
    /// still linked in `clear` (not yet drained).  This exercises the apple
    /// branch (Linked + stale epoch) on the clear list.
    #[test]
    fn reentrant_wake_drops_linked_stale_node() {
        let bufs: &[&[u8]] = &[b"initial", b"loaded"];
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

                // We need raw pointers to the pinned futures so the re-entrant
                // waker can poll fut1 and drop fut0 during the wake call.
                // SAFETY: both futures are pinned on the stack and outlive the
                // waker usage (the waker fires synchronously inside this scope).
                let fut0_storage: Cell<Option<strede_test_util::RawAction>> = Cell::new(None);
                let _fut1_storage: Cell<Option<strede_test_util::RawAction>> = Cell::new(None);

                // Poll fut1 with noop waker first (pushed first → tail of list).
                // Wrapped in ManuallyDrop so the re-entrant drop_in_place
                // doesn't cause a double-drop when the stack frame exits.
                let mut fut1 = core::pin::pin!(mem::ManuallyDrop::new(h1.next()));
                // SAFETY: projecting Pin<&mut ManuallyDrop<F>> → Pin<&mut F>.
                macro_rules! fut1_pin {
                    ($f:expr) => {
                        unsafe { $f.as_mut().map_unchecked_mut(|md| &mut **md) }
                    };
                }
                assert!(matches!(poll_once(fut1_pin!(fut1)), Poll::Pending));

                // Poll fut0 with the re-entrant waker (pushed second → head).
                // When woken, the closure will drop fut1 (still linked at tail).
                let mut fut0 = core::pin::pin!(h0.next());
                {
                    let w = strede_test_util::reentrant_waker(&fut0_storage);
                    let mut cx = Context::from_waker(&w);
                    assert!(matches!(fut0.as_mut().poll(&mut cx), Poll::Pending));
                }

                // Capture the unnameable future type via a phantom token,
                // then erase to a raw pointer + drop fn pair.
                fn type_token<T>(_: &T) -> core::marker::PhantomData<fn(*const T)> {
                    core::marker::PhantomData
                }
                fn drop_erased<T>(_: core::marker::PhantomData<fn(*const T)>, ptr: *mut ()) {
                    unsafe { core::ptr::drop_in_place(ptr as *mut T) };
                }
                let token = type_token(unsafe { &**fut1.as_mut().get_unchecked_mut() });
                let fut1_raw: *mut () =
                    unsafe { &mut **fut1.as_mut().get_unchecked_mut() } as *mut _ as *mut ();
                // SAFETY: the raw pointer targets a pinned stack future that
                // is alive when the re-entrant waker fires (synchronously
                // during load_and_wake, before this scope exits).
                fut0_storage.set(Some(strede_test_util::RawAction::new(move || {
                    drop_erased(token, fut1_raw);
                })));

                // h2 is the last handle → load_and_wake fires, drains clear
                // list.  fut1 (head) is woken first → re-entrant closure fires,
                // polls fut1 to Ready, drops fut0 (still linked in clear).
                let mut fut2 = core::pin::pin!(h2.next());
                let _h2_new = match poll_once(fut2.as_mut()) {
                    Poll::Ready(h) => h,
                    Poll::Pending => panic!("h2 should be Ready (last handle)"),
                };

                // fut1 was already drop_in_place'd inside the re-entrant
                // wake.  ManuallyDrop prevents a second drop at scope exit.
                // fut0 is still alive - its waker fired but it was never
                // polled to Ready.  Let it drop normally.
            },
        );
    }
}
