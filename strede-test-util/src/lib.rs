use core::{
    cell::Cell,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

pub fn noop_waker() -> Waker {
    static VTABLE: RawWakerVTable =
        RawWakerVTable::new(|p| RawWaker::new(p, &VTABLE), |_| {}, |_| {}, |_| {});
    unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
}

/// Poll a future to completion, panicking if it returns `Pending`.
/// Suitable for in-memory futures that never yield.
pub fn block_on<F: core::future::Future>(f: F) -> F::Output {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut fut = core::pin::pin!(f);
    match fut.as_mut().poll(&mut cx) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("future pending - unexpected for in-memory input"),
    }
}

/// Poll a future in a spin loop until it resolves.
/// Use for futures that may yield `Pending` cooperatively (e.g. chunked streaming).
pub fn block_on_loop<F: core::future::Future>(f: F) -> F::Output {
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let mut fut = core::pin::pin!(f);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => {}
        }
    }
}

/// A waker backed by a `Cell<Option<RawAction>>`.  When woken it takes
/// the stored action and calls it.  The action is a type-erased `FnOnce()`
/// stored as a raw pointer + caller pair, sidestepping `'static` bounds.
///
/// SAFETY contract: the action must be consumed or dropped before the
/// captured data goes out of scope.  All uses in these tests fire
/// synchronously inside `load_and_wake`, which returns before the
/// captured pinned futures are dropped.
pub struct RawAction {
    data: *mut (),
    call: unsafe fn(*mut ()),
}
// SAFETY: single-threaded tests only.
unsafe impl Send for RawAction {}

impl RawAction {
    /// Create from a concrete `FnOnce()`.  Caller must ensure `f` is
    /// not dropped before `call` is invoked (or the RawAction is
    /// discarded).
    pub fn new<F: FnOnce()>(f: F) -> Self {
        let boxed = Box::into_raw(Box::new(f));
        RawAction {
            data: boxed as *mut (),
            call: |ptr| unsafe { Box::from_raw(ptr as *mut F)() },
        }
    }
}

pub fn reentrant_waker(slot: &Cell<Option<RawAction>>) -> Waker {
    let ptr = slot as *const Cell<Option<RawAction>> as *const ();
    unsafe fn clone(ptr: *const ()) -> RawWaker {
        RawWaker::new(ptr, &VTABLE)
    }
    unsafe fn wake(ptr: *const ()) {
        let cell = unsafe { &*(ptr as *const Cell<Option<RawAction>>) };
        if let Some(action) = cell.take() {
            unsafe { (action.call)(action.data) };
        }
    }
    unsafe fn wake_ref(ptr: *const ()) {
        let cell = unsafe { &*(ptr as *const Cell<Option<RawAction>>) };
        if let Some(action) = cell.take() {
            unsafe { (action.call)(action.data) };
        }
    }
    unsafe fn drop(_: *const ()) {}
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_ref, drop);
    unsafe { Waker::from_raw(RawWaker::new(ptr, &VTABLE)) }
}
