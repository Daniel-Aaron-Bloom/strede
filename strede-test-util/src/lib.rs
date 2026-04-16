use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

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
        Poll::Pending => panic!("future pending — unexpected for in-memory input"),
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
