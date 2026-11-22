//! Proves the device path never allocates.
//!
//! A counting global allocator wraps the system one. The test builds a
//! [`Filter`], warms it up, streams values and forecasts, and asserts that
//! the count did not move. It runs without the libtest harness
//! (`harness = false`), so no harness thread allocates while it measures.

use backshift::{ArmaSpec, Filter};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The system allocator, counting every allocation.
struct Counting;

/// The number of allocations so far.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System` with the same arguments, so
// `System`'s guarantees carry over; the counter has no effect on memory.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, which `System` shares.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` via `alloc` above, with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn main() {
    let history = [0.3, -0.4, 1.1, 0.2, -0.9, 0.5, 0.0, 0.7];
    let mut mean = [0.0; 6];
    let mut variance = [0.0; 6];
    let before = ALLOCATIONS.load(Ordering::SeqCst);

    let spec = ArmaSpec {
        ar: &[0.5, -0.2],
        ma: &[0.4],
        sigma2: 1.3,
    };
    let mut filter = Filter::<4>::new(&spec).expect("stationary model");
    filter.warm_up(&history);
    for y in [0.1, -0.2, f64::NAN, 0.4] {
        std::hint::black_box(filter.update(y));
    }
    filter.forecast_into(&mut mean, &mut variance);

    let after = ALLOCATIONS.load(Ordering::SeqCst);
    std::hint::black_box((&mean, &variance));
    assert_eq!(
        after - before,
        0,
        "the device path allocated {} times",
        after - before
    );
    assert!(variance.iter().all(|v| v.is_finite() && *v > 0.0));
    println!("no_alloc: ok, 0 allocations");
}
