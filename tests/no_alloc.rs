//! Proves the device path never allocates.
//!
//! A counting global allocator wraps the system one. The test builds a
//! [`Filter`], warms it up, streams values, forecasts with intervals (also with an
//! `EtsFilter`), refits the model
//! on a window and swaps the new one in, and asserts that the count did not move. It runs without the libtest harness
//! (`harness = false`), so no harness thread allocates while it measures.

use backshift::{ArimaSpec, Filter, RefitOptions, Refitter, Seasonal};
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

/// The orders the refit is sized for: ARIMA(1, 1, 1) with drift.
const MODEL: ArimaSpec<'static> = ArimaSpec {
    ar: &[0.5],
    d: 1,
    ma: &[0.3],
    seasonal: Seasonal::NONE,
    exog: &[],
    mean: 0.0,
    sigma2: 1.0,
};

/// The refit window length.
const WINDOW: usize = 48;

/// Refit with a drift, as the model was fitted.
const OPTIONS: RefitOptions = RefitOptions {
    mean: true,
    max_iterations: 100,
};

/// The refitter, sized at compile time and kept out of the stack.
static REFITTER: std::sync::Mutex<
    Refitter<{ MODEL.refit_scratch_len(WINDOW, OPTIONS) }, { MODEL.coefficient_len() }>,
> = std::sync::Mutex::new(Refitter::new());

fn main() {
    let history = [0.3, -0.4, 1.1, 0.2, -0.9, 0.5, 0.0, 0.7];
    let mut mean = [0.0; 6];
    let mut variance = [0.0; 6];
    let before = ALLOCATIONS.load(Ordering::SeqCst);

    let spec = ArimaSpec {
        ar: &[0.5, -0.2],
        d: 1,
        ma: &[0.4],
        mean: 0.1,
        sigma2: 1.3,
        ..ArimaSpec::default()
    };
    let mut filter = Filter::<4>::new(&spec).expect("stationary model");
    filter.warm_up(&history);
    for y in [0.1, -0.2, f64::NAN, 0.4] {
        std::hint::black_box(filter.update(y));
    }
    filter.forecast_into(&mut mean, &mut variance);

    let mut steady = Filter::<4>::new(&spec)
        .expect("stationary model")
        .with_steady_state();
    for t in 0..64 {
        std::hint::black_box(steady.update(f64::from(t % 7) - 3.0));
    }
    steady.forecast_into(&mut mean, &mut variance);

    let mut conformal = backshift::Conformal::<3, 32>::new();
    let mut conformal_filter = Filter::<4>::new(&spec).expect("stationary model");
    let (mut conformal_mean, mut lower3, mut upper3) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    for t in 0..40 {
        let y = f64::from(t % 5) - 2.0;
        conformal.observe(y);
        conformal_filter.update(y);
        conformal_filter.forecast_into(&mut conformal_mean, &mut [0.0; 3]);
        conformal.record(&conformal_mean);
    }
    conformal.interval_into(0.9, &conformal_mean, &mut lower3, &mut upper3);
    std::hint::black_box((&lower3, &upper3));

    let mut online = backshift::OnlineAr::<4>::new(backshift::OnlineArOptions::default())
        .expect("default options are valid");
    for t in 0..64 {
        std::hint::black_box(online.update(f64::from(t % 9) / 9.0 - 0.5));
    }
    let mut online_forecast = [0.0; 3];
    online.forecast_into(&mut online_forecast);
    std::hint::black_box(&online_forecast);

    let cycles = [1.0, -1.0, 0.5, -0.5, 2.0, 0.0, -2.0];
    let mut seasonal = backshift::MstlFilter::<4, 7>::new(&spec, &[3, 4], &cycles)
        .expect("valid periods and cycles");
    for t in 0..20 {
        std::hint::black_box(seasonal.update(f64::from(t % 6)));
    }
    let (mut s_mean, mut s_lower, mut s_upper) = ([0.0; 5], [0.0; 5], [0.0; 5]);
    seasonal.forecast_interval_into(0.9, &mut s_mean, &mut s_lower, &mut s_upper);
    std::hint::black_box((&s_mean, &s_lower, &s_upper));

    let ets_seasonal = [1.0, -0.5, 0.5, -1.0];
    let ets_spec = backshift::EtsSpec {
        error: backshift::Form::Additive,
        season: backshift::Form::Additive,
        alpha: 0.3,
        trend: backshift::Trend::Damped {
            beta: 0.05,
            phi: 0.95,
        },
        gamma: 0.1,
        level: 10.0,
        slope: 0.2,
        seasonal: &ets_seasonal,
        sigma2: 1.0,
    };
    let mut ets = backshift::EtsFilter::<4>::new(&ets_spec).expect("valid model");
    for t in 0..20 {
        std::hint::black_box(ets.update(10.0 + f64::from(t % 4)));
    }
    let (mut e_mean, mut e_lower, mut e_upper) = ([0.0; 5], [0.0; 5], [0.0; 5]);
    ets.forecast_interval_into(0.9, &mut e_mean, &mut e_lower, &mut e_upper);
    std::hint::black_box((&e_mean, &e_lower, &e_upper));

    // A multiplicative season forecasts its intervals in caller scratch: 7 q^2 + 3 q + m
    // values for q = 2 m = 8.
    let factors = [1.1, 0.9, 1.05, 0.95];
    let multiplicative = backshift::EtsSpec {
        error: backshift::Form::Multiplicative,
        season: backshift::Form::Multiplicative,
        seasonal: &factors,
        sigma2: 0.01,
        ..ets_spec
    };
    let mut m_ets = backshift::EtsFilter::<4>::new(&multiplicative).expect("valid model");
    for t in 0..20 {
        std::hint::black_box(m_ets.update(10.0 + f64::from(t % 4)));
    }
    let mut ets_scratch = [0.0; 7 * 64 + 3 * 8 + 4];
    assert!(ets_scratch.len() >= m_ets.forecast_scratch_len());
    m_ets.forecast_interval_with(
        &mut ets_scratch,
        0.9,
        &mut e_mean,
        &mut e_lower,
        &mut e_upper,
    );
    assert!(e_lower.iter().all(|v| v.is_finite()));
    std::hint::black_box((&e_mean, &e_lower, &e_upper));

    let mut fourier = [0.0; 4];
    backshift::fourier_row(365.25, 2, 1_000_000, &mut fourier).expect("valid terms");
    std::hint::black_box(&fourier);
    let (mut lower, mut upper) = ([0.0; 6], [0.0; 6]);
    filter.forecast_interval_into(0.95, &mut mean, &mut lower, &mut upper);

    let exog_spec = ArimaSpec {
        ar: &[0.6],
        exog: &[1.5, -0.5],
        ..ArimaSpec::default()
    };
    let mut exog_filter = Filter::<4, 2>::new(&exog_spec).expect("stationary model");
    exog_filter.warm_up_exog(&history[..2], &[0.1, 0.2, 0.3, 0.4]);
    std::hint::black_box(exog_filter.update_exog(0.5, &[1.0, -1.0]));
    let (mut exog_mean, mut exog_variance) = ([0.0; 2], [0.0; 2]);
    exog_filter.forecast_exog_into(&[0.0, 1.0, 1.0, 0.0], &mut exog_mean, &mut exog_variance);
    std::hint::black_box((&exog_mean, &exog_variance));

    std::hint::black_box((
        backshift::ljung_box(&history, 3, 1),
        backshift::jarque_bera(&history),
        backshift::kpss(&history, backshift::KpssNull::Level, 2),
    ));

    // A ramp with an irregular sawtooth on top: plain arithmetic, and far from an exact fit.
    let window: [f64; WINDOW] = core::array::from_fn(|t| {
        let wobble = f64::from(u32::try_from((t * 7) % 5).expect("small"));
        0.2 * f64::from(u32::try_from(t).expect("small")) + wobble / 4.0
    });
    let mut refitter = REFITTER.lock().expect("not poisoned");
    let refit = refitter
        .refit(&MODEL, &window, OPTIONS)
        .expect("the window fits");
    let mut refitted = Filter::<4>::new(&refit.spec).expect("stationary model");
    refitted.warm_up(&window);
    std::hint::black_box(refitted.update(10.0));

    let after = ALLOCATIONS.load(Ordering::SeqCst);
    std::hint::black_box((&mean, &variance, &lower, &upper));
    assert_eq!(
        after - before,
        0,
        "the device path allocated {} times",
        after - before
    );
    assert!(variance.iter().all(|v| v.is_finite() && *v > 0.0));
    println!("no_alloc: ok, 0 allocations");
}
