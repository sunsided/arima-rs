//! Proves that `backshift` links into firmware with no heap.
//!
//! Without the `alloc` feature the crate must not pull in `alloc`; otherwise this
//! binary fails to link with "no global memory allocator found".

#![no_std]
#![no_main]

use core::panic::PanicInfo;

/// Halts on panic; firmware has nowhere to report it.
#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

/// The orders the refit is sized for: ARIMA(1, 1, 1) with drift.
const MODEL: backshift::ArimaSpec<'static> = backshift::ArimaSpec {
    ar: &[0.5],
    d: 1,
    ma: &[0.3],
    seasonal: backshift::Seasonal::NONE,
    exog: &[],
    mean: 0.0,
    sigma2: 1.0,
};

/// The refit window length.
const WINDOW: usize = 32;

/// Refit with a drift and a bounded number of iterations.
const OPTIONS: backshift::RefitOptions = backshift::RefitOptions {
    mean: true,
    max_iterations: 50,
};

/// Entry point: runs a statistic, the streaming filter and a refit so the linker keeps
/// their code.
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let x = [1.0, 2.0, 4.0, 3.0, 5.0];
    let series = backshift::TimeSeries::try_new(&x, backshift::Estimator::Biased);
    let r = series.ok().and_then(|s| backshift::acorr(&s, 1));
    core::hint::black_box(r);

    let spec = backshift::ArimaSpec {
        ar: &[0.5],
        d: 1,
        ma: &[0.3],
        ..backshift::ArimaSpec::default()
    };
    if let Ok(mut filter) = backshift::Filter::<4>::new(&spec) {
        filter.warm_up(&x);
        let innovation = filter.update(core::hint::black_box(2.5));
        let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
        filter.forecast_into(&mut mean, &mut variance);
        let (mut lower, mut upper) = ([0.0; 3], [0.0; 3]);
        filter.forecast_interval_into(0.95, &mut mean, &mut lower, &mut upper);
        core::hint::black_box((innovation, mean, variance, lower, upper));
    }

    let mut window = [0.0; WINDOW];
    for (t, value) in window.iter_mut().enumerate() {
        *value = core::hint::black_box(t as f64 * 0.5);
    }

    let mut refitter = backshift::Refitter::<
        { MODEL.refit_scratch_len(WINDOW, OPTIONS) },
        { MODEL.coefficient_len() },
    >::new();
    if let Ok(refit) = refitter.refit(&MODEL, &window, OPTIONS) {
        if let Ok(mut filter) = backshift::Filter::<4>::new(&refit.spec) {
            filter.warm_up(&window);
            core::hint::black_box(filter.update(core::hint::black_box(16.0)));
        }
    }

    loop {}
}
