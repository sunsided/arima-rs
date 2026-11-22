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

/// Entry point: runs a statistic and the streaming filter so the linker keeps their code.
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    let x = [1.0, 2.0, 4.0, 3.0, 5.0];
    let series = backshift::TimeSeries::try_new(&x, backshift::Estimator::Biased);
    let r = series.ok().and_then(|s| backshift::acorr(&s, 1));
    core::hint::black_box(r);

    let spec = backshift::ArmaSpec { ar: &[0.5], ma: &[0.3], sigma2: 1.0 };
    if let Ok(mut filter) = backshift::Filter::<4>::new(&spec) {
        filter.warm_up(&x);
        let innovation = filter.update(core::hint::black_box(2.5));
        let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
        filter.forecast_into(&mut mean, &mut variance);
        core::hint::black_box((innovation, mean, variance));
    }

    loop {}
}
