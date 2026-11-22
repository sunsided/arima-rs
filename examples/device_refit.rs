//! A device loop that notices when its model stops fitting and refits it in place.
//!
//! The series is an AR(1) whose coefficient jumps from 0.2 to 0.85 halfway through.
//! The filter streams every value. At the end of each non-overlapping window it runs
//! Ljung–Box on that window's standardised innovations; when they are no longer white,
//! it refits on the window, starting from the current model, and swaps in the new
//! filter, primed on the same window. Nothing in the loop allocates: the refitter's
//! buffers are sized at compile time, and the current coefficients live in an array.
//!
//! Run with `cargo run --example device_refit`.

use backshift::{ArimaSpec, Filter, RefitOptions, Refitter, Seasonal, ljung_box};

/// The orders the device runs, with the coefficients it was deployed with.
const DEPLOYED: ArimaSpec<'static> = ArimaSpec {
    ar: &[0.2],
    d: 0,
    ma: &[],
    seasonal: Seasonal::NONE,
    exog: &[],
    mean: 0.0,
    sigma2: 1.0,
};

/// Values per window: the drift check runs and the refit fits on this many.
const WINDOW: usize = 100;

/// Refit without a mean, as deployed, within a bounded number of iterations.
const OPTIONS: RefitOptions = RefitOptions {
    mean: false,
    max_iterations: 100,
};

/// The Ljung–Box lags of the drift check.
const LAGS: usize = 10;

/// The p-value below which the innovations count as no longer white. It is small
/// because the check runs every window: a refit is cheap, but each false alarm is one.
const ALPHA: f64 = 0.001;

fn main() {
    let series = simulate(1000);
    let mut refitter = Box::new(Refitter::<
        { DEPLOYED.refit_scratch_len(WINDOW, OPTIONS) },
        { DEPLOYED.coefficient_len() },
    >::new());
    // A refit's spec borrows the refitter, so the running model keeps its own copy.
    let mut ar = [DEPLOYED.ar[0]];
    let mut sigma2 = DEPLOYED.sigma2;
    let mut filter = Filter::<2>::new(&DEPLOYED).expect("the deployed model is stationary");
    let (mut window, mut standardised) = ([0.0; WINDOW], [0.0; WINDOW]);

    for (t, &y) in series.iter().enumerate() {
        let innovation = filter.update(y).expect("no differencing to prime");
        window[t % WINDOW] = y;
        standardised[t % WINDOW] = innovation.value / innovation.variance.sqrt();
        if t % WINDOW != WINDOW - 1 {
            continue;
        }

        let Some(check) = ljung_box(&standardised, LAGS, filter_parameters(&DEPLOYED)) else {
            continue;
        };

        if check.p_value >= ALPHA {
            println!("t = {t:4}: innovations white (p = {:.3})", check.p_value);
            continue;
        }

        let current = ArimaSpec {
            ar: &ar,
            sigma2,
            ..DEPLOYED
        };
        match refitter.refit(&current, &window, OPTIONS) {
            Ok(refit) => {
                ar.copy_from_slice(refit.spec.ar);
                sigma2 = refit.spec.sigma2;
                let mut refitted = Filter::<2>::new(&refit.spec).expect("refits are stationary");
                refitted.warm_up(&window);
                filter = refitted;
                println!(
                    "t = {t:4}: drift (p = {:.1e}), refitted ar = {:.3} in {} iterations",
                    check.p_value, ar[0], refit.iterations
                );
            }
            // The running filter stays in place, so a failed refit loses nothing.
            Err(error) => println!("t = {t:4}: drift, refit failed: {error}"),
        }
    }
}

/// The number of fitted ARMA coefficients, the degrees of freedom Ljung–Box removes.
fn filter_parameters(spec: &ArimaSpec<'_>) -> usize {
    spec.ar.len() + spec.ma.len() + spec.seasonal.ar.len() + spec.seasonal.ma.len()
}

/// An AR(1) with unit-variance noise whose coefficient jumps from 0.2 to 0.85 halfway.
fn simulate(n: usize) -> Vec<f64> {
    let mut state = 11_u64;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // The top 53 bits as a fraction in [0, 1).
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut normal = move || {
        // Box–Muller.
        let (u, v) = (uniform().max(f64::MIN_POSITIVE), uniform());
        (-2.0 * u.ln()).sqrt() * (core::f64::consts::TAU * v).cos()
    };

    let mut y = Vec::with_capacity(n);
    let mut previous = 0.0;
    for t in 0..n {
        let phi = if t < n / 2 { 0.2 } else { 0.85 };
        previous = phi * previous + normal();
        y.push(previous);
    }

    y
}
