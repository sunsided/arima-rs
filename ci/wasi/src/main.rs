//! Exercises the fit, the streaming filter, a refit and a diagnostic test on a WASI
//! target without threads, and checks the results.

use backshift::{Arima, AutoArima, Filter, RefitOptions, ljung_box};

fn main() {
    let y = series(240);

    let best = AutoArima::new().fit(&y).expect("the search finds a model");
    let fitted = Arima::new(1, 0, 1).fit(&y).expect("the series fits");
    assert!(
        fitted.ar[0].abs() < 1.0 && fitted.sigma2 > 0.0,
        "{fitted:?}"
    );

    let mut filter = Filter::<4>::new(&fitted.spec()).expect("the model fits the filter");
    filter.warm_up(&y);
    let innovation = filter.update(0.5).expect("no differencing to prime");
    assert!(innovation.variance > 0.0);
    let (mut mean, mut lower, mut upper) = ([0.0; 12], [0.0; 12], [0.0; 12]);
    filter.forecast_interval_into(0.95, &mut mean, &mut lower, &mut upper);
    assert!(
        mean.iter()
            .zip(&lower)
            .zip(&upper)
            .all(|((m, l), u)| l < m && m < u)
    );

    let spec = fitted.spec();
    let options = RefitOptions {
        mean: true,
        ..RefitOptions::default()
    };
    let mut coefficients = vec![0.0; spec.coefficient_len()];
    let mut scratch = vec![0.0; spec.refit_scratch_len(120, options)];
    let refit = spec
        .refit(&y[120..], options, &mut coefficients, &mut scratch)
        .expect("the window fits");
    assert!(refit.loglike.is_finite());

    let residuals: Vec<f64> = backshift::innovations(&y, &spec)
        .expect("the model is valid")
        .iter()
        .map(|i| i.value / i.variance.sqrt())
        .collect();
    let test = ljung_box(&residuals, 10, 2).expect("enough residuals");
    assert!((0.0..=1.0).contains(&test.p_value));

    println!(
        "wasi: ok ({} by search, AR {:.3} after refit, Ljung-Box p = {:.3})",
        best.fitted.ar.len(),
        refit.spec.ar[0],
        test.p_value
    );
}

/// A seeded ARMA(1, 1) series around 5.
fn series(n: usize) -> Vec<f64> {
    let mut state = 7_u32;
    let mut noise = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5
    };
    let (mut y, mut level, mut previous) = (Vec::with_capacity(n), 0.0, 0.0);
    for _ in 0..n {
        let e = noise();
        level = 0.6 * level + e + 0.3 * previous;
        previous = e;
        y.push(5.0 + level);
    }

    y
}
