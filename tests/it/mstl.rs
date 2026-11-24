//! Checks the MSTL decomposition against statsmodels.

use crate::mstl_data::ALL;
use backshift::{AutoArima, FitError, mstl};

/// Absolute tolerance against statsmodels, as for STL.
const TOLERANCE: f64 = 1e-9;

/// Asserts two equally long slices agree within [`TOLERANCE`].
fn assert_close(what: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= TOLERANCE, "{what}[{t}]: {a} vs {e}");
    }
}

#[test]
fn matches_statsmodels() {
    for case in ALL {
        let parts = mstl(case.values, case.periods, case.iterate).unwrap();
        assert_eq!(parts.seasonal.len(), case.kept);
        let n = case.values.len();
        for (i, component) in parts.seasonal.iter().enumerate() {
            let what = format!("periods {:?} seasonal {i}", case.periods);
            assert_close(&what, component, &case.seasonal[i * n..(i + 1) * n]);
        }

        assert_close("trend", &parts.trend, case.trend);
        assert_close("remainder", &parts.remainder, case.remainder);
    }
}

#[test]
fn rejects_what_statsmodels_cannot_decompose() {
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert!(mstl(&y, &[4], 2).is_none(), "4 is half the length");
    assert!(mstl(&y, &[1], 2).is_none());
    assert!(mstl(&y, &[2], 0).is_none());
    let mut gap = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    gap[4] = f64::INFINITY;
    assert!(mstl(&gap, &[2], 2).is_none());
}

/// A series with periods 7 and 30 on AR(1) noise: `n` values plus `horizon` held out.
fn two_seasons(n: usize) -> Vec<f64> {
    let weekly = [2.0, 1.0, 0.0, -0.5, -1.0, -2.0, 0.5];
    let mut state = 3_u32;
    let mut ar = 0.0;
    (0..n)
        .map(|t| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ar = 0.5 * ar + f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5;
            let monthly = 3.0 * (std::f64::consts::TAU * (t % 30) as f64 / 30.0).sin();
            10.0 + weekly[t % 7] + monthly + ar
        })
        .collect()
}

#[test]
fn mstl_forecast_tracks_both_seasons() {
    let (n, horizon) = (360, 30);
    let y = two_seasons(n + horizon);
    let model = AutoArima::new()
        .max_orders(2, 2, 0, 0)
        .fit_mstl(&y[..n], &[30, 7])
        .unwrap();
    let forecast = model.forecast::<8>(0.95, horizon).unwrap();

    let error: f64 = forecast
        .mean
        .iter()
        .zip(&y[n..])
        .map(|(f, a)| (f - a).abs())
        .sum::<f64>()
        / horizon as f64;
    // The noise has a standard deviation of about 0.33; the seasons swing by +-5.
    assert!(error < 0.6, "mean absolute error {error}");

    let covered = (0..horizon)
        .filter(|&h| forecast.lower[h] <= y[n + h] && y[n + h] <= forecast.upper[h])
        .count();
    assert!(
        covered >= 25,
        "{covered} of {horizon} inside the 95% interval"
    );
}

#[test]
fn mstl_fit_rejects_missing_values_and_small_filters() {
    let mut y = two_seasons(120);
    let model = AutoArima::new().max_orders(1, 1, 0, 0);
    assert!(
        model
            .fit_mstl(&y, &[7])
            .unwrap()
            .forecast::<0>(0.9, 3)
            .is_err()
    );
    y[10] = f64::NAN;
    assert_eq!(model.fit_mstl(&y, &[7]), Err(FitError::Decomposition));
}
