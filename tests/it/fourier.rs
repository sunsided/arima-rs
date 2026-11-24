//! Checks the Fourier regressors against statsmodels and through a fit and a forecast.

use crate::fourier_data::ALL;
use backshift::{Arima, AutoArima, Filter, FitError, fourier, fourier_row};

/// Absolute tolerance against statsmodels, whose angles are not reduced modulo the period.
const TOLERANCE: f64 = 1e-9;

#[test]
fn rows_match_statsmodels() {
    for case in ALL {
        let mut row = vec![0.0; case.columns];
        for (i, expected) in case.values.chunks_exact(case.columns).enumerate() {
            let t = case.t0 + i as u64;
            fourier_row(case.period, case.k, t, &mut row).unwrap();
            for (j, (a, e)) in row.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() < TOLERANCE,
                    "period {} t {t} column {j}: {a} vs {e}",
                    case.period
                );
            }
        }

        let batch = fourier(
            &[(case.period, case.k)],
            case.t0,
            case.values.len() / case.columns,
        );
        assert!(
            batch
                .unwrap()
                .iter()
                .zip(case.values)
                .all(|(a, e)| (a - e).abs() < TOLERANCE)
        );
    }
}

#[test]
fn fit_and_forecast_with_fourier_regressors() {
    // y_t = 3 sin(2 pi t / 12.5) - 2 cos(2 pi t / 12.5) + AR(1) noise, at a non-integer period.
    let (period, n, horizon) = (12.5, 400, 25);
    let mut state = 11_u32;
    let mut noise = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5
    };
    let x = fourier(&[(period, 1)], 0, n + horizon).unwrap();
    let mut ar = 0.0;
    let y: Vec<f64> = x
        .chunks_exact(2)
        .map(|row| {
            ar = 0.5 * ar + noise();
            3.0 * row[0] - 2.0 * row[1] + ar
        })
        .collect();

    let fitted = Arima::new(1, 0, 0)
        .fit_exog(&y[..n], &x[..2 * n], 2)
        .unwrap();
    assert!((fitted.exog[0] - 3.0).abs() < 0.1, "{:?}", fitted.exog);
    assert!((fitted.exog[1] + 2.0).abs() < 0.1, "{:?}", fitted.exog);

    let mut filter = Filter::<1, 2>::new(&fitted.spec()).unwrap();
    filter.warm_up_exog(&y[..n], &x[..2 * n]);
    let (mut mean, mut variance) = ([0.0; 25], [0.0; 25]);
    filter.forecast_exog_into(&x[2 * n..], &mut mean, &mut variance);
    // Far ahead, the forecast is the seasonal pattern alone.
    let seasonal = 3.0 * x[2 * (n + 24)] - 2.0 * x[2 * (n + 24) + 1];
    assert!(
        (mean[24] - seasonal).abs() < 0.2,
        "{} vs {seasonal}",
        mean[24]
    );
}

#[test]
fn aicc_selects_the_generating_harmonics() {
    // Two harmonics at period 7 and one at period 30.5, on AR(1) noise.
    let n = 300;
    let mut state = 5_u32;
    let mut noise = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5
    };
    let x = fourier(&[(7.0, 2), (30.5, 1)], 0, n).unwrap();
    let mut ar = 0.0;
    let y: Vec<f64> = x
        .chunks_exact(6)
        .map(|row| {
            ar = 0.4 * ar + noise();
            2.0 * row[0] + row[1] - 1.0 * row[2] + 0.8 * row[3] + 1.5 * row[4] - 0.5 * row[5] + ar
        })
        .collect();

    let best = AutoArima::new()
        .max_orders(2, 2, 0, 0)
        .fit_fourier(&y, &[(7.0, 3), (30.5, 2)])
        .unwrap();
    assert_eq!(best.terms, [(7.0, 2), (30.5, 1)]);
    assert_eq!(best.fit.fitted.exog.len(), 6);
    assert_eq!(best.candidates, 1 + 2 + 1);
}

#[test]
fn fourier_search_rejects_bad_periods() {
    let y = [1.0; 30];
    assert!(matches!(
        AutoArima::new().fit_fourier(&y, &[(1.5, 1)]),
        Err(FitError::Fourier(_))
    ));
}

#[test]
fn fourier_fit_matches_its_terms() {
    // Whatever the search picks, refitting the returned terms gives the returned fit.
    let y: Vec<f64> = (0..200)
        .map(|t| (std::f64::consts::TAU * f64::from(t) / 9.0).cos() + 0.3 * f64::from(t % 4))
        .collect();
    let search = AutoArima::new().max_orders(1, 1, 0, 0);
    let best = search.fit_fourier(&y, &[(9.0, 2), (4.5, 2)]).unwrap();
    let x = fourier(&best.terms, 0, y.len()).unwrap();
    let again = search.fit_exog(&y, &x, best.fit.fitted.exog.len()).unwrap();
    assert_eq!(again.fitted, best.fit.fitted);
}
