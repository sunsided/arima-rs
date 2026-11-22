//! Checks every statistic against the statsmodels vectors in [`crate::reference_data`].

use crate::reference_data::{Case, ALL};
use arima::{acorr, acov, pacf, Estimator, TimeSeries};

/// Absolute tolerance for comparisons with statsmodels.
///
/// Both sides sum the same terms in a possibly different order, so results
/// agree to a few ulps of the largest intermediate value, not exactly.
const TOLERANCE: f64 = 1e-9;

/// Asserts that `actual` and `expected` agree within [`TOLERANCE`], relative to the larger magnitude.
fn assert_close(what: &str, lag: usize, actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOLERANCE * scale,
        "{what} at lag {lag}: got {actual}, statsmodels {expected}"
    );
}

/// Builds a [`TimeSeries`] over the case's values with the given estimator.
fn series(case: &Case, estimator: Estimator) -> TimeSeries<'static> {
    TimeSeries::try_new(case.values, estimator).expect("reference series are non-empty")
}

#[test]
fn acorr_matches_statsmodels_acf() {
    for case in ALL {
        let ts = series(case, Estimator::Unbiased);
        for (lag, &expected) in case.acf.iter().enumerate() {
            assert_close("acorr", lag, acorr(&ts, lag).unwrap(), expected);
        }
    }
}

#[test]
fn pacf_matches_statsmodels_ldb() {
    for case in ALL {
        let ts = series(case, Estimator::Unbiased);
        let result = pacf(&ts, case.max_lags).unwrap();

        assert_eq!(result.len(), case.pacf.len());
        for (lag, (&actual, &expected)) in result.iter().zip(case.pacf).enumerate() {
            assert_close("pacf", lag, actual, expected);
        }
    }
}

#[test]
fn acov_matches_statsmodels_acovf() {
    for case in ALL {
        for (estimator, expected) in [
            (Estimator::Biased, case.acovf_biased),
            (Estimator::Unbiased, case.acovf_unbiased),
        ] {
            let ts = series(case, estimator);
            for (lag, &expected) in expected.iter().enumerate() {
                assert_close("acov", lag, acov(&ts, lag).unwrap(), expected);
            }
        }
    }
}

#[test]
fn var_matches_numpy() {
    for case in ALL {
        assert_close(
            "var biased",
            0,
            series(case, Estimator::Biased).var(),
            case.var_biased,
        );
        assert_close(
            "var unbiased",
            0,
            series(case, Estimator::Unbiased).var(),
            case.var_unbiased,
        );
    }
}
