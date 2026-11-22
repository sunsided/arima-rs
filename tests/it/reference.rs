//! Checks every statistic against the statsmodels vectors in [`crate::reference_data`].

use crate::reference_data::{ALL, Case};
use backshift::{ArCoefficients, Estimator, TimeSeries, acorr, acov, burg, pacf, yule_walker};

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

/// Asserts that a fit matches statsmodels' coefficients and innovation variance.
///
/// A `NaN` expected variance means statsmodels found `sigma2 <= 0`, so the fit must too.
fn assert_fit(what: &str, fit: &ArCoefficients, phi: &[f64], sigma2: f64) {
    assert_eq!(fit.phi.len(), phi.len(), "{what} order");
    for (lag, (&actual, &expected)) in fit.phi.iter().zip(phi).enumerate() {
        assert_close(what, lag + 1, actual, expected);
    }

    if sigma2.is_nan() {
        assert!(
            fit.sigma2 <= 0.0,
            "{what} sigma2: got {}, statsmodels <= 0",
            fit.sigma2
        );
    } else {
        // Relative, because the variances span 1e-6 to 10. 1e-8 rather than TOLERANCE: on the
        // near-deterministic sine series Burg's partial autocorrelations are close to +-1, and
        // its `(1 - phi^2) d - v^2 - u^2` update cancels about three digits.
        let error = (fit.sigma2 - sigma2).abs() / sigma2.abs();
        assert!(
            error <= 1e-8,
            "{what} sigma2: got {}, statsmodels {sigma2}",
            fit.sigma2
        );
    }
}

#[test]
fn yule_walker_matches_statsmodels() {
    for case in ALL {
        let mle = yule_walker(series(case, Estimator::Biased), case.ar_order).unwrap();
        assert_fit("yule_walker mle", &mle, case.yw_mle_phi, case.yw_mle_sigma2);

        let adjusted = yule_walker(series(case, Estimator::Unbiased), case.ar_order).unwrap();
        assert_fit(
            "yule_walker adjusted",
            &adjusted,
            case.yw_adjusted_phi,
            case.yw_adjusted_sigma2,
        );
    }
}

#[test]
fn burg_matches_statsmodels() {
    for case in ALL {
        let fit = burg(series(case, Estimator::Biased), case.ar_order).unwrap();
        assert_fit("burg", &fit, case.burg_phi, case.burg_sigma2);
    }
}
