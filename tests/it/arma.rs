//! Checks the ARMA filter and likelihood against the statsmodels vectors in [`crate::arma_data`].

use crate::arma_data::{ALL, ArmaCase, SPLIT};
use backshift::{ArmaSpec, Filter, Innovation, innovations, loglike};

/// Relative tolerance against statsmodels.
const TOLERANCE: f64 = 1e-9;

/// The capacity used for the fixed-size filter: larger than every case's state dimension,
/// so the stride differs from the runtime path's.
const CAPACITY: usize = 8;

/// Asserts that `actual` matches `expected` within [`TOLERANCE`], relative to `max(1, |expected|)`.
/// Two `NaN`s match.
fn assert_close(what: &str, index: usize, actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{what}[{index}]: got {actual}, statsmodels NaN"
        );
        return;
    }

    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= TOLERANCE * scale,
        "{what}[{index}]: got {actual}, statsmodels {expected}"
    );
}

/// Asserts innovations match statsmodels' errors and error variances.
fn assert_innovations(what: &str, actual: &[Innovation], values: &[f64], variances: &[f64]) {
    assert_eq!(actual.len(), values.len(), "{what} length");
    for (t, innovation) in actual.iter().enumerate() {
        assert_close(what, t, innovation.value, values[t]);
        assert_close(what, t, innovation.variance, variances[t]);
    }
}

/// The model of a case.
fn spec(case: &ArmaCase) -> ArmaSpec<'static> {
    ArmaSpec {
        ar: case.ar,
        ma: case.ma,
        sigma2: case.sigma2,
    }
}

/// A fixed-size filter at the stationary start of a case's model.
fn filter(case: &ArmaCase) -> Filter<CAPACITY> {
    Filter::new(&spec(case)).expect("reference models are stationary and fit")
}

#[test]
fn loglike_matches_statsmodels() {
    for case in ALL {
        assert_close(
            "loglike",
            0,
            loglike(case.values, &spec(case)).unwrap(),
            case.loglike,
        );
    }
}

#[test]
fn innovations_match_statsmodels() {
    for case in ALL {
        let result = innovations(case.values, &spec(case)).unwrap();
        assert_innovations(
            "innovations",
            &result,
            case.innovation_values,
            case.innovation_variances,
        );
    }
}

#[test]
fn fixed_filter_equals_runtime_path_bit_for_bit() {
    for case in ALL {
        let runtime = innovations(case.values, &spec(case)).unwrap();
        let mut fixed = filter(case);
        for (t, (&y, expected)) in case.values.iter().zip(&runtime).enumerate() {
            let actual = fixed.update(y);
            assert_eq!(
                actual.value.to_bits(),
                expected.value.to_bits(),
                "value at {t}"
            );
            assert_eq!(
                actual.variance.to_bits(),
                expected.variance.to_bits(),
                "variance at {t}"
            );
        }
    }
}

#[test]
fn streaming_matches_statsmodels_extend() {
    for case in ALL {
        let mut streaming = filter(case);
        streaming.warm_up(&case.values[..SPLIT]);
        let tail: Vec<Innovation> = case.values[SPLIT..]
            .iter()
            .map(|&y| streaming.update(y))
            .collect();
        assert_innovations("extend", &tail, case.extend_values, case.extend_variances);
    }
}

#[test]
fn forecasts_match_statsmodels() {
    for case in ALL {
        let mut streaming = filter(case);
        streaming.warm_up(case.values);

        let horizon = case.forecast_mean.len();
        let (mut mean, mut variance) = (vec![0.0; horizon], vec![0.0; horizon]);
        streaming.forecast_into(&mut mean, &mut variance);
        for h in 0..horizon {
            assert_close("forecast mean", h, mean[h], case.forecast_mean[h]);
            assert_close(
                "forecast variance",
                h,
                variance[h],
                case.forecast_variance[h],
            );
        }
    }
}
