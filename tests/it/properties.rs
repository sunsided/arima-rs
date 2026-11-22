//! Property tests: invariants that must hold for any non-constant series.

use backshift::{acorr, acov, corr, cov, difference, integrate, pacf, Estimator, TimeSeries};
use proptest::prelude::*;

/// Absolute slack for floating-point round-off in the invariants below.
const EPS: f64 = 1e-9;

/// Random series of 3 to 200 values in `[-1000, 1000]` that are not constant.
///
/// Constant series have no defined autocorrelation and are covered by unit tests.
fn series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1000.0..1000.0_f64, 3..200).prop_filter("non-constant", |v| {
        let first = v[0];
        v.iter().any(|&x| (x - first).abs() > 1e-6)
    })
}

/// Either estimator, so every property runs for both.
fn estimator() -> impl Strategy<Value = Estimator> {
    prop_oneof![Just(Estimator::Biased), Just(Estimator::Unbiased)]
}

/// Builds a [`TimeSeries`]; the strategies never produce empty input.
fn ts(values: &[f64], estimator: Estimator) -> TimeSeries<'_> {
    TimeSeries::try_new(values, estimator).expect("strategy yields non-empty series")
}

/// Converts a series length to `f64`; strategy lengths stay far below `u32::MAX`.
fn len_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("strategy lengths fit in u32"))
}

/// Asserts `|actual - expected| <= EPS * max(1, |expected|)`.
fn assert_close(actual: f64, expected: f64) -> Result<(), TestCaseError> {
    let scale = expected.abs().max(1.0);
    prop_assert!(
        (actual - expected).abs() <= EPS * scale,
        "got {actual}, expected {expected}"
    );
    Ok(())
}

proptest! {
    /// Autocorrelation is 1 at lag 0, lies in `[-1, 1]`, and is `None` from lag `n` on.
    #[test]
    fn acorr_is_bounded(values in series(), estimator in estimator()) {
        let x = ts(&values, estimator);
        assert_close(acorr(&x, 0).unwrap(), 1.0)?;

        for lag in 1..values.len() {
            let r = acorr(&x, lag).unwrap();
            prop_assert!(r.abs() <= 1.0 + EPS, "lag {lag}: {r}");
        }

        prop_assert_eq!(acorr(&x, values.len()), None);
    }

    /// Autocorrelation ignores shifting and scaling: `acorr(a x + b) == acorr(x)` for `a != 0`.
    #[test]
    fn acorr_is_affine_invariant(
        values in series(),
        a in prop_oneof![-10.0..-0.1_f64, 0.1..10.0_f64],
        b in -100.0..100.0_f64,
    ) {
        let scaled: Vec<f64> = values.iter().map(|v| a * v + b).collect();
        let x = ts(&values, Estimator::Biased);
        let y = ts(&scaled, Estimator::Biased);

        for lag in 0..values.len().min(20) {
            let expected = acorr(&x, lag).unwrap();
            let actual = acorr(&y, lag).unwrap();
            prop_assert!((actual - expected).abs() <= 1e-6, "lag {lag}: {actual} vs {expected}");
        }
    }

    /// With the biased estimator, autocovariance normalised by lag 0 is the autocorrelation.
    #[test]
    fn biased_acov_ratio_is_acorr(values in series()) {
        let x = ts(&values, Estimator::Biased);
        let c0 = acov(&x, 0).unwrap();

        for lag in 0..values.len() {
            assert_close(acov(&x, lag).unwrap() / c0, acorr(&x, lag).unwrap())?;
        }
    }

    /// The two autocovariance estimators differ only by their divisors, `n` and `n - lag`.
    #[test]
    fn acov_estimators_differ_by_divisor(values in series()) {
        let n = values.len();
        let biased = ts(&values, Estimator::Biased);
        let unbiased = ts(&values, Estimator::Unbiased);

        for lag in 0..n {
            let from_biased = acov(&biased, lag).unwrap() * len_f64(n);
            let from_unbiased = acov(&unbiased, lag).unwrap() * len_f64(n - lag);
            assert_close(from_unbiased, from_biased)?;
        }

        prop_assert_eq!(acov(&biased, n), None);
    }

    /// The covariance of a series with itself is its variance, and its correlation is 1.
    #[test]
    fn self_covariance_is_variance(values in series(), estimator in estimator()) {
        let x = ts(&values, estimator);
        assert_close(cov(&x, &x).unwrap(), x.var())?;
        assert_close(corr(&x, &x).unwrap(), 1.0)?;
    }

    /// Correlation with `a x + b` is the sign of `a`.
    #[test]
    fn corr_with_affine_copy_is_sign(
        values in series(),
        a in prop_oneof![-10.0..-0.1_f64, 0.1..10.0_f64],
        b in -100.0..100.0_f64,
        estimator in estimator(),
    ) {
        let scaled: Vec<f64> = values.iter().map(|v| a * v + b).collect();
        let r = corr(ts(&values, estimator), ts(&scaled, estimator)).unwrap();
        prop_assert!((r - a.signum()).abs() <= 1e-6, "{r}");
    }

    /// Variance estimators differ only by their divisors, `n` and `n - 1`.
    #[test]
    fn var_estimators_differ_by_divisor(values in series()) {
        let n = len_f64(values.len());
        let biased = ts(&values, Estimator::Biased).var() * n;
        let unbiased = ts(&values, Estimator::Unbiased).var() * (n - 1.0);
        assert_close(unbiased, biased)?;
    }

    /// Stationary slices keep the parent's mean and variance.
    #[test]
    fn slice_stationary_keeps_statistics(values in series(), estimator in estimator(), cut in 1usize..3) {
        let x = ts(&values, estimator);
        let slice = x.slice_stationary(cut..).unwrap();
        prop_assert_eq!(slice.len(), values.len() - cut);
        // The slice copies the cached values, so they must match bit for bit.
        prop_assert_eq!(slice.mean().to_bits(), x.mean().to_bits());
        prop_assert_eq!(slice.var().to_bits(), x.var().to_bits());
    }

    /// PACF has `min(max_lags, n - 1) + 1` entries, starts with 1 and the lag-1 autocorrelation,
    /// and stays in `[-1, 1]`.
    #[test]
    fn pacf_shape_and_bounds(values in series(), max_lags in 1usize..30, estimator in estimator()) {
        let x = ts(&values, estimator);
        let result = pacf(&x, max_lags).unwrap();

        prop_assert_eq!(result.len(), max_lags.min(values.len() - 1) + 1);
        prop_assert_eq!(result[0].to_bits(), 1.0_f64.to_bits());
        assert_close(result[1], acorr(&x, 1).unwrap())?;
        for (lag, phi) in result.iter().enumerate() {
            prop_assert!(phi.abs() <= 1.0 + 1e-6, "lag {lag}: {phi}");
        }
    }

    /// One pass of differencing is exactly `x[t + 1] - x[t]`, the same operation as `numpy.diff`.
    #[test]
    fn difference_is_exact(values in series()) {
        let expected: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        let once = difference(&values, 1);
        prop_assert_eq!(&once, &expected);
        prop_assert_eq!(difference(&values, 2), difference(&once, 1));
    }

    /// Integrating the differences after any split point restores the rest of the series.
    #[test]
    fn integrate_inverts_difference(values in series(), d in 0usize..4, split in 0usize..200) {
        prop_assume!(d < values.len());
        let split = d + split % (values.len() - d);
        let differenced = difference(&values, d);
        prop_assert_eq!(differenced.len(), values.len() - d);

        let restored = integrate(&values[..split], &differenced[split - d..], d).unwrap();
        prop_assert_eq!(restored.len(), values.len() - split);
        let scale = values.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (actual, expected) in restored.iter().zip(&values[split..]) {
            prop_assert!((actual - expected).abs() <= 1e-9 * scale * len_f64(values.len()), "{actual} vs {expected}");
        }
    }
}
