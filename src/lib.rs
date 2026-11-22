pub use crate::time_series::{TimeSeries, TimeSeriesError};
use std::borrow::Borrow;

mod time_series;

/// Defines the estimator to use.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Estimator {
    /// Divides by `n`: the statistics of the data itself.
    Biased,
    /// Divides by `n - 1`: the sample estimate of the population statistics.
    Unbiased,
}

/// Splits a series into its first `len - lag` and last `len - lag` values.
///
/// Returns [`None`] if `lag` is not less than the number of elements.
fn lagged<'a>(values: &TimeSeries<'a>, lag: usize) -> Option<(TimeSeries<'a>, TimeSeries<'a>)> {
    if lag >= values.len() {
        return None;
    }

    let length = values.len() - lag;
    let x = values.slice_stationary(..length).ok()?;
    let y = values.slice_stationary(lag..).ok()?;
    Some((x, y))
}

/// Sums the products of the residuals of two equally long series.
fn cross_sum(x: &TimeSeries, y: &TimeSeries) -> f64 {
    x.residuals().zip(y.residuals()).map(|(x, y)| x * y).sum()
}

/// Calculates the autocovariance with a given lag.
///
/// ## Arguments
/// * `values` - The values to calculate the autocovariance for.
/// * `lag` - The lag of the data. The lag must be less than the number of elements.
///
/// ## Returns
/// [`None`] if the lag is too large or too few values remain for the estimator,
/// or some value representing the autocovariance.
pub fn acov<'a, T: Borrow<TimeSeries<'a>>>(values: T, lag: usize) -> Option<f64> {
    let (x, y) = lagged(values.borrow(), lag)?;
    cov(x, y)
}

/// Calculates the covariance.
///
/// ## Arguments
/// * `x` - The first signal.
/// * `y` - The second signal.
///
/// ## Returns
/// [`None`] if the signals differ in length or, with [`Estimator::Unbiased`], hold
/// fewer than two values; otherwise some value representing the covariance.
pub fn cov<'x, 'y, X: Borrow<TimeSeries<'x>>, Y: Borrow<TimeSeries<'y>>>(
    x: X,
    y: Y,
) -> Option<f64> {
    let x = x.borrow();
    let y = y.borrow();
    debug_assert_eq!(x.estimator(), y.estimator());

    if x.len() != y.len() {
        return None;
    }

    let divisor = match x.estimator() {
        Estimator::Unbiased => x.len() - 1,
        Estimator::Biased => x.len(),
    };
    if divisor == 0 {
        return None;
    }

    Some(cross_sum(x, y) / divisor as f64)
}

/// Calculates the autocorrelation coefficient with a given lag.
///
/// Uses the standard estimate `sum((x_t - mean) * (x_t+lag - mean)) / sum((x_t - mean)^2)`,
/// which always lies in `[-1, 1]`. The estimator of the series is not used.
///
/// ## Arguments
/// * `values` - The values to calculate the autocorrelation for. Assumed to be a stationary time series.
/// * `lag` - The lag of the data. The lag must be less than the number of elements.
///
/// ## Returns
/// [`None`] if the lag is too large or the series is constant,
/// or some value representing the autocorrelation.
pub fn acorr<'a, T: Borrow<TimeSeries<'a>>>(values: T, lag: usize) -> Option<f64> {
    let values = values.borrow();
    let (x, y) = lagged(values, lag)?;
    let tss = cross_sum(values, values);
    if tss == 0.0 {
        return None;
    }
    Some(cross_sum(&x, &y) / tss)
}

/// Calculates the correlation coefficient.
///
/// ## Arguments
/// * `x` - The first signal.
/// * `y` - The second signal.
///
/// ## Returns
/// [`None`] if the covariance is undefined or either signal is constant,
/// or some value representing the correlation coefficient.
pub fn corr<'x, 'y, X: Borrow<TimeSeries<'x>>, Y: Borrow<TimeSeries<'y>>>(
    x: X,
    y: Y,
) -> Option<f64> {
    let x = x.borrow();
    let y = y.borrow();
    let std = x.std() * y.std();
    cov(x, y).filter(|_| std != 0.0).map(|phi| phi / std)
}

/// Calculates the partial autocorrelation for lags `0..=max_lags`.
///
/// `max_lags` is capped at `len - 1`.
///
/// ## Returns
/// [`None`] if no lag can be computed (zero lags, a single value or a constant series),
/// or the partial autocorrelations, starting with `1.0` for lag 0.
pub fn pacf<'x, X: Borrow<TimeSeries<'x>>>(values: X, max_lags: usize) -> Option<Vec<f64>> {
    let values = values.borrow();
    let max_lags = max_lags.min(values.len() - 1);
    if max_lags == 0 {
        return None;
    }

    let acf = (1..=max_lags)
        .map(|lag| acorr(values, lag))
        .collect::<Option<Vec<_>>>()?;
    Some(durbin_levinson(&acf))
}

/// Runs the Durbin-Levinson recursion.
///
/// `acf[k - 1]` holds the autocorrelation at lag `k`. Returns `[1.0, phi_11, phi_22, ...]`.
fn durbin_levinson(acf: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(acf.len() + 1);
    result.push(1.0);

    // phi_{k-1, j} for j = 1..k-1, stored at index j - 1.
    let mut prev: Vec<f64> = Vec::with_capacity(acf.len());
    let mut curr: Vec<f64> = Vec::with_capacity(acf.len());

    for k in 1..=acf.len() {
        let num = acf[k - 1] - (1..k).map(|j| prev[j - 1] * acf[k - j - 1]).sum::<f64>();
        let den = 1.0 - (1..k).map(|j| prev[j - 1] * acf[j - 1]).sum::<f64>();
        let phi_kk = num / den;

        curr.clear();
        curr.extend((1..k).map(|j| prev[j - 1] - phi_kk * prev[k - j - 1]));
        curr.push(phi_kk);
        std::mem::swap(&mut prev, &mut curr);

        result.push(phi_kk);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_acov() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 2;

        let result = acov(values, lag).unwrap();

        let expected: f64 = -2.0;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_acov_2() {
        let values: &[f64] = &[
            0.05, 0.01, -0.02, 0.03, -0.04, 0.06, 0.02, -0.01, -0.03, 0.04,
        ];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 1;

        let result = acov(values, lag).unwrap();

        let expected: f64 = -0.0005176249999999999;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_cov() {
        let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Unbiased).unwrap();

        let x = values
            .slice_stationary(..3)
            .expect("time series has enough data");
        let y = values
            .slice_stationary(2..)
            .expect("time series has enough data");

        let result = cov(x, y).unwrap();

        assert_approx_eq!(f64, result, -2.0);

        // Because we sliced the time series as stationary, these values now exist.
        assert_approx_eq!(f64, values.mean(), 6.0);
        assert_approx_eq!(f64, values.var(), 10.0);
        assert_approx_eq!(f64, values.std(), 3.1622776601683795);
    }

    #[test]
    fn test_acorr() {
        let values: &[f64] = &[
            0.05, 0.01, -0.02, 0.03, -0.04, 0.06, 0.02, -0.01, -0.03, 0.04,
        ];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 1;

        let result = acorr(values, lag).unwrap();

        let expected: f64 = -0.380257116620753; // statsmodels acf
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_pacf() {
        let values = signal_gen(1000);
        let values = TimeSeries::try_new(&values, Estimator::Unbiased).unwrap();
        let max_lags = 10;

        let result = pacf(values, max_lags).unwrap();

        // statsmodels.tsa.stattools.pacf(y, nlags=10, method="ldb")
        let expected = [
            1.0,
            0.9992026377192385,
            -0.9886934907501577,
            0.6421864105266735,
            0.3483789857330886,
            0.18317292590771958,
            0.06350364686564525,
            -0.024538003187321095,
            -0.08462920394055298,
            -0.12177379316633868,
            -0.14225955547380964,
        ];
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected) {
            assert_approx_eq!(f64, *r, e, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_acorr_is_bounded() {
        // statsmodels acf
        let expected = [1.0, 0.4, -0.1, -0.4, -0.4];
        for estimator in [Estimator::Biased, Estimator::Unbiased] {
            let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], estimator).unwrap();
            for (lag, e) in expected.iter().enumerate() {
                assert_approx_eq!(f64, acorr(&values, lag).unwrap(), *e, epsilon = 1e-12);
            }
            assert_eq!(acorr(&values, 5), None);
        }
    }

    #[test]
    fn test_edge_cases_return_none() {
        let short = TimeSeries::try_new(&[1.0, 2.0], Estimator::Unbiased).unwrap();
        assert_eq!(acov(&short, 1), None);

        let constant = TimeSeries::try_new(&[3.0; 5], Estimator::Unbiased).unwrap();
        assert_eq!(acorr(&constant, 1), None);
        assert_eq!(corr(&constant, &constant), None);
        assert_eq!(pacf(&constant, 3), None);

        let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Unbiased).unwrap();
        assert_eq!(pacf(&values, 0), None);
        assert_eq!(cov(&short, &values), None);
    }

    #[test]
    fn test_pacf_caps_lags() {
        let values = TimeSeries::try_new(&[2.0, 4.0, 6.0, 8.0, 10.0], Estimator::Unbiased).unwrap();
        let result = pacf(&values, usize::MAX).unwrap();
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|r| r.is_finite()));
    }

    #[test]
    fn test_durbin_levinson_ar1() {
        // An AR(1) process with phi = 0.5 has acf 0.5^k and pacf [1, 0.5, 0, 0, ...].
        let acf: Vec<f64> = (1..=5).map(|k| 0.5_f64.powi(k)).collect();
        let result = durbin_levinson(&acf);
        let expected = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        for (r, e) in result.iter().zip(expected) {
            assert_approx_eq!(f64, *r, e, epsilon = 1e-12);
        }
    }

    fn signal_gen(len: usize) -> Vec<f64> {
        let f_base = 2.5;
        let a_base = 4.0;
        let f_overlay = 16.0;
        let a_overlay = 0.5;

        let scale = 1.0 / (len - 1) as f64;
        let t = (0..len)
            .map(|t| 2.0_f64 * t as f64 * scale)
            .collect::<Vec<f64>>();
        let y: Vec<_> = t
            .iter()
            .map(|t| {
                a_base * (2.0 * std::f64::consts::PI * f_base * t).sin()
                    + a_overlay * (2.0 * std::f64::consts::PI * f_overlay * t).sin()
            })
            .collect();
        y
    }
}
