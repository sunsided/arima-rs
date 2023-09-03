pub use crate::time_series::{TimeSeries, TimeSeriesError};
use std::borrow::Borrow;

mod time_series;

/// Defines the estimator to use.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Estimator {
    /// A biased estimator (sample based).
    Biased,
    /// An unbiased estimator (population based).
    Unbiased,
}

/// Calculates the autocovariance with a given lag.
///
/// ## Arguments
/// * `values` - The values to calculate the autocovariance for.
/// * `lag` - The lag of the data. The lag must be less than the number of elements.
///
/// ## Returns
/// [`None`] if the values input is empty, or some value representing the autocovariance.
pub fn acov<'a, T: Borrow<TimeSeries<'a>>>(values: T, lag: usize) -> Option<f64> {
    let values = values.borrow();
    if values.is_empty() {
        return None;
    }

    if lag >= values.len() {
        return None;
    }

    // Obtain the lag-shifted signals.
    let length = values.len() - lag;
    let x = values
        .slice_stationary(..length)
        .expect("time series has enough data points");
    let y = values
        .slice_stationary(lag..)
        .expect("time series has enough data points");
    debug_assert_eq!(x.len(), length);
    debug_assert_eq!(y.len(), length);

    cov(x, y)
}

/// Calculates the covariance.
///
/// ## Arguments
/// * `x` - The first signal.
/// * `y` - The second signal.
///
/// ## Returns
/// [`None`] if the values input is empty, or some value representing the covariance.
pub fn cov<'x, 'y, X: Borrow<TimeSeries<'x>>, Y: Borrow<TimeSeries<'y>>>(
    x: X,
    y: Y,
) -> Option<f64> {
    let x = x.borrow();
    let y = y.borrow();
    assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.estimator(), y.estimator());

    if x.is_empty() || x.len() != y.len() {
        return None;
    }

    // Remove the mean from the original and the time-shifted signal.
    let x_error = x.residuals();
    let y_error = y.residuals();

    // Calculate the covariance.
    let tss: f64 = x_error.zip(y_error).map(|(x, y)| x * y).sum();

    let length = x.len();
    let phi = match x.estimator() {
        Estimator::Unbiased => tss / (length - 1) as f64,
        Estimator::Biased => tss / length as f64,
    };
    Some(phi)
}

/// Calculates the autocorrelation coefficient with a given lag.
///
/// ## Arguments
/// * `values` - The values to calculate the autocorrelation for. Assumed to be a stationary time series.
/// * `lag` - The lag of the data. The lag must be less than the number of elements.
///
/// ## Returns
/// [`None`] if the values input is empty, or some value representing the autocorrelation.
pub fn acorr<'a, T: Borrow<TimeSeries<'a>>>(values: T, lag: usize) -> Option<f64> {
    let values = values.borrow();
    if values.is_empty() {
        return None;
    }

    if lag >= values.len() {
        return None;
    }

    // Obtain the lag-shifted signals.
    let length = values.len() - lag;
    let x = values
        .slice_stationary(..length)
        .expect("time series has enough data points");
    let y = values
        .slice_stationary(lag..)
        .expect("time series has enough data points");

    corr(x, y)
}

/// Calculates the correlation coefficient.
///
/// ## Arguments
/// * `x` - The first signal.
/// * `y` - The second signal.
///
/// ## Returns
/// [`None`] if the values input is empty, or some value representing the correlation coefficient.
pub fn corr<'x, 'y, X: Borrow<TimeSeries<'x>>, Y: Borrow<TimeSeries<'y>>>(
    x: X,
    y: Y,
) -> Option<f64> {
    let x = x.borrow();
    let y = y.borrow();
    match cov(x, y) {
        None => None,
        Some(phi) => Some(phi / (x.std() * y.std())),
    }
}

/// Calculates the partial autocorrelation.
pub fn pacf<'x, X: Borrow<TimeSeries<'x>>>(values: X, mut max_lags: usize) -> Option<Vec<f64>> {
    let values = values.borrow();
    if values.is_empty() {
        return None;
    }

    max_lags = values.len().min(max_lags);

    // Pre-calculate the auto-correlation values.
    let mut acf = vec![f64::NAN; max_lags];
    for lag in 0..max_lags {
        acf[lag] = acorr(values, lag + 1).unwrap_or(0.0);
    }

    let mut pacf = vec![f64::NAN; max_lags * max_lags]; // square matrix

    // pacf[..max_lags].copy_from_slice(&acf);
    pacf[0] = acf[0];

    let idx = move |row, col| (row - 1) * max_lags + (col - 1); // MATLAB indexing

    for p in 2..=max_lags {
        let mut term = 0.0;
        let mut term2 = 0.0;
        for j in 1..=(p - 1) {
            // Durbin recursion
            let mut current = pacf[idx(p - 1, j)];
            if current.is_nan() {
                let a = pacf[idx(p - 2, j)];
                let b = pacf[idx(p - 1, p - 1)];
                let c = pacf[idx(p - 2, p - 1 - j)];

                current = a - b * c;
                pacf[idx(p - 1, j)] = current;
            }

            term += current * acf[p - j - (1)];
            term2 += current * acf[j - (1)];
        }

        let d = acf[p - (1)];
        let e = d - term;
        let f = 1.0 - term2;
        let value = e / f;
        pacf[idx(p, p)] = value;
    }

    let diag = std::iter::once(1.0)
        .chain((0..max_lags).map(|lag| pacf[lag * max_lags + lag]))
        .collect();
    Some(diag)
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

        let expected: f64 = -0.42778925619834696;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_pacf() {
        let values = signal_gen(1000);
        let values = TimeSeries::try_new(&values, Estimator::Unbiased).unwrap();
        let max_lags = 10;

        let result = pacf(values, max_lags).unwrap();

        let expected = [
            1.0,
            1.0002038427670525,
            3.870781320499196,
            -1.3414313085242502,
            -0.5864136419375363,
            -0.39778721734123756,
            -0.3281943113554122,
            -0.3108059438334501,
            -0.3322635124458095,
            -0.4044370178199057,
            -0.5921770731565059,
        ];
        assert_eq!(result, expected);
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
