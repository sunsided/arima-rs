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
/// * `mean` - The precalculated mean of `values`. Pass [`f32::NAN`], [`f32::INFINITY`] or [`f32::NEG_INFINITY`] to force calculation of the mean.
/// * `estimator` - Whether to use a biased or unbiased estimator.
///
/// ## Returns
/// [`None`] if the values input is empty, or some value representing the autocovariance.
pub fn acov<'a, T: Borrow<TimeSeries<'a>>>(values: T, lag: usize) -> Option<f32> {
    let values = values.borrow();
    if values.is_empty() {
        return None;
    }

    if lag >= values.len() {
        return None;
    }

    let mean = values.mean();

    // Obtain the lag-shifted signals.
    let length = values.len() - lag;
    let x = &values[..length];
    let y = &values[lag..];

    let x = TimeSeries::try_new_with(x, mean, f32::NAN, f32::NAN, values.estimator())
        .expect("time series has enough data points");
    let y = TimeSeries::try_new_with(y, mean, f32::NAN, f32::NAN, values.estimator())
        .expect("time series has enough data points");
    cov(x, y)
}

/// Calculates the covariance with a given lag.
///
/// ## Arguments
/// * `x` - The first signal.
/// * `y` - The second signal.
/// * `mean_x` - The mean of `x`. Pass [`f32::NAN`], [`f32::INFINITY`] or [`f32::NEG_INFINITY`] to force calculation of the mean.
/// * `mean_y` - The mean of `y`. Pass [`f32::NAN`], [`f32::INFINITY`] or [`f32::NEG_INFINITY`] to force calculation of the mean.
/// * `estimator` - Whether to use a biased or unbiased estimator.
///
/// ## Returns
/// [`None`] if the values input is empty, or some value representing the covariance.
pub fn cov<'x, 'y, X: Borrow<TimeSeries<'x>>, Y: Borrow<TimeSeries<'y>>>(
    x: X,
    y: Y,
) -> Option<f32> {
    let x = x.borrow();
    let y = y.borrow();

    if x.is_empty() || x.len() != y.len() {
        return None;
    }

    let mean_x = x.mean();
    let mean_y = y.mean();

    // Remove the mean from the original and the time-shifted signal.
    let x_error = x.iter().map(|x| x - mean_x);
    let y_error = y.iter().map(|x| x - mean_y);

    // Calculate the covariance.
    let tss: f32 = x_error.zip(y_error).map(|(x, y)| x * y).sum();
    if tss == 0.0 {
        return Some(0.0);
    }

    let length = x.len();
    let phi = match x.estimator() {
        Estimator::Unbiased => tss / length as f32,
        Estimator::Biased => tss / ((length + 1) as f32),
    };
    Some(phi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_acov() {
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 2;

        let result = acov(values, lag).unwrap();

        let expected: f32 = -1.3333334;
        assert_approx_eq!(f32, result, expected);
    }

    #[test]
    fn test_acov_2() {
        let values: &[f32] = &[
            0.05, 0.01, -0.02, 0.03, -0.04, 0.06, 0.02, -0.01, -0.03, 0.04,
        ];
        let values = TimeSeries::try_new(values, Estimator::Unbiased).unwrap();
        let lag: usize = 1;

        let result = acov(values, lag).unwrap();

        let expected: f32 = -0.00046011107;
        assert_approx_eq!(f32, result, expected);
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

        assert_approx_eq!(f32, result, -1.3333334);

        // Because we sliced the time series as stationary, these values now exist.
        assert_approx_eq!(f32, values.mean(), 6.0);
        assert_approx_eq!(f32, values.var(), 10.0);
        assert_approx_eq!(f32, values.std(), 3.1622776601683795);
    }
}
