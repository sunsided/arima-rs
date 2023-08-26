/// Defines the estimator to use.
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
pub fn acov(values: &[f32], lag: usize, mut mean: f32, estimator: Estimator) -> Option<f32> {
    if values.is_empty() {
        return None;
    }

    if lag >= values.len() {
        return None;
    }

    if !mean.is_finite() {
        // As AR(I)MA models are applied to stationary time series, the variance stays the
        // same regardless of the window we select. Because of that, and since the variance is based
        // on the mean, we can calculate the mean directly from the original sample instead of the
        // lagged versions.
        mean = mean_unchecked(&values);
    }

    // Obtain the lag-shifted signals.
    let length = values.len() - lag;
    let x = &values[..length];
    let y = &values[lag..];

    cov(x, y, mean, mean, estimator)
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
pub fn cov(
    x: &[f32],
    y: &[f32],
    mut mean_x: f32,
    mut mean_y: f32,
    estimator: Estimator,
) -> Option<f32> {
    if x.is_empty() || x.len() != y.len() {
        return None;
    }

    if !mean_x.is_finite() {
        mean_x = mean_unchecked(&x);
    }

    if !mean_y.is_finite() {
        mean_y = mean_unchecked(&y);
    }

    // Remove the mean from the original and the time-shifted signal.
    let x_error = x.iter().map(|x| x - mean_x);
    let y_error = y.iter().map(|x| x - mean_y);

    // Calculate the covariance.
    let tss: f32 = x_error.zip(y_error).map(|(x, y)| x * y).sum();
    if tss == 0.0 {
        return Some(0.0);
    }

    let length = x.len();
    let phi = match estimator {
        Estimator::Unbiased => tss / length as f32,
        Estimator::Biased => tss / ((length + 1) as f32),
    };
    Some(phi)
}

/// Calculates the mean of the values in the given slice.
/// Assumes that the slice is nonempty.
fn mean_unchecked(slice: &[f32]) -> f32 {
    debug_assert!(!slice.is_empty());
    let sum: f32 = slice.iter().sum();
    sum / slice.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_acov() {
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let lag: usize = 2;

        let result = acov(values, lag, f32::NAN, Estimator::Unbiased).unwrap();

        let expected: f32 = -1.3333334;
        assert_approx_eq!(f32, result, expected);
    }

    #[test]
    fn test_acov_2() {
        let values: &[f32] = &[
            0.05, 0.01, -0.02, 0.03, -0.04, 0.06, 0.02, -0.01, -0.03, 0.04,
        ];
        let lag: usize = 1;

        let result = acov(values, lag, f32::NAN, Estimator::Unbiased).unwrap();

        let expected: f32 = -0.00046011107;
        assert_approx_eq!(f32, result, expected);
    }

    #[test]
    fn test_cov() {
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let x: &[f32] = &values[0..3];
        let y: &[f32] = &values[2..];
        let mean = mean_unchecked(&values);

        let result = cov(x, y, mean, mean, Estimator::Unbiased).unwrap();

        let expected: f32 = -1.3333334;
        assert_approx_eq!(f32, result, expected);
    }
}
