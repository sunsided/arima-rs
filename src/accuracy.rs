//! Forecast accuracy measures, as R's `forecast::accuracy` computes them for a forecast
//! against the values that followed. Heap-free.

use crate::math;

/// The accuracy of a forecast against the actual values, as R's `accuracy` reports it on
/// the test set.
///
/// Errors are `actual - forecast`. The percentage measures divide by the actual values, so
/// they are not finite where an actual value is 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accuracy {
    /// The mean error (ME), the bias of the forecast.
    pub me: f64,
    /// The root mean squared error (RMSE).
    pub rmse: f64,
    /// The mean absolute error (MAE).
    pub mae: f64,
    /// The mean percentage error (MPE), in percent.
    pub mpe: f64,
    /// The mean absolute percentage error (MAPE), in percent.
    pub mape: f64,
    /// The mean absolute scaled error (MASE): the MAE divided by the mean absolute error of
    /// the seasonal naive forecast on the training values. Below 1, the forecast beats that
    /// naive forecast in sample.
    pub mase: f64,
}

/// Measures the accuracy of `forecast` against `actual`, with the MASE scaled by the
/// training values `training` and season length `period`, as R's
/// `accuracy(forecast, actual)` does for a forecast of `training`.
///
/// The scale of the MASE is the mean absolute difference of `training` at lag `period`
/// (lag 1 for a `period` of 0 or 1). Like R, the measures skip a pair where the actual
/// value or the forecast is not finite.
///
/// All measures are `NaN` if no pair is finite. The MASE is `NaN` if `training` has no
/// two finite values `period` apart, and infinite if its scale is 0.
///
/// # Examples
///
/// ```
/// use backshift::accuracy;
///
/// let training = [1.0, 2.0, 3.0, 4.0];
/// let scores = accuracy(&[5.0, 6.0], &[5.5, 6.5], &training, 1);
/// assert!((scores.mae - 0.5).abs() < 1e-12 && (scores.mase - 0.5).abs() < 1e-12);
/// ```
#[must_use]
pub fn accuracy(forecast: &[f64], actual: &[f64], training: &[f64], period: usize) -> Accuracy {
    let mut sums = [0.0; 5];
    let mut count = 0.0;
    for (f, a) in forecast.iter().zip(actual) {
        let error = a - f;
        if !error.is_finite() {
            continue;
        }

        let relative = 100.0 * error / a;
        sums[0] += error;
        sums[1] += error * error;
        sums[2] += error.abs();
        sums[3] += relative;
        sums[4] += relative.abs();
        count += 1.0;
    }

    let [me, mse, mae, mpe, mape] = sums.map(|sum| sum / count);
    Accuracy {
        me,
        rmse: math::sqrt(mse),
        mae,
        mpe,
        mape,
        mase: mae / naive_scale(training, period.max(1)),
    }
}

/// The mean absolute difference of the finite values of `training` at `lag`; `NaN` if
/// there is none.
fn naive_scale(training: &[f64], lag: usize) -> f64 {
    let (sum, count) = training
        .iter()
        .zip(training.iter().skip(lag))
        .map(|(earlier, later)| (later - earlier).abs())
        .filter(|difference| difference.is_finite())
        .fold((0.0, 0.0), |(sum, count), difference| {
            (sum + difference, count + 1.0)
        });

    sum / count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_missing_pairs_and_handles_empty_input() {
        let scores = accuracy(&[1.0, f64::NAN, 3.0], &[2.0, 5.0, 3.0], &[0.0, 2.0], 0);
        assert!((scores.me - 0.5).abs() < 1e-15);
        assert!((scores.mase - 0.25).abs() < 1e-15);

        let empty = accuracy(&[], &[], &[1.0], 12);
        assert!(empty.me.is_nan() && empty.mase.is_nan());
    }
}
