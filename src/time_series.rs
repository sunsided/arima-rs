use crate::{math, Estimator};
use core::cell::Cell;
use core::error::Error;
use core::fmt::{Display, Formatter};
use core::ops::Deref;

#[derive(Debug, Clone)]
pub struct TimeSeries<'a> {
    values: &'a [f64],
    mean: Cell<f64>,
    var: Cell<f64>,
    estimator: Estimator,
}

impl<'a> TimeSeries<'a> {
    /// Creates a new [`TimeSeries`] instance.
    ///
    /// ## Arguments
    /// * `values` - The time series data.
    /// * `estimator` - The estimator. [`Estimator::Unbiased`] divides by `n - 1` (the sample
    ///   estimate of the population statistics), while [`Estimator::Biased`] divides by `n`.
    pub fn try_new(values: &'a [f64], estimator: Estimator) -> Result<Self, TimeSeriesError> {
        Self::try_new_with(values, f64::NAN, f64::NAN, estimator)
    }

    /// Creates a new [`TimeSeries`] instance.
    ///
    /// ## Arguments
    /// * `values` - The time series data.
    /// * `mean` - The pre-calculated mean.
    ///   Pass [`f64::NAN`], [`f64::INFINITY`] or [`f64::NEG_INFINITY`] to force calculation of the mean.
    /// * `var` - The pre-calculated variance.
    ///   Pass [`f64::NAN`], [`f64::INFINITY`] or [`f64::NEG_INFINITY`] to force calculation of the variance.
    /// * `estimator` - The estimator. [`Estimator::Unbiased`] divides by `n - 1` (the sample
    ///   estimate of the population statistics), while [`Estimator::Biased`] divides by `n`.
    pub fn try_new_with(
        values: &'a [f64],
        mean: f64,
        var: f64,
        estimator: Estimator,
    ) -> Result<Self, TimeSeriesError> {
        if values.is_empty() {
            return Err(TimeSeriesError::NotEnoughData);
        }

        Ok(Self {
            values,
            mean: Cell::new(mean),
            var: Cell::new(var),
            estimator,
        })
    }

    /// Slices this time series assuming that it is a stationary series.
    ///
    /// This forces calculation of mean and variance if
    /// not already available from previous calculations in order to assure
    /// that these statistics remain the same regardless of the slice.
    pub fn slice_stationary<R: core::slice::SliceIndex<[f64], Output = [f64]>>(
        &self,
        range: R,
    ) -> Result<TimeSeries<'a>, TimeSeriesError> {
        TimeSeries::try_new_with(&self.values[range], self.mean(), self.var(), self.estimator)
    }

    pub fn estimator(&self) -> Estimator {
        self.estimator
    }

    /// Obtains or calculates the mean.
    pub fn mean(&self) -> f64 {
        let mean = self.mean.get();
        if mean.is_finite() {
            return mean;
        }

        let sum: f64 = self.values.iter().sum();
        let mean = sum / self.values.len() as f64;
        self.mean.set(mean);
        mean
    }

    /// Obtains or calculates the variance.
    ///
    /// Returns [`f64::NAN`] for a single value with [`Estimator::Unbiased`],
    /// since `n - 1` is zero there.
    pub fn var(&self) -> f64 {
        let var = self.var.get();
        if var.is_finite() {
            return var;
        }

        let tss: f64 = self.residuals().map(|r| r * r).sum();
        let divisor = match self.estimator {
            Estimator::Biased => self.values.len(),
            Estimator::Unbiased => self.values.len() - 1,
        };
        if divisor == 0 {
            return f64::NAN;
        }

        let var = tss / divisor as f64;
        self.var.set(var);
        var
    }

    /// Calculates the standard deviation.
    pub fn std(&self) -> f64 {
        math::sqrt(self.var())
    }

    /// Calculates the residuals, i.e. the deviations from the mean.
    pub fn residuals(&self) -> impl Iterator<Item = f64> + 'a {
        let mean = self.mean();
        self.values.iter().map(move |x| x - mean)
    }
}

impl<'a> TryFrom<&'a [f64]> for TimeSeries<'a> {
    type Error = TimeSeriesError;

    fn try_from(value: &'a [f64]) -> Result<Self, Self::Error> {
        TimeSeries::try_new(value, Estimator::Unbiased)
    }
}

impl Deref for TimeSeries<'_> {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        self.values
    }
}

impl AsRef<[f64]> for TimeSeries<'_> {
    fn as_ref(&self) -> &[f64] {
        self.values
    }
}

#[derive(Debug, Clone)]
pub enum TimeSeriesError {
    NotEnoughData,
}

impl Display for TimeSeriesError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            TimeSeriesError::NotEnoughData => write!(f, "Not enough data"),
        }
    }
}

impl Error for TimeSeriesError {}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_mean() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.mean();

        let expected: f64 = 6.0;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_var_unbiased() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.var();

        let expected: f64 = 10.0;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_std_unbiased() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.std();

        let expected: f64 = 3.1622776601683795;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_var_biased() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_new(values, Estimator::Biased).unwrap();

        let result = ts.var();

        let expected: f64 = 8.0;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_std_biased() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_new(values, Estimator::Biased).unwrap();

        let result = ts.std();

        let expected: f64 = 2.8284271247461903;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_var_unbiased_single_value_is_nan() {
        let ts = TimeSeries::try_new(&[1.0], Estimator::Unbiased).unwrap();
        assert!(ts.var().is_nan());
    }
}
