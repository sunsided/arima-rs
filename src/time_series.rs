use crate::Estimator;
use std::cell::Cell;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct TimeSeries<'a> {
    values: &'a [f32],
    mean: Cell<f32>,
    var: Cell<f32>,
    std: Cell<f32>,
    estimator: Estimator,
}

impl<'a> TimeSeries<'a> {
    /// Creates a new [`TimeSeries`] instance.
    ///
    /// ## Arguments
    /// * `values` - The time series data.
    /// * `estimator` - The estimator. Using [`Estimator::Unbiased`] calculates the population
    ///     statistics, while [`Estimator::Biased`] calculates the sample statistics.
    pub fn try_new(values: &'a [f32], estimator: Estimator) -> Result<Self, TimeSeriesError> {
        Self::try_new_with(values, f32::NAN, f32::NAN, f32::NAN, estimator)
    }

    /// Creates a new [`TimeSeries`] instance.
    ///
    /// ## Arguments
    /// * `values` - The time series data.
    /// * `mean` - The pre-calculated mean.
    ///     Pass [`f32::NAN`], [`f32::INFINITY`] or [`f32::NEG_INFINITY`] to force calculation of the mean.
    /// * `var` - The pre-calculated variance.
    ///     Pass [`f32::NAN`], [`f32::INFINITY`] or [`f32::NEG_INFINITY`] to force calculation of the mean.
    /// * `std` - The pre-calculated standard deviation.
    ///     Pass [`f32::NAN`], [`f32::INFINITY`] or [`f32::NEG_INFINITY`] to force calculation of the mean.
    /// * `estimator` - The estimator. Using [`Estimator::Unbiased`] calculates the population
    ///     statistics, while [`Estimator::Biased`] calculates the sample statistics.
    pub fn try_new_with(
        values: &'a [f32],
        mean: f32,
        var: f32,
        std: f32,
        estimator: Estimator,
    ) -> Result<Self, TimeSeriesError> {
        if values.is_empty() {
            return Err(TimeSeriesError::NotEnoughData);
        }

        Ok(Self {
            values,
            mean: Cell::new(mean),
            var: Cell::new(var),
            std: Cell::new(std),
            estimator,
        })
    }

    /// Slices this time series assuming that it is a stationary series.
    ///
    /// This forces calculation of mean, variance and standard deviation if
    /// not already available from previous calculations in order to assure
    /// that these statistics remain the same regardless of the slice.
    pub fn slice_stationary<R: std::slice::SliceIndex<[f32], Output = [f32]>>(
        &self,
        range: R,
    ) -> Result<TimeSeries, TimeSeriesError> {
        TimeSeries::try_new_with(
            &self.values[range],
            self.mean(),
            self.var(),
            self.std(),
            self.estimator,
        )
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        debug_assert!(!self.values.is_empty());
        false
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn estimator(&self) -> Estimator {
        self.estimator
    }

    /// Obtains or calculates the mean.
    pub fn mean(&self) -> f32 {
        let mean = self.mean.get();
        if mean.is_finite() {
            return mean;
        }

        debug_assert_ne!(self.values.len(), 0);
        let sum: f32 = self.values.iter().sum();
        let mean = sum / self.values.len() as f32;
        self.mean.set(mean);
        mean
    }

    /// Obtains or calculates the variation.
    pub fn var(&self) -> f32 {
        let var = self.var.get();
        if var.is_finite() {
            return var;
        }

        let mean = self.mean();
        let tss: f32 = self.values.iter().map(|x| (x - mean) * (x - mean)).sum();
        let var = match self.estimator {
            Estimator::Biased => tss / self.len() as f32,
            Estimator::Unbiased => tss / (self.len() - 1).max(1) as f32,
        };

        self.var.set(var);
        var
    }

    /// Obtains or calculates the standard deviation.
    pub fn std(&self) -> f32 {
        let std = self.std.get();
        if std.is_finite() {
            return std;
        }

        let var = self.var();
        let std = var.sqrt();
        self.std.set(std);
        std
    }
}

impl<'a> TryFrom<&'a [f32]> for TimeSeries<'a> {
    type Error = TimeSeriesError;

    fn try_from(value: &'a [f32]) -> Result<Self, Self::Error> {
        TimeSeries::try_new(value, Estimator::Unbiased)
    }
}

impl<'a> Deref for TimeSeries<'a> {
    type Target = &'a [f32];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<'a> AsRef<&'a [f32]> for TimeSeries<'a> {
    fn as_ref(&self) -> &&'a [f32] {
        &self.values
    }
}

#[derive(Debug, Clone)]
pub enum TimeSeriesError {
    NotEnoughData,
}

impl Display for TimeSeriesError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
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
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.mean();

        let expected: f32 = 6.0;
        assert_approx_eq!(f32, result, expected);
    }

    #[test]
    fn test_var_pop() {
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.var();

        let expected: f32 = 10.0;
        assert_approx_eq!(f32, result, expected);
    }

    #[test]
    fn test_std_pop() {
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.std();

        let expected: f32 = 3.1622776601683795;
        assert_approx_eq!(f32, result, expected);
    }

    #[test]
    fn test_var_sample() {
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_new(values, Estimator::Biased).unwrap();

        let result = ts.var();

        let expected: f32 = 8.0;
        assert_approx_eq!(f32, result, expected);
    }

    #[test]
    fn test_std_sample() {
        let values: &[f32] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_new(values, Estimator::Biased).unwrap();

        let result = ts.std();

        let expected: f32 = 2.8284271247461903;
        assert_approx_eq!(f32, result, expected);
    }
}
