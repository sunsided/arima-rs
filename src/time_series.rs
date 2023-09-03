use crate::Estimator;
use std::cell::Cell;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::ops::Deref;

#[derive(Debug, Clone)]
pub struct TimeSeries<'a> {
    values: &'a [f64],
    mean: Cell<f64>,
    var: Cell<f64>,
    std: Cell<f64>,
    orig_len: Cell<usize>,
    estimator: Estimator,
}

impl<'a> TimeSeries<'a> {
    /// Creates a new [`TimeSeries`] instance.
    ///
    /// ## Arguments
    /// * `values` - The time series data.
    /// * `estimator` - The estimator. Using [`Estimator::Unbiased`] calculates the population
    ///     statistics, while [`Estimator::Biased`] calculates the sample statistics.
    pub fn try_new(values: &'a [f64], estimator: Estimator) -> Result<Self, TimeSeriesError> {
        Self::try_new_with(
            values,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            values.len(),
            estimator,
        )
    }

    /// Creates a new [`TimeSeries`] instance.
    ///
    /// ## Arguments
    /// * `values` - The time series data.
    /// * `mean` - The pre-calculated mean.
    ///     Pass [`f64::NAN`], [`f64::INFINITY`] or [`f64::NEG_INFINITY`] to force calculation of the mean.
    /// * `var` - The pre-calculated variance.
    ///     Pass [`f64::NAN`], [`f64::INFINITY`] or [`f64::NEG_INFINITY`] to force calculation of the mean.
    /// * `std` - The pre-calculated standard deviation.
    ///     Pass [`f64::NAN`], [`f64::INFINITY`] or [`f64::NEG_INFINITY`] to force calculation of the mean.
    /// * `estimator` - The estimator. Using [`Estimator::Unbiased`] calculates the population
    ///     statistics, while [`Estimator::Biased`] calculates the sample statistics.
    pub fn try_new_with(
        values: &'a [f64],
        mean: f64,
        var: f64,
        std: f64,
        mut orig_len: usize,
        estimator: Estimator,
    ) -> Result<Self, TimeSeriesError> {
        if values.is_empty() {
            return Err(TimeSeriesError::NotEnoughData);
        }

        if orig_len == 0 {
            orig_len = values.len();
        }

        Ok(Self {
            values,
            mean: Cell::new(mean),
            var: Cell::new(var),
            std: Cell::new(std),
            orig_len: Cell::new(orig_len),
            estimator,
        })
    }

    /// Slices this time series assuming that it is a stationary series.
    ///
    /// This forces calculation of mean, variance and standard deviation if
    /// not already available from previous calculations in order to assure
    /// that these statistics remain the same regardless of the slice.
    pub fn slice_stationary<R: std::slice::SliceIndex<[f64], Output = [f64]>>(
        &self,
        range: R,
    ) -> Result<TimeSeries, TimeSeriesError> {
        TimeSeries::try_new_with(
            &self.values[range],
            self.mean(),
            self.var(),
            self.std(),
            self.orig_len(),
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

    pub fn orig_len(&self) -> usize {
        let orig_len = self.orig_len.get();
        if orig_len > 0 {
            return orig_len;
        }

        let orig_len = self.values.len();
        self.orig_len.set(orig_len);
        orig_len
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

        debug_assert_ne!(self.values.len(), 0);
        let sum: f64 = self.values.iter().sum();
        let mean = sum / self.values.len() as f64;
        self.mean.set(mean);
        mean
    }

    /// Obtains or calculates the variation.
    pub fn var(&self) -> f64 {
        let var = self.var.get();
        if var.is_finite() {
            return var;
        }

        let mean = self.mean();
        let tss: f64 = self
            .values
            .iter()
            .map(move |&x| (x - mean) * (x - mean))
            .sum();
        let var = match self.estimator {
            Estimator::Biased => tss / self.len() as f64,
            Estimator::Unbiased => tss / (self.len() - 1).max(1) as f64,
        };

        self.var.set(var);
        var
    }

    /// Obtains or calculates the standard deviation.
    pub fn std(&self) -> f64 {
        let std = self.std.get();
        if std.is_finite() {
            return std;
        }

        let var = self.var();
        let std = var.sqrt();
        self.std.set(std);
        std
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

impl<'a> Deref for TimeSeries<'a> {
    type Target = &'a [f64];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<'a> AsRef<&'a [f64]> for TimeSeries<'a> {
    fn as_ref(&self) -> &&'a [f64] {
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
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.mean();

        let expected: f64 = 6.0;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_var_pop() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.var();

        let expected: f64 = 10.0;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_std_pop() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_from(values).unwrap();

        let result = ts.std();

        let expected: f64 = 3.1622776601683795;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_var_sample() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_new(values, Estimator::Biased).unwrap();

        let result = ts.var();

        let expected: f64 = 8.0;
        assert_approx_eq!(f64, result, expected);
    }

    #[test]
    fn test_std_sample() {
        let values: &[f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
        let ts = TimeSeries::try_new(values, Estimator::Biased).unwrap();

        let result = ts.std();

        let expected: f64 = 2.8284271247461903;
        assert_approx_eq!(f64, result, expected);
    }
}
