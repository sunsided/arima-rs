//! A heap-free streaming filter for a fitted ARMA model.
//!
//! [`Filter`] is built from coefficients alone, so a model fitted offline (here or
//! in statsmodels) runs on a device without an allocator. Each [`Filter::update`]
//! returns the one-step prediction error for the new value and advances the state;
//! [`Filter::forecast_into`] predicts the next values. Results match statsmodels
//! `SARIMAX(..., trend="n").filter(params)` and `results.extend(new_values)`.

use crate::state_space::{self, Innovation, LyapunovScratch, System};
use core::error::Error;
use core::fmt::{Display, Formatter};

/// The parameters of an ARMA(p, q) model with zero mean.
///
/// `y_t = phi_1 y_(t-1) + ... + phi_p y_(t-p) + e_t + theta_1 e_(t-1) + ... + theta_q e_(t-q)`,
/// with `Var(e_t) = sigma2`. This is the sign convention of statsmodels and R.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmaSpec<'a> {
    /// The AR coefficients `phi_1, ..., phi_p`. May be empty.
    pub ar: &'a [f64],
    /// The MA coefficients `theta_1, ..., theta_q`. May be empty.
    pub ma: &'a [f64],
    /// The innovation variance. Must be positive and finite.
    pub sigma2: f64,
}

impl ArmaSpec<'_> {
    /// The state dimension of this model, `max(p, q + 1)`.
    #[must_use]
    pub fn states(&self) -> usize {
        self.ar.len().max(self.ma.len() + 1)
    }

    /// Checks that every parameter is finite and `sigma2` is positive.
    pub(crate) fn validate(&self) -> Result<(), FilterError> {
        let finite = self.ar.iter().chain(self.ma).all(|c| c.is_finite());
        if !finite || !self.sigma2.is_finite() || self.sigma2 <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }

        Ok(())
    }

    /// Writes the padded first column of `T` and the selection vector into the first `r` entries.
    pub(crate) fn pad_into(&self, phi: &mut [f64], selection: &mut [f64]) {
        let r = self.states();
        phi[..r].fill(0.0);
        selection[..r].fill(0.0);
        phi[..self.ar.len()].copy_from_slice(self.ar);
        selection[0] = 1.0;
        selection[1..=self.ma.len()].copy_from_slice(self.ma);
    }
}

/// Why a model could not be turned into a filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterError {
    /// The model needs more states than the filter's capacity `R`.
    Capacity {
        /// The states the model needs, `max(p, q + 1)`.
        needed: usize,
        /// The filter's capacity `R`.
        capacity: usize,
    },
    /// A coefficient is not finite, or `sigma2` is not positive and finite.
    InvalidParameter,
    /// The AR part is not stationary, so the model has no unconditional covariance
    /// to start from.
    NonStationary,
}

impl Display for FilterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            FilterError::Capacity { needed, capacity } => {
                write!(f, "model needs {needed} states, filter holds {capacity}")
            }
            FilterError::InvalidParameter => {
                write!(f, "coefficients must be finite and sigma2 positive")
            }
            FilterError::NonStationary => write!(f, "the AR part is not stationary"),
        }
    }
}

impl Error for FilterError {}

/// A streaming ARMA filter with room for `R` states, stored inline with no heap.
///
/// `R` must be at least [`ArmaSpec::states`]. Its size is about `8 (R² + 3R)` bytes,
/// and [`Filter::new`] uses about three times that on the stack while it computes the
/// starting covariance. The filter is `Copy`, so it can be snapshotted cheaply.
///
/// # Examples
///
/// ```
/// use backshift::{ArmaSpec, Filter};
///
/// let spec = ArmaSpec { ar: &[0.6], ma: &[0.3], sigma2: 1.0 };
/// let mut filter = Filter::<4>::new(&spec).unwrap();
/// for y in [0.2, -0.1, 0.5] {
///     let innovation = filter.update(y);
///     assert!(innovation.variance >= 1.0);
/// }
///
/// let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
/// filter.forecast_into(&mut mean, &mut variance);
/// assert!(variance[0] <= variance[2]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Filter<const R: usize> {
    /// The model's state dimension, `1 <= r <= R`.
    r: usize,
    /// The padded first column of the transition matrix.
    phi: [f64; R],
    /// The padded selection vector `[1, theta_1, ...]`.
    selection: [f64; R],
    /// The innovation variance.
    sigma2: f64,
    /// The predicted state for the next observation.
    a: [f64; R],
    /// The predicted state covariance for the next observation; the top-left `r x r` block is used.
    p: [[f64; R]; R],
}

impl<const R: usize> Filter<R> {
    /// Builds a filter at the model's stationary start, before any observation.
    ///
    /// This is the device path: it needs only the coefficients and does not allocate.
    ///
    /// # Errors
    ///
    /// - [`FilterError::InvalidParameter`] if a coefficient is not finite or `sigma2 <= 0`.
    /// - [`FilterError::Capacity`] if the model needs more than `R` states.
    /// - [`FilterError::NonStationary`] if the AR part is not stationary.
    pub fn new(spec: &ArmaSpec<'_>) -> Result<Self, FilterError> {
        spec.validate()?;
        let r = spec.states();
        if r > R {
            return Err(FilterError::Capacity {
                needed: r,
                capacity: R,
            });
        }

        let mut filter = Self {
            r,
            phi: [0.0; R],
            selection: [0.0; R],
            sigma2: spec.sigma2,
            a: [0.0; R],
            p: [[0.0; R]; R],
        };
        spec.pad_into(&mut filter.phi, &mut filter.selection);

        let (mut power, mut product, mut term) = ([[0.0; R]; R], [[0.0; R]; R], [[0.0; R]; R]);
        let scratch = LyapunovScratch {
            power: power.as_flattened_mut(),
            product: product.as_flattened_mut(),
            term: term.as_flattened_mut(),
        };
        let (system, _, p) = filter.parts();
        if !state_space::stationary_covariance(system, p, R, scratch) {
            return Err(FilterError::NonStationary);
        }

        Ok(filter)
    }

    /// Feeds one observation and returns its one-step prediction error.
    ///
    /// A `NaN` observation counts as missing: the filter only moves forward in time,
    /// as statsmodels does, and the returned value is `NaN`.
    pub fn update(&mut self, y: f64) -> Innovation {
        let r = self.r;
        let mut scratch = [0.0; R];
        let (system, a, p) = self.parts();

        let innovation = state_space::update(r, a, p, R, y, &mut scratch);
        state_space::predict(system, a, p, R, &mut scratch);

        innovation
    }

    /// Feeds every value of `history` in order, discarding the innovations.
    pub fn warm_up(&mut self, history: &[f64]) {
        for &y in history {
            self.update(y);
        }
    }

    /// Forecasts the next `mean.len()` values and their prediction variances.
    ///
    /// `mean[h]` and `variance[h]` describe the value `h + 1` steps ahead.
    ///
    /// # Panics
    ///
    /// Panics if `mean` and `variance` have different lengths.
    pub fn forecast_into(&self, mean: &mut [f64], variance: &mut [f64]) {
        assert_eq!(
            mean.len(),
            variance.len(),
            "mean and variance must have the same length"
        );
        let mut ahead = *self;
        let mut scratch = [0.0; R];
        let (system, a, p) = ahead.parts();

        for (h, (m, v)) in mean.iter_mut().zip(variance.iter_mut()).enumerate() {
            if h > 0 {
                state_space::predict(system, a, p, R, &mut scratch);
            }

            *m = a[0];
            *v = p[0];
        }
    }

    /// Splits the filter into its system matrices (trimmed to `r`), the state and the flat covariance.
    fn parts(&mut self) -> (System<'_>, &mut [f64], &mut [f64]) {
        let system = System {
            phi: &self.phi[..self.r],
            selection: &self.selection[..self.r],
            sigma2: self.sigma2,
        };

        (system, &mut self.a[..self.r], self.p.as_flattened_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    /// A filter with capacity 4 for the given model.
    fn filter(ar: &[f64], ma: &[f64], sigma2: f64) -> Filter<4> {
        Filter::new(&ArmaSpec { ar, ma, sigma2 }).unwrap()
    }

    #[test]
    fn first_variance_is_the_unconditional_variance() {
        // AR(1): sigma2 / (1 - phi^2).
        let innovation = filter(&[0.6], &[], 2.0).update(1.0);
        assert_approx_eq!(
            f64,
            innovation.variance,
            2.0 / (1.0 - 0.36),
            epsilon = 1e-12
        );
        assert_approx_eq!(f64, innovation.value, 1.0, epsilon = 1e-15);

        // MA(1): sigma2 (1 + theta^2).
        let innovation = filter(&[], &[0.5], 2.0).update(1.0);
        assert_approx_eq!(f64, innovation.variance, 2.0 * 1.25, epsilon = 1e-12);
    }

    #[test]
    fn pure_ar_innovations_are_exact_after_p_values() {
        let (phi, y) = ([0.5, -0.3], [0.4, -1.2, 0.7, 2.0, -0.5, 0.1]);
        let mut f = filter(&phi, &[], 1.5);
        for t in 0..y.len() {
            let innovation = f.update(y[t]);
            if t >= phi.len() {
                let expected = y[t] - phi[0] * y[t - 1] - phi[1] * y[t - 2];
                assert_approx_eq!(f64, innovation.value, expected, epsilon = 1e-12);
                assert_approx_eq!(f64, innovation.variance, 1.5, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn missing_value_only_predicts() {
        let mut f = filter(&[0.6], &[], 1.0);
        f.update(1.0);
        let missing = f.update(f64::NAN);
        assert!(missing.value.is_nan());
        assert_approx_eq!(f64, missing.variance, 1.0, epsilon = 1e-12);

        // Two steps without data: the variance is sigma2 (1 + phi^2).
        let next = f.update(0.0);
        assert_approx_eq!(f64, next.variance, 1.0 + 0.36, epsilon = 1e-12);
        assert!(next.value.is_finite());
    }

    #[test]
    fn forecasts_decay_towards_zero_mean() {
        let mut f = filter(&[0.6], &[], 1.0);
        f.update(2.0);
        let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
        f.forecast_into(&mut mean, &mut variance);
        assert_approx_eq!(f64, mean[0], 1.2, epsilon = 1e-12);
        assert_approx_eq!(f64, mean[2], 2.0 * 0.6_f64.powi(3), epsilon = 1e-12);
        assert_approx_eq!(f64, variance[1], 1.0 + 0.36, epsilon = 1e-12);
    }

    #[test]
    fn rejects_bad_models() {
        let new = |ar: &[f64], ma: &[f64], sigma2: f64| {
            Filter::<2>::new(&ArmaSpec { ar, ma, sigma2 }).err()
        };
        assert_eq!(
            new(&[0.1, 0.1, 0.1], &[], 1.0),
            Some(FilterError::Capacity {
                needed: 3,
                capacity: 2
            })
        );
        assert_eq!(
            new(&[], &[0.1, 0.1], 1.0),
            Some(FilterError::Capacity {
                needed: 3,
                capacity: 2
            })
        );
        assert_eq!(new(&[0.5], &[], 0.0), Some(FilterError::InvalidParameter));
        assert_eq!(
            new(&[f64::NAN], &[], 1.0),
            Some(FilterError::InvalidParameter)
        );
        assert_eq!(new(&[1.0], &[], 1.0), Some(FilterError::NonStationary));
        assert_eq!(new(&[1.2], &[], 1.0), Some(FilterError::NonStationary));
        assert_eq!(new(&[0.5, 0.6], &[], 1.0), Some(FilterError::NonStationary));
        assert!(new(&[0.99], &[], 1.0).is_none());
    }
}
