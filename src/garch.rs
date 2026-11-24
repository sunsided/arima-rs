//! GARCH(1,1) conditional variance (Bollerslev 1986), with a heap-free filter.
//!
//! The variance of each value given the ones before it follows
//!
//! ```text
//! h_t = omega + alpha x_(t-1)^2 + beta h_(t-1)
//! ```
//!
//! for a zero-mean series `x`: returns, or the innovations of an ARIMA or ETS filter. The
//! variance forecast `k` steps ahead decays towards `omega / (1 - alpha - beta)`:
//! `h_(t+k) = omega + (alpha + beta) h_(t+k-1)`.

use crate::filter::FilterError;
use crate::state_space::Innovation;

/// A GARCH(1,1) model at its current state.
///
/// # Examples
///
/// ```
/// use backshift::{GarchFilter, GarchSpec};
///
/// let spec = GarchSpec { omega: 0.1, alpha: 0.1, beta: 0.8, variance: 1.0 };
/// let mut filter = GarchFilter::new(&spec).unwrap();
/// let innovation = filter.update(2.0);
/// assert_eq!(innovation.variance, 1.0);
/// assert!((filter.variance() - (0.1 + 0.1 * 4.0 + 0.8)).abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GarchSpec {
    /// The constant `omega > 0`.
    pub omega: f64,
    /// The weight `alpha >= 0` of the last squared value.
    pub alpha: f64,
    /// The weight `beta >= 0` of the last variance; `alpha + beta < 1`.
    pub beta: f64,
    /// The conditional variance of the next value, `> 0`.
    pub variance: f64,
}

/// A streaming GARCH(1,1) filter: the conditional variance of each value given the ones
/// before it, with `O(1)` work and no heap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GarchFilter {
    /// The constant.
    omega: f64,
    /// The weight of the last squared value.
    alpha: f64,
    /// The weight of the last variance.
    beta: f64,
    /// The conditional variance of the next value.
    variance: f64,
}

impl GarchFilter {
    /// Builds a filter at the state of `spec`. This is the device path: it does not allocate.
    ///
    /// # Errors
    ///
    /// [`FilterError::InvalidParameter`] if a value is not finite, `omega` or `variance` is
    /// not positive, `alpha` or `beta` is negative, or `alpha + beta >= 1` (the variance
    /// would have no finite long-run level).
    pub fn new(spec: &GarchSpec) -> Result<Self, FilterError> {
        let finite = [spec.omega, spec.alpha, spec.beta, spec.variance]
            .iter()
            .all(|v| v.is_finite());
        let valid = finite
            && spec.omega > 0.0
            && spec.variance > 0.0
            && spec.alpha >= 0.0
            && spec.beta >= 0.0
            && spec.alpha + spec.beta < 1.0;
        if !valid {
            return Err(FilterError::InvalidParameter);
        }

        Ok(Self {
            omega: spec.omega,
            alpha: spec.alpha,
            beta: spec.beta,
            variance: spec.variance,
        })
    }

    /// Feeds one value and returns it with its conditional variance, the one before the
    /// update; `value / sqrt(variance)` is the standardised value.
    ///
    /// A value that is not finite, or whose square is not, counts as missing: its innovation
    /// is `NaN` and the variance moves to its expectation, `omega + (alpha + beta) h`.
    pub fn update(&mut self, x: f64) -> Innovation {
        let square = x * x;
        let observed = square.is_finite();
        let innovation = Innovation {
            value: if observed { x } else { f64::NAN },
            variance: self.variance,
        };
        let shock = if observed { square } else { self.variance };
        self.variance = self.omega + self.alpha * shock + self.beta * self.variance;

        innovation
    }

    /// Feeds `history` through [`GarchFilter::update`], discarding the innovations.
    pub fn warm_up(&mut self, history: &[f64]) {
        for &x in history {
            self.update(x);
        }
    }

    /// Forecasts the conditional variances of the next `variance.len()` values.
    pub fn forecast_into(&self, variance: &mut [f64]) {
        let persistence = self.alpha + self.beta;
        let mut h = self.variance;
        for value in variance.iter_mut() {
            *value = h;
            h = self.omega + persistence * h;
        }
    }

    /// The conditional variance of the next value.
    #[must_use]
    pub const fn variance(&self) -> f64 {
        self.variance
    }

    /// The long-run variance `omega / (1 - alpha - beta)` the forecasts decay towards.
    #[must_use]
    pub fn unconditional_variance(&self) -> f64 {
        self.omega / (1.0 - self.alpha - self.beta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spec with typical weights and the given next variance.
    const fn spec(variance: f64) -> GarchSpec {
        GarchSpec {
            omega: 0.2,
            alpha: 0.1,
            beta: 0.7,
            variance,
        }
    }

    #[test]
    fn forecasts_decay_to_the_long_run_variance() {
        let filter = GarchFilter::new(&spec(5.0)).unwrap();
        let mut variance = [0.0; 200];
        filter.forecast_into(&mut variance);
        assert_eq!(variance[0], 5.0);
        assert!((variance[1] - (0.2 + 0.8 * 5.0)).abs() < 1e-15);
        assert!((variance[199] - filter.unconditional_variance()).abs() < 1e-12);
    }

    #[test]
    fn missing_and_overflowing_values_use_the_expectation() {
        for missing in [f64::NAN, f64::INFINITY, 1e200] {
            let mut filter = GarchFilter::new(&spec(2.0)).unwrap();
            assert!(filter.update(missing).value.is_nan());
            assert!((filter.variance() - (0.2 + 0.8 * 2.0)).abs() < 1e-15);
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        let bad = [
            GarchSpec {
                omega: 0.0,
                ..spec(1.0)
            },
            GarchSpec {
                alpha: -0.1,
                ..spec(1.0)
            },
            GarchSpec {
                beta: 0.9,
                ..spec(1.0)
            },
            GarchSpec {
                variance: f64::NAN,
                ..spec(1.0)
            },
        ];
        for spec in bad {
            assert_eq!(GarchFilter::new(&spec), Err(FilterError::InvalidParameter));
        }
    }
}
