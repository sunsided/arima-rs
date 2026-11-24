//! Exponential smoothing in the innovations state-space form (ETS), with a heap-free filter.
//!
//! The models are those of Hyndman, Koehler, Ord and Snyder (2008) as R's `forecast::ets`
//! fits them: a level, an optional additive or damped trend, and an optional additive season,
//! all updated by one source of error. [`EtsSpec`] holds the smoothing parameters and the
//! current states. [`EtsFilter`] carries them forward one value at a time without a heap,
//! as [`Filter`](crate::Filter) does for ARIMA, and forecasts with R's analytic intervals.
//!
//! The states are a level `l`, a slope `b` and `m` seasonal states. With `y` the new value
//! and `f = l + phi b + s_m` its one-step forecast, one step is
//!
//! ```text
//! l' = l + phi b + alpha (y - f)
//! b' = phi b + beta (y - f)
//! s' = s_m + gamma (y - f), and the other seasonal states move one place on
//! ```
//!
//! which is R's update in the error-correction form. `phi = 1` without damping.

mod recursion;

pub(crate) use recursion::Smoothing;
#[cfg(feature = "alloc")]
pub(crate) use recursion::unroll;

use crate::filter::FilterError;
use crate::math;
use crate::normal::normal_quantile;
use crate::state_space::Innovation;

/// The trend of an ETS model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Trend {
    /// No trend: the level carries the forecast.
    None,
    /// An additive trend with slope smoothing parameter `beta`.
    Additive {
        /// The slope smoothing parameter, in R's form (`beta < alpha`).
        beta: f64,
    },
    /// An additive trend damped by `phi`: the slope decays by `phi` each step.
    Damped {
        /// The slope smoothing parameter, in R's form (`beta < alpha`).
        beta: f64,
        /// The damping parameter, `0 < phi <= 1`; R fits it in `[0.8, 0.98]`.
        phi: f64,
    },
}

impl Trend {
    /// The slope smoothing parameter, or 0 without a trend.
    pub(crate) const fn beta(self) -> f64 {
        match self {
            Trend::None => 0.0,
            Trend::Additive { beta } | Trend::Damped { beta, .. } => beta,
        }
    }

    /// The damping parameter: `phi` for a damped trend, 1 otherwise.
    pub(crate) const fn phi(self) -> f64 {
        match self {
            Trend::Damped { phi, .. } => phi,
            Trend::None | Trend::Additive { .. } => 1.0,
        }
    }
}

/// An ETS model with its smoothing parameters and current states.
///
/// A fitted model gives its states after the last value
/// ([`EtsFit::spec`](crate::EtsFit::spec)), so a filter built from it continues where the
/// fit ended and needs no warm-up.
///
/// # Examples
///
/// ```
/// use backshift::{EtsFilter, EtsSpec, Trend};
///
/// // Holt's linear trend: level 10, slope 0.5.
/// let spec = EtsSpec {
///     alpha: 0.3,
///     trend: Trend::Additive { beta: 0.1 },
///     level: 10.0,
///     slope: 0.5,
///     ..EtsSpec::default()
/// };
/// let filter = EtsFilter::<1>::new(&spec).unwrap();
/// let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
/// filter.forecast_into(&mut mean, &mut variance);
/// assert_eq!(mean, [10.5, 11.0, 11.5]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EtsSpec<'a> {
    /// The level smoothing parameter, `0 < alpha < 1`.
    pub alpha: f64,
    /// The trend and its parameters.
    pub trend: Trend,
    /// The seasonal smoothing parameter `gamma`; unused without seasonal states.
    pub gamma: f64,
    /// The level.
    pub level: f64,
    /// The slope; unused without a trend.
    pub slope: f64,
    /// The additive seasonal states, most recent first: `seasonal[m - 1]` belongs to the
    /// next value. Empty for a model without a season; its length is the period `m`.
    pub seasonal: &'a [f64],
    /// The innovation variance. Must be positive and finite.
    pub sigma2: f64,
}

impl Default for EtsSpec<'_> {
    /// Simple exponential smoothing at level 0 with `alpha = 0.5` and unit variance.
    fn default() -> Self {
        Self {
            alpha: 0.5,
            trend: Trend::None,
            gamma: 0.0,
            level: 0.0,
            slope: 0.0,
            seasonal: &[],
            sigma2: 1.0,
        }
    }
}

impl EtsSpec<'_> {
    /// Checks that every parameter and state is finite and `sigma2` is positive.
    fn validate(&self) -> Result<(), FilterError> {
        let values = [
            self.alpha,
            self.trend.beta(),
            self.trend.phi(),
            self.gamma,
            self.level,
            self.slope,
            self.sigma2,
        ];
        let finite = values.iter().chain(self.seasonal).all(|v| v.is_finite());
        if !finite || self.sigma2 <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }

        Ok(())
    }

    /// The smoothing parameters as the recursion takes them.
    fn smoothing(&self) -> Smoothing {
        Smoothing {
            alpha: self.alpha,
            beta: self.trend.beta(),
            gamma: self.gamma,
            phi: self.trend.phi(),
            trend: !matches!(self.trend, Trend::None),
        }
    }
}

/// A streaming ETS filter with room for `M` seasonal states, stored inline with no heap.
///
/// `M` must be at least the period of the model (1 for a model without a season). The size
/// is about `8 (M + 9)` bytes, and each update costs `O(1)`: the seasonal states form a
/// ring, so no update shifts them.
///
/// # Examples
///
/// ```
/// use backshift::{EtsFilter, EtsSpec};
///
/// let spec = EtsSpec { alpha: 0.5, level: 10.0, ..EtsSpec::default() };
/// let mut filter = EtsFilter::<1>::new(&spec).unwrap();
/// let innovation = filter.update(12.0);
/// assert_eq!(innovation.value, 2.0);
/// assert_eq!(filter.level(), 11.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EtsFilter<const M: usize> {
    /// The smoothing parameters.
    smoothing: Smoothing,
    /// The innovation variance.
    sigma2: f64,
    /// The level.
    level: f64,
    /// The slope; 0 without a trend.
    slope: f64,
    /// The seasonal states, most recent first; the first `m` entries are used.
    seasonal: [f64; M],
    /// The period `m`; 0 without a season.
    m: usize,
    /// Where the most recent seasonal state sits in the ring `seasonal[..m]`.
    head: usize,
}

impl<const M: usize> EtsFilter<M> {
    /// Builds a filter at the states of `spec`. This is the device path: it does not
    /// allocate.
    ///
    /// # Errors
    ///
    /// - [`FilterError::InvalidParameter`] if a parameter or state is not finite or
    ///   `sigma2` is not positive.
    /// - [`FilterError::Capacity`] if the model has more seasonal states than `M`.
    pub fn new(spec: &EtsSpec<'_>) -> Result<Self, FilterError> {
        spec.validate()?;
        let m = spec.seasonal.len();
        if m > M {
            return Err(FilterError::Capacity {
                needed: m,
                capacity: M,
            });
        }

        let mut seasonal = [0.0; M];
        seasonal[..m].copy_from_slice(spec.seasonal);
        Ok(Self {
            smoothing: spec.smoothing(),
            sigma2: spec.sigma2,
            level: spec.level,
            slope: if matches!(spec.trend, Trend::None) {
                0.0
            } else {
                spec.slope
            },
            seasonal,
            m,
            head: 0,
        })
    }

    /// Feeds one value and returns its one-step forecast error and variance.
    ///
    /// A non-finite value counts as missing: its error is `NaN`, and the states move on as
    /// their forecast, as if the value had been forecast exactly.
    pub fn update(&mut self, y: f64) -> Innovation {
        let forecast = self
            .smoothing
            .forecast(self.level, self.slope, self.seasonal(), self.head);
        let innovation = Innovation {
            value: if y.is_finite() {
                y - forecast
            } else {
                f64::NAN
            },
            variance: self.sigma2,
        };
        let observed = if y.is_finite() { y } else { forecast };
        let m = self.m;
        self.smoothing.update(
            &mut self.level,
            &mut self.slope,
            &mut self.seasonal[..m],
            &mut self.head,
            observed,
        );

        innovation
    }

    /// The current level.
    #[must_use]
    pub const fn level(&self) -> f64 {
        self.level
    }

    /// Forecasts the next `mean.len()` values and their prediction variances.
    ///
    /// These are R's analytic forecasts for additive-error models (its class 1):
    /// `sigma2 (1 + c_1^2 + ... + c_(h-1)^2)` with `c_j = alpha + beta phi_j + gamma` where
    /// `j` is a multiple of `m`, and `phi_j = phi + ... + phi^j`.
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
        self.smoothing
            .forecast_means(self.level, self.slope, self.seasonal(), self.head, mean);
        self.smoothing
            .forecast_variances(self.m, self.sigma2, variance);
    }

    /// Forecasts the next `mean.len()` values with central prediction intervals at `level`:
    /// `mean ± z sqrt(variance)` with `z = Φ⁻¹((1 + level) / 2)`, as R's `forecast.ets`.
    ///
    /// # Panics
    ///
    /// Panics if `mean`, `lower` and `upper` do not all have the same length.
    pub fn forecast_interval_into(
        &self,
        level: f64,
        mean: &mut [f64],
        lower: &mut [f64],
        upper: &mut [f64],
    ) {
        assert!(
            mean.len() == lower.len() && mean.len() == upper.len(),
            "mean, lower and upper must have the same length"
        );

        // `upper` holds the variances until it is overwritten below.
        self.forecast_into(mean, upper);
        let z = normal_quantile(f64::midpoint(1.0, level));
        for ((m, lo), up) in mean.iter().zip(lower.iter_mut()).zip(upper.iter_mut()) {
            let half_width = z * math::sqrt(*up);
            *lo = m - half_width;
            *up = m + half_width;
        }
    }

    /// The ring of seasonal states in use; the most recent sits at `head`.
    fn seasonal(&self) -> &[f64] {
        &self.seasonal[..self.m]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_values_move_the_states_on_as_forecast() {
        let spec = EtsSpec {
            alpha: 0.4,
            trend: Trend::Damped {
                beta: 0.1,
                phi: 0.9,
            },
            level: 10.0,
            slope: 1.0,
            ..EtsSpec::default()
        };
        let mut filter = EtsFilter::<1>::new(&spec).unwrap();
        assert!(filter.update(f64::NAN).value.is_nan());
        assert!((filter.level() - 10.9).abs() < 1e-12);
        assert!(filter.update(f64::INFINITY).value.is_nan());
        assert!(filter.update(12.0).value.is_finite());
    }

    #[test]
    fn rejects_bad_models() {
        let seasonal = [1.0, -1.0];
        let spec = EtsSpec {
            seasonal: &seasonal,
            ..EtsSpec::default()
        };
        assert_eq!(
            EtsFilter::<1>::new(&spec).unwrap_err(),
            FilterError::Capacity {
                needed: 2,
                capacity: 1
            }
        );
        let bad = EtsSpec {
            sigma2: 0.0,
            ..EtsSpec::default()
        };
        assert_eq!(
            EtsFilter::<1>::new(&bad).unwrap_err(),
            FilterError::InvalidParameter
        );
    }
}
