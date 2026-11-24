//! The Theta method (Assimakopoulos and Nikolopoulos 2000) as R's `forecast::thetaf` forecasts
//! it, with a heap-free filter.
//!
//! Hyndman and Billah (2003) showed it is simple exponential smoothing with a drift of half
//! the slope of the series. On a seasonal series it works on the values divided by fixed
//! multiplicative seasonal indices, then multiplies the forecasts by them. The forecast `h`
//! steps ahead of `n` values is
//!
//! ```text
//! (l + b (h - 1 + (1 - (1 - alpha)^n) / alpha)) s_h
//! ```
//!
//! with the smoothed level `l`, the drift `b` and the index `s_h`; its variance is
//! `sigma2 (1 + (h - 1) alpha^2)`, which R does not scale by the index.

use crate::filter::FilterError;
use crate::math;
use crate::normal::normal_quantile;
use crate::state_space::Innovation;

/// A Theta model at its current state.
///
/// # Examples
///
/// ```
/// use backshift::{ThetaFilter, ThetaSpec};
///
/// let spec = ThetaSpec { alpha: 0.5, drift: 0.2, level: 10.0, count: 50, sigma2: 1.0, seasonal: &[] };
/// let filter = ThetaFilter::<1>::new(&spec).unwrap();
/// let (mut mean, mut variance) = ([0.0; 2], [0.0; 2]);
/// filter.forecast_into(&mut mean, &mut variance);
/// assert!((mean[1] - mean[0] - 0.2).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThetaSpec<'a> {
    /// The smoothing parameter of the level, `0 < alpha <= 1`.
    pub alpha: f64,
    /// The drift `b`, half the least-squares slope of the (deseasonalised) series.
    pub drift: f64,
    /// The smoothed level after the last value.
    pub level: f64,
    /// How many values the level has seen, `n`.
    pub count: usize,
    /// The innovation variance of the smoothing.
    pub sigma2: f64,
    /// The multiplicative seasonal indices of the next `m` values, next first; empty
    /// without a season.
    pub seasonal: &'a [f64],
}

/// A streaming Theta filter with room for `M` seasonal indices, stored inline with no heap.
///
/// The indices stay fixed, as R estimates them once; the level keeps learning.
#[derive(Debug, Clone, Copy)]
pub struct ThetaFilter<const M: usize> {
    /// The smoothing parameter.
    alpha: f64,
    /// The drift.
    drift: f64,
    /// The smoothed level of the deseasonalised values.
    level: f64,
    /// How many values the level has seen.
    count: usize,
    /// The innovation variance.
    sigma2: f64,
    /// The seasonal indices, a ring with the index of the next value at `head`.
    seasonal: [f64; M],
    /// The period; 0 without a season.
    m: usize,
    /// Where the index of the next value sits.
    head: usize,
}

impl<const M: usize> ThetaFilter<M> {
    /// Builds a filter at the state of `spec`. This is the device path: it does not allocate.
    ///
    /// # Errors
    ///
    /// - [`FilterError::InvalidParameter`] if a value is not finite, `alpha` is not in
    ///   `(0, 1]`, `sigma2` is not positive, or a seasonal index is not positive.
    /// - [`FilterError::Capacity`] if there are more seasonal indices than `M`.
    pub fn new(spec: &ThetaSpec<'_>) -> Result<Self, FilterError> {
        let finite = [spec.alpha, spec.drift, spec.level, spec.sigma2]
            .iter()
            .all(|v| v.is_finite());
        let valid = finite
            && spec.alpha > 0.0
            && spec.alpha <= 1.0
            && spec.sigma2 > 0.0
            && spec.seasonal.iter().all(|s| s.is_finite() && *s > 0.0);
        if !valid {
            return Err(FilterError::InvalidParameter);
        }

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
            alpha: spec.alpha,
            drift: spec.drift,
            level: spec.level,
            count: spec.count,
            sigma2: spec.sigma2,
            seasonal,
            m,
            head: 0,
        })
    }

    /// Feeds one value and returns its error against the one-step forecast, with the
    /// variance `sigma2`.
    ///
    /// The level learns from the value divided by its seasonal index. A non-finite value
    /// counts as missing: its error is `NaN`, the level stays, and time moves on.
    pub fn update(&mut self, y: f64) -> Innovation {
        let mut forecast = [0.0; 1];
        self.means_into(&mut forecast);
        let innovation = Innovation {
            value: if y.is_finite() {
                y - forecast[0]
            } else {
                f64::NAN
            },
            variance: self.sigma2,
        };
        if y.is_finite() {
            let adjusted = y / self.index(0);
            self.level += self.alpha * (adjusted - self.level);
        }

        self.count = self.count.saturating_add(1);
        if self.m > 0 {
            self.head = (self.head + 1) % self.m;
        }

        innovation
    }

    /// Forecasts the next `mean.len()` values and their variances, as R's `thetaf`.
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
        self.means_into(mean);
        let alpha2 = self.alpha * self.alpha;
        let mut steps = 0.0;
        for value in variance.iter_mut() {
            *value = self.sigma2 * (1.0 + steps * alpha2);
            steps += 1.0;
        }
    }

    /// Forecasts with central prediction intervals at `level`, `mean ± z sqrt(variance)`.
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

    /// The current smoothed level.
    #[must_use]
    pub const fn level(&self) -> f64 {
        self.level
    }

    /// Writes the forecast means `(l + b (h - 1 + w)) s_h` with `w = (1 - (1 - alpha)^n) /
    /// alpha`.
    fn means_into(&self, mean: &mut [f64]) {
        // `(1 - alpha)^n` through `exp(n ln(1 - alpha))`; with no values it is 1 (and
        // `0 ln 0` would be `NaN` at `alpha = 1`).
        let decay = if self.count == 0 {
            1.0
        } else {
            math::exp(count_as_f64(self.count) * math::ln(1.0 - self.alpha))
        };
        let weight = (1.0 - decay) / self.alpha;
        let mut steps = 0.0;
        for (h, value) in mean.iter_mut().enumerate() {
            *value = (self.level + self.drift * (steps + weight)) * self.index(h);
            steps += 1.0;
        }
    }

    /// The seasonal index `h` values after the next one, or 1 without a season.
    fn index(&self, h: usize) -> f64 {
        if self.m == 0 {
            1.0
        } else {
            self.seasonal[(self.head + h) % self.m]
        }
    }
}

/// A count as a float; exact up to 2^53 values, far beyond any series.
#[expect(
    clippy::cast_precision_loss,
    reason = "value counts stay far below 2^53, where the conversion is exact"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}
