//! A heap-free streaming filter for a fitted ARIMA model.
//!
//! [`Filter`] is built from coefficients alone, so a model fitted offline (here or
//! in statsmodels) runs on a device without an allocator. Each [`Filter::update`]
//! takes a raw value, differences it `d` times, removes the mean, and returns the
//! one-step prediction error of the result; [`Filter::forecast_into`] predicts the
//! next raw values with their variances.
//!
//! Conventions follow statsmodels with simple differencing, like R's `arima`:
//!
//! - Innovations and the likelihood are those of the ARMA model of
//!   `diff(y, d) - mean`, so the first `d` values only prime the differencing.
//! - `mean` is the mean of the differenced series: statsmodels ARIMA `trend="c"`
//!   for `d = 0` and the drift `trend="t"` for `d = 1`.
//! - Forecasts and their variances are on the original scale and match statsmodels
//!   `ARIMA(y, order=(p, d, q), trend=...).filter(params).get_forecast(h)`.

use crate::math;
use crate::normal::normal_quantile;
use crate::state_space::{self, Innovation, LyapunovScratch, System};
use core::error::Error;
use core::fmt::{Display, Formatter};

/// The parameters of an ARIMA(p, d, q) model.
///
/// With `w_t = (1 - B)^d y_t - mean`, the differenced series follows
/// `w_t = phi_1 w_(t-1) + ... + phi_p w_(t-p) + e_t + theta_1 e_(t-1) + ... + theta_q e_(t-q)`,
/// with `Var(e_t) = sigma2`. This is the sign convention of statsmodels and R.
///
/// # Examples
///
/// ```
/// use backshift::ArimaSpec;
///
/// // ARIMA(1, 1, 1) with drift 0.2.
/// let spec = ArimaSpec { ar: &[0.5], d: 1, ma: &[0.3], mean: 0.2, sigma2: 1.0 };
/// assert_eq!(spec.states(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArimaSpec<'a> {
    /// The AR coefficients `phi_1, ..., phi_p`. May be empty.
    pub ar: &'a [f64],
    /// How often the series is differenced before the ARMA part applies.
    pub d: usize,
    /// The MA coefficients `theta_1, ..., theta_q`. May be empty.
    pub ma: &'a [f64],
    /// The mean of the differenced series: the mean for `d = 0`, the drift for `d = 1`.
    pub mean: f64,
    /// The innovation variance. Must be positive and finite.
    pub sigma2: f64,
}

impl ArimaSpec<'_> {
    /// The smallest capacity `R` a [`Filter`] needs for this model: `max(p, q + 1) + d`.
    #[must_use]
    pub fn states(&self) -> usize {
        self.arma_states() + self.d
    }

    /// The state dimension of the ARMA part, `max(p, q + 1)`.
    pub(crate) fn arma_states(&self) -> usize {
        self.ar.len().max(self.ma.len() + 1)
    }

    /// Checks that every parameter is finite and `sigma2` is positive.
    pub(crate) fn validate(&self) -> Result<(), FilterError> {
        let finite = self.ar.iter().chain(self.ma).all(|c| c.is_finite());
        if !finite || !self.mean.is_finite() || !self.sigma2.is_finite() || self.sigma2 <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }

        Ok(())
    }

    /// Writes the padded first column of `T` and the selection vector into the first `r` entries.
    pub(crate) fn pad_into(&self, phi: &mut [f64], selection: &mut [f64]) {
        let r = self.arma_states();
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
        /// The states the model needs, [`ArimaSpec::states`].
        needed: usize,
        /// The filter's capacity `R`.
        capacity: usize,
    },
    /// A coefficient or the mean is not finite, or `sigma2` is not positive and finite.
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
                write!(
                    f,
                    "coefficients and mean must be finite and sigma2 positive"
                )
            }
            FilterError::NonStationary => write!(f, "the AR part is not stationary"),
        }
    }
}

impl Error for FilterError {}

/// A streaming ARIMA filter with room for `R` states, stored inline with no heap.
///
/// `R` must be at least [`ArimaSpec::states`]. Its size is about `8 (R² + 4R)` bytes.
/// [`Filter::new`] uses about three times that on the stack while it computes the
/// starting covariance, and [`Filter::forecast_into`] about twice. The filter is
/// `Copy`, so it can be snapshotted cheaply.
///
/// # Examples
///
/// ```
/// use backshift::{ArimaSpec, Filter};
///
/// let spec = ArimaSpec { ar: &[0.6], d: 1, ma: &[0.3], mean: 0.0, sigma2: 1.0 };
/// let mut filter = Filter::<4>::new(&spec).unwrap();
/// assert!(filter.update(10.0).is_none()); // primes the differencing
/// for y in [10.2, 10.1, 10.5] {
///     let innovation = filter.update(y).unwrap();
///     assert!(innovation.variance >= 1.0);
/// }
///
/// let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
/// filter.forecast_into(&mut mean, &mut variance);
/// assert!(variance[0] < variance[2]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Filter<const R: usize> {
    /// The ARMA state dimension, `1 <= r` and `r + d <= R`.
    r: usize,
    /// The padded first column of the transition matrix.
    phi: [f64; R],
    /// The padded selection vector `[1, theta_1, ...]`.
    selection: [f64; R],
    /// The innovation variance.
    sigma2: f64,
    /// The predicted ARMA state for the next differenced, demeaned value.
    a: [f64; R],
    /// The predicted ARMA state covariance; the top-left `r x r` block is used.
    p: [[f64; R]; R],
    /// The differencing order.
    d: usize,
    /// The mean of the differenced series.
    mean: f64,
    /// `levels[j]` is the last raw value differenced `j` times, for `j < d`.
    levels: [f64; R],
    /// How many raw values were fed, capped at `d + 1`. Values beyond `d` produce innovations.
    seen: usize,
}

impl<const R: usize> Filter<R> {
    /// Builds a filter at the model's stationary start, before any observation.
    ///
    /// This is the device path: it needs only the coefficients and does not allocate.
    ///
    /// # Errors
    ///
    /// - [`FilterError::InvalidParameter`] if a coefficient or the mean is not finite,
    ///   or `sigma2 <= 0`.
    /// - [`FilterError::Capacity`] if the model needs more than `R` states.
    /// - [`FilterError::NonStationary`] if the AR part is not stationary.
    pub fn new(spec: &ArimaSpec<'_>) -> Result<Self, FilterError> {
        spec.validate()?;
        if spec.states() > R {
            return Err(FilterError::Capacity {
                needed: spec.states(),
                capacity: R,
            });
        }

        let mut filter = Self {
            r: spec.arma_states(),
            phi: [0.0; R],
            selection: [0.0; R],
            sigma2: spec.sigma2,
            a: [0.0; R],
            p: [[0.0; R]; R],
            d: spec.d,
            mean: spec.mean,
            levels: [0.0; R],
            seen: 0,
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

    /// Feeds one raw observation and returns the one-step prediction error of its
    /// differenced, demeaned value.
    ///
    /// Returns [`None`] for the first `d` observations, which only prime the
    /// differencing. A `NaN` observation counts as missing, as in statsmodels: its
    /// innovation value is `NaN` and the filter only moves forward in time. With
    /// `d > 0` a missing value also makes the next `d` differences missing.
    pub fn update(&mut self, y: f64) -> Option<Innovation> {
        let w = self.difference(y)? - self.mean;

        let r = self.r;
        let mut scratch = [0.0; R];
        let (system, a, p) = self.parts();
        let innovation = state_space::update(r, a, p, R, w, &mut scratch);
        state_space::predict(system, a, p, R, &mut scratch);

        Some(innovation)
    }

    /// Feeds every value of `history` in order, discarding the innovations.
    pub fn warm_up(&mut self, history: &[f64]) {
        for &y in history {
            self.update(y);
        }
    }

    /// Forecasts the next `mean.len()` raw values and their prediction variances.
    ///
    /// `mean[h]` and `variance[h]` describe the value `h + 1` steps ahead, on the
    /// original (undifferenced) scale. Before `d` values were fed there is nothing
    /// to integrate from, and both are filled with `NaN`.
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
        if self.seen < self.d {
            mean.fill(f64::NAN);
            variance.fill(f64::NAN);
            return;
        }

        Forecast::start(self).run(mean, variance);
    }

    /// Forecasts the next `mean.len()` raw values with central prediction intervals.
    ///
    /// `lower[h]` and `upper[h]` bound the value `h + 1` steps ahead with probability
    /// `level` under the model's Gaussian errors: `mean ± z sqrt(variance)` with
    /// `z = Φ⁻¹((1 + level) / 2)`, as statsmodels `conf_int(alpha=1 - level)`. A
    /// `level` outside `[0, 1]` gives `NaN` bounds; `level = 1` gives infinite ones.
    ///
    /// # Panics
    ///
    /// Panics if `mean`, `lower` and `upper` do not all have the same length.
    ///
    /// # Examples
    ///
    /// ```
    /// use backshift::{ArimaSpec, Filter};
    ///
    /// let spec = ArimaSpec { ar: &[0.5], d: 0, ma: &[], mean: 0.0, sigma2: 1.0 };
    /// let filter = Filter::<2>::new(&spec).unwrap();
    /// let (mut mean, mut lower, mut upper) = ([0.0; 1], [0.0; 1], [0.0; 1]);
    /// filter.forecast_interval_into(0.95, &mut mean, &mut lower, &mut upper);
    /// // The stationary variance of this AR(1) is 1 / (1 - 0.25).
    /// let half_width = 1.959963984540054 * (1.0_f64 / 0.75).sqrt();
    /// assert!((upper[0] - half_width).abs() < 1e-12);
    /// ```
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

    /// Differences `y` through the stored levels and returns `(1 - B)^d y_t`, or [`None`]
    /// while fewer than `d + 1` values were seen.
    fn difference(&mut self, y: f64) -> Option<f64> {
        let mut value = y;
        for level in &mut self.levels[..self.d] {
            let previous = *level;
            *level = value;
            value -= previous;
        }

        self.seen = (self.seen + 1).min(self.d + 1);

        (self.seen > self.d).then_some(value)
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

/// The joint forecast of the ARMA state and the `d` integration levels.
///
/// The joint vector is `x = [a_0, ..., a_(r-1), L_0, ..., L_(d-1)]`, where `L_j` is the
/// forecast of the raw series differenced `j` times, so `L_0` is the forecast itself.
/// Each step moves the ARMA part forward, then integrates:
/// `L_(d-1) += a_0 + mean`, then `L_j += L_(j+1)` for `j = d-2, ..., 0`. The past is
/// known, so the levels start with zero variance; their covariance with the ARMA state
/// then accumulates exactly, which gives the variance of the integrated forecast.
struct Forecast<'f, const R: usize> {
    /// The filter being forecast from.
    filter: &'f Filter<R>,
    /// The forecast mean of `x`.
    x: [f64; R],
    /// The forecast covariance of `x`; the top-left `(r + d) x (r + d)` block is used.
    cov: [[f64; R]; R],
}

impl<'f, const R: usize> Forecast<'f, R> {
    /// The joint state right after the last observation.
    fn start(filter: &'f Filter<R>) -> Self {
        let (r, d) = (filter.r, filter.d);
        let mut x = [0.0; R];
        x[..r].copy_from_slice(&filter.a[..r]);
        x[r..r + d].copy_from_slice(&filter.levels[..d]);

        let mut cov = [[0.0; R]; R];
        for (row, source) in cov.iter_mut().zip(&filter.p).take(r) {
            row[..r].copy_from_slice(&source[..r]);
        }

        Self { filter, x, cov }
    }

    /// Writes one forecast per horizon into `mean` and `variance`.
    fn run(mut self, mean: &mut [f64], variance: &mut [f64]) {
        let (r, d) = (self.filter.r, self.filter.d);
        for (h, (m, v)) in mean.iter_mut().zip(variance.iter_mut()).enumerate() {
            if h > 0 {
                self.step_arma();
            }

            self.integrate();
            if d == 0 {
                *m = self.x[0] + self.filter.mean;
                *v = self.cov[0][0];
            } else {
                *m = self.x[r];
                *v = self.cov[r][r];
            }
        }
    }

    /// Moves the ARMA part one step forward: its mean and covariance as in the filter,
    /// and its covariance with the levels by `T`.
    fn step_arma(&mut self) {
        let (r, d) = (self.filter.r, self.filter.d);
        let system = System {
            phi: &self.filter.phi[..r],
            selection: &self.filter.selection[..r],
            sigma2: self.filter.sigma2,
        };

        let mut scratch = [0.0; R];
        state_space::predict(
            system,
            &mut self.x[..r],
            self.cov.as_flattened_mut(),
            R,
            &mut scratch,
        );

        for level in r..r + d {
            for (value, row) in scratch.iter_mut().zip(&self.cov[..r]) {
                *value = row[level];
            }

            state_space::transition_vector(system, &mut scratch[..r]);
            for (i, &value) in scratch[..r].iter().enumerate() {
                self.cov[i][level] = value;
                self.cov[level][i] = value;
            }
        }
    }

    /// Adds each level's new difference to it, from the deepest level outwards.
    fn integrate(&mut self) {
        let (r, d) = (self.filter.r, self.filter.d);
        if d == 0 {
            return;
        }

        self.x[r + d - 1] += self.x[0] + self.filter.mean;
        self.add_covariance(r + d - 1, 0);
        for j in (0..d - 1).rev() {
            self.x[r + j] += self.x[r + j + 1];
            self.add_covariance(r + j, r + j + 1);
        }
    }

    /// Applies `x_k += x_m` to the covariance: row `k` += row `m`, then column `k` += column `m`.
    fn add_covariance(&mut self, k: usize, m: usize) {
        let n = self.filter.r + self.filter.d;
        for j in 0..n {
            self.cov[k][j] += self.cov[m][j];
        }

        for row in &mut self.cov[..n] {
            row[k] += row[m];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    /// A filter with capacity 4 for the given ARMA model (`d = 0`, zero mean).
    fn filter(ar: &[f64], ma: &[f64], sigma2: f64) -> Filter<4> {
        arima(ar, 0, ma, 0.0, sigma2)
    }

    /// A filter with capacity 4 for the given ARIMA model.
    fn arima(ar: &[f64], d: usize, ma: &[f64], mean: f64, sigma2: f64) -> Filter<4> {
        Filter::new(&ArimaSpec {
            ar,
            d,
            ma,
            mean,
            sigma2,
        })
        .unwrap()
    }

    #[test]
    fn first_variance_is_the_unconditional_variance() {
        // AR(1): sigma2 / (1 - phi^2).
        let innovation = filter(&[0.6], &[], 2.0).update(1.0).unwrap();
        assert_approx_eq!(
            f64,
            innovation.variance,
            2.0 / (1.0 - 0.36),
            epsilon = 1e-12
        );
        assert_approx_eq!(f64, innovation.value, 1.0, epsilon = 1e-15);

        // MA(1): sigma2 (1 + theta^2).
        let innovation = filter(&[], &[0.5], 2.0).update(1.0).unwrap();
        assert_approx_eq!(f64, innovation.variance, 2.0 * 1.25, epsilon = 1e-12);
    }

    #[test]
    fn pure_ar_innovations_are_exact_after_p_values() {
        let (phi, y) = ([0.5, -0.3], [0.4, -1.2, 0.7, 2.0, -0.5, 0.1]);
        let mut f = filter(&phi, &[], 1.5);
        for t in 0..y.len() {
            let innovation = f.update(y[t]).unwrap();
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
        let missing = f.update(f64::NAN).unwrap();
        assert!(missing.value.is_nan());
        assert_approx_eq!(f64, missing.variance, 1.0, epsilon = 1e-12);

        // Two steps without data: the variance is sigma2 (1 + phi^2).
        let next = f.update(0.0).unwrap();
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
            Filter::<2>::new(&ArimaSpec {
                ar,
                d: 0,
                ma,
                mean: 0.0,
                sigma2,
            })
            .err()
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

    #[test]
    fn differencing_primes_then_matches_arma_on_differences() {
        let y = [10.0, 10.5, 10.2, 11.0, 11.7, 11.1];
        let mut integrated = arima(&[0.6], 1, &[], 0.0, 1.0);
        let mut arma = filter(&[0.6], &[], 1.0);
        assert!(integrated.update(y[0]).is_none());
        for t in 1..y.len() {
            let expected = arma.update(y[t] - y[t - 1]).unwrap();
            assert_eq!(integrated.update(y[t]), Some(expected));
        }
    }

    #[test]
    fn mean_is_removed_and_added_back() {
        let mut with_mean = arima(&[0.6], 0, &[], 5.0, 1.0);
        let mut plain = filter(&[0.6], &[], 1.0);
        for y in [5.4, 4.1, 6.0] {
            assert_eq!(with_mean.update(y), plain.update(y - 5.0));
        }

        let (mut mean, mut variance) = ([0.0; 2], [0.0; 2]);
        let (mut plain_mean, mut plain_variance) = ([0.0; 2], [0.0; 2]);
        with_mean.forecast_into(&mut mean, &mut variance);
        plain.forecast_into(&mut plain_mean, &mut plain_variance);
        assert_approx_eq!(f64, mean[1], plain_mean[1] + 5.0, epsilon = 1e-12);
        assert_eq!(variance, plain_variance);
    }

    #[test]
    fn random_walk_with_drift_forecasts() {
        // ARIMA(0, 1, 0) with drift 0.5: y_(n+h) = y_n + 0.5 h, variance h sigma2.
        let mut f = arima(&[], 1, &[], 0.5, 2.0);
        f.warm_up(&[1.0, 3.0, 2.0]);
        let (mut mean, mut variance) = ([0.0; 3], [0.0; 3]);
        f.forecast_into(&mut mean, &mut variance);
        for h in 0..3 {
            let steps = (h + 1) as f64;
            assert_approx_eq!(f64, mean[h], 2.0 + 0.5 * steps, epsilon = 1e-12);
            assert_approx_eq!(f64, variance[h], 2.0 * steps, epsilon = 1e-12);
        }
    }

    #[test]
    fn forecast_needs_d_values() {
        let mut f = arima(&[], 2, &[], 0.0, 1.0);
        let (mut mean, mut variance) = ([0.0; 2], [0.0; 2]);
        f.update(1.0);
        f.forecast_into(&mut mean, &mut variance);
        assert!(mean.iter().chain(&variance).all(|v| v.is_nan()));

        f.update(2.0);
        f.forecast_into(&mut mean, &mut variance);
        assert!(mean.iter().chain(&variance).all(|v| v.is_finite()));
    }

    #[test]
    fn capacity_counts_the_differencing() {
        let spec = ArimaSpec {
            ar: &[0.5],
            d: 2,
            ma: &[],
            mean: 0.0,
            sigma2: 1.0,
        };
        assert_eq!(spec.states(), 3);
        assert_eq!(
            Filter::<2>::new(&spec).err(),
            Some(FilterError::Capacity {
                needed: 3,
                capacity: 2
            })
        );
        assert!(Filter::<3>::new(&spec).is_ok());
    }
}
