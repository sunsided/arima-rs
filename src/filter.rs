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

use crate::normal::normal_quantile;
use crate::state_space::{self, Innovation, LyapunovScratch, SteadyBuffers, SteadyState, System};
use crate::{math, polynomial};
use core::error::Error;
use core::fmt::{Display, Formatter};

/// The parameters of a seasonal ARIMA(p, d, q)(P, D, Q, s) model.
///
/// With `w_t = (1 - B)^d (1 - B^s)^D y_t - mean`, the differenced series follows
/// `phi(B) Phi(B^s) w_t = theta(B) Theta(B^s) e_t` with `Var(e_t) = sigma2`, where
/// `phi(B) = 1 - phi_1 B - ... - phi_p B^p` and `theta(B) = 1 + theta_1 B + ... + theta_q B^q`
/// (and likewise for the seasonal `Phi` and `Theta`). This is the sign convention of
/// statsmodels and R. Leave [`ArimaSpec::seasonal`] at [`Seasonal::NONE`] for a
/// non-seasonal model.
///
/// # Examples
///
/// ```
/// use backshift::{ArimaSpec, Seasonal};
///
/// // ARIMA(1, 1, 1) with drift 0.2.
/// let spec = ArimaSpec { ar: &[0.5], d: 1, ma: &[0.3], mean: 0.2, ..ArimaSpec::default() };
/// assert_eq!(spec.states(), 3);
///
/// // The airline model ARIMA(0, 1, 1)(0, 1, 1, 12).
/// let seasonal = Seasonal { ar: &[], d: 1, ma: &[-0.6], period: 12 };
/// let airline = ArimaSpec { d: 1, ma: &[-0.4], seasonal, ..ArimaSpec::default() };
/// assert_eq!(airline.states(), 14 + 13);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArimaSpec<'a> {
    /// The AR coefficients `phi_1, ..., phi_p`. May be empty.
    pub ar: &'a [f64],
    /// How often the series is differenced (at lag 1) before the ARMA part applies.
    pub d: usize,
    /// The MA coefficients `theta_1, ..., theta_q`. May be empty.
    pub ma: &'a [f64],
    /// The seasonal part; [`Seasonal::NONE`] for a non-seasonal model.
    pub seasonal: Seasonal<'a>,
    /// Regression coefficients `beta` for exogenous regressors; empty for none.
    ///
    /// The ARIMA part then models `u_t = y_t - x_t . beta` instead of `y_t`, so each
    /// observation needs a row `x_t` of `exog.len()` regressors.
    pub exog: &'a [f64],
    /// The mean of the differenced series: the mean for `d = D = 0`, the drift for `d = 1, D = 0`.
    pub mean: f64,
    /// The innovation variance. Must be positive and finite.
    pub sigma2: f64,
}

/// The seasonal part `(P, D, Q, s)` of a seasonal ARIMA model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seasonal<'a> {
    /// The seasonal AR coefficients `Phi_1, ..., Phi_P`, at lags `s, 2s, ...`.
    pub ar: &'a [f64],
    /// How often the series is differenced at lag `s`.
    pub d: usize,
    /// The seasonal MA coefficients `Theta_1, ..., Theta_Q`, at lags `s, 2s, ...`.
    pub ma: &'a [f64],
    /// The season length `s`. At least 1 unless the seasonal part is empty.
    pub period: usize,
}

impl Seasonal<'static> {
    /// No seasonal part.
    pub const NONE: Self = Self {
        ar: &[],
        d: 0,
        ma: &[],
        period: 0,
    };
}

impl Default for Seasonal<'_> {
    fn default() -> Self {
        Seasonal::NONE
    }
}

impl Seasonal<'_> {
    /// Whether the seasonal part has any term.
    fn is_empty(&self) -> bool {
        self.ar.is_empty() && self.d == 0 && self.ma.is_empty()
    }
}

impl Default for ArimaSpec<'_> {
    /// White noise with unit variance: no AR, MA, differencing or mean.
    fn default() -> Self {
        Self {
            ar: &[],
            d: 0,
            ma: &[],
            seasonal: Seasonal::NONE,
            exog: &[],
            mean: 0.0,
            sigma2: 1.0,
        }
    }
}

impl ArimaSpec<'_> {
    /// The number of states the model needs: `max(p + P s, q + Q s + 1) + d + D s`.
    ///
    /// This is the smallest state capacity `R` of a [`Filter`] for this model. The
    /// regressor capacity `M` must hold `exog.len()`.
    #[must_use]
    pub fn states(&self) -> usize {
        self.arma_states() + self.differencing_order()
    }

    /// The state dimension of the expanded ARMA part, `max(p + P s, q + Q s + 1)`.
    pub(crate) fn arma_states(&self) -> usize {
        let s = self.seasonal.period;
        let ar = self.ar.len() + self.seasonal.ar.len() * s;
        let ma = self.ma.len() + self.seasonal.ma.len() * s;
        ar.max(ma + 1)
    }

    /// The degree of the differencing polynomial, `d + D s`.
    pub(crate) const fn differencing_order(&self) -> usize {
        self.d + self.seasonal.d * self.seasonal.period
    }

    /// Checks that every parameter is finite, `sigma2` is positive, and a non-empty
    /// seasonal part has a period.
    pub(crate) fn validate(&self) -> Result<(), FilterError> {
        let seasonal = &self.seasonal;
        let coefficients = self
            .ar
            .iter()
            .chain(self.ma)
            .chain(seasonal.ar)
            .chain(seasonal.ma)
            .chain(self.exog);
        let finite = coefficients.copied().all(f64::is_finite);
        let period_ok = seasonal.is_empty() || seasonal.period > 0;
        if !finite
            || !period_ok
            || !self.mean.is_finite()
            || !self.sigma2.is_finite()
            || self.sigma2 <= 0.0
        {
            return Err(FilterError::InvalidParameter);
        }

        Ok(())
    }

    /// Expands the ARMA part into the first column of `T` and the selection vector:
    /// `phi[..r]` and `selection[..r]`.
    ///
    /// Requires `phi` and `selection` to hold `r = arma_states()` values.
    pub(crate) fn expand_arma_into(&self, phi: &mut [f64], selection: &mut [f64]) {
        let r = self.arma_states();
        let s = self.seasonal.period;
        phi[..r].fill(0.0);
        selection[..r].fill(0.0);

        polynomial::seasonal_product(self.ar, self.seasonal.ar, s, -1.0, phi);
        selection[0] = 1.0;
        polynomial::seasonal_product(self.ma, self.seasonal.ma, s, 1.0, &mut selection[1..]);
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
    /// The model has more regression coefficients than the filter's regressor capacity `M`.
    Regressors {
        /// The number of regression coefficients, `exog.len()`.
        needed: usize,
        /// The filter's regressor capacity `M`.
        capacity: usize,
    },
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
            FilterError::Regressors { needed, capacity } => {
                write!(f, "model has {needed} regressors, filter holds {capacity}")
            }
        }
    }
}

impl Error for FilterError {}

/// A streaming ARIMA filter with room for `R` states and `M` regressors, stored inline
/// with no heap.
///
/// `R` must be at least [`ArimaSpec::states`] and `M` at least the number of regression
/// coefficients; `M` defaults to 0, a model without regressors. The regressors do not
/// enlarge the covariance, so the size is about `8 (R² + 6R + M)` bytes.
/// [`Filter::new`] uses about three times that on the stack while it computes the
/// starting covariance, and [`Filter::forecast_into`] about twice. The filter is
/// `Copy`, so it can be snapshotted cheaply.
///
/// # Examples
///
/// ```
/// use backshift::{ArimaSpec, Filter};
///
/// let spec = ArimaSpec { ar: &[0.6], d: 1, ma: &[0.3], ..ArimaSpec::default() };
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
pub struct Filter<const R: usize, const M: usize = 0> {
    /// The ARMA state dimension, `1 <= r` and `r + k <= R`.
    r: usize,
    /// The degree `k` of the differencing polynomial.
    k: usize,
    /// The padded first column of the transition matrix (the expanded AR polynomial).
    phi: [f64; R],
    /// The padded selection vector `[1, theta_1, ...]` (the expanded MA polynomial).
    selection: [f64; R],
    /// The innovation variance.
    sigma2: f64,
    /// The predicted ARMA state for the next differenced, demeaned value.
    a: [f64; R],
    /// The predicted ARMA state covariance; the top-left `r x r` block is used.
    p: [[f64; R]; R],
    /// The differencing polynomial `delta_0 = 1, ..., delta_k`.
    delta: [f64; R],
    /// `lags[j]` is the raw value `j + 1` steps back, for `j < k`.
    lags: [f64; R],
    /// The mean of the differenced series.
    mean: f64,
    /// How many raw values were fed, capped at `k + 1`. Values beyond `k` produce innovations.
    seen: usize,
    /// The regression coefficients; the first `m` entries are used.
    beta: [f64; M],
    /// The number of regressors.
    m: usize,
    /// The steady-state shortcut and its frozen gain, if [`Filter::with_steady_state`]
    /// enabled it.
    steady: Option<(SteadyState, [f64; R])>,
}

impl<const R: usize, const M: usize> Filter<R, M> {
    /// Builds a filter at the model's stationary start, before any observation.
    ///
    /// This is the device path: it needs only the coefficients and does not allocate.
    ///
    /// # Errors
    ///
    /// - [`FilterError::InvalidParameter`] if a coefficient or the mean is not finite,
    ///   `sigma2 <= 0`, or a non-empty seasonal part has period 0.
    /// - [`FilterError::Capacity`] if the model needs more than `R` states.
    /// - [`FilterError::Regressors`] if the model has more than `M` regression coefficients.
    /// - [`FilterError::NonStationary`] if the AR part is not stationary.
    pub fn new(spec: &ArimaSpec<'_>) -> Result<Self, FilterError> {
        spec.validate()?;
        if spec.states() > R {
            return Err(FilterError::Capacity {
                needed: spec.states(),
                capacity: R,
            });
        }

        if spec.exog.len() > M {
            return Err(FilterError::Regressors {
                needed: spec.exog.len(),
                capacity: M,
            });
        }

        let mut filter = Self {
            r: spec.arma_states(),
            k: spec.differencing_order(),
            phi: [0.0; R],
            selection: [0.0; R],
            sigma2: spec.sigma2,
            a: [0.0; R],
            p: [[0.0; R]; R],
            delta: [0.0; R],
            lags: [0.0; R],
            mean: spec.mean,
            seen: 0,
            beta: [0.0; M],
            m: spec.exog.len(),
            steady: None,
        };
        filter.beta[..filter.m].copy_from_slice(spec.exog);
        spec.expand_arma_into(&mut filter.phi, &mut filter.selection);
        polynomial::differencing(
            spec.d,
            spec.seasonal.d,
            spec.seasonal.period,
            &mut filter.delta,
        );

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

    /// Turns on statsmodels' steady-state shortcut, its default `ssm.tolerance = 1e-19`.
    ///
    /// Once the predicted covariance changes by less than `1e-19` (the sum of its squared
    /// changes) in one step, the innovation variance and gain are frozen and reused. Each
    /// update then costs `O(R)` instead of `O(R²)`, which saves time on a device that runs a
    /// filter for a long time. The innovations and forecasts then match statsmodels with its
    /// default tolerance, such as `results.extend()` or `filter()` on the whole series, and
    /// move by up to about 1e-9 against the full recursion (the default here). A missing
    /// value resumes the full recursion until the covariance converges again.
    ///
    /// Each update before convergence takes `8 R²` more bytes of stack for the comparison.
    ///
    /// # Examples
    ///
    /// ```
    /// use backshift::{ArimaSpec, Filter};
    ///
    /// let spec = ArimaSpec { ar: &[0.5], ma: &[0.3], ..ArimaSpec::default() };
    /// let mut exact = Filter::<2>::new(&spec).unwrap();
    /// let mut steady = Filter::<2>::new(&spec).unwrap().with_steady_state();
    /// for t in 0..200 {
    ///     let y = (t as f64 * 0.7).sin();
    ///     let (a, b) = (exact.update(y).unwrap(), steady.update(y).unwrap());
    ///     assert!((a.value - b.value).abs() < 1e-8);
    /// }
    /// ```
    #[must_use]
    pub fn with_steady_state(mut self) -> Self {
        self.steady = Some((SteadyState::NEW, [0.0; R]));
        self
    }

    /// Feeds one raw observation and returns the one-step prediction error of its
    /// differenced, demeaned value.
    ///
    /// Returns [`None`] for the first `k = d + D s` observations, which only prime the
    /// differencing. A `NaN` observation counts as missing, as in statsmodels, and so does
    /// an infinite one (as in fitting): its
    /// innovation value is `NaN` and the filter only moves forward in time. A missing
    /// value also makes every difference that uses it missing.
    ///
    /// # Panics
    ///
    /// Panics if the model has regressors; use [`Filter::update_exog`] then.
    pub fn update(&mut self, y: f64) -> Option<Innovation> {
        self.update_exog(y, &[])
    }

    /// Like [`Filter::update`], for a model with regressors: `x` is the regressor row
    /// that goes with `y`, and the filter works on `y - x . beta`.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have one entry per regression coefficient.
    pub fn update_exog(&mut self, y: f64, x: &[f64]) -> Option<Innovation> {
        let u = y - self.regression(x);
        // An infinite value would poison the state for good, so it counts as missing.
        let u = if u.is_finite() { u } else { f64::NAN };
        let w = self.difference(u)? - self.mean;

        let r = self.r;
        let mut steady = self.steady;
        let mut scratch = [0.0; R];
        let (system, a, p) = self.parts();
        let innovation = if let Some((steady, gain)) = &mut steady {
            // A converged step never compares covariances, so it skips the R² buffer.
            let mut storage;
            let previous: &mut [f64] = if steady.converged() {
                &mut []
            } else {
                storage = [[0.0; R]; R];
                storage.as_flattened_mut()
            };
            let buffers = SteadyBuffers {
                gain,
                previous,
                scratch: &mut scratch,
            };
            steady.step(system, a, p, R, w, buffers)
        } else {
            let innovation = state_space::update(r, a, p, R, w, &mut scratch);
            state_space::predict(system, a, p, R, &mut scratch);
            innovation
        };

        self.steady = steady;
        Some(innovation)
    }

    /// Feeds every value of `history` in order, discarding the innovations.
    ///
    /// # Panics
    ///
    /// Panics if the model has regressors; use [`Filter::warm_up_exog`] then.
    pub fn warm_up(&mut self, history: &[f64]) {
        self.warm_up_exog(history, &[]);
    }

    /// Like [`Filter::warm_up`], for a model with regressors. `regressors` holds one row
    /// per value of `history`, row-major.
    ///
    /// # Panics
    ///
    /// Panics if `regressors` does not hold `history.len()` rows of `exog.len()` entries.
    pub fn warm_up_exog(&mut self, history: &[f64], regressors: &[f64]) {
        assert_eq!(
            regressors.len(),
            history.len() * self.m,
            "one regressor row per value"
        );
        for (t, &y) in history.iter().enumerate() {
            self.update_exog(y, &regressors[t * self.m..(t + 1) * self.m]);
        }
    }

    /// Forecasts the next `mean.len()` raw values and their prediction variances.
    ///
    /// `mean[h]` and `variance[h]` describe the value `h + 1` steps ahead, on the
    /// original (undifferenced) scale. Before `k = d + D s` values were fed there is
    /// nothing to integrate from, and both are filled with `NaN`.
    ///
    /// # Panics
    ///
    /// Panics if `mean` and `variance` have different lengths, or the model has
    /// regressors (use [`Filter::forecast_exog_into`] then).
    pub fn forecast_into(&self, mean: &mut [f64], variance: &mut [f64]) {
        self.forecast_exog_into(&[], mean, variance);
    }

    /// Like [`Filter::forecast_into`], for a model with regressors: `future` holds the
    /// regressor rows of the forecast horizon, row-major, and `x . beta` is added to each
    /// mean. The variances treat `beta` as known.
    ///
    /// # Panics
    ///
    /// Panics if `mean` and `variance` have different lengths, or `future` does not hold
    /// `mean.len()` rows of `exog.len()` entries.
    pub fn forecast_exog_into(&self, future: &[f64], mean: &mut [f64], variance: &mut [f64]) {
        assert_eq!(
            mean.len(),
            variance.len(),
            "mean and variance must have the same length"
        );
        assert_eq!(
            future.len(),
            mean.len() * self.m,
            "one regressor row per horizon"
        );
        if self.seen < self.k {
            mean.fill(f64::NAN);
            variance.fill(f64::NAN);
            return;
        }

        Forecast::start(self).run(mean, variance);
        for (h, value) in mean.iter_mut().enumerate() {
            *value += self.regression(&future[h * self.m..(h + 1) * self.m]);
        }
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
    /// Panics if `mean`, `lower` and `upper` do not all have the same length, or the model
    /// has regressors (use [`Filter::forecast_interval_exog_into`] then).
    ///
    /// # Examples
    ///
    /// ```
    /// use backshift::{ArimaSpec, Filter};
    ///
    /// let spec = ArimaSpec { ar: &[0.5], ..ArimaSpec::default() };
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
        self.forecast_interval_exog_into(level, &[], mean, lower, upper);
    }

    /// Like [`Filter::forecast_interval_into`], for a model with regressors; `future` as
    /// in [`Filter::forecast_exog_into`].
    ///
    /// # Panics
    ///
    /// Panics if `mean`, `lower` and `upper` do not all have the same length, or `future`
    /// does not hold `mean.len()` rows of `exog.len()` entries.
    pub fn forecast_interval_exog_into(
        &self,
        level: f64,
        future: &[f64],
        mean: &mut [f64],
        lower: &mut [f64],
        upper: &mut [f64],
    ) {
        assert!(
            mean.len() == lower.len() && mean.len() == upper.len(),
            "mean, lower and upper must have the same length"
        );

        // `upper` holds the variances until it is overwritten below.
        self.forecast_exog_into(future, mean, upper);
        let z = normal_quantile(f64::midpoint(1.0, level));
        for ((m, lo), up) in mean.iter().zip(lower.iter_mut()).zip(upper.iter_mut()) {
            let half_width = z * math::sqrt(*up);
            *lo = m - half_width;
            *up = m + half_width;
        }
    }

    /// `x . beta` for one regressor row.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have one entry per regression coefficient.
    fn regression(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.m, "one regressor per regression coefficient");
        x.iter().zip(&self.beta[..self.m]).map(|(x, b)| x * b).sum()
    }

    /// Applies the differencing polynomial to `y` and its stored lags, then stores `y`.
    ///
    /// Returns `sum_j delta_j y_(t-j)`, or [`None`] while fewer than `k + 1` values were seen.
    fn difference(&mut self, y: f64) -> Option<f64> {
        let k = self.k;
        let value = polynomial::apply(&self.delta[..=k], y, self.lags[..k].iter().copied());
        if k > 0 {
            self.lags.copy_within(..k - 1, 1);
            self.lags[0] = y;
        }

        self.seen = (self.seen + 1).min(k + 1);

        (self.seen > k).then_some(value)
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

/// The joint forecast of the ARMA state and the last `k` raw values.
///
/// The joint vector is `x = [a_0, ..., a_(r-1), y_t, y_(t-1), ..., y_(t-k+1)]`. Each step
/// moves the ARMA part forward, then integrates: the next raw value is
/// `y_(t+1) = a_0 + mean - sum_(j>=1) delta_j y_(t+1-j)`, which is pushed onto the lags.
/// The past is known, so the lags start with zero variance; their covariance with the
/// ARMA state then accumulates exactly, which gives the variance of the integrated
/// forecast for any differencing polynomial.
struct Forecast<'f, const R: usize, const M: usize> {
    /// The filter being forecast from.
    filter: &'f Filter<R, M>,
    /// The forecast mean of `x`.
    x: [f64; R],
    /// The forecast covariance of `x`; the top-left `(r + k) x (r + k)` block is used.
    cov: [[f64; R]; R],
}

impl<'f, const R: usize, const M: usize> Forecast<'f, R, M> {
    /// The joint state right after the last observation.
    fn start(filter: &'f Filter<R, M>) -> Self {
        let (r, k) = (filter.r, filter.k);
        let mut x = [0.0; R];
        x[..r].copy_from_slice(&filter.a[..r]);
        x[r..r + k].copy_from_slice(&filter.lags[..k]);

        let mut cov = [[0.0; R]; R];
        for (row, source) in cov.iter_mut().zip(&filter.p).take(r) {
            row[..r].copy_from_slice(&source[..r]);
        }

        Self { filter, x, cov }
    }

    /// Writes one forecast per horizon into `mean` and `variance`.
    fn run(mut self, mean: &mut [f64], variance: &mut [f64]) {
        let (r, k) = (self.filter.r, self.filter.k);
        for (h, (m, v)) in mean.iter_mut().zip(variance.iter_mut()).enumerate() {
            if h > 0 {
                self.step_arma();
            }

            if k == 0 {
                *m = self.x[0] + self.filter.mean;
                *v = self.cov[0][0];
            } else {
                self.integrate();
                *m = self.x[r];
                *v = self.cov[r][r];
            }
        }
    }

    /// Moves the ARMA part one step forward: its mean and covariance as in the filter,
    /// and its covariance with the lags by `T`.
    fn step_arma(&mut self) {
        let (r, k) = (self.filter.r, self.filter.k);
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

        for lag in r..r + k {
            for (value, row) in scratch.iter_mut().zip(&self.cov[..r]) {
                *value = row[lag];
            }

            state_space::transition_vector(system, &mut scratch[..r]);
            for (i, &value) in scratch[..r].iter().enumerate() {
                self.cov[i][lag] = value;
                self.cov[lag][i] = value;
            }
        }
    }

    /// Pushes the next raw value onto the lags.
    ///
    /// As a linear map, the new `y` is `v . x` with `v = e_0 - sum_j delta_j e_(r+j-1)`,
    /// and the older lags shift down by one. The covariance becomes `M cov M'`: rows first,
    /// then columns.
    fn integrate(&mut self) {
        let (r, k) = (self.filter.r, self.filter.k);
        let n = r + k;
        let delta = &self.filter.delta[1..=k];
        let combine = |values: &dyn Fn(usize) -> f64| {
            delta
                .iter()
                .enumerate()
                .fold(values(0), |sum, (j, d)| sum - d * values(r + j))
        };

        let next = combine(&|i| self.x[i]) + self.filter.mean;
        self.x.copy_within(r..n - 1, r + 1);
        self.x[r] = next;

        let mut row = [0.0; R];
        for (c, value) in row.iter_mut().enumerate().take(n) {
            *value = combine(&|i| self.cov[i][c]);
        }

        self.cov.copy_within(r..n - 1, r + 1);
        self.cov[r] = row;

        let mut column = [0.0; R];
        for (i, value) in column.iter_mut().enumerate().take(n) {
            *value = combine(&|c| self.cov[i][c]);
        }

        for (i, cov_row) in self.cov.iter_mut().enumerate().take(n) {
            cov_row.copy_within(r..n - 1, r + 1);
            cov_row[r] = column[i];
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
            ..ArimaSpec::default()
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
                ma,
                sigma2,
                ..ArimaSpec::default()
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
            ..ArimaSpec::default()
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

    #[test]
    fn regressors_are_subtracted_before_differencing() {
        let beta = [2.0, -1.0];
        let spec = ArimaSpec {
            ar: &[0.5],
            d: 1,
            exog: &beta,
            ..ArimaSpec::default()
        };
        let plain_spec = ArimaSpec { exog: &[], ..spec };
        let mut with_exog = Filter::<4, 2>::new(&spec).unwrap();
        let mut plain = Filter::<4>::new(&plain_spec).unwrap();

        let rows = [[1.0, 0.5], [0.2, 0.1], [-0.3, 0.9]];
        for (y, x) in [3.0, 2.5, 1.0].into_iter().zip(rows) {
            let u = y - (x[0] * beta[0] + x[1] * beta[1]);
            assert_eq!(with_exog.update_exog(y, &x), plain.update(u));
        }

        let future = [0.4, 0.2];
        let (mut mean, mut variance) = ([0.0; 1], [0.0; 1]);
        let (mut plain_mean, mut plain_variance) = ([0.0; 1], [0.0; 1]);
        with_exog.forecast_exog_into(&future, &mut mean, &mut variance);
        plain.forecast_into(&mut plain_mean, &mut plain_variance);
        assert_approx_eq!(
            f64,
            mean[0],
            plain_mean[0] + 0.4 * 2.0 - 0.2,
            epsilon = 1e-12
        );
        assert_eq!(variance, plain_variance);
    }

    #[test]
    #[should_panic(expected = "one regressor per regression coefficient")]
    fn update_without_regressors_panics_for_exog_models() {
        let spec = ArimaSpec {
            exog: &[1.0],
            ..ArimaSpec::default()
        };
        Filter::<2, 1>::new(&spec).unwrap().update(1.0);
    }

    #[test]
    fn too_many_regressors() {
        let spec = ArimaSpec {
            exog: &[1.0, 2.0, 3.0],
            ..ArimaSpec::default()
        };
        assert_eq!(spec.states(), 1);
        assert!(Filter::<1, 3>::new(&spec).is_ok());
        assert_eq!(
            Filter::<1, 2>::new(&spec).err(),
            Some(FilterError::Regressors {
                needed: 3,
                capacity: 2
            })
        );
    }

    #[test]
    fn infinite_values_count_as_missing() {
        let mut filter = filter(&[0.5], &[0.3], 1.0);
        filter.update(0.1);
        assert!(filter.update(f64::INFINITY).unwrap().value.is_nan());
        assert!(filter.update(0.2).unwrap().value.is_finite());

        let (mut mean, mut variance) = ([0.0; 2], [0.0; 2]);
        filter.forecast_into(&mut mean, &mut variance);
        assert!(mean.iter().all(|m| m.is_finite()));
    }

    #[test]
    fn converged_steady_state_matches_the_full_recursion() {
        let spec = ArimaSpec {
            ar: &[0.5],
            ma: &[0.3],
            ..ArimaSpec::default()
        };
        let mut exact = Filter::<2>::new(&spec).unwrap();
        let mut steady = Filter::<2>::new(&spec).unwrap().with_steady_state();
        for t in 0..300 {
            let y = (f64::from(t) * 0.7).sin();
            let (a, b) = (exact.update(y).unwrap(), steady.update(y).unwrap());
            assert!((a.value - b.value).abs() < 1e-8);
        }
        assert!(steady.steady.unwrap().0.converged());
    }
}
