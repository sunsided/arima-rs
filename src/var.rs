//! Vector autoregression (VAR) for several series at once, with a heap-free online filter.
//!
//! A VAR(p) on `k` series models `y_t = c + A_1 y_(t-1) + ... + A_p y_(t-p) + u_t` with
//! `Cov(u_t) = Sigma`. [`VarSpec`] holds `c`, the lag matrices and `Sigma`.
//! [`OnlineVar`] carries a model forward one row of values at a time without a heap: it
//! returns the one-step errors and their squared Mahalanobis distance, an anomaly score for
//! values that are only unusual together, and forecasts with intervals. It can also keep
//! learning by recursive least squares ([`OnlineVar::with_learning`]); started from a fit's
//! inverse Gram matrix with no forgetting, its coefficients stay those of least squares on
//! all the values so far, exactly.

use crate::filter::FilterError;
use crate::linalg;
use crate::math;
use crate::normal::normal_quantile;

/// A VAR(p) model on `k` series.
///
/// # Examples
///
/// ```
/// use backshift::{OnlineVar, VarSpec};
///
/// // Two series that each follow their own past and pull on each other.
/// let spec = VarSpec {
///     intercept: &[1.0, 0.5],
///     lags: &[0.5, 0.1, 0.2, 0.3], // A_1, row-major: row i is the equation of series i
///     sigma: &[1.0, 0.2, 0.2, 0.5],
/// };
/// let mut filter = OnlineVar::<2, 3>::new(&spec).unwrap();
/// assert_eq!(filter.update(&[2.0, 1.0], &mut [0.0; 2]), None); // primes the lag
/// let score = filter.update(&[2.3, 1.4], &mut [0.0; 2]).unwrap();
/// assert!(score >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarSpec<'a> {
    /// The intercepts `c`, one per series; their count is `k`.
    pub intercept: &'a [f64],
    /// The lag matrices `A_1, ..., A_p`, each `k x k` row-major, back to back: entry
    /// `(i, j)` of `A_l` is the weight of series `j`, `l` steps back, in the equation of
    /// series `i`. Their count `p` is `lags.len() / k^2`.
    pub lags: &'a [f64],
    /// The innovation covariance `Sigma`, `k x k` row-major. Must be positive definite.
    pub sigma: &'a [f64],
}

impl VarSpec<'_> {
    /// The number of series `k`.
    #[must_use]
    pub const fn series(&self) -> usize {
        self.intercept.len()
    }

    /// The lag order `p`, or 0 if `lags` does not hold whole `k x k` matrices.
    #[must_use]
    pub const fn order(&self) -> usize {
        let k = self.series();
        if k == 0 || self.lags.len() % (k * k) != 0 {
            return 0;
        }
        self.lags.len() / (k * k)
    }

    /// The weight `A_l[i][j]` of series `j`, `l` steps back (`l >= 1`), in equation `i`.
    fn weight(&self, l: usize, i: usize, j: usize) -> f64 {
        let k = self.series();
        self.lags[(l - 1) * k * k + i * k + j]
    }
}

/// The recursive least squares state of [`OnlineVar::with_learning`].
#[derive(Debug, Clone, Copy)]
struct Learning<const D: usize> {
    /// The forgetting factor `lambda` in `(0, 1]`; 1 weighs all values alike.
    forgetting: f64,
    /// The inverse Gram matrix of the regressors, `(X'X)^-1`; the top-left `d x d` block is
    /// used.
    gram_inverse: [[f64; D]; D],
}

/// A streaming VAR filter for `K` series with room for `D` regressors (`D >= K p + 1`),
/// stored inline with no heap.
///
/// The regressors of each step are `[1, y_(t-1), ..., y_(t-p)]`, so `D` bounds the lag
/// order. The size is about `8 (K D + D + 2 K^2)` bytes, plus `8 D^2` for learning.
#[derive(Debug, Clone, Copy)]
pub struct OnlineVar<const K: usize, const D: usize> {
    /// The lag order `p`.
    p: usize,
    /// The coefficients: row `r` holds the weight of regressor `r` in each equation.
    coefficients: [[f64; K]; D],
    /// The regressors of the next step: 1, then the last `p` rows, most recent first.
    regressors: [f64; D],
    /// How many rows were fed, capped at `p`.
    seen: usize,
    /// The innovation covariance.
    sigma: [[f64; K]; K],
    /// The lower Cholesky factor of `sigma`.
    factor: [[f64; K]; K],
    /// The learning state, if learning is on.
    learning: Option<Learning<D>>,
}

impl<const K: usize, const D: usize> OnlineVar<K, D> {
    /// Builds a filter for `spec`, before any value. This is the device path: it does not
    /// allocate.
    ///
    /// # Errors
    ///
    /// - [`FilterError::InvalidParameter`] if `spec` does not have `K` series, `lags` does
    ///   not hold whole `K x K` matrices, a coefficient is not finite, or `sigma` is not
    ///   positive definite.
    /// - [`FilterError::Capacity`] if the model needs more than `D` regressors (`K p + 1`).
    pub fn new(spec: &VarSpec<'_>) -> Result<Self, FilterError> {
        let p = spec.order();
        let finite = spec
            .intercept
            .iter()
            .chain(spec.lags)
            .chain(spec.sigma)
            .all(|v| v.is_finite());
        let shaped = spec.series() == K
            && K > 0
            && spec.lags.len() == p * K * K
            && spec.sigma.len() == K * K;
        if !finite || !shaped {
            return Err(FilterError::InvalidParameter);
        }

        let needed = K * p + 1;
        if needed > D {
            return Err(FilterError::Capacity {
                needed,
                capacity: D,
            });
        }

        let mut sigma = [[0.0; K]; K];
        for (row, source) in sigma.iter_mut().zip(spec.sigma.chunks_exact(K)) {
            row.copy_from_slice(source);
        }
        let mut factor = sigma;
        if !linalg::cholesky(factor.as_flattened_mut(), K, K) {
            return Err(FilterError::InvalidParameter);
        }

        let mut coefficients = [[0.0; K]; D];
        for i in 0..K {
            coefficients[0][i] = spec.intercept[i];
            for l in 1..=p {
                for j in 0..K {
                    coefficients[1 + (l - 1) * K + j][i] = spec.weight(l, i, j);
                }
            }
        }

        let mut regressors = [0.0; D];
        regressors[0] = 1.0;
        Ok(Self {
            p,
            coefficients,
            regressors,
            seen: 0,
            sigma,
            factor,
            learning: None,
        })
    }

    /// Turns on learning by recursive least squares with forgetting factor `forgetting`,
    /// from the inverse Gram matrix `gram_inverse` of the regressors so far (`(K p + 1)^2`
    /// values, row-major; [`VarFit::gram_inverse`](crate::VarFit::gram_inverse) gives it).
    ///
    /// Each value then moves the coefficients by its error. With `forgetting = 1` they stay
    /// exactly those of least squares on every value so far; below 1, older values weigh
    /// less, so the model follows dynamics that drift. `sigma` stays fixed.
    ///
    /// The guarantee holds only when the filter's coefficients are the least-squares
    /// solution for the rows that gave `gram_inverse`, as a [`VarFit`](crate::VarFit)'s
    /// `spec` and `gram_inverse` are. The shapes are checked, the pairing cannot be: with an
    /// unrelated matrix the filter still learns, from a wrong starting curvature.
    ///
    /// # Errors
    ///
    /// [`FilterError::InvalidParameter`] if `forgetting` is not in `(0, 1]`, or
    /// `gram_inverse` does not hold `(K p + 1)^2` finite values.
    pub fn with_learning(
        mut self,
        gram_inverse: &[f64],
        forgetting: f64,
    ) -> Result<Self, FilterError> {
        let d = self.dimension();
        let valid = forgetting > 0.0
            && forgetting <= 1.0
            && gram_inverse.len() == d * d
            && gram_inverse.iter().all(|v| v.is_finite());
        if !valid {
            return Err(FilterError::InvalidParameter);
        }

        let mut inverse = [[0.0; D]; D];
        for (row, source) in inverse.iter_mut().zip(gram_inverse.chunks_exact(d)) {
            row[..d].copy_from_slice(source);
        }
        self.learning = Some(Learning {
            forgetting,
            gram_inverse: inverse,
        });
        Ok(self)
    }

    /// Feeds one row of `K` values and writes its one-step errors into `innovation`.
    ///
    /// Returns [`None`] for the first `p` rows, which only fill the lags, and then the
    /// squared Mahalanobis distance `u' Sigma^-1 u` of the errors: about `chi^2_K` for a
    /// well-specified model, so it flags rows that are unusual together even where each
    /// value alone is not. A non-finite value counts as missing: its error is `NaN`, the
    /// distance is `NaN`, nothing is learned, and its forecast takes its place in the lags.
    ///
    /// # Panics
    ///
    /// Panics if `y` or `innovation` does not hold `K` values.
    pub fn update(&mut self, y: &[f64], innovation: &mut [f64]) -> Option<f64> {
        assert!(
            y.len() == K && innovation.len() == K,
            "one value per series"
        );
        if self.seen < self.p {
            self.push(y);
            self.seen += 1;
            innovation.fill(f64::NAN);
            return None;
        }

        let mut forecast = [0.0; K];
        self.predict(&self.regressors, &mut forecast);
        let mut observed = [0.0; K];
        let mut complete = true;
        for i in 0..K {
            if y[i].is_finite() {
                innovation[i] = y[i] - forecast[i];
                observed[i] = y[i];
            } else {
                innovation[i] = f64::NAN;
                observed[i] = forecast[i];
                complete = false;
            }
        }

        let distance = if complete {
            self.learn(innovation);
            self.mahalanobis(innovation)
        } else {
            f64::NAN
        };
        self.push(&observed);
        Some(distance)
    }

    /// Feeds every row of `history` (row-major, `K` values per row), discarding the
    /// errors.
    ///
    /// # Panics
    ///
    /// Panics if `history` does not hold whole rows of `K` values.
    pub fn warm_up(&mut self, history: &[f64]) {
        assert_eq!(history.len() % K, 0, "whole rows of K values");
        let mut innovation = [0.0; K];
        for row in history.chunks_exact(K) {
            self.update(row, &mut innovation);
        }
    }

    /// Forecasts the next `mean.len() / K` rows and the variance of each value.
    ///
    /// `mean` and `variance` are row-major, `K` values per horizon. The variances are the
    /// diagonals of `Phi_0 Sigma Phi_0' + ... + Phi_(h-1) Sigma Phi_(h-1)'`, with the moving
    /// average weights `Phi_i` of the model, as statsmodels' `forecast_cov`. Before `p` rows
    /// were fed both are `NaN`.
    ///
    /// # Panics
    ///
    /// Panics if `mean` and `variance` differ in length or do not hold whole rows.
    pub fn forecast_into(&self, mean: &mut [f64], variance: &mut [f64]) {
        assert!(
            mean.len() == variance.len() && mean.len() % K == 0,
            "mean and variance must hold the same whole rows"
        );
        if self.seen < self.p {
            mean.fill(f64::NAN);
            variance.fill(f64::NAN);
            return;
        }

        if mean.is_empty() {
            return;
        }

        let mut regressors = self.regressors;
        for row in mean.chunks_exact_mut(K) {
            let mut forecast = [0.0; K];
            self.predict(&regressors, &mut forecast);
            row.copy_from_slice(&forecast);
            shift_in(&mut regressors[..self.dimension()], &forecast);
        }

        let mut history = [[0.0; K]; D];
        let mut work = [[0.0; K]; K];
        let mut covariance = [[0.0; K]; K];
        let weight = |l: usize, i: usize, j: usize| self.coefficients[1 + (l - 1) * K + j][i];
        let scratch = MaScratch {
            history: &mut history.as_flattened_mut()[..self.p.max(1) * K * K],
            work: work.as_flattened_mut(),
            covariance: covariance.as_flattened_mut(),
        };
        let horizons = variance.len() / K;
        forecast_covariances(
            K,
            self.p,
            weight,
            self.sigma.as_flattened(),
            scratch,
            |h, cov| {
                for i in 0..K {
                    variance[h * K + i] = cov[i * K + i];
                }
                h + 1 < horizons
            },
        );
    }

    /// Forecasts with central prediction intervals at `level` for each value:
    /// `mean ± z sqrt(variance)` with `z = Φ⁻¹((1 + level) / 2)`, as statsmodels'
    /// `forecast_interval`.
    ///
    /// # Panics
    ///
    /// Panics if `mean`, `lower` and `upper` differ in length or do not hold whole rows.
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

    /// The coefficients in [`VarSpec`] form: writes the intercepts into `intercept` (`K`
    /// values) and the lag matrices into `lags` (`p K^2` values). With learning on they move
    /// with each value.
    ///
    /// # Panics
    ///
    /// Panics if `intercept` or `lags` has the wrong length.
    pub fn coefficients_into(&self, intercept: &mut [f64], lags: &mut [f64]) {
        assert!(
            intercept.len() == K && lags.len() == self.p * K * K,
            "K intercepts and p K^2 lag weights"
        );
        for i in 0..K {
            intercept[i] = self.coefficients[0][i];
            for l in 1..=self.p {
                for j in 0..K {
                    lags[(l - 1) * K * K + i * K + j] = self.coefficients[1 + (l - 1) * K + j][i];
                }
            }
        }
    }

    /// The number of regressors in use, `K p + 1`.
    const fn dimension(&self) -> usize {
        K * self.p + 1
    }

    /// The one-step forecast `B' z` from the regressors `z`.
    fn predict(&self, regressors: &[f64; D], forecast: &mut [f64; K]) {
        forecast.fill(0.0);
        for (z, row) in regressors
            .iter()
            .zip(&self.coefficients)
            .take(self.dimension())
        {
            for (f, b) in forecast.iter_mut().zip(row) {
                *f += z * b;
            }
        }
    }

    /// Pushes the row `y` onto the lags of the regressors.
    fn push(&mut self, y: &[f64]) {
        let d = self.dimension();
        shift_in(&mut self.regressors[..d], y);
    }

    /// One recursive least squares step with the errors `error` of the current regressors:
    /// `g = P z / (lambda + z' P z)`, `B += g e'`, `P = (P - g z' P) / lambda`.
    fn learn(&mut self, error: &[f64]) {
        let d = self.dimension();
        let z = self.regressors;
        let Some(learning) = self.learning.as_mut() else {
            return;
        };

        let mut pz = [0.0; D];
        for (value, row) in pz.iter_mut().zip(&learning.gram_inverse).take(d) {
            *value = row[..d].iter().zip(&z[..d]).map(|(p, z)| p * z).sum();
        }
        let denominator =
            learning.forgetting + z[..d].iter().zip(&pz[..d]).map(|(z, p)| z * p).sum::<f64>();
        let mut gain = [0.0; D];
        for (g, p) in gain.iter_mut().zip(&pz).take(d) {
            *g = p / denominator;
        }

        for (row, g) in self.coefficients.iter_mut().zip(&gain).take(d) {
            for (b, e) in row.iter_mut().zip(error) {
                *b += g * e;
            }
        }

        // `P z` is also `z' P` (P is symmetric), so `g z' P` is the outer product `g (P z)'`.
        for (i, row) in learning.gram_inverse.iter_mut().enumerate().take(d) {
            for (j, value) in row.iter_mut().enumerate().take(d) {
                *value = (*value - gain[i] * pz[j]) / learning.forgetting;
            }
        }
    }

    /// `u' Sigma^-1 u` through the Cholesky factor of `Sigma`.
    fn mahalanobis(&self, error: &[f64]) -> f64 {
        let mut solved = [0.0; K];
        solved.copy_from_slice(error);
        linalg::solve_lower(self.factor.as_flattened(), K, K, &mut solved);
        solved.iter().map(|v| v * v).sum()
    }
}

/// Moves the lags of `regressors` (`[1, y_(t-1), ..., y_(t-p)]`) one row back and puts
/// `row` first.
fn shift_in(regressors: &mut [f64], row: &[f64]) {
    let k = row.len();
    let d = regressors.len();
    if d > 1 + k {
        regressors.copy_within(1..d - k, 1 + k);
    }
    if d > 1 {
        regressors[1..=k].copy_from_slice(row);
    }
}

/// Working memory for [`forecast_covariances`].
pub(crate) struct MaScratch<'a> {
    /// The last `p` moving average weights `Phi`, each `k x k`: `p k^2` values (at least
    /// `k^2`).
    pub(crate) history: &'a mut [f64],
    /// A `k x k` matrix.
    pub(crate) work: &'a mut [f64],
    /// The running forecast covariance, `k x k`.
    pub(crate) covariance: &'a mut [f64],
}

/// Hands `sink` the forecast covariance of horizon `h = 0, 1, ...` as `k x k` row-major,
/// `sum_(i <= h) Phi_i Sigma Phi_i'` with `Phi_0 = I` and `Phi_i = A_1 Phi_(i-1) + ... +
/// A_p Phi_(i-p)`, as statsmodels' `forecast_cov`, until `sink` returns `false`.
///
/// `weight(l, i, j)` is `A_l[i][j]`.
pub(crate) fn forecast_covariances(
    k: usize,
    p: usize,
    weight: impl Fn(usize, usize, usize) -> f64,
    sigma: &[f64],
    scratch: MaScratch<'_>,
    mut sink: impl FnMut(usize, &[f64]) -> bool,
) {
    let MaScratch {
        history,
        work,
        covariance,
    } = scratch;
    let kk = k * k;
    let slots = p.max(1);
    covariance.fill(0.0);
    // `history` is a ring of the last `p` weights; `Phi_i` sits in slot `i % slots`.
    for h in 0.. {
        let slot = h % slots;
        if h == 0 {
            history[..kk].fill(0.0);
            for i in 0..k {
                history[i * k + i] = 1.0;
            }
        } else {
            work.fill(0.0);
            for l in 1..=p.min(h) {
                let source = ((h - l) % slots) * kk;
                for i in 0..k {
                    for j in 0..k {
                        let value: f64 = (0..k)
                            .map(|m| weight(l, i, m) * history[source + m * k + j])
                            .sum();
                        work[i * k + j] += value;
                    }
                }
            }
            history[slot * kk..slot * kk + kk].copy_from_slice(work);
        }

        // `covariance += Phi Sigma Phi'`.
        let phi = &history[slot * kk..slot * kk + kk];
        for i in 0..k {
            for j in 0..k {
                let mut value = 0.0;
                for a in 0..k {
                    for b in 0..k {
                        value += phi[i * k + a] * sigma[a * k + b] * phi[j * k + b];
                    }
                }
                covariance[i * k + j] += value;
            }
        }

        if !sink(h, covariance) {
            return;
        }
    }
}
