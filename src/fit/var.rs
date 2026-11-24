//! Least-squares fitting of VAR and VARX models, and lag order selection, as statsmodels'
//! `VAR(y, exog).fit(p)` and `select_order`.
//!
//! Every equation shares the regressors `[1, y_(t-1), ..., y_(t-p), x_t]`, so one Gram
//! matrix serves all of them: the coefficients are `(X'X)^-1 X'Y`, solved by Cholesky. The
//! inverse Gram matrix is kept, so an [`OnlineVar`](crate::OnlineVar) can continue the fit
//! value by value with exact recursive least squares.

use super::FitError;
use crate::linalg;
use crate::math;
use crate::normal::normal_quantile;
use crate::var::{MaScratch, VarSpec, forecast_covariances};
use alloc::vec;
use alloc::vec::Vec;

/// A VAR(p) model to fit, optionally with exogenous regressors (VARX).
///
/// # Examples
///
/// ```
/// use backshift::Var;
///
/// // Two series, row-major: each row holds one value of each.
/// let mut y = vec![0.0; 2 * 120];
/// for t in 1..120 {
///     let shock = ((t * 7919) % 13) as f64 / 13.0 - 0.5;
///     y[2 * t] = 0.5 * y[2 * t - 2] + 0.2 * y[2 * t - 1] + shock;
///     y[2 * t + 1] = 0.3 * y[2 * t - 1] + 0.5 * shock;
/// }
/// let fit = Var::new(1).fit(&y, 2).unwrap();
/// assert_eq!(fit.lags.len(), 4);
/// let forecast = fit.forecast(3, &[]);
/// assert_eq!(forecast.mean.len(), 6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var {
    /// The lag order `p`.
    p: usize,
}

impl Var {
    /// A VAR(`p`) with an intercept in each equation.
    #[must_use]
    pub const fn new(p: usize) -> Self {
        Self { p }
    }

    /// Fits the model to `y`, row-major with `k` values per row.
    ///
    /// # Errors
    ///
    /// As [`Var::fit_exog`].
    pub fn fit(&self, y: &[f64], k: usize) -> Result<VarFit, FitError> {
        self.fit_exog(y, k, &[], 0)
    }

    /// Fits the model with `columns` exogenous regressors (a VARX): `regressors` holds one
    /// row of `columns` values per row of `y`, row-major, and enters every equation, as
    /// statsmodels' `VAR(y, exog=x)`.
    ///
    /// # Errors
    ///
    /// - [`FitError::RegressorShape`] if `y` does not hold whole rows of `k > 0` values, or
    ///   `regressors` does not hold one row per row of `y`.
    /// - [`FitError::Unsupported`] if a value is not finite.
    /// - [`FitError::TooShort`] if fewer rows remain after the first `p` than regressors
    ///   per equation, `k p + 1 + columns`.
    /// - [`FitError::Degenerate`] if the regressors are collinear or the residuals have no
    ///   positive definite covariance.
    pub fn fit_exog(
        &self,
        y: &[f64],
        k: usize,
        regressors: &[f64],
        columns: usize,
    ) -> Result<VarFit, FitError> {
        let data = Data::new(y, k, regressors, columns)?;
        estimate(&data, self.p, self.p)
    }
}

/// The checked input of a fit: the series and its exogenous regressors.
struct Data<'a> {
    /// The series, row-major.
    y: &'a [f64],
    /// The number of series.
    k: usize,
    /// The exogenous regressors, row-major.
    x: &'a [f64],
    /// The number of exogenous regressors.
    columns: usize,
    /// The number of rows.
    n: usize,
}

impl<'a> Data<'a> {
    /// Checks the shapes and values of a fit's input; see [`Var::fit_exog`].
    fn new(y: &'a [f64], k: usize, x: &'a [f64], columns: usize) -> Result<Self, FitError> {
        if k == 0 || y.len() % k != 0 {
            return Err(FitError::RegressorShape {
                expected: y.len().next_multiple_of(k.max(1)),
                got: y.len(),
            });
        }

        let n = y.len() / k;
        if x.len() != n * columns {
            return Err(FitError::RegressorShape {
                expected: n * columns,
                got: x.len(),
            });
        }

        if !math::all_finite(y) || !math::all_finite(x) {
            return Err(FitError::Unsupported);
        }

        Ok(Self {
            y,
            k,
            x,
            columns,
            n,
        })
    }

    /// The regressors of row `t >= p`: 1, the rows `t - 1, ..., t - p`, then `x_t`.
    fn regressors(&self, t: usize, p: usize, out: &mut [f64]) {
        let k = self.k;
        let (constant, rest) = out.split_at_mut(1);
        constant[0] = 1.0;
        let (lags, exogenous) = rest.split_at_mut(p * k);
        for (l, slot) in lags.chunks_exact_mut(k).enumerate() {
            let source = (t - 1 - l) * k;
            slot.copy_from_slice(&self.y[source..source + k]);
        }
        let row = t * self.columns;
        exogenous.copy_from_slice(&self.x[row..row + self.columns]);
    }
}

/// Fits a VAR(`p`) on the rows `start..n` (`start >= p`); statsmodels' `select_order` starts
/// every order at the largest one, so all see the same rows.
fn estimate(data: &Data<'_>, p: usize, start: usize) -> Result<VarFit, FitError> {
    let (k, total) = (data.k, data.n);
    let d = k * p + 1 + data.columns;
    let rows = total.saturating_sub(start);
    if rows <= d {
        return Err(FitError::TooShort {
            needed: start + d + 1,
            present: total,
        });
    }

    // `X'X` and `X'Y` over the rows, then the coefficients `B = (X'X)^-1 X'Y`.
    let mut gram = vec![0.0; d * d];
    let mut moment = vec![0.0; d * k];
    let mut regressors = vec![0.0; d];
    for t in start..total {
        data.regressors(t, p, &mut regressors);
        let target = &data.y[t * k..(t + 1) * k];
        for (i, &zi) in regressors.iter().enumerate() {
            for (g, &zj) in gram[i * d..(i + 1) * d].iter_mut().zip(&regressors) {
                *g += zi * zj;
            }
            for (m, &yj) in moment[i * k..(i + 1) * k].iter_mut().zip(target) {
                *m += zi * yj;
            }
        }
    }

    let mut factor = gram;
    if !linalg::cholesky(&mut factor, d, d) {
        return Err(FitError::Degenerate);
    }

    let coefficients = solve_columns(&factor, d, &moment, k);
    let mut identity = vec![0.0; d * d];
    for i in 0..d {
        identity[i * d + i] = 1.0;
    }
    let gram_inverse = solve_columns(&factor, d, &identity, d);

    let covariance = residual_covariance(data, p, start, &coefficients, d);
    VarFit::new(data, p, rows, &coefficients, gram_inverse, &covariance)
}

/// Solves `G X = R` for the `d x c` matrix `X`, column by column, with the Cholesky factor
/// of `G`. `rhs` and the result are row-major `d x c`.
fn solve_columns(factor: &[f64], d: usize, rhs: &[f64], c: usize) -> Vec<f64> {
    let mut out = vec![0.0; d * c];
    let mut column = vec![0.0; d];
    for j in 0..c {
        for (i, value) in column.iter_mut().enumerate() {
            *value = rhs[i * c + j];
        }
        linalg::solve_lower(factor, d, d, &mut column);
        linalg::solve_upper_transposed(factor, d, d, &mut column);
        for (i, value) in column.iter().enumerate() {
            out[i * c + j] = *value;
        }
    }

    out
}

/// `U'U`, the cross products of the residuals of the rows `start..n`, `k x k` row-major.
fn residual_covariance(
    data: &Data<'_>,
    p: usize,
    start: usize,
    coefficients: &[f64],
    d: usize,
) -> Vec<f64> {
    let k = data.k;
    let mut cross = vec![0.0; k * k];
    let mut z = vec![0.0; d];
    let mut residual = vec![0.0; k];
    for t in start..data.n {
        data.regressors(t, p, &mut z);
        for (j, r) in residual.iter_mut().enumerate() {
            let fitted: f64 = z
                .iter()
                .enumerate()
                .map(|(i, zi)| zi * coefficients[i * k + j])
                .sum();
            *r = data.y[t * k + j] - fitted;
        }
        for (i, &ri) in residual.iter().enumerate() {
            for (c, &rj) in cross[i * k..(i + 1) * k].iter_mut().zip(&residual) {
                *c += ri * rj;
            }
        }
    }

    cross
}

/// A fitted VAR or VARX model.
///
/// The covariances and criteria follow statsmodels: `sigma` divides the residual cross
/// products by `nobs - d`, `sigma_mle` by `nobs`, with `d = k p + 1 + columns` regressors per
/// equation.
#[derive(Debug, Clone, PartialEq)]
pub struct VarFit {
    /// The number of series `k`.
    pub k: usize,
    /// The lag order `p`.
    pub p: usize,
    /// The number of exogenous regressors.
    pub columns: usize,
    /// The intercepts, one per series.
    pub intercept: Vec<f64>,
    /// The lag matrices `A_1, ..., A_p`, each `k x k` row-major, as in [`VarSpec::lags`].
    pub lags: Vec<f64>,
    /// The exogenous coefficients, `k x columns` row-major: row `i` is equation `i`.
    pub beta: Vec<f64>,
    /// The residual covariance over `nobs - d` (statsmodels' `sigma_u`).
    pub sigma: Vec<f64>,
    /// The residual covariance over `nobs` (statsmodels' `sigma_u_mle`).
    pub sigma_mle: Vec<f64>,
    /// The inverse Gram matrix `(X'X)^-1` of the regressors `[1, y lags, x]`, `d x d`
    /// row-major. Without exogenous regressors it starts the learning of
    /// [`OnlineVar::with_learning`](crate::OnlineVar::with_learning).
    pub gram_inverse: Vec<f64>,
    /// The number of rows the fit used, after the first `p`.
    pub nobs: usize,
    /// The Gaussian log-likelihood at `sigma_mle`.
    pub loglike: f64,
    /// The last `p` rows of the series, row-major, oldest first, which forecasts start from.
    pub last: Vec<f64>,
    /// `ln det sigma_mle`, which every criterion starts from.
    log_det: f64,
}

impl VarFit {
    /// Collects the fit from its coefficients `B` (`d x k`, rows `[1, lags, x]`).
    fn new(
        data: &Data<'_>,
        p: usize,
        rows: usize,
        coefficients: &[f64],
        gram_inverse: Vec<f64>,
        cross: &[f64],
    ) -> Result<Self, FitError> {
        let (k, columns) = (data.k, data.columns);
        let d = k * p + 1 + columns;
        let n = count_as_f64(rows);
        let sigma: Vec<f64> = cross.iter().map(|c| c / count_as_f64(rows - d)).collect();
        let sigma_mle: Vec<f64> = cross.iter().map(|c| c / n).collect();
        let mut factor = sigma_mle.clone();
        if !linalg::cholesky(&mut factor, k, k) {
            return Err(FitError::Degenerate);
        }
        let log_det = linalg::log_det(&factor, k, k);

        let weight = |row: usize, equation: usize| coefficients[row * k + equation];
        let intercept = (0..k).map(|i| weight(0, i)).collect();
        let mut lags = vec![0.0; p * k * k];
        for l in 1..=p {
            for i in 0..k {
                for j in 0..k {
                    lags[(l - 1) * k * k + i * k + j] = weight(1 + (l - 1) * k + j, i);
                }
            }
        }
        let mut beta = vec![0.0; k * columns];
        for i in 0..k {
            for c in 0..columns {
                beta[i * columns + c] = weight(1 + p * k + c, i);
            }
        }

        let kf = count_as_f64(k);
        Ok(Self {
            k,
            p,
            columns,
            intercept,
            lags,
            beta,
            sigma,
            sigma_mle,
            gram_inverse,
            nobs: rows,
            loglike: -0.5 * n * kf * (math::ln(core::f64::consts::TAU) + 1.0) - 0.5 * n * log_det,
            last: data.y[(data.n - p) * k..].to_vec(),
            log_det,
        })
    }

    /// The number of regressors per equation, `d = k p + 1 + columns`.
    const fn regressors(&self) -> usize {
        self.k * self.p + 1 + self.columns
    }

    /// statsmodels' free parameters, `k d`.
    fn free(&self) -> f64 {
        count_as_f64(self.k * self.regressors())
    }

    /// The Akaike criterion `ln det sigma_mle + 2 k d / nobs`, statsmodels' `aic`.
    #[must_use]
    pub fn aic(&self) -> f64 {
        self.log_det + 2.0 * self.free() / count_as_f64(self.nobs)
    }

    /// The Bayesian criterion `ln det sigma_mle + ln(nobs) k d / nobs`, statsmodels' `bic`.
    #[must_use]
    pub fn bic(&self) -> f64 {
        let n = count_as_f64(self.nobs);
        self.log_det + math::ln(n) * self.free() / n
    }

    /// The Hannan–Quinn criterion `ln det sigma_mle + 2 ln(ln nobs) k d / nobs`.
    #[must_use]
    pub fn hqic(&self) -> f64 {
        let n = count_as_f64(self.nobs);
        self.log_det + 2.0 * math::ln(math::ln(n)) * self.free() / n
    }

    /// The final prediction error `((nobs + d) / (nobs - d))^k det sigma_mle`.
    #[must_use]
    pub fn fpe(&self) -> f64 {
        let (n, d) = (count_as_f64(self.nobs), count_as_f64(self.regressors()));
        let ratio = math::ln((n + d) / (n - d));
        math::exp(count_as_f64(self.k) * ratio + self.log_det)
    }

    /// The fitted model, for an [`OnlineVar`](crate::OnlineVar). The exogenous part stays
    /// with the fit: the online filter models the series alone.
    #[must_use]
    pub fn spec(&self) -> VarSpec<'_> {
        VarSpec {
            intercept: &self.intercept,
            lags: &self.lags,
            sigma: &self.sigma,
        }
    }

    /// Forecasts `h` rows after the series, with `future` holding the exogenous rows of the
    /// horizon (`h` rows of `columns` values; empty without regressors), as statsmodels'
    /// `forecast` and `forecast_cov`.
    ///
    /// # Panics
    ///
    /// Panics if `future` does not hold `h` rows of `columns` values.
    #[must_use]
    pub fn forecast(&self, h: usize, future: &[f64]) -> VarForecast {
        assert_eq!(
            future.len(),
            h * self.columns,
            "one exogenous row per horizon"
        );
        let k = self.k;
        let mut history = self.last.clone();
        let mut mean = vec![0.0; h * k];
        for (step, row) in mean.chunks_exact_mut(k).enumerate() {
            for (i, value) in row.iter_mut().enumerate() {
                let mut sum = self.intercept[i];
                for l in 1..=self.p {
                    let lagged = &history[history.len() - l * k..history.len() - (l - 1) * k];
                    let weights =
                        &self.lags[(l - 1) * k * k + i * k..(l - 1) * k * k + (i + 1) * k];
                    sum += weights.iter().zip(lagged).map(|(a, y)| a * y).sum::<f64>();
                }
                let x = &future[step * self.columns..(step + 1) * self.columns];
                let beta = &self.beta[i * self.columns..(i + 1) * self.columns];
                *value = sum + beta.iter().zip(x).map(|(b, x)| b * x).sum::<f64>();
            }
            history.extend_from_slice(row);
        }

        let mut covariance = vec![0.0; h * k * k];
        if h > 0 {
            let (mut phi, mut work, mut running) = (
                vec![0.0; self.p.max(1) * k * k],
                vec![0.0; k * k],
                vec![0.0; k * k],
            );
            let weight = |l: usize, i: usize, j: usize| self.lags[(l - 1) * k * k + i * k + j];
            let scratch = MaScratch {
                history: &mut phi,
                work: &mut work,
                covariance: &mut running,
            };
            forecast_covariances(k, self.p, weight, &self.sigma, scratch, |step, cov| {
                covariance[step * k * k..(step + 1) * k * k].copy_from_slice(cov);
                step + 1 < h
            });
        }

        VarForecast {
            k,
            mean,
            covariance,
        }
    }
}

/// The forecasts of a [`VarFit`].
#[derive(Debug, Clone, PartialEq)]
pub struct VarForecast {
    /// The number of series.
    k: usize,
    /// The means, row-major: `k` values per horizon.
    pub mean: Vec<f64>,
    /// The forecast covariance of each horizon, `k x k` row-major, back to back.
    pub covariance: Vec<f64>,
}

impl VarForecast {
    /// Writes central intervals at `level` for every value, `mean ± z sqrt(variance)`, as
    /// statsmodels' `forecast_interval`, laid out like [`VarForecast::mean`].
    ///
    /// # Panics
    ///
    /// Panics if `lower` or `upper` does not have the length of `mean`.
    pub fn interval_into(&self, level: f64, lower: &mut [f64], upper: &mut [f64]) {
        assert!(
            lower.len() == self.mean.len() && upper.len() == self.mean.len(),
            "one bound per forecast value"
        );
        let k = self.k;
        let z = normal_quantile(f64::midpoint(1.0, level));
        for (index, (lo, up)) in lower.iter_mut().zip(upper.iter_mut()).enumerate() {
            let (h, i) = (index / k, index % k);
            let half_width = z * math::sqrt(self.covariance[h * k * k + i * k + i]);
            *lo = self.mean[index] - half_width;
            *up = self.mean[index] + half_width;
        }
    }
}

/// The lag orders each criterion picks, as statsmodels' `select_order(maxlags)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarOrders {
    /// The order with the lowest AIC.
    pub aic: usize,
    /// The order with the lowest BIC.
    pub bic: usize,
    /// The order with the lowest HQIC.
    pub hqic: usize,
    /// The order with the lowest FPE.
    pub fpe: usize,
}

/// Picks the VAR lag order of `y` (row-major, `k` values per row) up to `max_lags`, as
/// statsmodels' `VAR(y, exog).select_order(max_lags)`: every order from 0 is fitted on the
/// same rows, those after the first `max_lags`, and each criterion picks its lowest, the
/// smallest order on a tie.
///
/// # Errors
///
/// As [`Var::fit_exog`] for the largest order.
pub fn select_var_order(
    y: &[f64],
    k: usize,
    regressors: &[f64],
    columns: usize,
    max_lags: usize,
) -> Result<VarOrders, FitError> {
    let data = Data::new(y, k, regressors, columns)?;
    let fits = (0..=max_lags)
        .map(|p| estimate(&data, p, max_lags))
        .collect::<Result<Vec<_>, _>>()?;
    let best = |criterion: fn(&VarFit) -> f64| {
        fits.iter()
            .enumerate()
            .min_by(|a, b| {
                criterion(a.1)
                    .total_cmp(&criterion(b.1))
                    .then(a.0.cmp(&b.0))
            })
            .map_or(0, |(p, _)| p)
    };

    Ok(VarOrders {
        aic: best(VarFit::aic),
        bic: best(VarFit::bic),
        hqic: best(VarFit::hqic),
        fpe: best(VarFit::fpe),
    })
}

/// A count as a float. Counts here are row and parameter counts, far below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "row and parameter counts stay far below 2^53, where the conversion is exact"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}
