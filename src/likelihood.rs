//! The exact Gaussian log-likelihood of an ARIMA model, for any state dimension.
//!
//! This is the runtime-sized twin of [`Filter`](crate::Filter): the same recursion
//! from [`state_space`](crate::state_space), on `Vec` buffers sized to the model.
//! The series is differenced first (simple differencing, as R's `arima` does).

use crate::filter::{ArimaSpec, FilterError};
use crate::state_space::{self, Innovation, LyapunovScratch, System};
use crate::{math, polynomial};
use alloc::vec;
use alloc::vec::Vec;

/// Runs the filter over `diff(y, d) - mean` from the stationary start and returns
/// every innovation: `y.len() - d` of them, or none if `y` is shorter.
///
/// Matches statsmodels `SARIMAX(diff(y, d) - mean, order=(p, 0, q), trend="n").filter(params)`:
/// `value` is `forecasts_error` and `variance` is `forecasts_error_cov`. `NaN`
/// observations count as missing. For `d = 0` this is statsmodels
/// `ARIMA(y, order=(p, 0, q), trend="c")` with `const = mean`.
///
/// # Errors
///
/// [`FilterError::InvalidParameter`] or [`FilterError::NonStationary`], as for
/// [`Filter::new`](crate::Filter::new). There is no capacity limit.
///
/// # Panics
///
/// Panics if the model has regressors; use [`innovations_exog`] then.
pub fn innovations(y: &[f64], spec: &ArimaSpec<'_>) -> Result<Vec<Innovation>, FilterError> {
    innovations_exog(y, &[], spec)
}

/// Like [`innovations`], for a model with regressors: `regressors` holds one row of
/// `spec.exog.len()` values per observation, row-major, and the ARIMA part models
/// `y - x . beta`. Matches statsmodels `ARIMA(y, exog=X, ...)` with simple differencing.
///
/// # Errors
///
/// As for [`innovations`].
///
/// # Panics
///
/// Panics if `regressors` does not hold `y.len()` rows of `spec.exog.len()` values.
pub fn innovations_exog(
    y: &[f64],
    regressors: &[f64],
    spec: &ArimaSpec<'_>,
) -> Result<Vec<Innovation>, FilterError> {
    let w = demeaned_difference(&regressed(y, regressors, spec.exog), spec);
    let mut result = Vec::with_capacity(w.len());
    Workspace::new(spec)?.run(&w, |innovation| result.push(innovation));

    Ok(result)
}

/// The exact Gaussian log-likelihood of `y` under the model.
///
/// `-0.5 * sum(ln(2 pi) + ln F_t + v_t^2 / F_t)` over the innovations `v_t` with
/// variances `F_t` of [`innovations`], skipping missing (`NaN`) observations.
///
/// For `d > 0` this differs from statsmodels' default `ARIMA(y, order=(p, d, q))`,
/// which keeps the first `d` values in the likelihood through a diffuse start, by
/// a small amount (about 1e-5 on 60 values). It equals statsmodels with
/// `simple_differencing=True` and R's `arima`.
///
/// # Errors
///
/// As for [`innovations`].
///
/// # Panics
///
/// Panics if the model has regressors; use [`loglike_exog`] then.
///
/// # Examples
///
/// ```
/// use backshift::{loglike, ArimaSpec};
///
/// // White noise with variance 1: each term is ln(2 pi) + y^2.
/// let spec = ArimaSpec::default();
/// let llf = loglike(&[0.0, 0.0], &spec).unwrap();
/// assert!((llf + std::f64::consts::TAU.ln()).abs() < 1e-12);
/// ```
pub fn loglike(y: &[f64], spec: &ArimaSpec<'_>) -> Result<f64, FilterError> {
    loglike_exog(y, &[], spec)
}

/// Like [`loglike`], for a model with regressors; `regressors` as in [`innovations_exog`].
///
/// # Errors
///
/// As for [`innovations`].
///
/// # Panics
///
/// Panics if `regressors` does not hold `y.len()` rows of `spec.exog.len()` values.
pub fn loglike_exog(
    y: &[f64],
    regressors: &[f64],
    spec: &ArimaSpec<'_>,
) -> Result<f64, FilterError> {
    let w = demeaned_difference(&regressed(y, regressors, spec.exog), spec);
    let ln_two_pi = math::ln(core::f64::consts::TAU);
    let mut sum = 0.0;
    Workspace::new(spec)?.run(&w, |Innovation { value, variance }| {
        if !value.is_nan() {
            sum += ln_two_pi + math::ln(variance) + value * value / variance;
        }
    });

    Ok(-0.5 * sum)
}

/// The log-likelihood of the already differenced, demeaned series `w` under the ARMA part
/// of `shape` (its AR and MA terms, seasonal ones included), with `sigma2` concentrated out.
///
/// With `steady` it takes statsmodels' steady-state shortcut (see [`Workspace::run_steady`]),
/// which the fitting objective uses; without it the full recursion runs, which the fit
/// uses for the likelihood and `sigma2` it reports.
///
/// All innovation variances scale with `sigma2`, so the filter runs once with
/// `sigma2 = 1`, and the maximising `sigma2 = sum(v_t^2 / F_t) / n` follows in closed
/// form over the `n` non-missing values. Returns `(log-likelihood, sigma2, n)`, the
/// same maximum as the full likelihood, or [`None`] if the model is invalid, no
/// value is present, or `sigma2` is not positive and finite (the model fits `w` exactly, or
/// the values are so large that their squares overflow).
pub(crate) fn concentrated(
    w: &[f64],
    shape: &ArimaSpec<'_>,
    steady: bool,
) -> Option<(f64, f64, usize)> {
    let spec = ArimaSpec {
        d: 0,
        mean: 0.0,
        sigma2: 1.0,
        ..*shape
    };
    let (mut count, mut log_variances, mut weighted) = (0_usize, 0.0, 0.0);
    let sink = |Innovation { value, variance }: Innovation| {
        if !value.is_nan() {
            count += 1;
            log_variances += math::ln(variance);
            weighted += value * value / variance;
        }
    };
    let workspace = Workspace::new(&spec).ok()?;
    if steady {
        workspace.run_steady(w, sink);
    } else {
        workspace.run(w, sink);
    }
    if count == 0 {
        return None;
    }

    let n = count as f64;
    let sigma2 = weighted / n;
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return None;
    }

    let llf =
        -0.5 * (n * (math::ln(core::f64::consts::TAU) + 1.0 + math::ln(sigma2)) + log_variances);

    Some((llf, sigma2, count))
}

/// `u_t = y_t - x_t . beta`, with the same arithmetic as [`Filter`](crate::Filter).
///
/// # Panics
///
/// Panics if `regressors` does not hold `y.len()` rows of `beta.len()` values.
pub(crate) fn regressed(y: &[f64], regressors: &[f64], beta: &[f64]) -> Vec<f64> {
    let m = beta.len();
    assert_eq!(
        regressors.len(),
        y.len() * m,
        "one regressor row per observation"
    );
    y.iter()
        .enumerate()
        .map(|(t, &value)| {
            let row = &regressors[t * m..(t + 1) * m];
            value - row.iter().zip(beta).map(|(x, b)| x * b).sum::<f64>()
        })
        .collect()
}

/// `(1 - B)^d (1 - B^s)^D y - mean`: the series the ARMA part models.
fn demeaned_difference(y: &[f64], spec: &ArimaSpec<'_>) -> Vec<f64> {
    let mut w = differenced(y, spec);
    for value in &mut w {
        *value -= spec.mean;
    }

    w
}

/// `(1 - B)^d (1 - B^s)^D y`, through the same arithmetic as [`Filter`](crate::Filter):
/// `y.len() - k` values for `k = d + D s`, or none if `y` is shorter.
pub(crate) fn differenced(y: &[f64], spec: &ArimaSpec<'_>) -> Vec<f64> {
    let k = spec.differencing_order();
    let mut delta = vec![0.0; k + 1];
    polynomial::differencing(spec.d, spec.seasonal.d, spec.seasonal.period, &mut delta);

    (k..y.len())
        .map(|t| polynomial::apply(&delta, y[t], (1..=k).map(|j| y[t - j])))
        .collect()
}

/// statsmodels' default `ssm.tolerance`: the change in the innovation variance below
/// which the filter counts as converged.
const STEADY_STATE_TOLERANCE: f64 = 1e-19;

/// Heap buffers for one filter run, sized to the model's state dimension `r`.
struct Workspace {
    /// The padded first column of the transition matrix. Length `r`.
    phi: Vec<f64>,
    /// The padded selection vector. Length `r`.
    selection: Vec<f64>,
    /// The innovation variance.
    sigma2: f64,
    /// The predicted state. Length `r`.
    a: Vec<f64>,
    /// The predicted state covariance, `r x r` row-major.
    p: Vec<f64>,
    /// Scratch for the recursion. Length `r`.
    scratch: Vec<f64>,
}

impl Workspace {
    /// Allocates the buffers and computes the stationary starting covariance.
    fn new(spec: &ArimaSpec<'_>) -> Result<Self, FilterError> {
        spec.validate()?;
        let r = spec.arma_states();
        let mut workspace = Self {
            phi: vec![0.0; r],
            selection: vec![0.0; r],
            sigma2: spec.sigma2,
            a: vec![0.0; r],
            p: vec![0.0; r * r],
            scratch: vec![0.0; r],
        };
        let mut delta = vec![0.0; spec.differencing_order() + 1];
        spec.expand_into(&mut workspace.phi, &mut workspace.selection, &mut delta);

        let (mut power, mut product, mut term) =
            (vec![0.0; r * r], vec![0.0; r * r], vec![0.0; r * r]);
        let scratch = LyapunovScratch {
            power: &mut power,
            product: &mut product,
            term: &mut term,
        };
        let (system, _, p, _) = workspace.parts();
        if !state_space::stationary_covariance(system, p, r, scratch) {
            return Err(FilterError::NonStationary);
        }

        Ok(workspace)
    }

    /// Feeds every observation and hands each innovation to `sink`.
    fn run(mut self, y: &[f64], mut sink: impl FnMut(Innovation)) {
        let r = self.phi.len();
        let (system, a, p, scratch) = self.parts();
        for &value in y {
            sink(state_space::update(r, a, p, r, value, scratch));
            state_space::predict(system, a, p, r, scratch);
        }
    }

    /// Like [`Workspace::run`], but once the innovation variance stops changing (by less
    /// than [`STEADY_STATE_TOLERANCE`]) the covariance and gain are frozen, and each further
    /// step only moves the state: `O(r)` instead of `O(r^2)`.
    ///
    /// statsmodels takes the same shortcut by default; it moves results by about 1e-9, so
    /// it is used for the fitting objective only. A missing value after convergence resumes
    /// the full recursion from the frozen (still valid) covariance.
    fn run_steady(mut self, y: &[f64], mut sink: impl FnMut(Innovation)) {
        let r = self.phi.len();
        let (system, a, p, scratch) = self.parts();
        let mut steady = false;
        let mut gain = vec![0.0; r];
        let mut previous = f64::NAN;
        for &value in y {
            if steady && !value.is_nan() {
                let variance = p[0];
                let innovation = value - a[0];
                for (state, k) in a.iter_mut().zip(&gain) {
                    *state += k * innovation / variance;
                }

                state_space::transition_vector(system, a);
                sink(Innovation {
                    value: innovation,
                    variance,
                });
                continue;
            }

            if steady {
                // A missing value breaks the steady state; continue from the frozen covariance.
                steady = false;
                previous = f64::NAN;
            }

            sink(state_space::update(r, a, p, r, value, scratch));
            state_space::predict(system, a, p, r, scratch);
            if (p[0] - previous).abs() < STEADY_STATE_TOLERANCE {
                steady = true;
                // The steady-state update uses the predicted covariance's first column.
                gain.copy_from_slice(&p[..r]);
            }

            previous = p[0];
        }
    }

    /// Splits the workspace into the system matrices, the state, the covariance and scratch.
    fn parts(&mut self) -> (System<'_>, &mut [f64], &mut [f64], &mut [f64]) {
        let system = System {
            phi: &self.phi,
            selection: &self.selection,
            sigma2: self.sigma2,
        };

        (system, &mut self.a, &mut self.p, &mut self.scratch)
    }
}
