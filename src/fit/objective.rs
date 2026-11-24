//! The fitting objective on caller-owned memory: unconstrained parameters in, the
//! concentrated log-likelihood out.
//!
//! The optimiser's vector `u` is laid out as `[ar..., ma..., seasonal ar..., seasonal ma...,
//! mean, beta...]`, with the mean present only when it is estimated. Each evaluation maps
//! `u` to stationary and invertible coefficients, removes the regression and the mean from
//! the differenced series, and filters the result. Nothing here allocates.

use super::{Arima, FitError, bfgs, transform};
use crate::buffer::take;
use crate::filter::{ArimaSpec, Seasonal};
use crate::likelihood::{concentrated, workspace_len};
use crate::math;

/// Constrained parameters, borrowed from an [`Objective`]'s buffer.
pub(crate) struct Parameters<'a> {
    /// The AR coefficients.
    pub(crate) ar: &'a [f64],
    /// The MA coefficients.
    pub(crate) ma: &'a [f64],
    /// The seasonal AR coefficients.
    pub(crate) seasonal_ar: &'a [f64],
    /// The seasonal MA coefficients.
    pub(crate) seasonal_ma: &'a [f64],
    /// The mean of the differenced series, or 0.
    pub(crate) mean: f64,
    /// The regression coefficients.
    pub(crate) exog: &'a [f64],
}

/// The concentrated log-likelihood of one model on one differenced series, as a function
/// of the unconstrained parameters.
pub(crate) struct Objective<'a> {
    /// The model being fitted.
    model: &'a Arima,
    /// The differenced series, `NaN` where missing.
    dy: &'a [f64],
    /// The differenced regressors, column-major: column `j` is `dx[j n..(j + 1) n]`.
    dx: &'a [f64],
    /// The number of regressors.
    columns: usize,
    /// Coefficients held fixed, laid out like `u` in constrained form: `NaN` marks a free
    /// one. Empty when every coefficient is free. A fixed coefficient overrides what `u`
    /// maps to, so the objective is flat in that coordinate and the optimiser leaves it.
    pinned: &'a [f64],
    /// The constrained parameters of the last evaluation, laid out like `u`.
    params: &'a mut [f64],
    /// Scratch for [`transform::constrain`].
    work: &'a mut [f64],
    /// The differenced series with the regression and the mean removed.
    centered: &'a mut [f64],
    /// Scratch for the filter.
    filter: &'a mut [f64],
}

impl<'a> Objective<'a> {
    /// The scratch an objective for `model` needs, for `n` differenced values and
    /// `columns` regressors.
    pub(crate) const fn scratch_len(model: &Arima, n: usize, columns: usize) -> usize {
        let parameters = model.parameter_count(columns);
        2 * parameters + n + workspace_len(model.arma_states())
    }

    /// An objective over the differenced series `dy` and the column-major differenced
    /// regressors `dx`, working in `scratch`, with the coefficients of `pinned` held fixed
    /// (see [`Objective::pinned`]).
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds fewer than [`Objective::scratch_len`] values.
    pub(crate) fn new(
        model: &'a Arima,
        dy: &'a [f64],
        dx: &'a [f64],
        columns: usize,
        pinned: &'a [f64],
        mut scratch: &'a mut [f64],
    ) -> Self {
        let parameters = model.parameter_count(columns);
        Self {
            model,
            dy,
            dx,
            columns,
            pinned,
            params: take(&mut scratch, parameters),
            work: take(&mut scratch, parameters),
            centered: take(&mut scratch, dy.len()),
            filter: scratch,
        }
    }

    /// The concentrated log-likelihood at the unconstrained `u`: `(loglike, sigma2, n)`.
    ///
    /// With `steady` it takes the steady-state shortcut, as the optimiser does. Returns
    /// [`None`] where the likelihood cannot be evaluated (see [`concentrated`]).
    pub(crate) fn evaluate(&mut self, u: &[f64], steady: bool) -> Option<(f64, f64, usize)> {
        self.constrain(u);
        let parameters = split(self.model, self.params);
        residual_into(
            self.dy,
            self.dx,
            self.columns,
            parameters.exog,
            parameters.mean,
            self.centered,
        );

        concentrated(
            self.centered,
            &shape(self.model, &parameters),
            steady,
            self.filter,
        )
    }

    /// The scratch [`estimate`]'s CSS phase needs for `model` on `n` differenced values and
    /// `columns` regressors: the expanded AR and MA polynomials, the residuals, and a copy
    /// of the start to fall back to.
    #[cfg(feature = "alloc")]
    pub(crate) const fn css_scratch_len(model: &Arima, n: usize, columns: usize) -> usize {
        2 * model.arma_states() + n + model.parameter_count(columns)
    }

    /// The conditional-sum-of-squares objective at `u`, `0.5 ln(css / m)`: R's `CSS`, which
    /// conditions on the first `p + P s` values and zero earlier shocks, scaled like the
    /// exact objective `-loglike / n` so that both share their curvature near the optimum.
    ///
    /// It costs `O(n r)` against the exact likelihood's `O(n r²)`. Returns [`None`] if a
    /// value is missing, none is left after conditioning, or the residuals vanish.
    /// `scratch` holds at least [`Objective::css_scratch_len`] values.
    #[cfg(feature = "alloc")]
    pub(crate) fn css(&mut self, u: &[f64], mut scratch: &mut [f64]) -> Option<f64> {
        self.constrain(u);
        let parameters = split(self.model, self.params);
        residual_into(
            self.dy,
            self.dx,
            self.columns,
            parameters.exog,
            parameters.mean,
            self.centered,
        );

        let shape = shape(self.model, &parameters);
        let r = shape.arma_states();
        let phi = take(&mut scratch, r);
        let selection = take(&mut scratch, r);
        shape.expand_arma_into(phi, selection);

        let w = &*self.centered;
        let conditioned = self.model.p + self.model.seasonal_p * self.model.period;
        let residuals = &mut scratch[..w.len()];
        let mut sum = 0.0;
        for t in 0..w.len() {
            if t < conditioned {
                residuals[t] = 0.0;
                continue;
            }

            let mut e = w[t];
            for i in 1..=conditioned {
                e -= phi[i - 1] * w[t - i];
            }
            for j in 1..r.min(t + 1) {
                e -= selection[j] * residuals[t - j];
            }
            residuals[t] = e;
            sum += e * e;
        }

        let m = w.len().saturating_sub(conditioned);
        if m == 0 || !sum.is_finite() || sum <= 0.0 {
            return None;
        }

        Some(0.5 * math::ln(sum / m as f64))
    }

    /// The range (largest minus smallest) of the non-missing differenced values: the
    /// spread of the data, whatever its offset. Zero when they are all equal or none is
    /// present; the floor then only rejects `sigma2 = 0`, and a constant series is left to
    /// the callers (`refit` rejects it, `AutoArima` picks white noise for it).
    fn range(&self) -> f64 {
        let (low, high) = self
            .dy
            .iter()
            .filter(|v| !v.is_nan())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &v| {
                (low.min(v), high.max(v))
            });
        if high >= low { high - low } else { 0.0 }
    }

    /// The constrained parameters of the last [`Objective::evaluate`].
    pub(crate) fn parameters(&self) -> Parameters<'_> {
        split(self.model, self.params)
    }

    /// Maps `u` onto `params`.
    ///
    /// AR parts are negated after the transform and MA parts are not, as in statsmodels
    /// SARIMAX, so that both come out stationary or invertible. The mean and `beta` are
    /// copied unchanged.
    fn constrain(&mut self, u: &[f64]) {
        let model = self.model;
        let parts = [
            (model.p, true),
            (model.q, false),
            (model.seasonal_p, true),
            (model.seasonal_q, false),
        ];
        let mut start = 0;
        for (len, negate) in parts {
            let range = start..start + len;
            transform::constrain(
                &u[range.clone()],
                &mut self.params[range.clone()],
                self.work,
            );
            if negate {
                for value in &mut self.params[range] {
                    *value = -*value;
                }
            }

            start += len;
        }

        self.params[start..].copy_from_slice(&u[start..]);
        for (value, &pin) in self.params.iter_mut().zip(self.pinned) {
            if !pin.is_nan() {
                *value = pin;
            }
        }
    }
}

/// The innovation standard deviation, relative to the range of the differenced values,
/// below which a fit counts as exact: the model explains all but `1e-18` of the variance,
/// the likelihood is unbounded there, and how close the optimiser gets depends only on
/// when it stops. Relative to the range, an offset does not matter: `1e9` plus noise of
/// standard deviation `1e-4` is far above it.
const NOISE_FLOOR: f64 = 1e-9;

/// What [`estimate`] found at the optimum, with `sigma2` concentrated out.
pub(crate) struct Estimate {
    /// The exact log-likelihood of the differenced series.
    pub(crate) loglike: f64,
    /// The innovation variance.
    pub(crate) sigma2: f64,
    /// The number of non-missing values the likelihood covers.
    pub(crate) nobs: usize,
    /// The optimiser iterations taken.
    pub(crate) iterations: usize,
    /// Whether the optimiser met its tolerance rather than its iteration limit.
    pub(crate) converged: bool,
}

/// Maximises the likelihood of `objective` from the unconstrained start `x`, which is
/// overwritten with the optimum; [`Objective::parameters`] then holds its coefficients.
///
/// The optimiser takes the steady-state shortcut; the reported values do not.
/// `bfgs_scratch` holds at least [`bfgs::scratch_len`]`(x.len())` values.
///
/// With [`Css::WarmStart`] it first minimises the conditional sum of squares, as R's
/// default `CSS-ML` does, and hands both its optimum and its inverse Hessian to the exact
/// likelihood. The two objectives share their curvature near the optimum, so the expensive
/// phase starts nearly converged. If the CSS optimum is worse than `x` under the exact
/// likelihood (another local optimum), the exact phase starts from `x` as without it. The
/// result is the exact-likelihood optimum either way. With [`Css::Only`] it stops at the
/// CSS optimum, as R's `CSS`, and evaluates the exact likelihood there once.
///
/// # Errors
///
/// [`FitError::Degenerate`] if the likelihood cannot be evaluated at the optimum, or the
/// innovations left there are negligible (below [`NOISE_FLOOR`] of the data's range).
pub(crate) fn estimate(
    objective: &mut Objective<'_>,
    x: &mut [f64],
    bfgs_scratch: &mut [f64],
    max_iterations: usize,
    css: Css,
    css_scratch: &mut [f64],
) -> Result<Estimate, FitError> {
    let options = bfgs::Options {
        max_iterations,
        ..bfgs::Options::default()
    };

    // Without `alloc` only `Css::Off` exists, and nothing reads the scratch.
    #[cfg(not(feature = "alloc"))]
    let _ = css_scratch;
    let (iterations, converged) = match css {
        Css::Off => exact_phase(objective, x, bfgs_scratch, options, 0, false),
        #[cfg(feature = "alloc")]
        Css::WarmStart => {
            let (css_iterations, warm) =
                css_phase(objective, x, bfgs_scratch, options, css_scratch);
            exact_phase(objective, x, bfgs_scratch, options, css_iterations, warm)
        }
        #[cfg(feature = "alloc")]
        Css::Only => {
            let minimum = bfgs::minimize(
                |u| objective.css(u, css_scratch).unwrap_or(f64::INFINITY),
                x,
                bfgs_scratch,
                options,
            );
            (minimum.iterations, minimum.converged)
        }
    };

    let (loglike, sigma2, nobs) = objective.evaluate(x, false).ok_or(FitError::Degenerate)?;
    if math::sqrt(sigma2) <= NOISE_FLOOR * objective.range() {
        return Err(FitError::Degenerate);
    }

    Ok(Estimate {
        loglike,
        sigma2,
        nobs,
        iterations,
        converged,
    })
}

/// What [`estimate`] does with the conditional sum of squares. Both CSS modes take their
/// working memory from `css_scratch`, at least [`Objective::css_scratch_len`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Css {
    /// Nothing: the exact likelihood from the given start.
    Off,
    /// A warm start for the exact likelihood.
    #[cfg(feature = "alloc")]
    WarmStart,
    /// The CSS optimum is the estimate; the exact likelihood is only evaluated there.
    #[cfg(feature = "alloc")]
    Only,
}

/// Minimises the exact objective from `x`, seeded with the inverse Hessian in `bfgs_scratch`
/// when `warm`. Returns the iterations, `earlier` included, and whether it converged.
fn exact_phase(
    objective: &mut Objective<'_>,
    x: &mut [f64],
    bfgs_scratch: &mut [f64],
    options: bfgs::Options,
    earlier: usize,
    warm: bool,
) -> (usize, bool) {
    let minimum = bfgs::minimize(
        |u| exact(objective, u),
        x,
        bfgs_scratch,
        bfgs::Options {
            warm_hessian: warm,
            ..options
        },
    );
    (earlier + minimum.iterations, minimum.converged)
}

/// The exact objective the optimiser minimises: `-loglike / n` with the steady-state
/// shortcut, or `+inf` where it cannot be evaluated.
fn exact(objective: &mut Objective<'_>, u: &[f64]) -> f64 {
    match objective.evaluate(u, true) {
        Some((llf, _, n)) => -llf / n as f64,
        None => f64::INFINITY,
    }
}

/// The CSS phase of [`estimate`]: moves `x` to the CSS optimum if that is better under the
/// exact likelihood, leaving the CSS inverse Hessian in `bfgs_scratch`. Returns the CSS
/// iterations and whether that Hessian should seed the exact phase.
#[cfg(feature = "alloc")]
fn css_phase(
    objective: &mut Objective<'_>,
    x: &mut [f64],
    bfgs_scratch: &mut [f64],
    options: bfgs::Options,
    mut scratch: &mut [f64],
) -> (usize, bool) {
    let start = take(&mut scratch, x.len());
    start.copy_from_slice(x);
    let minimum = bfgs::minimize(
        |u| objective.css(u, scratch).unwrap_or(f64::INFINITY),
        x,
        bfgs_scratch,
        options,
    );

    if minimum.iterations > 0 && exact(objective, x) <= exact(objective, start) {
        return (minimum.iterations, true);
    }

    x.copy_from_slice(start);
    (minimum.iterations, false)
}

/// Writes the unconstrained form of `coefficients` into `out`, negated first for an AR
/// part, or zeros if they are not stationary (or not invertible), as statsmodels starts.
///
/// `work` holds at least `3 coefficients.len()` values.
pub(crate) fn unconstrain_part(
    coefficients: &[f64],
    negate: bool,
    out: &mut [f64],
    work: &mut [f64],
) {
    let n = coefficients.len();
    let (input, work) = work.split_at_mut(n);
    for (value, &c) in input.iter_mut().zip(coefficients) {
        *value = if negate { -c } else { c };
    }

    if !transform::unconstrain(input, out, work) {
        out[..n].fill(0.0);
    }
}

/// The ARMA shape of `model` at `parameters`, without mean or differencing.
fn shape<'p>(model: &Arima, parameters: &Parameters<'p>) -> ArimaSpec<'p> {
    ArimaSpec {
        ar: parameters.ar,
        d: model.d,
        ma: parameters.ma,
        seasonal: Seasonal {
            ar: parameters.seasonal_ar,
            d: model.seasonal_d,
            ma: parameters.seasonal_ma,
            period: model.period,
        },
        ..ArimaSpec::default()
    }
}

/// Splits a parameter vector laid out like `u` into its parts for `model`.
fn split<'p>(model: &Arima, params: &'p [f64]) -> Parameters<'p> {
    let (ar, rest) = params.split_at(model.p);
    let (ma, rest) = rest.split_at(model.q);
    let (seasonal_ar, rest) = rest.split_at(model.seasonal_p);
    let (seasonal_ma, rest) = rest.split_at(model.seasonal_q);
    let mean = if model.mean { rest[0] } else { 0.0 };
    Parameters {
        ar,
        ma,
        seasonal_ar,
        seasonal_ma,
        mean,
        exog: &rest[usize::from(model.mean)..],
    }
}

/// Writes `dy_t - dx_t . beta - mean` into `out`, with `dx` column-major.
pub(crate) fn residual_into(
    dy: &[f64],
    dx: &[f64],
    columns: usize,
    beta: &[f64],
    mean: f64,
    out: &mut [f64],
) {
    let n = dy.len();
    for (t, value) in out.iter_mut().enumerate().take(n) {
        let regression = (0..columns).map(|j| dx[j * n + t] * beta[j]).sum::<f64>();
        *value = dy[t] - regression - mean;
    }
}
