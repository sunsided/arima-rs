//! Fitting GARCH(1,1) by Gaussian maximum likelihood, as R's `tseries::garch`.
//!
//! The likelihood is tseries': the variance starts at the mean square `v` of the series,
//! and the values from the second on contribute `-(ln h_t + x_t^2 / h_t) / 2` plus the
//! normal constant. tseries' optimiser often stops short of the maximum, so this fit runs
//! BFGS from a few starts instead. It works on unconstrained coordinates that keep the
//! model valid and do not depend on the scale of the series:
//!
//! ```text
//! u = (ln(omega / v), logit(alpha + beta), logit(alpha / (alpha + beta)))
//! ```

use alloc::vec::Vec;

use super::FitError;
use super::bfgs;
use crate::garch::{GarchFilter, GarchSpec};
use crate::math;

/// The fewest values a fit accepts: enough for three parameters to be more than noise.
const MIN_VALUES: usize = 10;

/// The `(alpha, beta)` pairs the optimiser starts from, `omega` set so that the long-run
/// variance is the mean square. They span low and high persistence, and a weight on
/// either term.
const STARTS: [(f64, f64); 4] = [(0.05, 0.9), (0.1, 0.8), (0.2, 0.5), (0.1, 0.1)];

/// GARCH(1,1) to fit to a zero-mean series.
///
/// # Examples
///
/// ```
/// use backshift::{Garch, GarchFilter};
///
/// let x: Vec<f64> = (0..300).map(|t| ((t * 37 % 101) as f64 - 50.0) / 25.0).collect();
/// let fit = Garch::new().fit(&x).unwrap();
/// assert!(fit.alpha + fit.beta < 1.0);
/// let filter = GarchFilter::new(&fit.spec()).unwrap();
/// let mut variance = [0.0; 5];
/// filter.forecast_into(&mut variance);
/// assert!(variance.iter().all(|v| *v > 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Garch;

/// A fitted GARCH(1,1) model.
#[derive(Debug, Clone, PartialEq)]
pub struct GarchFit {
    /// The constant `omega`.
    pub omega: f64,
    /// The weight of the last squared value.
    pub alpha: f64,
    /// The weight of the last variance.
    pub beta: f64,
    /// The log-likelihood over the values from the second on, as R's `logLik` of a
    /// `tseries::garch` fit.
    pub loglike: f64,
    /// The conditional variance of each value given the ones before it; the first is the
    /// mean square of the series.
    pub variances: Vec<f64>,
    /// The conditional variance of the next value.
    pub variance: f64,
}

impl Garch {
    /// GARCH(1,1) without a mean: fit it to returns, or to the innovations of a fitted
    /// ARIMA or ETS model.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Fits the model to `x` by maximum likelihood.
    ///
    /// # Errors
    ///
    /// - [`FitError::Unsupported`] if a value is not finite.
    /// - [`FitError::TooShort`] for fewer than 10 values.
    /// - [`FitError::Degenerate`] if the values are all 0, or no start can be scored.
    pub fn fit(&self, x: &[f64]) -> Result<GarchFit, FitError> {
        if !math::all_finite(x) {
            return Err(FitError::Unsupported);
        }

        if x.len() < MIN_VALUES {
            return Err(FitError::TooShort {
                needed: MIN_VALUES,
                present: x.len(),
            });
        }

        let scale = x.iter().map(|v| v * v).sum::<f64>() / count_as_f64(x.len());
        if !(scale > 0.0 && scale.is_finite()) {
            return Err(FitError::Degenerate);
        }

        let mut scratch = alloc::vec![0.0; bfgs::scratch_len(3)];
        let u = STARTS
            .iter()
            .map(|&(alpha, beta)| {
                let mut u = coordinates(alpha, beta);
                bfgs::minimize(
                    |u| deviance(x, scale, u),
                    &mut u,
                    &mut scratch,
                    bfgs::Options::default(),
                );
                (deviance(x, scale, &u), u)
            })
            .filter(|(value, _)| value.is_finite())
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, u)| u)
            .ok_or(FitError::Degenerate)?;

        Self::result(x, scale, &u).ok_or(FitError::Degenerate)
    }

    /// The fit at the optimum `u`, with the conditional variances; `None` if the
    /// parameters round to an invalid model (persistence 1).
    fn result(x: &[f64], scale: f64, u: &[f64; 3]) -> Option<GarchFit> {
        let spec = spec(scale, u);
        let mut filter = GarchFilter::new(&spec).ok()?;
        let variances = x.iter().map(|&v| filter.update(v).variance).collect();
        let terms = count_as_f64(x.len() - 1);

        Some(GarchFit {
            omega: spec.omega,
            alpha: spec.alpha,
            beta: spec.beta,
            loglike: -0.5 * deviance(x, scale, u) - 0.5 * terms * math::ln(core::f64::consts::TAU),
            variances,
            variance: filter.variance(),
        })
    }
}

impl GarchFit {
    /// The fitted model at the end of the series, for a
    /// [`GarchFilter`] that continues from there.
    #[must_use]
    pub const fn spec(&self) -> GarchSpec {
        GarchSpec {
            omega: self.omega,
            alpha: self.alpha,
            beta: self.beta,
            variance: self.variance,
        }
    }
}

/// The unconstrained coordinates of the start `(alpha, beta)`, with `omega` chosen so that
/// the long-run variance is the mean square: `omega / v = 1 - alpha - beta`.
fn coordinates(alpha: f64, beta: f64) -> [f64; 3] {
    let persistence = alpha + beta;
    [
        math::ln(1.0 - persistence),
        logit(persistence),
        logit(alpha / persistence),
    ]
}

/// The model at the coordinates `u`, for mean square `scale`, with the variance of the
/// first value at `scale`.
fn spec(scale: f64, u: &[f64]) -> GarchSpec {
    let persistence = logistic(u[1]);
    let alpha = persistence * logistic(u[2]);
    GarchSpec {
        omega: scale * math::exp(u[0]),
        alpha,
        beta: persistence - alpha,
        variance: scale,
    }
}

/// Twice the negative log-likelihood without its constant: `sum_(t >= 1) ln h_t + x_t^2 /
/// h_t`, with `h_0 = scale`. Infinite where the parameters are not a valid model.
fn deviance(x: &[f64], scale: f64, u: &[f64]) -> f64 {
    let Ok(mut filter) = GarchFilter::new(&spec(scale, u)) else {
        return f64::INFINITY;
    };
    let Some((&first, rest)) = x.split_first() else {
        return f64::INFINITY;
    };

    filter.update(first);
    rest.iter()
        .map(|&v| {
            let h = filter.update(v).variance;
            math::ln(h) + v * v / h
        })
        .sum()
}

/// The logistic function, `1 / (1 + e^-u)`.
fn logistic(u: f64) -> f64 {
    1.0 / (1.0 + math::exp(-u))
}

/// The inverse of [`logistic`], for `0 < p < 1`.
fn logit(p: f64) -> f64 {
    math::ln(p / (1.0 - p))
}

/// A count as a float; exact up to 2^53 values, far beyond any series.
#[expect(
    clippy::cast_precision_loss,
    reason = "value counts stay far below 2^53, where the conversion is exact"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}
