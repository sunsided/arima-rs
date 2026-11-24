//! Refitting a model on new data without a heap, for devices that adapt in place.
//!
//! [`ArimaSpec::refit`] re-estimates a model's coefficients by exact maximum likelihood
//! on a window of recent raw values. It starts from the model's current coefficients
//! (a warm start, so a model that still fits converges in a few iterations), runs the
//! same objective and optimiser as [`Arima::fit`](crate::Arima::fit), and writes the
//! new coefficients into a caller-owned buffer. All working memory comes from one
//! caller-owned scratch slice, sized by [`ArimaSpec::refit_scratch_len`]; a
//! [`Refitter`] holds both buffers inline.
//!
//! The running model is never touched: a failed refit (a stuck sensor gives
//! [`FitError::Degenerate`]) leaves it in place.
//!
//! ```mermaid
//! flowchart LR
//!     W["window"] -->|"difference"| D["w"]
//!     S["current spec"] -->|"unconstrain"| U["start u"]
//!     D --> B["BFGS on -llf / n"]
//!     U --> B
//!     B -->|constrain| N["new spec"]
//!     N -->|"Filter::new, warm_up"| F["new filter"]
//! ```

use super::objective::{self, Objective};
use super::{Arima, FitError, bfgs};
use crate::buffer::take;
use crate::filter::{ArimaSpec, Seasonal};
use crate::polynomial;

/// How [`ArimaSpec::refit`] estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefitOptions {
    /// Whether to estimate the mean of the differenced series (the mean for `d = D = 0`,
    /// the drift for `d = 1, D = 0`), starting from the spec's `mean`. Without it the
    /// mean is fixed at zero.
    pub mean: bool,
    /// The largest number of optimiser iterations, which bounds the time a refit takes.
    pub max_iterations: usize,
}

impl Default for RefitOptions {
    /// No mean, and the 200 iterations [`Arima::fit`](crate::Arima::fit) allows.
    fn default() -> Self {
        Self {
            mean: false,
            max_iterations: bfgs::Options::DEFAULT_MAX_ITERATIONS,
        }
    }
}

/// The result of a refit: the new model and how well it fits the window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Refit<'o> {
    /// The new model, with the orders and differencing of the old one. Its coefficients
    /// live in the `coefficients` buffer passed to the refit.
    pub spec: ArimaSpec<'o>,
    /// The maximised exact log-likelihood of the differenced window.
    pub loglike: f64,
    /// The number of non-missing differenced values the likelihood covers.
    pub nobs: usize,
    /// The optimiser iterations taken.
    pub iterations: usize,
    /// Whether the optimiser met its tolerance rather than its iteration limit.
    pub converged: bool,
}

impl ArimaSpec<'_> {
    /// The length of the `coefficients` buffer a refit writes:
    /// `p + q + P + Q + exog.len()`.
    #[must_use]
    pub const fn coefficient_len(&self) -> usize {
        self.ar.len()
            + self.ma.len()
            + self.seasonal.ar.len()
            + self.seasonal.ma.len()
            + self.exog.len()
    }

    /// The length of the `scratch` buffer a refit of this model over `window` raw values
    /// needs.
    ///
    /// It is roughly `4 r² + (exog.len() + 2) n` for `r` ARMA states and `n` differenced
    /// values, plus a little per parameter. As a `const fn` it can size a static array.
    #[must_use]
    pub const fn refit_scratch_len(&self, window: usize, options: RefitOptions) -> usize {
        let model = self.refit_model(options);
        let k = self.differencing_order();
        let n = window.saturating_sub(k);
        let columns = self.exog.len();
        let parameters = model.parameter_count(columns);

        (k + 1)
            + n
            + n * columns
            + parameters
            + 3 * self.largest_part()
            + bfgs::scratch_len(parameters)
            + Objective::scratch_len(&model, n, columns)
    }

    /// Re-estimates the coefficients on `window`, the most recent raw values.
    ///
    /// Like [`Arima::fit`](crate::Arima::fit), but heap-free and started from this
    /// spec's coefficients instead of Hannan–Rissanen. `NaN` and infinite values count
    /// as missing. A part of the start that is not stationary (or not invertible) starts
    /// at zero. The new coefficients go into `coefficients`, which the returned
    /// [`Refit::spec`] borrows.
    ///
    /// # Errors
    ///
    /// - [`FitError::BufferTooSmall`] if `coefficients` holds fewer than
    ///   [`ArimaSpec::coefficient_len`] values or `scratch` fewer than
    ///   [`ArimaSpec::refit_scratch_len`].
    /// - [`FitError::RegressorShape`] if the model has regressors; use
    ///   [`ArimaSpec::refit_exog`] then.
    /// - [`FitError::TooShort`] if the differenced window has fewer than
    ///   `p + q + P + Q + exog.len() + 2` non-missing values (one more with a mean).
    /// - [`FitError::Degenerate`] if the differenced window is constant (a stuck sensor)
    ///   or the model fits it exactly.
    pub fn refit<'o>(
        &self,
        window: &[f64],
        options: RefitOptions,
        coefficients: &'o mut [f64],
        scratch: &mut [f64],
    ) -> Result<Refit<'o>, FitError> {
        self.refit_exog(window, &[], options, coefficients, scratch)
    }

    /// Like [`ArimaSpec::refit`], for a model with regressors: `regressors` holds one row
    /// of `exog.len()` values per value of `window`, row-major. `beta` starts from the
    /// spec's `exog` and is estimated with the rest.
    ///
    /// # Errors
    ///
    /// As for [`ArimaSpec::refit`]; [`FitError::RegressorShape`] if `regressors` does not
    /// hold `window.len()` rows.
    pub fn refit_exog<'o>(
        &self,
        window: &[f64],
        regressors: &[f64],
        options: RefitOptions,
        coefficients: &'o mut [f64],
        scratch: &mut [f64],
    ) -> Result<Refit<'o>, FitError> {
        let columns = self.exog.len();
        if regressors.len() != window.len() * columns {
            return Err(FitError::RegressorShape {
                expected: window.len() * columns,
                got: regressors.len(),
            });
        }

        check_len(coefficients.len(), self.coefficient_len())?;
        check_len(scratch.len(), self.refit_scratch_len(window.len(), options))?;

        let model = self.refit_model(options);
        let k = self.differencing_order();
        let n = window.len().saturating_sub(k);
        let mut scratch = scratch;
        let delta = take(&mut scratch, k + 1);
        polynomial::differencing(self.d, self.seasonal.d, self.seasonal.period, delta);
        let dy = take(&mut scratch, n);
        difference_into(|t| window[t], delta, dy);
        let dx = take(&mut scratch, n * columns);
        for (j, column) in dx.chunks_exact_mut(n.max(1)).enumerate().take(columns) {
            difference_into(|t| regressors[t * columns + j], delta, column);
        }

        let present = (0..n)
            .filter(|&t| !dy[t].is_nan() && (0..columns).all(|j| !dx[j * n + t].is_nan()))
            .count();
        let needed = model.parameter_count(columns) + 2;
        if present < needed {
            return Err(FitError::TooShort { needed, present });
        }

        if is_constant(dy) {
            return Err(FitError::Degenerate);
        }

        let x = take(&mut scratch, model.parameter_count(columns));
        let work = take(&mut scratch, 3 * self.largest_part());
        self.start_into(options, x, work);
        let bfgs_scratch = take(&mut scratch, bfgs::scratch_len(x.len()));
        let mut objective = Objective::new(&model, dy, dx, columns, scratch);
        let estimate =
            objective::estimate(&mut objective, x, bfgs_scratch, options.max_iterations)?;

        let parameters = objective.parameters();
        let parts = [
            parameters.ar,
            parameters.ma,
            parameters.seasonal_ar,
            parameters.seasonal_ma,
            parameters.exog,
        ];
        let mut out = &mut coefficients[..];
        for part in parts {
            take(&mut out, part.len()).copy_from_slice(part);
        }

        let coefficients: &'o [f64] = coefficients;
        let (ar, rest) = coefficients.split_at(self.ar.len());
        let (ma, rest) = rest.split_at(self.ma.len());
        let (seasonal_ar, rest) = rest.split_at(self.seasonal.ar.len());
        let (seasonal_ma, rest) = rest.split_at(self.seasonal.ma.len());
        let spec = ArimaSpec {
            ar,
            d: self.d,
            ma,
            seasonal: Seasonal {
                ar: seasonal_ar,
                d: self.seasonal.d,
                ma: seasonal_ma,
                period: self.seasonal.period,
            },
            exog: &rest[..columns],
            mean: parameters.mean,
            sigma2: estimate.sigma2,
        };

        Ok(Refit {
            spec,
            loglike: estimate.loglike,
            nobs: estimate.nobs,
            iterations: estimate.iterations,
            converged: estimate.converged,
        })
    }

    /// The model this spec's orders describe, for the shared fitting core.
    pub(super) const fn refit_model(&self, options: RefitOptions) -> Arima {
        Arima {
            p: self.ar.len(),
            d: self.d,
            q: self.ma.len(),
            seasonal_p: self.seasonal.ar.len(),
            seasonal_d: self.seasonal.d,
            seasonal_q: self.seasonal.ma.len(),
            period: self.seasonal.period,
            mean: options.mean,
        }
    }

    /// The longest of the AR, MA, seasonal AR and seasonal MA parts.
    pub(super) const fn largest_part(&self) -> usize {
        let parts = [
            self.ar.len(),
            self.ma.len(),
            self.seasonal.ar.len(),
            self.seasonal.ma.len(),
        ];
        let mut largest = 0;
        let mut i = 0;
        while i < parts.len() {
            if parts[i] > largest {
                largest = parts[i];
            }

            i += 1;
        }

        largest
    }

    /// Writes the unconstrained warm start `[ar..., ma..., seasonal ar..., seasonal ma...,
    /// mean, beta...]` into `x`, with `work` as scratch of `3 largest_part()` values.
    pub(super) fn start_into(&self, options: RefitOptions, x: &mut [f64], work: &mut [f64]) {
        let parts = [
            (self.ar, true),
            (self.ma, false),
            (self.seasonal.ar, true),
            (self.seasonal.ma, false),
        ];
        let mut rest = x;
        for (part, negate) in parts {
            objective::unconstrain_part(part, negate, take(&mut rest, part.len()), work);
        }

        if options.mean {
            take(&mut rest, 1)[0] = self.mean;
        }

        rest.copy_from_slice(self.exog);
    }
}

/// Whether every present (non-`NaN`) value of `values` is the same, as from a stuck
/// sensor. Such a window has no finite likelihood maximum: a model either fits it exactly
/// or runs into a unit root, and neither is worth swapping in.
fn is_constant(values: &[f64]) -> bool {
    let mut present = values.iter().filter(|v| !v.is_nan());
    present
        .next()
        .is_none_or(|&first| present.all(|&v| v == first))
}

/// [`FitError::BufferTooSmall`] unless a buffer of `got` values holds `needed`.
fn check_len(got: usize, needed: usize) -> Result<(), FitError> {
    if got < needed {
        return Err(FitError::BufferTooSmall { needed, got });
    }

    Ok(())
}

/// Writes `sum_j delta_j v_(t-j)` for `t = k, k + 1, ...` into `out`, where `value(t)`
/// is the raw value at `t` and `k = delta.len() - 1`.
///
/// Non-finite raw values count as missing (`NaN`), and the arithmetic is that of
/// [`polynomial::apply`], so the result matches the differencing of `Arima::fit`.
fn difference_into(value: impl Fn(usize) -> f64, delta: &[f64], out: &mut [f64]) {
    let k = delta.len() - 1;
    let finite = |t: usize| {
        let v = value(t);
        if v.is_finite() { v } else { f64::NAN }
    };
    for (i, o) in out.iter_mut().enumerate() {
        let t = i + k;
        *o = polynomial::apply(delta, finite(t), (1..=k).map(|j| finite(t - j)));
    }
}

/// A heap-free refitter with its buffers inline: `SCRATCH` values of scratch and room
/// for `COEFFICIENTS` coefficients.
///
/// Size it with the `const fn`s of the model it refits. It is large (8 bytes per value),
/// so on a device keep it in a `static` or another long-lived place rather than on a
/// small stack.
///
/// # Examples
///
/// ```
/// use backshift::{ArimaSpec, Filter, RefitOptions, Refitter, Seasonal};
///
/// // The model's orders fix the buffer sizes; the coefficients are only a start.
/// const MODEL: ArimaSpec<'static> = ArimaSpec {
///     ar: &[0.5],
///     d: 0,
///     ma: &[0.2],
///     seasonal: Seasonal::NONE,
///     exog: &[],
///     mean: 0.0,
///     sigma2: 1.0,
/// };
/// const WINDOW: usize = 64;
/// const OPTIONS: RefitOptions = RefitOptions { mean: false, max_iterations: 100 };
///
/// let window: [f64; WINDOW] = core::array::from_fn(|t| ((t * 7919) % 13) as f64 / 6.0 - 1.0);
/// let mut refitter =
///     Refitter::<{ MODEL.refit_scratch_len(WINDOW, OPTIONS) }, { MODEL.coefficient_len() }>::new();
/// let refit = refitter.refit(&MODEL, &window, OPTIONS).unwrap();
/// assert!(refit.spec.sigma2 > 0.0);
///
/// // Swap in the new model, primed on the window it was fitted to.
/// let mut filter = Filter::<4>::new(&refit.spec).unwrap();
/// filter.warm_up(&window);
/// ```
#[derive(Debug, Clone)]
pub struct Refitter<const SCRATCH: usize, const COEFFICIENTS: usize> {
    /// The scratch every refit works in.
    scratch: [f64; SCRATCH],
    /// Holds the coefficients of the last refit, which its [`Refit::spec`] borrows.
    coefficients: [f64; COEFFICIENTS],
}

impl<const SCRATCH: usize, const COEFFICIENTS: usize> Refitter<SCRATCH, COEFFICIENTS> {
    /// A refitter with zeroed buffers; `const`, so it can initialise a `static`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            scratch: [0.0; SCRATCH],
            coefficients: [0.0; COEFFICIENTS],
        }
    }

    /// [`ArimaSpec::refit`] on the inline buffers.
    ///
    /// # Errors
    ///
    /// As for [`ArimaSpec::refit`].
    pub fn refit(
        &mut self,
        spec: &ArimaSpec<'_>,
        window: &[f64],
        options: RefitOptions,
    ) -> Result<Refit<'_>, FitError> {
        spec.refit(window, options, &mut self.coefficients, &mut self.scratch)
    }

    /// [`ArimaSpec::refit_exog`] on the inline buffers.
    ///
    /// # Errors
    ///
    /// As for [`ArimaSpec::refit_exog`].
    pub fn refit_exog(
        &mut self,
        spec: &ArimaSpec<'_>,
        window: &[f64],
        regressors: &[f64],
        options: RefitOptions,
    ) -> Result<Refit<'_>, FitError> {
        spec.refit_exog(
            window,
            regressors,
            options,
            &mut self.coefficients,
            &mut self.scratch,
        )
    }
}

impl<const SCRATCH: usize, const COEFFICIENTS: usize> Default for Refitter<SCRATCH, COEFFICIENTS> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::Arima;
    use std::vec;
    use std::vec::Vec;

    /// A seeded ARMA(1, 1) series with mean 2: `y_t = 2 + 0.6 (y_(t-1) - 2) + e_t + 0.3 e_(t-1)`.
    fn arma(n: usize, phi: f64) -> Vec<f64> {
        let mut state = 7_u32;
        let mut noise = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5
        };
        let (mut y, mut previous, mut e_previous) = (Vec::with_capacity(n), 0.0, 0.0);
        for _ in 0..n {
            let e = noise();
            previous = phi * previous + e + 0.3 * e_previous;
            e_previous = e;
            y.push(2.0 + previous);
        }

        y
    }

    /// Refits `spec` on `window` with exactly sized fresh buffers.
    fn refit(
        spec: &ArimaSpec<'_>,
        window: &[f64],
        options: RefitOptions,
    ) -> (Vec<f64>, Result<(f64, ArimaSpec<'static>), FitError>) {
        let mut coefficients = vec![0.0; spec.coefficient_len()];
        let mut scratch = vec![0.0; spec.refit_scratch_len(window.len(), options)];
        let result = spec
            .refit(window, options, &mut coefficients, &mut scratch)
            .map(|r| {
                (
                    r.loglike,
                    ArimaSpec {
                        ar: &[],
                        ma: &[],
                        seasonal: Seasonal::NONE,
                        exog: &[],
                        ..r.spec
                    },
                )
            });
        (coefficients, result)
    }

    #[test]
    fn warm_start_from_the_optimum_stays_there() {
        let y = arma(200, 0.6);
        let fitted = Arima::new(1, 0, 1).fit(&y).unwrap();
        let options = RefitOptions {
            mean: true,
            ..RefitOptions::default()
        };
        let (coefficients, result) = refit(&fitted.spec(), &y, options);
        let (loglike, spec) = result.unwrap();

        assert!(
            loglike >= fitted.loglike - 1e-9,
            "{loglike} vs {}",
            fitted.loglike
        );
        assert!(
            (coefficients[0] - fitted.ar[0]).abs() < 1e-5,
            "{coefficients:?}"
        );
        assert!(
            (coefficients[1] - fitted.ma[0]).abs() < 1e-5,
            "{coefficients:?}"
        );
        assert!((spec.mean - fitted.mean).abs() < 1e-5);
        assert!((spec.sigma2 / fitted.sigma2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cold_start_reaches_the_fit_maximum() {
        let y = arma(200, 0.6);
        let fitted = Arima::new(1, 0, 1).fit(&y).unwrap();
        let cold = ArimaSpec {
            ar: &[0.0],
            ma: &[0.0],
            ..ArimaSpec::default()
        };
        let options = RefitOptions {
            mean: true,
            ..RefitOptions::default()
        };
        let (_, result) = refit(&cold, &y, options);
        let (loglike, _) = result.unwrap();
        assert!(
            loglike >= fitted.loglike - 1e-6,
            "{loglike} vs {}",
            fitted.loglike
        );
    }

    #[test]
    fn tracks_a_changed_process() {
        // Fitted on phi = 0.2, refitted on data from phi = 0.8.
        let old = Arima::new(1, 0, 1).fit(&arma(200, 0.2)).unwrap();
        let window = arma(200, 0.8);
        let options = RefitOptions {
            mean: true,
            ..RefitOptions::default()
        };
        let (coefficients, result) = refit(&old.spec(), &window, options);
        result.unwrap();
        let reference = Arima::new(1, 0, 1).fit(&window).unwrap();
        assert!(
            (coefficients[0] - reference.ar[0]).abs() < 1e-4,
            "{coefficients:?} vs {reference:?}"
        );
    }

    #[test]
    fn seasonal_model_with_missing_values() {
        let mut y: Vec<f64> = arma(120, 0.5)
            .iter()
            .enumerate()
            .map(|(t, v)| v + [1.0, -0.5, 0.3, -0.8][t % 4] + 0.05 * t as f64)
            .collect();
        y[30] = f64::NAN;
        y[71] = f64::INFINITY;
        let model = Arima::new(0, 1, 1).seasonal(0, 1, 1, 4);
        let fitted = model.fit(&y).unwrap();
        let (coefficients, result) = refit(&fitted.spec(), &y, RefitOptions::default());
        let (loglike, _) = result.unwrap();
        assert!(
            loglike >= fitted.loglike - 1e-9,
            "{loglike} vs {}",
            fitted.loglike
        );
        assert!(
            (coefficients[1] - fitted.seasonal_ma[0]).abs() < 1e-4,
            "{coefficients:?}"
        );
    }

    #[test]
    fn regressors_are_refitted() {
        let n = 150;
        let x: Vec<f64> = (0..n)
            .map(|t| f64::from(u32::try_from(t % 7).unwrap()))
            .collect();
        let y: Vec<f64> = arma(n, 0.5)
            .iter()
            .zip(&x)
            .map(|(e, x)| e + 1.5 * x)
            .collect();
        let fitted = Arima::new(1, 0, 0).fit_exog(&y, &x, 1).unwrap();
        let spec = fitted.spec();
        let options = RefitOptions {
            mean: true,
            ..RefitOptions::default()
        };
        let mut coefficients = vec![0.0; spec.coefficient_len()];
        let mut scratch = vec![0.0; spec.refit_scratch_len(n, options)];
        let refit = spec
            .refit_exog(&y, &x, options, &mut coefficients, &mut scratch)
            .unwrap();
        assert!(refit.loglike >= fitted.loglike - 1e-9);
        assert!(
            (refit.spec.exog[0] - fitted.exog[0]).abs() < 1e-4,
            "{refit:?}"
        );

        let mut scratch = vec![0.0; spec.refit_scratch_len(n, options)];
        assert_eq!(
            spec.refit(&y, options, &mut coefficients, &mut scratch)
                .unwrap_err(),
            FitError::RegressorShape {
                expected: n,
                got: 0
            }
        );
    }

    #[test]
    fn errors_instead_of_panics() {
        let spec = ArimaSpec {
            ar: &[0.5],
            ..ArimaSpec::default()
        };
        let options = RefitOptions::default();
        let needed = spec.refit_scratch_len(30, options);
        let (mut coefficients, mut scratch) = (vec![0.0; 1], vec![0.0; needed]);

        assert_eq!(
            spec.refit(
                &[1.0; 30],
                options,
                &mut coefficients,
                &mut scratch[..needed - 1]
            ),
            Err(FitError::BufferTooSmall {
                needed,
                got: needed - 1
            })
        );
        assert_eq!(
            spec.refit(&[1.0; 30], options, &mut [], &mut scratch),
            Err(FitError::BufferTooSmall { needed: 1, got: 0 })
        );
        assert_eq!(
            spec.refit(&[1.0, 2.0], options, &mut coefficients, &mut scratch),
            Err(FitError::TooShort {
                needed: 3,
                present: 2
            })
        );
        let with_mean = RefitOptions {
            mean: true,
            ..options
        };
        let mut scratch = vec![0.0; spec.refit_scratch_len(30, with_mean)];
        assert_eq!(
            spec.refit(&[5.0; 30], with_mean, &mut coefficients, &mut scratch),
            Err(FitError::Degenerate)
        );
    }

    #[test]
    fn stuck_sensor_is_degenerate() {
        // Without a mean and with d = 0, an AR(1) on a constant window would run into a
        // unit root that `Filter::new` still accepts.
        let mut window = [5.0; 30];
        window[4] = f64::NAN;
        for spec in [
            ArimaSpec {
                ar: &[0.5],
                ..ArimaSpec::default()
            },
            ArimaSpec {
                ma: &[0.3],
                ..ArimaSpec::default()
            },
            ArimaSpec {
                ar: &[0.5],
                d: 1,
                ..ArimaSpec::default()
            },
        ] {
            let (_, result) = refit(&spec, &window, RefitOptions::default());
            assert_eq!(result.unwrap_err(), FitError::Degenerate, "{spec:?}");
        }
    }

    #[test]
    fn exact_regression_hits_the_noise_floor() {
        // y = 3 + 2 x exactly: the window varies, but from beta = 0 the optimiser runs
        // towards the exact fit, where only rounding noise is left.
        let x: Vec<f64> = (0..40).map(|t| f64::from((t * 7) % 11)).collect();
        let y: Vec<f64> = x.iter().map(|x| 3.0 + 2.0 * x).collect();
        let spec = ArimaSpec {
            ar: &[0.2],
            exog: &[0.0],
            ..ArimaSpec::default()
        };
        let options = RefitOptions {
            mean: true,
            ..RefitOptions::default()
        };
        let mut coefficients = vec![0.0; spec.coefficient_len()];
        let mut scratch = vec![0.0; spec.refit_scratch_len(y.len(), options)];
        assert_eq!(
            spec.refit_exog(&y, &x, options, &mut coefficients, &mut scratch),
            Err(FitError::Degenerate)
        );
    }

    #[test]
    fn large_offset_is_not_degenerate() {
        let window: Vec<f64> = arma(100, 0.5).iter().map(|v| 1e9 + 1e-4 * v).collect();
        let spec = ArimaSpec {
            ar: &[0.5],
            ..ArimaSpec::default()
        };
        let options = RefitOptions {
            mean: true,
            ..RefitOptions::default()
        };
        let (_, result) = refit(&spec, &window, options);
        result.unwrap();
    }
}
