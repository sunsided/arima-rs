//! Maximum-likelihood fitting of ARIMA(p, d, q) models.
//!
//! [`Arima::fit`] differences the series, takes Hannan–Rissanen starting values, and
//! maximises the exact Gaussian likelihood with BFGS. Parameters are optimised in
//! an unconstrained space that maps onto stationary AR and invertible MA
//! coefficients (statsmodels' transforms), and `sigma2` is concentrated out, so the
//! optimiser only sees the AR, MA and mean parameters.
//!
//! ```mermaid
//! flowchart LR
//!     Y["y"] -->|"difference d times"| W["w"]
//!     W -->|Hannan–Rissanen| S["start"]
//!     S -->|unconstrain| U["u"]
//!     U -->|"BFGS on -llf / n"| U2["u*"]
//!     U2 -->|constrain| F["Fitted"]
//! ```

#[cfg(feature = "alloc")]
mod auto;
mod bfgs;
#[cfg(feature = "alloc")]
mod hannan_rissanen;
mod objective;
mod refit;
#[cfg(feature = "alloc")]
mod roots;
mod transform;

#[cfg(feature = "alloc")]
pub use auto::{AutoArima, AutoFit, Criterion, FourierFit, nsdiffs};
#[cfg(feature = "alloc")]
pub use hannan_rissanen::{ArmaCoefficients, hannan_rissanen};
pub use refit::{Refit, RefitOptions, Refitter};

#[cfg(feature = "alloc")]
use crate::filter::{ArimaSpec, Seasonal};
use crate::fourier::FourierError;
#[cfg(feature = "alloc")]
use crate::likelihood::differenced;
#[cfg(feature = "alloc")]
use crate::math;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::error::Error;
use core::fmt::{Display, Formatter};
#[cfg(feature = "alloc")]
use objective::Objective;

/// A seasonal ARIMA(p, d, q)(P, D, Q, s) model to fit.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "alloc")]
/// # fn main() {
/// use backshift::Arima;
///
/// let y = [0.3, 1.1, 0.4, -0.2, 0.9, 1.4, 0.6, 0.1, -0.5, 0.2, 0.8, 1.0, 0.3, -0.1, 0.4];
/// let fitted = Arima::new(1, 0, 0).fit(&y).unwrap();
/// assert!(fitted.ar[0].abs() < 1.0);
/// assert!(fitted.sigma2 > 0.0);
///
/// // ARIMA(1, 0, 0)(1, 0, 0, 4) on the same series.
/// let seasonal = Arima::new(1, 0, 0).seasonal(1, 0, 0, 4).fit(&y).unwrap();
/// assert_eq!(seasonal.seasonal_ar.len(), 1);
/// # }
/// # #[cfg(not(feature = "alloc"))]
/// # fn main() {}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arima {
    /// The AR order.
    p: usize,
    /// The differencing order.
    d: usize,
    /// The MA order.
    q: usize,
    /// The seasonal AR order.
    seasonal_p: usize,
    /// The seasonal differencing order.
    seasonal_d: usize,
    /// The seasonal MA order.
    seasonal_q: usize,
    /// The season length; 0 without a seasonal part.
    period: usize,
    /// Whether to estimate the mean of the differenced series.
    mean: bool,
}

impl Arima {
    /// The number of AR and MA coefficients, seasonal ones included.
    const fn coefficients(&self) -> usize {
        self.p + self.q + self.seasonal_p + self.seasonal_q
    }

    /// The number of optimised parameters with `columns` regressors: the coefficients,
    /// the mean if estimated, and `beta`.
    const fn parameter_count(&self, columns: usize) -> usize {
        self.coefficients() + self.mean as usize + columns
    }

    /// The state dimension of the expanded ARMA part, `max(p + P s, q + Q s + 1)`.
    const fn arma_states(&self) -> usize {
        let ar = self.p + self.seasonal_p * self.period;
        let ma = self.q + self.seasonal_q * self.period;
        if ar > ma { ar } else { ma + 1 }
    }
}

#[cfg(feature = "alloc")]
impl Arima {
    /// An ARIMA(`p`, `d`, `q`) model. It estimates a mean for `d = D = 0` and none
    /// otherwise, like statsmodels ARIMA and R's `arima`; change that with [`Arima::mean`].
    #[must_use]
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            period: 0,
            mean: d == 0,
        }
    }

    /// Adds a seasonal part `(P, D, Q, s)`. With `D > 0` the mean is no longer estimated
    /// by default.
    ///
    /// # Panics
    ///
    /// Panics if `period` is 0 while `P`, `D` or `Q` is not.
    #[must_use]
    pub fn seasonal(self, p: usize, d: usize, q: usize, period: usize) -> Self {
        assert!(
            period > 0 || p + d + q == 0,
            "a seasonal part needs a period of at least 1"
        );
        Self {
            seasonal_p: p,
            seasonal_d: d,
            seasonal_q: q,
            period,
            mean: self.mean && d == 0,
            ..self
        }
    }

    /// Whether to estimate the mean of the differenced series: the mean for `d = D = 0`,
    /// the drift for `d = 1, D = 0`.
    #[must_use]
    pub fn mean(self, include: bool) -> Self {
        Self {
            mean: include,
            ..self
        }
    }

    /// Fits the model to `y` by exact maximum likelihood.
    ///
    /// `NaN` and infinite values count as missing. A fit that stops at the iteration limit is still
    /// returned, with [`Fitted::converged`] set to `false`.
    ///
    /// # Errors
    ///
    /// - [`FitError::TooShort`] if the differenced series has fewer than
    ///   `p + q + P + Q + 2` non-missing values (one more with a mean).
    /// - [`FitError::Degenerate`] if the model fits the series exactly.
    pub fn fit(&self, y: &[f64]) -> Result<Fitted, FitError> {
        self.fit_exog(y, &[], 0)
    }

    /// Fits a regression with ARIMA errors: `y_t = x_t . beta + u_t`, where `u` follows
    /// this model. `regressors` holds `columns` values per observation, row-major.
    ///
    /// Like R's `arima(xreg=)` and statsmodels `ARIMA(exog=)` with simple differencing,
    /// the regressors are differenced along with `y`. `beta` starts from least squares
    /// on the differenced data and is estimated jointly with the rest. A `NaN` or infinite
    /// regressor makes its observation missing.
    ///
    /// # Errors
    ///
    /// - [`FitError::RegressorShape`] if `regressors` does not hold `y.len()` rows.
    /// - [`FitError::TooShort`] if the differenced series has fewer than
    ///   `p + q + P + Q + columns + 2` non-missing values (one more with a mean).
    /// - [`FitError::Degenerate`] if the model fits the series exactly.
    pub fn fit_exog(
        &self,
        y: &[f64],
        regressors: &[f64],
        columns: usize,
    ) -> Result<Fitted, FitError> {
        self.fit_exog_with(y, regressors, columns, None)
    }

    /// Like [`Arima::fit`], but the optimiser starts from the coefficients of `start`
    /// instead of Hannan–Rissanen, as R's `arima(init=)` and statsmodels' `start_params` do.
    ///
    /// `start` must have this model's orders, differencing and period, for example a
    /// previous fit of the same model ([`Fitted::spec`]) or one fitted to a related series.
    /// Its `mean` seeds the mean if this model estimates one; its `sigma2` is not used. A
    /// part of `start` that is not stationary (or not invertible) starts at zero.
    ///
    /// The benefit is speed and reproducibility: no source shows that supplied start values
    /// improve forecasts over the automatic start.
    ///
    /// # Errors
    ///
    /// [`FitError::StartMismatch`] if `start` has other orders than this model, otherwise
    /// as for [`Arima::fit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use backshift::Arima;
    ///
    /// let y = [0.3, 1.1, 0.4, -0.2, 0.9, 1.4, 0.6, 0.1, -0.5, 0.2, 0.8, 1.0, 0.3, -0.1, 0.4];
    /// let model = Arima::new(1, 0, 1);
    /// let first = model.fit(&y).unwrap();
    /// let again = model.fit_from(&y, &first.spec()).unwrap();
    /// assert!(again.iterations <= first.iterations);
    /// assert!((again.loglike - first.loglike).abs() < 1e-6);
    /// ```
    pub fn fit_from(&self, y: &[f64], start: &ArimaSpec<'_>) -> Result<Fitted, FitError> {
        self.fit_exog_from(y, &[], 0, start)
    }

    /// Like [`Arima::fit_exog`], started from `start` as in [`Arima::fit_from`]. `start.exog`
    /// seeds `beta` and must hold `columns` values.
    ///
    /// # Errors
    ///
    /// [`FitError::StartMismatch`] if `start` has other orders than this model or not
    /// `columns` regression coefficients, otherwise as for [`Arima::fit_exog`].
    pub fn fit_exog_from(
        &self,
        y: &[f64],
        regressors: &[f64],
        columns: usize,
        start: &ArimaSpec<'_>,
    ) -> Result<Fitted, FitError> {
        let options = RefitOptions {
            mean: self.mean,
            ..RefitOptions::default()
        };
        if start.refit_model(options) != *self || start.exog.len() != columns {
            return Err(FitError::StartMismatch);
        }

        self.fit_exog_with(y, regressors, columns, Some(start))
    }

    /// [`Arima::fit_exog`] from `start`, or from Hannan–Rissanen and least squares without one.
    fn fit_exog_with(
        &self,
        y: &[f64],
        regressors: &[f64],
        columns: usize,
        start: Option<&ArimaSpec<'_>>,
    ) -> Result<Fitted, FitError> {
        if regressors.len() != y.len() * columns {
            return Err(FitError::RegressorShape {
                expected: y.len() * columns,
                got: regressors.len(),
            });
        }

        let differencing = self.differencing();
        let dy = differenced(&missing_if_infinite(y), &differencing);
        let dx: Vec<f64> = (0..columns)
            .flat_map(|j| {
                let column: Vec<f64> = (0..y.len()).map(|t| regressors[t * columns + j]).collect();
                differenced(&missing_if_infinite(&column), &differencing)
            })
            .collect();

        let present = (0..dy.len())
            .filter(|&t| !residual_is_missing(&dy, &dx, columns, t))
            .count();
        let needed = self.coefficients() + columns + usize::from(self.mean) + 2;
        if present < needed {
            return Err(FitError::TooShort { needed, present });
        }

        let mut x = match start {
            Some(spec) => self.start_from(spec),
            None => {
                let beta_start = self.least_squares_start(&dy, &dx, columns);
                let mut residual = vec![0.0; dy.len()];
                objective::residual_into(&dy, &dx, columns, &beta_start, 0.0, &mut residual);
                self.start(&residual, &beta_start)
            }
        };

        let mut scratch = vec![0.0; Objective::scratch_len(self, dy.len(), columns)];
        let mut objective = Objective::new(self, &dy, &dx, columns, &mut scratch);
        let mut bfgs_scratch = vec![0.0; bfgs::scratch_len(x.len())];
        let max_iterations = bfgs::Options::default().max_iterations;
        let estimate =
            objective::estimate(&mut objective, &mut x, &mut bfgs_scratch, max_iterations)?;
        let parameters = objective.parameters();

        Ok(Fitted {
            ar: parameters.ar.to_vec(),
            d: self.d,
            ma: parameters.ma.to_vec(),
            seasonal_ar: parameters.seasonal_ar.to_vec(),
            seasonal_d: self.seasonal_d,
            seasonal_ma: parameters.seasonal_ma.to_vec(),
            period: self.period,
            exog: parameters.exog.to_vec(),
            with_mean: self.mean,
            mean: parameters.mean,
            sigma2: estimate.sigma2,
            loglike: estimate.loglike,
            nobs: estimate.nobs,
            iterations: estimate.iterations,
            converged: estimate.converged,
        })
    }

    /// Least-squares regression coefficients of the differenced `dy` on the differenced
    /// regressors `dx`, with an intercept when the mean is estimated. Zeros if the
    /// regression fails (too few complete rows or collinear regressors). `dx` is
    /// column-major with `columns` columns.
    fn least_squares_start(&self, dy: &[f64], dx: &[f64], columns: usize) -> Vec<f64> {
        if columns == 0 {
            return Vec::new();
        }

        let complete: Vec<usize> = (0..dy.len())
            .filter(|&t| !residual_is_missing(dy, dx, columns, t))
            .collect();
        let rows: Vec<Vec<f64>> = complete
            .iter()
            .map(|&t| {
                let mut row: Vec<f64> = (0..columns).map(|j| dx[j * dy.len() + t]).collect();
                if self.mean {
                    row.push(1.0);
                }
                row
            })
            .collect();
        let targets: Vec<f64> = complete.iter().map(|&t| dy[t]).collect();

        hannan_rissanen::least_squares(&rows, &targets)
            .map(|mut b| {
                b.truncate(columns);
                b
            })
            .unwrap_or_else(|| vec![0.0; columns])
    }

    /// A spec with this model's differencing orders and no coefficients.
    fn differencing(&self) -> ArimaSpec<'static> {
        ArimaSpec {
            d: self.d,
            seasonal: Seasonal {
                d: self.seasonal_d,
                period: self.period,
                ..Seasonal::NONE
            },
            ..ArimaSpec::default()
        }
    }

    /// Unconstrained starting values from the coefficients of `spec`, which has this
    /// model's orders, through the same path as a refit's warm start.
    fn start_from(&self, spec: &ArimaSpec<'_>) -> Vec<f64> {
        let options = RefitOptions {
            mean: self.mean,
            ..RefitOptions::default()
        };
        let mut x = vec![0.0; self.parameter_count(spec.exog.len())];
        let mut work = vec![0.0; 3 * spec.largest_part()];
        spec.start_into(options, &mut x, &mut work);
        x
    }

    /// Unconstrained starting values `[ar..., ma..., seasonal ar..., seasonal ma..., mean, beta...]`.
    ///
    /// Hannan–Rissanen supplies the non-seasonal AR and MA parts and the sample mean the
    /// mean; the seasonal parts start at zero. A part that comes out non-stationary (or
    /// non-invertible), or a failed regression, starts at zero instead, as statsmodels does.
    fn start(&self, w: &[f64], beta: &[f64]) -> Vec<f64> {
        let present: Vec<f64> = w.iter().copied().filter(|v| !v.is_nan()).collect();
        let sample_mean = present.iter().sum::<f64>() / present.len() as f64;
        let mean = if self.mean { sample_mean } else { 0.0 };
        let centered: Vec<f64> = w.iter().map(|v| v - mean).collect();

        let estimate = hannan_rissanen(&centered, self.p, self.q);
        let (ar, ma) = estimate.map_or((Vec::new(), Vec::new()), |s| (s.ar, s.ma));
        let mut start = vec![0.0; self.coefficients()];
        let mut work = vec![0.0; 3 * self.p.max(self.q)];
        if ar.len() == self.p {
            objective::unconstrain_part(&ar, true, &mut start[..self.p], &mut work);
        }

        if ma.len() == self.q {
            let ma_start = &mut start[self.p..self.p + self.q];
            objective::unconstrain_part(&ma, false, ma_start, &mut work);
        }

        if self.mean {
            start.push(mean);
        }

        start.extend_from_slice(beta);
        start
    }
}

/// `values` with every infinite entry replaced by `NaN`, so the fit treats it as missing,
/// as [`AutoArima`]'s differencing tests already do.
#[cfg(feature = "alloc")]
fn missing_if_infinite(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .map(|&v| if v.is_finite() { v } else { f64::NAN })
        .collect()
}

/// Whether observation `t` of the differenced data has a missing value; `dx` is
/// column-major with `columns` columns.
#[cfg(feature = "alloc")]
fn residual_is_missing(dy: &[f64], dx: &[f64], columns: usize, t: usize) -> bool {
    dy[t].is_nan() || (0..columns).any(|j| dx[j * dy.len() + t].is_nan())
}

/// A fitted seasonal ARIMA model.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct Fitted {
    /// The AR coefficients `phi_1, ..., phi_p`. Always stationary.
    pub ar: Vec<f64>,
    /// The differencing order.
    pub d: usize,
    /// The MA coefficients `theta_1, ..., theta_q`. Always invertible.
    pub ma: Vec<f64>,
    /// The seasonal AR coefficients `Phi_1, ..., Phi_P`. Always stationary.
    pub seasonal_ar: Vec<f64>,
    /// The seasonal differencing order.
    pub seasonal_d: usize,
    /// The seasonal MA coefficients `Theta_1, ..., Theta_Q`. Always invertible.
    pub seasonal_ma: Vec<f64>,
    /// The season length; 0 without a seasonal part.
    pub period: usize,
    /// The regression coefficients `beta`; empty without regressors.
    pub exog: Vec<f64>,
    /// Whether the mean of the differenced series was estimated.
    pub with_mean: bool,
    /// The mean of the differenced series; `0.0` if it was not estimated.
    pub mean: f64,
    /// The innovation variance at the optimum.
    pub sigma2: f64,
    /// The maximised exact log-likelihood of the differenced series.
    pub loglike: f64,
    /// The number of non-missing differenced values the likelihood covers.
    pub nobs: usize,
    /// The optimiser iterations taken.
    pub iterations: usize,
    /// Whether the optimiser met its tolerance rather than its iteration limit.
    pub converged: bool,
}

#[cfg(feature = "alloc")]
impl Fitted {
    /// The number of estimated parameters: `p + q + P + Q`, the regression coefficients,
    /// one for the mean if estimated, and one for `sigma2`. This is the `k` of the
    /// information criteria.
    #[must_use]
    pub fn parameters(&self) -> usize {
        self.ar.len()
            + self.ma.len()
            + self.seasonal_ar.len()
            + self.seasonal_ma.len()
            + self.exog.len()
            + usize::from(self.with_mean)
            + 1
    }

    /// Akaike's information criterion, `-2 loglike + 2k`.
    #[must_use]
    pub fn aic(&self) -> f64 {
        -2.0 * self.loglike + 2.0 * self.parameters() as f64
    }

    /// The small-sample corrected AIC, `AIC + 2k(k + 1) / (n - k - 1)` with `n = nobs`.
    /// Infinite when `n <= k + 1`.
    #[must_use]
    pub fn aicc(&self) -> f64 {
        let k = self.parameters() as f64;
        let n = self.nobs as f64;
        if n <= k + 1.0 {
            return f64::INFINITY;
        }

        self.aic() + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    }

    /// The Bayesian information criterion, `-2 loglike + k ln(n)` with `n = nobs`.
    ///
    /// `n` counts only non-missing values, as R's `arima` does. statsmodels also counts
    /// missing ones, so its BIC and AICc differ when the series has gaps.
    #[must_use]
    pub fn bic(&self) -> f64 {
        -2.0 * self.loglike + self.parameters() as f64 * math::ln(self.nobs as f64)
    }

    /// The fitted model as a spec, for [`Filter::new`](crate::Filter::new),
    /// [`loglike`](crate::loglike) and [`innovations`](crate::innovations).
    ///
    /// After [`Arima::fit_exog`] the spec carries the regression coefficients, so use
    /// [`loglike_exog`](crate::loglike_exog) and [`innovations_exog`](crate::innovations_exog)
    /// with the original regressors instead; the plain forms panic for it.
    #[must_use]
    pub fn spec(&self) -> ArimaSpec<'_> {
        ArimaSpec {
            ar: &self.ar,
            d: self.d,
            ma: &self.ma,
            seasonal: Seasonal {
                ar: &self.seasonal_ar,
                d: self.seasonal_d,
                ma: &self.seasonal_ma,
                period: self.period,
            },
            exog: &self.exog,
            mean: self.mean,
            sigma2: self.sigma2,
        }
    }
}

/// Why a model could not be fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitError {
    /// `regressors` does not hold one row of `columns` values per observation.
    RegressorShape {
        /// `y.len() * columns`.
        expected: usize,
        /// `regressors.len()`.
        got: usize,
    },
    /// The differenced series has too few non-missing values for the model.
    TooShort {
        /// The values the model needs: `p + q + P + Q + columns + 2`, plus one with a mean.
        needed: usize,
        /// The non-missing values after differencing.
        present: usize,
    },
    /// The likelihood has no finite maximum: the model explains the series exactly, so the
    /// innovations are zero or rounding noise (a constant series with a mean, or a straight
    /// line with a drift), or the values are so large (beyond about `1e154`) that the
    /// variance overflows. From `AutoArima`, also when every fitted candidate was rejected.
    Degenerate,
    /// A caller-provided buffer of a refit is too small.
    BufferTooSmall {
        /// The values the buffer must hold.
        needed: usize,
        /// The values it holds.
        got: usize,
    },
    /// The Fourier terms of a search are invalid.
    Fourier(FourierError),
    /// The start values of [`Arima::fit_from`](crate::Arima::fit_from) have other orders,
    /// differencing, period or regressor count than the model.
    StartMismatch,
}

impl From<FourierError> for FitError {
    fn from(error: FourierError) -> Self {
        FitError::Fourier(error)
    }
}

impl Display for FitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            FitError::RegressorShape { expected, got } => {
                write!(f, "expected {expected} regressor values, got {got}")
            }
            FitError::TooShort { needed, present } => {
                write!(
                    f,
                    "model needs {needed} values after differencing, got {present}"
                )
            }
            FitError::Degenerate => write!(f, "the series leaves no innovation variance"),
            FitError::BufferTooSmall { needed, got } => {
                write!(f, "buffer holds {got} values, needs {needed}")
            }
            FitError::Fourier(error) => write!(f, "invalid Fourier terms: {error}"),
            FitError::StartMismatch => write!(f, "the start values do not match the model"),
        }
    }
}

impl Error for FitError {}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn near_unit_root_stays_stationary() {
        // A random walk fitted as AR(1): the estimate approaches 1 but must stay below it.
        let mut y = alloc::vec![0.0; 80];
        let mut state = 1_u32;
        for t in 1..y.len() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5;
            y[t] = y[t - 1] + noise;
        }

        let fitted = Arima::new(1, 0, 0).fit(&y).unwrap();
        assert!(fitted.ar[0] > 0.8 && fitted.ar[0] < 1.0, "{fitted:?}");
        assert!(fitted.loglike.is_finite() && fitted.sigma2 > 0.0);
    }

    #[test]
    fn too_short() {
        let error = Arima::new(1, 1, 1).fit(&[1.0, 2.0, 3.0]).unwrap_err();
        assert_eq!(
            error,
            FitError::TooShort {
                needed: 4,
                present: 2
            }
        );
    }

    #[test]
    fn exact_fits_are_degenerate() {
        assert_eq!(
            Arima::new(1, 0, 0).fit(&[5.0; 30]),
            Err(FitError::Degenerate)
        );

        let line: Vec<f64> = (0..30).map(f64::from).collect();
        assert_eq!(
            Arima::new(0, 1, 1).mean(true).fit(&line),
            Err(FitError::Degenerate)
        );

        let huge: Vec<f64> = (0..30).map(|t| 1e200 * f64::from(t % 3)).collect();
        assert_eq!(Arima::new(1, 0, 0).fit(&huge), Err(FitError::Degenerate));
    }

    #[test]
    fn infinite_values_count_as_missing() {
        let y: Vec<f64> = (0..40).map(|t| f64::from((t * 7) % 11)).collect();
        let x: Vec<f64> = (0..40).map(|t| f64::from(t % 5)).collect();
        let mut with_infinity = y.clone();
        with_infinity[3] = f64::INFINITY;
        let mut with_nan = y;
        with_nan[3] = f64::NAN;

        let model = Arima::new(1, 0, 0);
        let fitted = model.fit_exog(&with_infinity, &x, 1).unwrap();
        assert_eq!(fitted, model.fit_exog(&with_nan, &x, 1).unwrap());
        assert!(fitted.loglike.is_finite() && fitted.sigma2 > 0.0);
    }

    #[test]
    fn criteria_formulas() {
        let fitted = Fitted {
            ar: alloc::vec![0.5],
            d: 0,
            ma: alloc::vec![0.2, 0.1],
            seasonal_ar: alloc::vec![],
            seasonal_d: 0,
            seasonal_ma: alloc::vec![],
            period: 0,
            exog: alloc::vec![],
            with_mean: true,
            mean: 1.0,
            sigma2: 1.0,
            loglike: -50.0,
            nobs: 40,
            iterations: 3,
            converged: true,
        };
        assert_eq!(fitted.parameters(), 5);
        assert_eq!(fitted.aic(), 110.0);
        assert!((fitted.aicc() - (110.0 + 60.0 / 34.0)).abs() < 1e-12);
        assert!((fitted.bic() - (100.0 + 5.0 * 40.0_f64.ln())).abs() < 1e-12);

        let tiny = Fitted { nobs: 6, ..fitted };
        assert_eq!(tiny.aicc(), f64::INFINITY);
    }

    #[test]
    fn reported_loglike_is_exact() {
        // A long series, so the objective's steady-state shortcut does kick in.
        let mut y = alloc::vec![0.0; 400];
        let mut state = 5_u32;
        for t in 1..y.len() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            y[t] = 0.5 * y[t - 1] + f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5;
        }

        let fitted = Arima::new(1, 0, 1).fit(&y).unwrap();
        let exact = crate::loglike(&y, &fitted.spec()).unwrap();
        assert!(
            (fitted.loglike - exact).abs() < 1e-10,
            "{} vs {exact}",
            fitted.loglike
        );
    }

    #[test]
    fn regressor_shape_is_checked() {
        let error = Arima::new(1, 0, 0)
            .fit_exog(&[1.0, 2.0, 3.0], &[1.0, 2.0], 1)
            .unwrap_err();
        assert_eq!(
            error,
            FitError::RegressorShape {
                expected: 3,
                got: 2
            }
        );
    }

    #[test]
    fn mean_defaults_follow_statsmodels() {
        assert!(Arima::new(1, 0, 0).mean);
        assert!(!Arima::new(1, 1, 0).mean);
        assert!(Arima::new(1, 1, 0).mean(true).mean);
    }
}
