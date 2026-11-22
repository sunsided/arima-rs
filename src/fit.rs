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

mod bfgs;
mod hannan_rissanen;
mod transform;

pub use hannan_rissanen::{ArmaCoefficients, hannan_rissanen};

use crate::filter::{ArimaSpec, Seasonal};
use crate::likelihood::{concentrated, differenced};
use crate::math;
use alloc::vec;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt::{Display, Formatter};

/// A seasonal ARIMA(p, d, q)(P, D, Q, s) model to fit.
///
/// # Examples
///
/// ```
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

/// Constrained parameters, in the order the optimiser stores them unconstrained.
struct Parameters {
    /// The AR coefficients.
    ar: Vec<f64>,
    /// The MA coefficients.
    ma: Vec<f64>,
    /// The seasonal AR coefficients.
    seasonal_ar: Vec<f64>,
    /// The seasonal MA coefficients.
    seasonal_ma: Vec<f64>,
    /// The mean of the differenced series, or 0.
    mean: f64,
}

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
    /// `NaN` values count as missing. A fit that stops at the iteration limit is still
    /// returned, with [`Fitted::converged`] set to `false`.
    ///
    /// # Errors
    ///
    /// [`FitError::TooShort`] if the differenced series has fewer than
    /// `p + q + P + Q + 2` non-missing values (one more with a mean).
    pub fn fit(&self, y: &[f64]) -> Result<Fitted, FitError> {
        let w = differenced(y, &self.shape(&[], &[], &[], &[]));
        let present = w.iter().filter(|v| !v.is_nan()).count();
        let needed = self.coefficients() + usize::from(self.mean) + 2;
        if present < needed {
            return Err(FitError::TooShort { needed, present });
        }

        let start = self.start(&w);
        let evaluate = |parameters: &Parameters| {
            let centered: Vec<f64> = w.iter().map(|v| v - parameters.mean).collect();
            let shape = self.shape(
                &parameters.ar,
                &parameters.ma,
                &parameters.seasonal_ar,
                &parameters.seasonal_ma,
            );
            concentrated(&centered, &shape)
        };
        let objective = |u: &[f64]| match evaluate(&self.constrain(u)) {
            Some((llf, _, n)) => -llf / n as f64,
            None => f64::INFINITY,
        };
        let minimum = bfgs::minimize(objective, start, bfgs::Options::default());

        let parameters = self.constrain(&minimum.x);
        let (loglike, sigma2, nobs) =
            evaluate(&parameters).expect("the optimum was evaluated finite");

        Ok(Fitted {
            ar: parameters.ar,
            d: self.d,
            ma: parameters.ma,
            seasonal_ar: parameters.seasonal_ar,
            seasonal_d: self.seasonal_d,
            seasonal_ma: parameters.seasonal_ma,
            period: self.period,
            with_mean: self.mean,
            mean: parameters.mean,
            sigma2,
            loglike,
            nobs,
            iterations: minimum.iterations,
            converged: minimum.converged,
        })
    }

    /// The number of AR and MA coefficients, seasonal ones included.
    fn coefficients(&self) -> usize {
        self.p + self.q + self.seasonal_p + self.seasonal_q
    }

    /// A spec with this model's orders and the given coefficients, zero mean and unit variance.
    fn shape<'a>(
        &self,
        ar: &'a [f64],
        ma: &'a [f64],
        seasonal_ar: &'a [f64],
        seasonal_ma: &'a [f64],
    ) -> ArimaSpec<'a> {
        ArimaSpec {
            ar,
            d: self.d,
            ma,
            seasonal: Seasonal {
                ar: seasonal_ar,
                d: self.seasonal_d,
                ma: seasonal_ma,
                period: self.period,
            },
            ..ArimaSpec::default()
        }
    }

    /// Unconstrained starting values `[ar..., ma..., seasonal ar..., seasonal ma..., mean]`.
    ///
    /// Hannan–Rissanen supplies the non-seasonal AR and MA parts and the sample mean the
    /// mean; the seasonal parts start at zero. A part that comes out non-stationary (or
    /// non-invertible), or a failed regression, starts at zero instead, as statsmodels does.
    fn start(&self, w: &[f64]) -> Vec<f64> {
        let present: Vec<f64> = w.iter().copied().filter(|v| !v.is_nan()).collect();
        let sample_mean = present.iter().sum::<f64>() / present.len() as f64;
        let mean = if self.mean { sample_mean } else { 0.0 };
        let centered: Vec<f64> = w.iter().map(|v| v - mean).collect();

        let estimate = hannan_rissanen(&centered, self.p, self.q);
        let (ar, ma) = estimate.map_or((Vec::new(), Vec::new()), |s| (s.ar, s.ma));
        let negated: Vec<f64> = ar.iter().map(|phi| -phi).collect();
        let ar_start = transform::unconstrain(&negated)
            .filter(|_| ar.len() == self.p)
            .unwrap_or_else(|| vec![0.0; self.p]);
        let ma_start = transform::unconstrain(&ma)
            .filter(|_| ma.len() == self.q)
            .unwrap_or_else(|| vec![0.0; self.q]);

        let mut start = ar_start;
        start.extend(ma_start);
        start.extend(core::iter::repeat_n(0.0, self.seasonal_p + self.seasonal_q));
        if self.mean {
            start.push(mean);
        }

        start
    }

    /// Maps unconstrained parameters to constrained ones.
    ///
    /// AR parts are negated after the transform and MA parts are not, as in statsmodels
    /// SARIMAX, so that both come out stationary or invertible.
    fn constrain(&self, u: &[f64]) -> Parameters {
        let (ar, rest) = u.split_at(self.p);
        let (ma, rest) = rest.split_at(self.q);
        let (seasonal_ar, rest) = rest.split_at(self.seasonal_p);
        let (seasonal_ma, rest) = rest.split_at(self.seasonal_q);
        let negated = |x: &[f64]| transform::constrain(x).iter().map(|y| -y).collect();

        Parameters {
            ar: negated(ar),
            ma: transform::constrain(ma),
            seasonal_ar: negated(seasonal_ar),
            seasonal_ma: transform::constrain(seasonal_ma),
            mean: if self.mean { rest[0] } else { 0.0 },
        }
    }
}

/// A fitted seasonal ARIMA model.
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

impl Fitted {
    /// The number of estimated parameters: `p + q + P + Q`, one for the mean if estimated,
    /// and one for `sigma2`. This is the `k` of the information criteria.
    #[must_use]
    pub fn parameters(&self) -> usize {
        self.ar.len()
            + self.ma.len()
            + self.seasonal_ar.len()
            + self.seasonal_ma.len()
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
            mean: self.mean,
            sigma2: self.sigma2,
        }
    }
}

/// Why a model could not be fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitError {
    /// The differenced series has too few non-missing values for the model.
    TooShort {
        /// The values the model needs: `p + q + P + Q + 2`, plus one with a mean.
        needed: usize,
        /// The non-missing values after differencing.
        present: usize,
    },
}

impl Display for FitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            FitError::TooShort { needed, present } => {
                write!(
                    f,
                    "model needs {needed} values after differencing, got {present}"
                )
            }
        }
    }
}

impl Error for FitError {}

#[cfg(test)]
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
    fn criteria_formulas() {
        let fitted = Fitted {
            ar: alloc::vec![0.5],
            d: 0,
            ma: alloc::vec![0.2, 0.1],
            seasonal_ar: alloc::vec![],
            seasonal_d: 0,
            seasonal_ma: alloc::vec![],
            period: 0,
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
    fn mean_defaults_follow_statsmodels() {
        assert!(Arima::new(1, 0, 0).mean);
        assert!(!Arima::new(1, 1, 0).mean);
        assert!(Arima::new(1, 1, 0).mean(true).mean);
    }
}
