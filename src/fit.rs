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

use crate::difference::difference;
use crate::filter::ArimaSpec;
use crate::likelihood::concentrated;
use alloc::vec::Vec;
use core::error::Error;
use core::fmt::{Display, Formatter};

/// An ARIMA(p, d, q) model to fit.
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
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arima {
    /// The AR order.
    p: usize,
    /// The differencing order.
    d: usize,
    /// The MA order.
    q: usize,
    /// Whether to estimate the mean of the differenced series.
    mean: bool,
}

impl Arima {
    /// An ARIMA(`p`, `d`, `q`) model. It estimates a mean for `d = 0` and none otherwise,
    /// like statsmodels ARIMA and R's `arima`; change that with [`Arima::mean`].
    #[must_use]
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            mean: d == 0,
        }
    }

    /// Whether to estimate the mean of the differenced series: the mean for `d = 0`,
    /// the drift for `d = 1`.
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
    /// `p + q + 2` non-missing values (one more with a mean).
    pub fn fit(&self, y: &[f64]) -> Result<Fitted, FitError> {
        let w = difference(y, self.d);
        let present = w.iter().filter(|v| !v.is_nan()).count();
        let needed = self.p + self.q + usize::from(self.mean) + 2;
        if present < needed {
            return Err(FitError::TooShort { needed, present });
        }

        let start = self.start(&w);
        let objective = |u: &[f64]| {
            let (ar, ma, mean) = self.constrain(u);
            let centered: Vec<f64> = w.iter().map(|v| v - mean).collect();
            match concentrated(&centered, &ar, &ma) {
                Some((llf, _, n)) => -llf / n as f64,
                None => f64::INFINITY,
            }
        };
        let minimum = bfgs::minimize(objective, start, bfgs::Options::default());

        let (ar, ma, mean) = self.constrain(&minimum.x);
        let centered: Vec<f64> = w.iter().map(|v| v - mean).collect();
        let (loglike, sigma2, nobs) =
            concentrated(&centered, &ar, &ma).expect("the optimum was evaluated finite");

        Ok(Fitted {
            ar,
            d: self.d,
            ma,
            mean,
            sigma2,
            loglike,
            nobs,
            iterations: minimum.iterations,
            converged: minimum.converged,
        })
    }

    /// Unconstrained starting values `[ar..., ma..., mean]`.
    ///
    /// Hannan–Rissanen supplies the AR and MA parts and the sample mean the mean. A part
    /// that comes out non-stationary (or non-invertible), or a failed regression, starts
    /// at zero instead, as statsmodels does.
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
            .unwrap_or_else(|| alloc::vec![0.0; self.p]);
        let ma_start = transform::unconstrain(&ma)
            .filter(|_| ma.len() == self.q)
            .unwrap_or_else(|| alloc::vec![0.0; self.q]);

        let mut start = ar_start;
        start.extend(ma_start);
        if self.mean {
            start.push(mean);
        }

        start
    }

    /// Maps unconstrained parameters to `(ar, ma, mean)`.
    fn constrain(&self, u: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
        let ar = transform::constrain(&u[..self.p])
            .iter()
            .map(|y| -y)
            .collect();
        let ma = transform::constrain(&u[self.p..self.p + self.q]);
        let mean = if self.mean { u[self.p + self.q] } else { 0.0 };

        (ar, ma, mean)
    }
}

/// A fitted ARIMA model.
#[derive(Debug, Clone, PartialEq)]
pub struct Fitted {
    /// The AR coefficients `phi_1, ..., phi_p`. Always stationary.
    pub ar: Vec<f64>,
    /// The differencing order.
    pub d: usize,
    /// The MA coefficients `theta_1, ..., theta_q`. Always invertible.
    pub ma: Vec<f64>,
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
    /// The fitted model as a spec, for [`Filter::new`](crate::Filter::new),
    /// [`loglike`](crate::loglike) and [`innovations`](crate::innovations).
    #[must_use]
    pub fn spec(&self) -> ArimaSpec<'_> {
        ArimaSpec {
            ar: &self.ar,
            d: self.d,
            ma: &self.ma,
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
        /// The values the model needs: `p + q + 2`, plus one with a mean.
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
    fn mean_defaults_follow_statsmodels() {
        assert!(Arima::new(1, 0, 0).mean);
        assert!(!Arima::new(1, 1, 0).mean);
        assert!(Arima::new(1, 1, 0).mean(true).mean);
    }
}
