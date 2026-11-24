//! PMM2: the second-order polynomial maximisation method for ARIMA models with skewed
//! innovations (Zabolotnii et al., arXiv:2511.07059, Algorithm 1).
//!
//! Exact Gaussian maximum likelihood is efficient for Gaussian innovations. When the
//! innovations are skewed, PMM2 uses their third and fourth moments as well and estimates
//! the coefficients with less variance. In theory its variance is that of least squares
//! divided by `(2 + g4) / (2 + g4 - g3^2)`, for skewness `g3` and excess kurtosis `g4`.
//!
//! 1. Fit the model by exact maximum likelihood; keep its innovations `e_t`.
//! 2. Build the design `x_t = (1, z_(t-1), ..., z_(t-p), e_(t-1), ..., e_(t-q))` on the
//!    differenced series `z`, with the intercept only if the model has a mean.
//! 3. Take the central moments `mu2, mu3, mu4` of the innovations.
//! 4. Solve the PMM2 estimating equations by Newton's method from the maximum-likelihood
//!    coefficients, with the design and moments fixed.
//!
//! The paper starts from conditional sum of squares; this port starts from exact
//! maximum likelihood, which the paper also allows.

use super::hannan_rissanen::cholesky_solve;
use super::roots::max_inverse_root;
use super::{Arima, FitError, Fitted};
use crate::filter::ArimaSpec;
use crate::likelihood::{concentrated, differenced, workspace_len};
use crate::{innovations, math};
use alloc::vec;
use alloc::vec::Vec;

/// The largest number of Newton steps.
const MAX_STEPS: usize = 100;

/// The step norm below which the Newton iteration has converged.
const STEP_TOLERANCE: f64 = 1e-10;

/// A PMM2 fit and the innovation moments it used.
#[derive(Debug, Clone, PartialEq)]
pub struct Pmm2Fit {
    /// The PMM2 model. `sigma2` and `loglike` are the exact Gaussian ones at its
    /// coefficients (post hoc, as in the paper), so the information criteria compare
    /// with maximum likelihood fits. `iterations` counts Newton steps.
    pub fitted: Fitted,
    /// The maximum-likelihood fit PMM2 started from.
    pub start: Fitted,
    /// The skewness `g3` of the starting innovations.
    pub skewness: f64,
    /// The excess kurtosis `g4` of the starting innovations.
    pub excess_kurtosis: f64,
}

impl Pmm2Fit {
    /// The theoretical variance reduction against least squares,
    /// `(2 + g4) / (2 + g4 - g3^2)`: 1 for symmetric innovations, about 1.67 for
    /// Gamma(2, 1) ones.
    ///
    /// The paper recommends PMM2 over the start only when this is above about 1.2 and
    /// the series has at least 200 values.
    #[must_use]
    pub fn efficiency(&self) -> f64 {
        let base = 2.0 + self.excess_kurtosis;
        base / (base - self.skewness * self.skewness)
    }
}

impl Arima {
    /// Fits the model by PMM2, for innovations that are skewed (Zabolotnii et al.,
    /// arXiv:2511.07059).
    ///
    /// It starts from [`Arima::fit`] and its innovations, then solves the PMM2 estimating
    /// equations. On simulated series of 500 values with Gamma(2, 1), chi-squared(3) or
    /// lognormal innovations, the AR coefficients of AR(1), ARIMA(1, 1, 0), AR(2) and
    /// ARMA(1, 1) models had 1.4 to 2 times less mean squared error than exact maximum
    /// likelihood, near [`Pmm2Fit::efficiency`]. With Gaussian innovations both were equal
    /// for AR models.
    ///
    /// The MA part uses the starting innovations as fixed regressors, as the paper does.
    /// That linearisation costs efficiency: for an MA(1), PMM2 gained only a factor 1.2
    /// with Gamma(2, 1) innovations and lost a factor 1.25 with Gaussian ones. Check
    /// [`Pmm2Fit::efficiency`] before you prefer PMM2.
    ///
    /// # Errors
    ///
    /// - [`FitError::Unsupported`] for a seasonal model.
    /// - [`FitError::Degenerate`] if the innovation moments give no PMM2 solution
    ///   (`mu2 (mu4 - mu2^2) <= mu3^2`), Newton's method fails, or its result is not
    ///   stationary or not invertible. The last happens for models near the boundary, such
    ///   as an ARMA(1, 1) whose AR and MA roots nearly cancel (1 of 200 simulated series
    ///   above); keep the maximum-likelihood fit then.
    /// - As for [`Arima::fit`].
    ///
    /// # Examples
    ///
    /// ```
    /// use backshift::Arima;
    ///
    /// // An AR(1) with skewed (centred exponential) innovations.
    /// let mut state = 1_u32;
    /// let mut y = vec![0.0; 400];
    /// for t in 1..y.len() {
    ///     state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ///     let u = (f64::from(state >> 8) + 0.5) / f64::from(1_u32 << 24);
    ///     y[t] = 0.7 * y[t - 1] - u.ln() - 1.0;
    /// }
    ///
    /// let pmm2 = Arima::new(1, 0, 0).fit_pmm2(&y).unwrap();
    /// assert!(pmm2.skewness > 1.0);
    /// assert!(pmm2.efficiency() > 1.2);
    /// assert!((pmm2.fitted.ar[0] - 0.7).abs() < 0.05);
    /// ```
    pub fn fit_pmm2(&self, y: &[f64]) -> Result<Pmm2Fit, FitError> {
        if self.seasonal_p + self.seasonal_d + self.seasonal_q > 0 {
            return Err(FitError::Unsupported);
        }

        let start = self.fit(y)?;
        let spec = start.spec();
        let z = differenced(y, &spec);
        let residuals: Vec<f64> = innovations(y, &spec)
            .map_err(|_| FitError::Degenerate)?
            .iter()
            .map(|innovation| innovation.value)
            .collect();
        let design = Design::new(self, &z, &residuals);
        let moments = Moments::of(&design.residuals).ok_or(FitError::Degenerate)?;

        let mut theta = self.start_coefficients(&start);
        let iterations = newton(&design, &moments, &mut theta).ok_or(FitError::Degenerate)?;
        let fitted = self.pmm2_fitted(&theta, &z, iterations.0, iterations.1)?;

        let g3 = moments.mu3 / math::sqrt(moments.mu2 * moments.mu2 * moments.mu2);
        let g4 = moments.mu4 / (moments.mu2 * moments.mu2) - 3.0;
        Ok(Pmm2Fit {
            fitted,
            start,
            skewness: g3,
            excess_kurtosis: g4,
        })
    }

    /// The Newton start `[intercept, ar..., ma...]` from a maximum-likelihood fit: the
    /// intercept is `mean (1 - sum ar)`.
    fn start_coefficients(&self, fit: &Fitted) -> Vec<f64> {
        let mut theta = Vec::with_capacity(usize::from(self.mean) + self.p + self.q);
        if self.mean {
            theta.push(fit.mean * (1.0 - fit.ar.iter().sum::<f64>()));
        }

        theta.extend_from_slice(&fit.ar);
        theta.extend_from_slice(&fit.ma);
        theta
    }

    /// The fit at the PMM2 coefficients `[intercept, ar..., ma...]`, with the exact
    /// Gaussian `sigma2` and log-likelihood there.
    fn pmm2_fitted(
        &self,
        theta: &[f64],
        z: &[f64],
        iterations: usize,
        converged: bool,
    ) -> Result<Fitted, FitError> {
        let (intercept, coefficients) = theta.split_at(usize::from(self.mean));
        let (ar, ma) = coefficients.split_at(self.p);
        let negated: Vec<f64> = ar.iter().map(|a| -a).collect();
        if max_inverse_root(&negated) >= 1.0 || max_inverse_root(ma) >= 1.0 {
            return Err(FitError::Degenerate);
        }

        let mean = intercept
            .first()
            .map_or(0.0, |c| c / (1.0 - ar.iter().sum::<f64>()));
        let w: Vec<f64> = z.iter().map(|v| v - mean).collect();
        let shape = ArimaSpec {
            ar,
            ma,
            ..ArimaSpec::default()
        };
        let mut buffer = vec![0.0; workspace_len(shape.arma_states())];
        let (loglike, sigma2, nobs) =
            concentrated(&w, &shape, false, &mut buffer).ok_or(FitError::Degenerate)?;

        Ok(Fitted {
            ar: ar.to_vec(),
            d: self.d,
            ma: ma.to_vec(),
            seasonal_ar: Vec::new(),
            seasonal_d: 0,
            seasonal_ma: Vec::new(),
            period: self.period,
            exog: Vec::new(),
            with_mean: self.mean,
            mean,
            sigma2,
            loglike,
            nobs,
            iterations,
            converged,
        })
    }
}

/// The fixed PMM2 regression: one row `x_t` per usable time step, and its target `z_t`.
struct Design {
    /// The rows `(1, z_(t-1..t-p), e_(t-1..t-q))`, without the 1 when there is no mean.
    rows: Vec<Vec<f64>>,
    /// The targets `z_t`.
    targets: Vec<f64>,
    /// The starting innovations at the same time steps, for the moments.
    residuals: Vec<f64>,
}

impl Design {
    /// The rows from `t = max(p, q)` on, skipping any with a missing value.
    fn new(model: &Arima, z: &[f64], e: &[f64]) -> Self {
        let lags = model.p.max(model.q);
        let (mut rows, mut targets, mut residuals) = (Vec::new(), Vec::new(), Vec::new());
        for t in lags..z.len() {
            let mut row = Vec::with_capacity(usize::from(model.mean) + model.p + model.q);
            if model.mean {
                row.push(1.0);
            }

            row.extend((1..=model.p).map(|i| z[t - i]));
            row.extend((1..=model.q).map(|j| e[t - j]));
            if !z[t].is_nan() && !e[t].is_nan() && row.iter().all(|v| !v.is_nan()) {
                rows.push(row);
                targets.push(z[t]);
                residuals.push(e[t]);
            }
        }

        Self {
            rows,
            targets,
            residuals,
        }
    }
}

/// The central moments of the starting innovations and the PMM2 determinant.
struct Moments {
    /// The variance.
    mu2: f64,
    /// The third central moment.
    mu3: f64,
    /// The fourth central moment.
    mu4: f64,
    /// `mu2 (mu4 - mu2^2) - mu3^2`. Positive.
    delta: f64,
}

impl Moments {
    /// The moments of `e`, or [`None`] if `delta` is not positive and finite.
    fn of(e: &[f64]) -> Option<Self> {
        let n = e.len() as f64;
        let mean = e.iter().sum::<f64>() / n;
        let (mut mu2, mut mu3, mut mu4) = (0.0, 0.0, 0.0);
        for v in e {
            let centred = v - mean;
            let square = centred * centred;
            mu2 += square / n;
            mu3 += square * centred / n;
            mu4 += square * square / n;
        }

        let delta = mu2 * (mu4 - mu2 * mu2) - mu3 * mu3;
        (delta.is_finite() && delta > 0.0).then_some(Self {
            mu2,
            mu3,
            mu4,
            delta,
        })
    }
}

/// Solves the PMM2 equations by Newton's method from `theta`, in place.
///
/// Returns the steps taken and whether the step norm fell below [`STEP_TOLERANCE`], or
/// [`None`] if a Newton system cannot be solved.
fn newton(design: &Design, m: &Moments, theta: &mut [f64]) -> Option<(usize, bool)> {
    let k = theta.len();
    for step in 1..=MAX_STEPS {
        // g = sum x_t [a(eta) (z - eta) - b (z^2 - eta^2 - mu2)], J = sum lambda_t x_t x_t'.
        // J is negative definite near the solution, so solve (-J) s = -g by Cholesky.
        let (mut gradient, mut negative_jacobian) = (vec![0.0; k], vec![vec![0.0; k]; k]);
        for (row, &z) in design.rows.iter().zip(&design.targets) {
            let eta: f64 = row.iter().zip(theta.iter()).map(|(x, t)| x * t).sum();
            let a = (m.mu4 - m.mu2 * m.mu2 + 2.0 * m.mu3 * eta) / m.delta;
            let b = m.mu3 / m.delta;
            let score = a * (z - eta) - b * (z * z - eta * eta - m.mu2);
            let lambda = (2.0 * m.mu3 * (z - eta) - (m.mu4 - m.mu2 * m.mu2)) / m.delta;
            for i in 0..k {
                gradient[i] -= row[i] * score;
                for j in 0..=i {
                    negative_jacobian[i][j] -= lambda * row[i] * row[j];
                }
            }
        }

        let newton_step = cholesky_solve(negative_jacobian, gradient)?;
        let mut norm = 0.0;
        for (t, s) in theta.iter_mut().zip(&newton_step) {
            *t -= s;
            norm += s * s;
        }

        if math::sqrt(norm) < STEP_TOLERANCE {
            return Some((step, true));
        }
    }

    Some((MAX_STEPS, false))
}
