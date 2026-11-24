//! An adaptive online AR predictor: ARMA-ONS of Anava, Hazan, Mannor & Shamir,
//! "Online Learning for Time Series Prediction" (2013, arXiv:1302.6927).
//!
//! Instead of filtering with fixed maximum-likelihood coefficients, [`OnlineAr`] learns
//! the coefficients of an AR(`M`) predictor one value at a time. Each step takes an
//! online Newton step on the squared one-step error, so the predictor follows
//! coefficients that drift or switch. The paper shows that an AR(`m + k`) predictor
//! learned this way competes with the best ARMA(`k`, `q`) model in hindsight.
//!
//! It is a second mode next to the exact [`Filter`](crate::Filter), not a replacement: it
//! gives up exact likelihood-based forecasts, prediction variances and the statsmodels
//! parity, and gains adaptation without a refit.
//!
//! Deviations from the paper's algorithm:
//!
//! - The projection onto the coefficient box `|gamma_i| <= 1` clips each coefficient.
//!   That is the Euclidean projection; the paper's projection in the norm of `A_t` is a
//!   quadratic program with no closed form.
//! - An optional forgetting factor `lambda` scales the matrix before each update,
//!   `A_t = lambda A_(t-1) + g g'`. The paper's algorithm is `lambda = 1`. Then the steps
//!   shrink like `1 / t`, and the predictor adapts slowly to changes late in a long stream.
//!
//! Like the paper, the predictor has no intercept and assumes values of order 1: remove
//! the mean (or difference the series) and scale it first.

use core::error::Error;
use core::fmt::{Display, Formatter};

/// The tuning of an [`OnlineAr`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnlineArOptions {
    /// The learning rate `eta` of the Newton step `gamma -= A^-1 g / eta`. Larger is slower
    /// and steadier. Positive and finite.
    pub eta: f64,
    /// The starting matrix `A_0 = epsilon I`. Smaller lets the first steps move further.
    /// Positive and finite.
    pub epsilon: f64,
    /// The forgetting factor `lambda` in `(0, 1]`: 1 is the paper's algorithm, smaller
    /// values forget old gradients and follow drifting coefficients faster.
    pub forgetting: f64,
}

impl Default for OnlineArOptions {
    /// `eta = 1`, `epsilon = 1` and `forgetting = 0.999`, tuned on the drifting and
    /// switching AR series of the tests (the paper does not report its values).
    fn default() -> Self {
        Self {
            eta: 1.0,
            epsilon: 1.0,
            forgetting: 0.999,
        }
    }
}

/// Why an [`OnlineAr`] could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnlineArError {
    /// `eta` or `epsilon` is not positive and finite, or `forgetting` is not in `(0, 1]`.
    InvalidOption,
}

impl Display for OnlineArError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "eta and epsilon must be positive and finite, forgetting in (0, 1]"
        )
    }
}

impl Error for OnlineArError {}

/// An AR(`M`) predictor that learns its coefficients online (ARMA-ONS), inline with no
/// heap.
///
/// The size is about `8 (M² + 2M)` bytes; each update costs `O(M²)`.
///
/// # Examples
///
/// ```
/// use backshift::{OnlineAr, OnlineArOptions};
///
/// let mut ar = OnlineAr::<4>::new(OnlineArOptions::default()).unwrap();
/// let (mut y, mut previous) = (0.0, 0.0);
/// for t in 0..2000 {
///     // The AR coefficient switches sign halfway.
///     let phi = if t < 1000 { 0.8 } else { -0.8 };
///     let noise = (((t * 7919) % 101) as f64 / 101.0 - 0.5) * 0.2;
///     (previous, y) = (y, phi * y + noise);
///     ar.update(previous);
/// }
///
/// ar.update(y);
/// assert!(ar.coefficients()[0] < -0.5);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OnlineAr<const M: usize> {
    /// The coefficients `gamma`, one per lag, inside the box `|gamma_i| <= 1`.
    gamma: [f64; M],
    /// The last `M` values, the most recent first.
    lags: [f64; M],
    /// `A_t^-1`, kept by Sherman–Morrison updates. Symmetric.
    inverse: [[f64; M]; M],
    /// The tuning.
    options: OnlineArOptions,
}

impl<const M: usize> OnlineAr<M> {
    /// A predictor with zero coefficients and zero history.
    ///
    /// # Errors
    ///
    /// [`OnlineArError::InvalidOption`] if `eta` or `epsilon` is not positive and finite,
    /// or `forgetting` is not in `(0, 1]`.
    pub fn new(options: OnlineArOptions) -> Result<Self, OnlineArError> {
        let positive = |x: f64| x.is_finite() && x > 0.0;
        let forgetting_ok = options.forgetting > 0.0 && options.forgetting <= 1.0;
        if !positive(options.eta) || !positive(options.epsilon) || !forgetting_ok {
            return Err(OnlineArError::InvalidOption);
        }

        let mut inverse = [[0.0; M]; M];
        for (i, row) in inverse.iter_mut().enumerate() {
            row[i] = 1.0 / options.epsilon;
        }

        Ok(Self {
            gamma: [0.0; M],
            lags: [0.0; M],
            inverse,
            options,
        })
    }

    /// The prediction of the next value: `sum_i gamma_i y_(t-i)`.
    #[must_use]
    pub fn predict(&self) -> f64 {
        self.gamma.iter().zip(&self.lags).map(|(g, y)| g * y).sum()
    }

    /// Feeds the next value, learns from its error, and returns that one-step error
    /// `y - predict()`.
    ///
    /// A non-finite value (`NaN` or infinite) counts as missing: nothing is learned, its
    /// error is `NaN`, and its prediction takes its place in the history.
    pub fn update(&mut self, y: f64) -> f64 {
        let prediction = self.predict();
        if !y.is_finite() {
            self.push(prediction);
            return f64::NAN;
        }

        let error = y - prediction;
        // The gradient of (y - gamma . x)^2 is -2 error x.
        let mut gradient = self.lags;
        for g in &mut gradient {
            *g *= -2.0 * error;
        }

        self.update_inverse(&gradient);
        self.newton_step(&gradient);
        self.push(y);
        error
    }

    /// Forecasts the next `out.len()` values by running the learned AR recursion forward.
    pub fn forecast_into(&self, out: &mut [f64]) {
        let mut ahead = *self;
        for value in out {
            *value = ahead.predict();
            ahead.push(*value);
        }
    }

    /// The learned coefficients, the most recent lag first.
    #[must_use]
    pub fn coefficients(&self) -> &[f64; M] {
        &self.gamma
    }

    /// `A^-1 <- (P - P g g' P / (lambda + g' P g)) / lambda` with `P` the old inverse:
    /// Sherman–Morrison for `A = lambda A_old + g g'`.
    ///
    /// Forgetting is skipped (`lambda = 1`) while it would lift a diagonal entry of the
    /// inverse above its start `1 / epsilon`. Without that cap, a stretch of zero
    /// gradients (a constant zero input) would grow the inverse by `1 / lambda` per step
    /// until it overflows, the "windup" of recursive least squares.
    fn update_inverse(&mut self, g: &[f64; M]) {
        let largest = (0..M).map(|i| self.inverse[i][i]).fold(0.0, f64::max);
        let lambda = if largest / self.options.forgetting > 1.0 / self.options.epsilon {
            1.0
        } else {
            self.options.forgetting
        };
        let mut pg = [0.0; M];
        for (value, row) in pg.iter_mut().zip(&self.inverse) {
            *value = row.iter().zip(g).map(|(p, g)| p * g).sum();
        }

        let denominator = lambda + g.iter().zip(&pg).map(|(g, p)| g * p).sum::<f64>();
        for (i, row) in self.inverse.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = (*value - pg[i] * pg[j] / denominator) / lambda;
            }
        }
    }

    /// `gamma <- clip(gamma - A^-1 g / eta)` onto the box `|gamma_i| <= 1`.
    fn newton_step(&mut self, g: &[f64; M]) {
        for (gamma, row) in self.gamma.iter_mut().zip(&self.inverse) {
            let step: f64 = row.iter().zip(g).map(|(p, g)| p * g).sum();
            *gamma = (*gamma - step / self.options.eta).clamp(-1.0, 1.0);
        }
    }

    /// Pushes `y` onto the history, dropping the oldest value.
    fn push(&mut self, y: f64) {
        if M > 0 {
            self.lags.copy_within(..M - 1, 1);
            self.lags[0] = y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_options() {
        let bad = |eta, epsilon, forgetting| {
            OnlineAr::<2>::new(OnlineArOptions {
                eta,
                epsilon,
                forgetting,
            })
            .err()
        };
        assert_eq!(bad(0.0, 1.0, 1.0), Some(OnlineArError::InvalidOption));
        assert_eq!(bad(1.0, f64::NAN, 1.0), Some(OnlineArError::InvalidOption));
        assert_eq!(bad(1.0, 1.0, 1.5), Some(OnlineArError::InvalidOption));
        assert_eq!(bad(1.0, 1.0, 0.0), Some(OnlineArError::InvalidOption));
        assert!(bad(1.0, 1.0, 1.0).is_none());
    }

    #[test]
    fn learns_a_noiseless_ar1() {
        let mut ar = OnlineAr::<1>::new(OnlineArOptions::default()).unwrap();
        let mut y = 1.0;
        for _ in 0..500 {
            ar.update(y);
            y *= 0.5;
            if y.abs() < 1e-3 {
                y = 1.0;
            }
        }

        assert!(
            (ar.coefficients()[0] - 0.5).abs() < 1e-3,
            "{:?}",
            ar.coefficients()
        );
    }

    #[test]
    fn a_long_zero_input_does_not_wind_up() {
        let mut ar = OnlineAr::<2>::new(OnlineArOptions::default()).unwrap();
        for _ in 0..1_000_000 {
            ar.update(0.0);
        }

        let largest = ar
            .inverse
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest <= 1.0, "{largest}");
        ar.update(0.5);
        assert!(ar.update(0.2).is_finite());
    }

    #[test]
    fn missing_values_keep_the_coefficients() {
        let mut ar = OnlineAr::<2>::new(OnlineArOptions::default()).unwrap();
        for y in [0.3, -0.2, 0.5, 0.1] {
            ar.update(y);
        }

        let before = *ar.coefficients();
        let prediction = ar.predict();
        assert!(ar.update(f64::NAN).is_nan());
        assert_eq!(*ar.coefficients(), before);
        assert_eq!(ar.lags[0], prediction);
    }

    #[test]
    fn infinite_values_count_as_missing() {
        let mut ar = OnlineAr::<2>::new(OnlineArOptions::default()).unwrap();
        for t in 0..20 {
            ar.update((f64::from(t) * 0.7).sin());
        }
        assert!(ar.update(f64::INFINITY).is_nan());
        assert!(ar.update(0.3).is_finite());
        assert!(ar.predict().is_finite());
    }
}
