//! Maps between unconstrained optimiser parameters and stationary AR (or invertible MA) coefficients.
//!
//! The optimiser works on any real vector. Each entry `x_k` becomes a partial
//! autocorrelation `r_k = x_k / sqrt(1 + x_k^2)` in `(-1, 1)`, and the Durbin–Levinson
//! step-up turns those into polynomial coefficients. Every coefficient vector this
//! produces is stationary, so the likelihood never sees an invalid model. This is
//! statsmodels' `constrain_stationary_univariate` and its inverse (Monahan 1984).

use crate::math;
use alloc::vec;
use alloc::vec::Vec;

/// The `y` coefficients of statsmodels' step-up recursion for the unconstrained `x`.
///
/// Callers negate the result for AR coefficients and keep it for MA coefficients,
/// as statsmodels SARIMAX does.
pub(crate) fn constrain(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut prev = vec![0.0; n];
    let mut curr = vec![0.0; n];
    for k in 0..n {
        let r = x[k] / math::sqrt(1.0 + x[k] * x[k]);
        for i in 0..k {
            curr[i] = prev[i] + r * prev[k - i - 1];
        }

        curr[k] = r;
        prev[..=k].copy_from_slice(&curr[..=k]);
    }

    prev
}

/// The inverse of [`constrain`]. Returns [`None`] if `y` is not stationary, so some
/// partial autocorrelation is not inside `(-1, 1)`.
pub(crate) fn unconstrain(y: &[f64]) -> Option<Vec<f64>> {
    let n = y.len();
    let mut row = y.to_vec();
    let mut r = vec![0.0; n];
    for k in (0..n).rev() {
        let rk = row[k];
        if rk.is_nan() || rk.abs() >= 1.0 {
            return None;
        }

        r[k] = rk;
        let scale = 1.0 - rk * rk;
        let previous: Vec<f64> = (0..k)
            .map(|i| (row[i] - rk * row[k - i - 1]) / scale)
            .collect();
        row[..k].copy_from_slice(&previous);
    }

    Some(r.iter().map(|&rk| rk / math::sqrt(1.0 - rk * rk)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn matches_statsmodels() {
        // statsmodels constrain_stationary_univariate([0.5, -1.2, 2.0]) is -y.
        let y = constrain(&[0.5, -1.2, 2.0]);
        let expected = [0.5834634062650049, 0.6755097914363262, -0.8944271909999159];
        for (a, e) in y.iter().zip(expected) {
            assert_approx_eq!(f64, -a, e, epsilon = 1e-15);
        }
    }

    #[test]
    fn round_trips() {
        let x = [0.3, -2.5, 1.1, 0.0];
        let back = unconstrain(&constrain(&x)).unwrap();
        for (a, b) in x.iter().zip(&back) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1e-10);
        }
    }

    #[test]
    fn rejects_non_stationary() {
        // AR coefficients phi = -y: phi = [1.0] is a unit root, phi = [0.5, 0.6] is explosive.
        assert_eq!(unconstrain(&[-1.0]), None);
        assert_eq!(unconstrain(&[-0.5, -0.6]), None);
        assert!(unconstrain(&[-0.5, 0.3]).is_some());
    }
}
