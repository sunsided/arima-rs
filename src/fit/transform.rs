//! Maps between unconstrained optimiser parameters and stationary AR (or invertible MA) coefficients.
//!
//! The optimiser works on any real vector. Each entry `x_k` becomes a partial
//! autocorrelation `r_k = x_k / sqrt(1 + x_k^2)` in `(-1, 1)`, and the Durbin–Levinson
//! step-up turns those into polynomial coefficients. Every coefficient vector this
//! produces is stationary, so the likelihood never sees an invalid model. This is
//! statsmodels' `constrain_stationary_univariate` and its inverse (Monahan 1984).

use crate::math;

/// Writes the `y` coefficients of statsmodels' step-up recursion for the unconstrained
/// `x` into `out`, with `work` as scratch. Both hold at least `x.len()` values.
///
/// Callers negate the result for AR coefficients and keep it for MA coefficients,
/// as statsmodels SARIMAX does.
pub(crate) fn constrain(x: &[f64], out: &mut [f64], work: &mut [f64]) {
    let n = x.len();
    let (prev, curr) = (&mut out[..n], &mut work[..n]);
    for k in 0..n {
        let r = x[k] / math::sqrt(1.0 + x[k] * x[k]);
        for i in 0..k {
            curr[i] = prev[i] + r * prev[k - i - 1];
        }

        curr[k] = r;
        prev[..=k].copy_from_slice(&curr[..=k]);
    }
}

/// Writes the inverse of [`constrain`] for `y` into `out`, with `work` as scratch of
/// `2 y.len()` values; `out` holds at least `y.len()`.
///
/// Returns `false`, leaving `out` unspecified, if `y` is not stationary, so some partial
/// autocorrelation is not inside `(-1, 1)`.
pub(crate) fn unconstrain(y: &[f64], out: &mut [f64], work: &mut [f64]) -> bool {
    let n = y.len();
    let (row, previous) = work[..2 * n].split_at_mut(n);
    let r = &mut out[..n];
    row.copy_from_slice(y);
    for k in (0..n).rev() {
        let rk = row[k];
        if rk.is_nan() || rk.abs() >= 1.0 {
            return false;
        }

        r[k] = rk;
        let scale = 1.0 - rk * rk;
        for i in 0..k {
            previous[i] = (row[i] - rk * row[k - i - 1]) / scale;
        }

        row[..k].copy_from_slice(&previous[..k]);
    }

    for value in r {
        *value /= math::sqrt(1.0 - *value * *value);
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use std::vec;
    use std::vec::Vec;

    /// [`constrain`] into a fresh vector.
    fn constrained(x: &[f64]) -> Vec<f64> {
        let (mut out, mut work) = (vec![0.0; x.len()], vec![0.0; x.len()]);
        constrain(x, &mut out, &mut work);
        out
    }

    /// [`unconstrain`] into a fresh vector, or [`None`] if `y` is not stationary.
    fn unconstrained(y: &[f64]) -> Option<Vec<f64>> {
        let (mut out, mut work) = (vec![0.0; y.len()], vec![0.0; 2 * y.len()]);
        unconstrain(y, &mut out, &mut work).then_some(out)
    }

    #[test]
    fn matches_statsmodels() {
        // statsmodels constrain_stationary_univariate([0.5, -1.2, 2.0]) is -y.
        let y = constrained(&[0.5, -1.2, 2.0]);
        let expected = [0.5834634062650049, 0.6755097914363262, -0.8944271909999159];
        for (a, e) in y.iter().zip(expected) {
            assert_approx_eq!(f64, -a, e, epsilon = 1e-15);
        }
    }

    #[test]
    fn round_trips() {
        let x = [0.3, -2.5, 1.1, 0.0];
        let back = unconstrained(&constrained(&x)).unwrap();
        for (a, b) in x.iter().zip(&back) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1e-10);
        }
    }

    #[test]
    fn rejects_non_stationary() {
        // AR coefficients phi = -y: phi = [1.0] is a unit root, phi = [0.5, 0.6] is explosive.
        assert_eq!(unconstrained(&[-1.0]), None);
        assert_eq!(unconstrained(&[-0.5, -0.6]), None);
        assert!(unconstrained(&[-0.5, 0.3]).is_some());
    }
}
