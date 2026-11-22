//! Hannan–Rissanen starting values for an ARMA(p, q) fit.
//!
//! 1. Fit a long AR(m) model by Yule–Walker and take its residuals as estimates of the
//!    unobserved innovations.
//! 2. Regress the series on its own `p` lags and the `q` lags of those residuals.
//!
//! This is statsmodels `hannan_rissanen(..., unbiased=False)` (Hannan & Rissanen 1982),
//! with its default long-AR order `m = max(floor(ln(n)^2), 2 max(p, q))`. statsmodels'
//! optional third, bias-correcting step is not needed for starting values.

use crate::math::dot;
use crate::{Estimator, TimeSeries, math, yule_walker};
use alloc::vec;
use alloc::vec::Vec;

/// AR and MA coefficients of an ARMA(p, q) model.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmaCoefficients {
    /// `phi_1, ..., phi_p`.
    pub ar: Vec<f64>,
    /// `theta_1, ..., theta_q`.
    pub ma: Vec<f64>,
}

/// Hannan–Rissanen estimates of an ARMA(`p`, `q`) model for the series `x`, which is
/// demeaned first.
///
/// Fast and closed-form, but not efficient: [`Arima::fit`](crate::Arima::fit) uses
/// these as starting values for maximum likelihood. The estimates are not guaranteed
/// to be stationary or invertible.
///
/// Returns [`None`] if `x` is too short for the regressions, contains non-finite
/// values, or a regression is singular.
///
/// # Examples
///
/// ```
/// use backshift::hannan_rissanen;
///
/// let x = [0.3, 1.1, 0.4, -0.2, 0.9, 1.4, 0.6, 0.1, -0.5, 0.2, 0.8, 1.0, 0.3, -0.1, 0.4];
/// let estimate = hannan_rissanen(&x, 1, 0).unwrap();
/// assert_eq!(estimate.ar.len(), 1);
/// ```
#[must_use]
pub fn hannan_rissanen(x: &[f64], p: usize, q: usize) -> Option<ArmaCoefficients> {
    if !x.iter().all(|v| v.is_finite()) {
        return None;
    }

    let x = demeaned(x);
    if p == 0 && q == 0 {
        return Some(ArmaCoefficients {
            ar: vec![],
            ma: vec![],
        });
    }

    if q == 0 {
        // Pure AR: least squares on the lagged series.
        let rows: Vec<Vec<f64>> = (p..x.len()).map(|t| lags(&x, t, p)).collect();
        let ar = least_squares(&rows, &x[p..])?;
        return Some(ArmaCoefficients { ar, ma: vec![] });
    }

    let n = x.len();
    let m = initial_ar_order(n, p, q);
    if m + q >= n {
        return None;
    }

    let series = TimeSeries::try_new(&x, Estimator::Biased).ok()?;
    let long = yule_walker(&series, m)?;
    let residuals: Vec<f64> = (m..n)
        .map(|t| x[t] - dot(&long.phi, &lags(&x, t, m)))
        .collect();

    // For t >= m + q: [x_(t-1), ..., x_(t-p), e_(t-1), ..., e_(t-q)], with e_s = residuals[s - m].
    let rows: Vec<Vec<f64>> = (m + q..n)
        .map(|t| {
            let mut row = lags(&x, t, p);
            row.extend((1..=q).map(|j| residuals[t - j - m]));
            row
        })
        .collect();
    let mut coefficients = least_squares(&rows, &x[m + q..])?;
    let ma = coefficients.split_off(p);

    Some(ArmaCoefficients {
        ar: coefficients,
        ma,
    })
}

/// statsmodels' default long-AR order, `max(floor(ln(n)^2), 2 max(p, q))` (Gómez & Maravall 2001).
fn initial_ar_order(n: usize, p: usize, q: usize) -> usize {
    let ln = math::ln(n as f64);
    // Truncation is the floor for this non-negative value.
    let squared = (ln * ln) as usize;
    squared.max(2 * p.max(q))
}

/// `x - mean(x)`.
fn demeaned(x: &[f64]) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| v - mean).collect()
}

/// `[x_(t-1), ..., x_(t-k)]`.
fn lags(x: &[f64], t: usize, k: usize) -> Vec<f64> {
    (1..=k).map(|j| x[t - j]).collect()
}

/// Ordinary least squares: the `b` that minimises `|y - X b|`, from the normal
/// equations `X'X b = X'y` solved by Cholesky. [`None`] if `X'X` is not positive definite
/// or there are no more rows than columns.
pub(crate) fn least_squares(rows: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let k = rows.first()?.len();
    if rows.len() <= k {
        return None;
    }

    let mut gram = vec![vec![0.0; k]; k];
    let mut moment = vec![0.0; k];
    for (row, &target) in rows.iter().zip(y) {
        for i in 0..k {
            moment[i] += row[i] * target;
            for j in 0..=i {
                gram[i][j] += row[i] * row[j];
            }
        }
    }

    cholesky_solve(gram, moment)
}

/// Solves `A b = c` for symmetric positive definite `A`, given its lower triangle.
fn cholesky_solve(mut a: Vec<Vec<f64>>, mut c: Vec<f64>) -> Option<Vec<f64>> {
    let k = c.len();
    for j in 0..k {
        let diagonal = a[j][j] - (0..j).map(|l| a[j][l] * a[j][l]).sum::<f64>();
        if diagonal.is_nan() || diagonal <= 0.0 {
            return None;
        }

        a[j][j] = math::sqrt(diagonal);
        for i in j + 1..k {
            a[i][j] = (a[i][j] - (0..j).map(|l| a[i][l] * a[j][l]).sum::<f64>()) / a[j][j];
        }
    }

    for i in 0..k {
        c[i] = (c[i] - (0..i).map(|l| a[i][l] * c[l]).sum::<f64>()) / a[i][i];
    }

    for i in (0..k).rev() {
        c[i] = (c[i] - (i + 1..k).map(|l| a[l][i] * c[l]).sum::<f64>()) / a[i][i];
    }

    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;

    #[test]
    fn least_squares_recovers_exact_coefficients() {
        let rows = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 5.0],
            vec![0.5, -1.0],
        ];
        let y: Vec<f64> = rows.iter().map(|r| 0.7 * r[0] - 1.3 * r[1]).collect();
        let b = least_squares(&rows, &y).unwrap();
        assert_approx_eq!(f64, b[0], 0.7, epsilon = 1e-12);
        assert_approx_eq!(f64, b[1], -1.3, epsilon = 1e-12);
    }

    #[test]
    fn initial_order_follows_statsmodels() {
        // floor(ln(60)^2) = 16; 2 max(p, q) wins for large orders.
        assert_eq!(initial_ar_order(60, 1, 1), 16);
        assert_eq!(initial_ar_order(60, 3, 9), 18);
    }

    #[test]
    fn rejects_short_or_non_finite_input() {
        assert!(hannan_rissanen(&[1.0, 2.0, f64::NAN, 3.0], 1, 0).is_none());
        assert!(hannan_rissanen(&[1.0, 2.0, 3.0], 1, 1).is_none());
    }
}
