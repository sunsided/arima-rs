//! Autoregressive (AR) model fitting: Yule–Walker and Burg.
//!
//! An AR(p) model writes each value as a weighted sum of the `p` values before
//! it plus white noise: `x_t = phi_1 x_(t-1) + ... + phi_p x_(t-p) + e_t`, with
//! `Var(e_t) = sigma2`. Both estimators here work on the demeaned series and
//! match statsmodels (`yule_walker` and `burg` in `statsmodels.regression.linear_model`).

#[cfg(doc)]
use crate::Estimator;
use crate::math::{dot, is_constant};
use crate::{TimeSeries, acov};
use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

/// The fitted coefficients of an AR(p) model.
#[derive(Debug, Clone, PartialEq)]
pub struct ArCoefficients {
    /// `phi_1, ..., phi_p`, the weights of the `p` previous values. Length `p`.
    pub phi: Vec<f64>,
    /// The innovation variance, `Var(e_t)`.
    ///
    /// Yule–Walker with [`Estimator::Unbiased`] can return zero or a negative value,
    /// because its autocovariance matrix is not guaranteed to be positive definite.
    pub sigma2: f64,
}

/// The output of the Durbin–Levinson recursion over `p` lags.
pub(crate) struct Levinson {
    /// Partial autocorrelations `[1, phi_11, ..., phi_pp]`. Length `p + 1`.
    pub(crate) pacf: Vec<f64>,
    /// The AR(p) coefficients `phi_p1, ..., phi_pp`, the last row of the recursion. Length `p`.
    pub(crate) phi: Vec<f64>,
}

/// Fits an AR(`p`) model by solving the Yule–Walker equations.
///
/// The autocovariances come from [`acov`], so the series' estimator picks the
/// method: [`Estimator::Biased`] divides by `n` (statsmodels `method="mle"`),
/// [`Estimator::Unbiased`] by `n - lag` (statsmodels `method="adjusted"`).
/// `sigma2` is `c_0 - sum(phi_j c_j)`; statsmodels returns its square root.
///
/// Returns [`None`] if `p` is not less than the number of values, or the series is
/// constant or contains non-finite values.
///
/// # Examples
///
/// ```
/// use backshift::{yule_walker, Estimator, TimeSeries};
///
/// let x = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 8.0];
/// let fit = yule_walker(TimeSeries::try_new(&x, Estimator::Biased).unwrap(), 2).unwrap();
/// assert_eq!(fit.phi.len(), 2);
/// ```
#[must_use]
pub fn yule_walker<'a, T: Borrow<TimeSeries<'a>>>(values: T, p: usize) -> Option<ArCoefficients> {
    let values = values.borrow();
    if is_constant(values) {
        return None;
    }

    let c: Vec<f64> = (0..=p)
        .map(|lag| acov(values, lag))
        .collect::<Option<_>>()?;
    if !c[0].is_finite() || c[0] == 0.0 {
        return None;
    }

    let phi = durbin_levinson(&c).phi;
    let sigma2 = c[0] - phi.iter().zip(&c[1..]).map(|(a, b)| a * b).sum::<f64>();

    Some(ArCoefficients { phi, sigma2 })
}

/// Fits an AR(`p`) model with Burg's method.
///
/// Burg minimises the forward and backward prediction errors directly instead of
/// going through autocovariances, which gives better estimates on short series and
/// always yields a stationary model. This is a port of statsmodels `pacf_burg`
/// followed by the step-up recursion, so it matches statsmodels `burg`. The
/// series' estimator is not used.
///
/// Returns [`None`] if `p` is zero or not less than the number of values, or the
/// series is constant or contains non-finite values.
///
/// # Examples
///
/// ```
/// use backshift::{burg, Estimator, TimeSeries};
///
/// let x = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 8.0];
/// let fit = burg(TimeSeries::try_new(&x, Estimator::Biased).unwrap(), 2).unwrap();
/// assert!(fit.phi.iter().all(|phi| phi.abs() < 2.0));
/// ```
#[must_use]
pub fn burg<'a, T: Borrow<TimeSeries<'a>>>(values: T, p: usize) -> Option<ArCoefficients> {
    let values = values.borrow();
    if p == 0 || p >= values.len() || is_constant(values) {
        return None;
    }

    let x: Vec<f64> = values.residuals().collect();
    let energy = dot(&x, &x);
    if !energy.is_finite() || energy == 0.0 {
        return None;
    }

    let (pacf, sigma2) = burg_pacf(&x, p);
    let phi = phi_from_pacf(&pacf[1..]);

    Some(ArCoefficients { phi, sigma2 })
}

/// Runs the Durbin–Levinson recursion on the autocovariances `c_0, ..., c_p`.
///
/// Autocorrelations work too, since the recursion only uses ratios of `c`.
/// Requires `c.len() >= 1` and `c[0] != 0`.
pub(crate) fn durbin_levinson(c: &[f64]) -> Levinson {
    let p = c.len() - 1;
    let mut pacf = Vec::with_capacity(p + 1);
    pacf.push(1.0);

    // phi_{k-1, j} for j = 1..k-1, stored at index j - 1.
    let mut prev: Vec<f64> = Vec::with_capacity(p);
    let mut curr: Vec<f64> = Vec::with_capacity(p);

    for k in 1..=p {
        let num = c[k] - (1..k).map(|j| prev[j - 1] * c[k - j]).sum::<f64>();
        let den = c[0] - (1..k).map(|j| prev[j - 1] * c[j]).sum::<f64>();
        let phi_kk = num / den;

        step_up(&prev, phi_kk, &mut curr);
        core::mem::swap(&mut prev, &mut curr);

        pacf.push(phi_kk);
    }

    Levinson { pacf, phi: prev }
}

/// Converts partial autocorrelations `phi_11, ..., phi_pp` into AR(p) coefficients.
///
/// This is statsmodels `levinson_durbin_pacf` without the autocorrelations.
fn phi_from_pacf(pacf: &[f64]) -> Vec<f64> {
    let mut prev = Vec::with_capacity(pacf.len());
    let mut curr = Vec::with_capacity(pacf.len());
    for &phi_kk in pacf {
        step_up(&prev, phi_kk, &mut curr);
        core::mem::swap(&mut prev, &mut curr);
    }

    prev
}

/// One step of the Levinson step-up recursion.
///
/// Given the AR(k-1) coefficients `prev` and the new partial autocorrelation
/// `phi_kk`, writes the AR(k) coefficients
/// `phi_kj = phi_(k-1)j - phi_kk * phi_(k-1)(k-j)`, followed by `phi_kk`, into `next`.
fn step_up(prev: &[f64], phi_kk: f64, next: &mut Vec<f64>) {
    let k = prev.len() + 1;
    next.clear();
    next.extend((1..k).map(|j| prev[j - 1] - phi_kk * prev[k - j - 1]));
    next.push(phi_kk);
}

/// Burg's recursion over the demeaned series `x`, for `1 <= p < x.len()`.
///
/// Returns the partial autocorrelations `[1, phi_11, ..., phi_pp]` and the AR(p)
/// innovation variance. Index for index a port of statsmodels `pacf_burg`, which
/// runs on the reversed series; `u` and `v` hold the forward and backward errors.
fn burg_pacf(x: &[f64], p: usize) -> (Vec<f64>, f64) {
    let n = x.len();
    let mut u: Vec<f64> = x.iter().rev().copied().collect();
    let mut v = u.clone();
    let mut last_u = u.clone();
    let mut last_v = v.clone();

    let mut d = vec![0.0; p + 1];
    let mut pacf = vec![0.0; p + 1];
    d[0] = 2.0 * dot(x, x);
    d[1] = dot(&u[..n - 1], &u[..n - 1]) + dot(&v[1..], &v[1..]);
    pacf[1] = 2.0 / d[1] * dot(&v[1..], &u[..n - 1]);

    for i in 1..p {
        last_u.clone_from(&u);
        last_v.clone_from(&v);
        for j in 1..n {
            u[j] = last_u[j - 1] - pacf[i] * last_v[j];
            v[j] = last_v[j] - pacf[i] * last_u[j - 1];
        }

        d[i + 1] = (1.0 - pacf[i] * pacf[i]) * d[i] - v[i] * v[i] - u[n - 1] * u[n - 1];
        pacf[i + 1] = 2.0 / d[i + 1] * dot(&v[i + 1..], &u[i..n - 1]);
    }

    let sigma2 = (1.0 - pacf[p] * pacf[p]) * d[p] / (2.0 * (n - p) as f64);
    pacf[0] = 1.0;

    (pacf, sigma2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Estimator;
    use float_cmp::assert_approx_eq;

    #[test]
    fn durbin_levinson_on_ar1() {
        // An AR(1) process with phi = 0.5 has acf 0.5^k and pacf [1, 0.5, 0, 0, ...].
        let c: Vec<f64> = (0..=5).map(|k| 0.5_f64.powi(k)).collect();
        let result = durbin_levinson(&c);
        let expected = [1.0, 0.5, 0.0, 0.0, 0.0, 0.0];
        for (r, e) in result.pacf.iter().zip(expected) {
            assert_approx_eq!(f64, *r, e, epsilon = 1e-12);
        }

        assert_eq!(result.phi.len(), 5);
        assert_approx_eq!(f64, result.phi[0], 0.5, epsilon = 1e-12);
        assert!(result.phi[1..].iter().all(|phi| phi.abs() < 1e-12));
    }

    #[test]
    fn phi_from_pacf_inverts_durbin_levinson() {
        let c = [2.0, 1.1, 0.4, -0.2];
        let levinson = durbin_levinson(&c);
        let phi = phi_from_pacf(&levinson.pacf[1..]);
        for (a, b) in phi.iter().zip(&levinson.phi) {
            assert_approx_eq!(f64, *a, *b, epsilon = 1e-12);
        }
    }

    #[test]
    fn rejects_bad_orders_and_constant_series() {
        let x = [1.0, 2.0, 4.0];
        let ts = TimeSeries::try_new(&x, Estimator::Biased).unwrap();
        assert_eq!(yule_walker(&ts, 3), None);
        assert_eq!(burg(&ts, 0), None);
        assert_eq!(burg(&ts, 3), None);

        let constant = TimeSeries::try_new(&[2.0; 4], Estimator::Biased).unwrap();
        assert_eq!(yule_walker(&constant, 1), None);
        assert_eq!(burg(&constant, 1), None);
    }

    #[test]
    fn yule_walker_order_zero_is_the_variance() {
        let x = [1.0, 2.0, 4.0];
        let ts = TimeSeries::try_new(&x, Estimator::Biased).unwrap();
        let fit = yule_walker(&ts, 0).unwrap();
        assert!(fit.phi.is_empty());
        assert_approx_eq!(f64, fit.sigma2, ts.var(), epsilon = 1e-12);
    }

    #[test]
    fn constant_series_with_an_inexact_mean() {
        let x = [0.1; 50];
        let ts = TimeSeries::try_new(&x, Estimator::Biased).unwrap();
        assert!(yule_walker(&ts, 2).is_none());
        assert!(burg(&ts, 2).is_none());
    }
}
