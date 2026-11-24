//! The Box–Cox transformation, `(y^lambda - 1) / lambda` (`ln y` at `lambda = 0`), and its
//! maximum-likelihood `lambda`, as scipy's `stats.boxcox` computes them.
//!
//! It stabilises a variance that grows with the level of a positive series. statsmodels'
//! `MSTL(lmbda=...)` transforms the series this way before it decomposes it; here the two
//! steps compose: `mstl(&box_cox(&y, box_cox_lambda(&y)?)?, periods, iterate)`.

use crate::math;
use alloc::vec::Vec;

/// The Box–Cox transform of every value: `expm1(lambda ln y) / lambda`, or `ln y` at
/// `lambda = 0`, as scipy's `special.boxcox`.
///
/// Returns [`None`] unless every value is positive and finite and `lambda` is finite.
///
/// # Examples
///
/// ```
/// use backshift::box_cox;
///
/// let y = box_cox(&[1.0, 4.0, 9.0], 0.5).unwrap();
/// assert_eq!(y, [0.0, 2.0, 4.0]);
/// ```
#[must_use]
pub fn box_cox(values: &[f64], lambda: f64) -> Option<Vec<f64>> {
    if !lambda.is_finite() || !all_positive(values) {
        return None;
    }

    Some(
        values
            .iter()
            .map(|&y| {
                let log = math::ln(y);
                if lambda == 0.0 {
                    log
                } else {
                    math::exp_m1(lambda * log) / lambda
                }
            })
            .collect(),
    )
}

/// The `lambda` that maximises the Box–Cox log-likelihood of `values`, as scipy's
/// `stats.boxcox(values)` (and statsmodels' `MSTL(lmbda="auto")`) chooses it.
///
/// The log-likelihood is `(lambda - 1) sum(ln y) - n/2 ln(var(y^lambda / lambda))` with the
/// population variance. It is maximised by golden-section search on `[-10, 10]`, where it
/// is unimodal for the series this is meant for. scipy uses Brent's method with a relative
/// tolerance of about 1.5e-8, so the two agree to about that.
///
/// Returns [`None`] unless `values` holds at least two values, all positive and finite, and
/// not all equal.
///
/// # Examples
///
/// ```
/// use backshift::box_cox_lambda;
///
/// // Exponential growth: the logarithm (lambda near 0) makes it linear.
/// let y: Vec<f64> = (0..50).map(|t| (0.1 * t as f64).exp() * (1.0 + 0.05 * ((t % 3) as f64 - 1.0))).collect();
/// assert!(box_cox_lambda(&y).unwrap().abs() < 0.2);
/// ```
#[must_use]
pub fn box_cox_lambda(values: &[f64]) -> Option<f64> {
    let first = *values.first()?;
    if values.len() < 2 || !all_positive(values) || values.iter().all(|&y| y == first) {
        return None;
    }

    let logs: Vec<f64> = values.iter().map(|&y| math::ln(y)).collect();
    let log_sum: f64 = logs.iter().sum();
    let likelihood = |lambda: f64| log_likelihood(&logs, log_sum, lambda);
    Some(golden_section_maximum(likelihood, -10.0, 10.0))
}

/// scipy's `boxcox_llf` on the logarithms of the data.
fn log_likelihood(logs: &[f64], log_sum: f64, lambda: f64) -> f64 {
    let n = logs.len() as f64;
    let transformed = |log: f64| {
        if lambda == 0.0 {
            log
        } else {
            math::exp(lambda * log) / lambda
        }
    };
    let mean = logs.iter().map(|&l| transformed(l)).sum::<f64>() / n;
    let variance = logs
        .iter()
        .map(|&l| {
            let deviation = transformed(l) - mean;
            deviation * deviation
        })
        .sum::<f64>()
        / n;
    (lambda - 1.0) * log_sum - n / 2.0 * math::ln(variance)
}

/// The maximiser of a unimodal `f` on `[low, high]`, to about 1e-12.
fn golden_section_maximum(f: impl Fn(f64) -> f64, mut low: f64, mut high: f64) -> f64 {
    // 1 / golden ratio.
    const INVERSE_PHI: f64 = 0.618_033_988_749_894_9;
    let mut left = high - INVERSE_PHI * (high - low);
    let mut right = low + INVERSE_PHI * (high - low);
    let (mut f_left, mut f_right) = (f(left), f(right));
    // Each step shrinks the interval by 0.618, so 100 steps reach 20 * 0.618^100 < 1e-19.
    for _ in 0..100 {
        if f_left >= f_right {
            high = right;
            (right, f_right) = (left, f_left);
            left = high - INVERSE_PHI * (high - low);
            f_left = f(left);
        } else {
            low = left;
            (left, f_left) = (right, f_right);
            right = low + INVERSE_PHI * (high - low);
            f_right = f(right);
        }

        if high - low < 1e-12 {
            break;
        }
    }

    f64::midpoint(low, high)
}

/// Whether every value is positive and finite.
fn all_positive(values: &[f64]) -> bool {
    values.iter().all(|&y| y.is_finite() && y > 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_values_it_cannot_transform() {
        assert!(box_cox(&[1.0, 0.0], 0.5).is_none());
        assert!(box_cox(&[1.0, -2.0], 0.0).is_none());
        assert!(box_cox(&[1.0, 2.0], f64::NAN).is_none());
        assert!(box_cox_lambda(&[3.0, 3.0, 3.0]).is_none());
        assert!(box_cox_lambda(&[3.0]).is_none());
        assert!(box_cox_lambda(&[]).is_none());
    }

    #[test]
    fn lambda_zero_is_the_logarithm() {
        let y = box_cox(&[1.0, core::f64::consts::E], 0.0).unwrap();
        assert_eq!(y[0], 0.0);
        assert!((y[1] - 1.0).abs() < 1e-15);
    }
}
