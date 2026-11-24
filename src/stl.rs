//! STL: seasonal-trend decomposition using LOESS (Cleveland, Cleveland, McRae & Terpenning 1990).
//!
//! A port of the non-robust path of the original NETLIB Fortran (`stlstp`, `stlss`,
//! `stless`, `stlest`, `stlfts`, `stlma`), the code statsmodels' `STL` also wraps, with
//! every LOESS evaluated at every point (jumps of 1). Positions are 1-based inside the
//! LOESS helpers, as in the Fortran, so the arithmetic stays the same.
//!
//! ```mermaid
//! flowchart LR
//!     Y["y - trend"] -->|"cycle-subseries LOESS"| C["C (n + 2s)"]
//!     C -->|"MA s, s, 3 + LOESS"| L["low-pass L"]
//!     C --> S["seasonal = C - L"]
//!     S -->|"y - seasonal, LOESS"| T["trend"]
//!     T -->|"next inner pass"| Y
//! ```

use alloc::vec;
use alloc::vec::Vec;

mod mstl;
mod period;

pub use mstl::{MSTL_ITERATIONS, MultiDecomposition, mstl};
pub use period::{PeriodCandidate, find_period};

/// The seasonal strength above which a series counts as seasonal: the threshold of R's
/// `nsdiffs` with its default seasonal-strength test, and of coreforecast.
pub const SEASONAL_STRENGTH_THRESHOLD: f64 = 0.64;

/// The components of an STL decomposition: `values = seasonal + trend + remainder`.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    /// The seasonal component.
    pub seasonal: Vec<f64>,
    /// The trend component.
    pub trend: Vec<f64>,
    /// What is left: `values - seasonal - trend`.
    pub remainder: Vec<f64>,
}

impl Decomposition {
    /// The seasonal strength `max(0, 1 - Var(remainder) / Var(seasonal + remainder))`
    /// (Wang, Smith & Hyndman 2006): near 1 for a strong seasonal pattern, near 0 for none.
    #[must_use]
    pub fn seasonal_strength(&self) -> f64 {
        let detrended: Vec<f64> = self
            .seasonal
            .iter()
            .zip(&self.remainder)
            .map(|(s, r)| s + r)
            .collect();
        let total = variance(&detrended);
        if total == 0.0 {
            return 0.0;
        }

        (1.0 - variance(&self.remainder) / total).max(0.0)
    }
}

/// STL with statsmodels' defaults: a seasonal smoother of 7, the trend and low-pass windows
/// derived from `period`, degree-1 LOESS throughout, and five non-robust inner passes.
///
/// Matches statsmodels `STL(values, period=period).fit()`. Returns [`None`] unless
/// `period >= 2` and `values` holds at least two full seasons of finite values. A constant series decomposes exactly
/// into its level as the trend, with zero seasonal and remainder parts; statsmodels leaves
/// rounding noise there, which would read as seasonality.
///
/// # Examples
///
/// ```
/// use backshift::stl;
///
/// let y: Vec<f64> = (0..48).map(|t| [3.0, -1.0, 0.5, -2.5][t % 4] + 0.1 * t as f64).collect();
/// let parts = stl(&y, 4).unwrap();
/// assert!(parts.seasonal_strength() > 0.99);
/// ```
#[must_use]
pub fn stl(values: &[f64], period: usize) -> Option<Decomposition> {
    stl_with(values, period, DEFAULT_SEASONAL_WINDOW)
}

/// STL like [`stl`], with the seasonal smoother `seasonal_window` instead of 7.
///
/// The trend window follows from it as in statsmodels, the smallest odd integer at least
/// `1.5 period / (1 - 1.5 / seasonal_window)`. Matches statsmodels
/// `STL(values, period=period, seasonal=seasonal_window).fit()`. A longer window gives a
/// seasonal pattern that changes more slowly. Returns [`None`] unless `seasonal_window` is
/// odd and at least 3, `period >= 2`, and `values` holds at least two full seasons of
/// finite values. A constant series decomposes exactly
/// into its level as the trend, with zero seasonal and remainder parts; statsmodels leaves
/// rounding noise there, which would read as seasonality.
///
/// # Examples
///
/// ```
/// use backshift::stl_with;
///
/// let y: Vec<f64> = (0..48).map(|t| [3.0, -1.0, 0.5, -2.5][t % 4] + 0.1 * t as f64).collect();
/// let parts = stl_with(&y, 4, 11).unwrap();
/// assert!(parts.seasonal_strength() > 0.99);
/// assert!(stl_with(&y, 4, 8).is_none()); // even window
/// ```
#[must_use]
pub fn stl_with(values: &[f64], period: usize, seasonal_window: usize) -> Option<Decomposition> {
    let n = values.len();
    let window_ok = seasonal_window >= 3 && seasonal_window % 2 == 1;
    if !window_ok || period < 2 || n < 2 * period || !crate::math::all_finite(values) {
        return None;
    }

    if crate::math::is_constant(values) {
        // The smoothers would leave rounding noise that reads as seasonality.
        return Some(Decomposition {
            seasonal: vec![0.0; n],
            trend: values.to_vec(),
            remainder: vec![0.0; n],
        });
    }

    let windows = Windows::new(period, seasonal_window);
    let mut seasonal = vec![0.0; n];
    let mut trend = vec![0.0; n];
    for _ in 0..INNER_PASSES {
        inner_pass(values, period, &windows, &mut seasonal, &mut trend);
    }

    let remainder = (0..n).map(|i| values[i] - seasonal[i] - trend[i]).collect();
    Some(Decomposition {
        seasonal,
        trend,
        remainder,
    })
}

/// statsmodels' default seasonal smoother.
const DEFAULT_SEASONAL_WINDOW: usize = 7;

/// statsmodels' number of inner passes without robustness weights.
const INNER_PASSES: usize = 5;

/// The LOESS window lengths, all odd.
struct Windows {
    /// The seasonal (cycle-subseries) smoother.
    seasonal: usize,
    /// The trend smoother.
    trend: usize,
    /// The low-pass smoother.
    low_pass: usize,
}

impl Windows {
    /// statsmodels' windows for `period` and the odd seasonal window `seasonal >= 3`: trend
    /// the smallest odd integer at least `1.5 period / (1 - 1.5 / seasonal)`, low-pass the
    /// smallest odd integer above `period`.
    fn new(period: usize, seasonal: usize) -> Self {
        let trend_minimum = 1.5 * period as f64 / (1.0 - 1.5 / seasonal as f64);
        let next_odd_above = |x: f64| {
            // The first odd integer strictly greater than x (x is positive).
            let mut candidate = x as usize + 1;
            if candidate % 2 == 0 {
                candidate += 1;
            }
            candidate
        };

        // The trend window may equal its minimum when that is an odd integer (period 11
        // gives exactly 21), so round up first; the low-pass window is strictly above `period`.
        let truncated = trend_minimum as usize;
        let mut trend = truncated + usize::from(trend_minimum > truncated as f64);
        if trend % 2 == 0 {
            trend += 1;
        }

        Self {
            seasonal,
            trend,
            low_pass: next_odd_above(period as f64),
        }
    }
}

/// One inner pass of `stlstp`: detrend, smooth the cycle-subseries, remove their
/// low-pass part, then re-estimate the trend.
fn inner_pass(
    y: &[f64],
    period: usize,
    windows: &Windows,
    seasonal: &mut [f64],
    trend: &mut [f64],
) {
    let n = y.len();
    let detrended: Vec<f64> = y.iter().zip(trend.iter()).map(|(y, t)| y - t).collect();
    let cycle = cycle_subseries(&detrended, period, windows.seasonal);

    let low_pass_input = low_pass_averages(&cycle, period);
    let low_pass = loess_smooth(&low_pass_input, windows.low_pass);
    for i in 0..n {
        seasonal[i] = cycle[period + i] - low_pass[i];
    }

    let deseasonalized: Vec<f64> = y.iter().zip(seasonal.iter()).map(|(y, s)| y - s).collect();
    trend.copy_from_slice(&loess_smooth(&deseasonalized, windows.trend));
}

/// `stlss`: LOESS-smooths each of the `period` cycle-subseries and extends each by one
/// value at both ends. Returns `n + 2 period` values: position `(m - 1) period + j` holds
/// the `m`-th value of subseries `j`, `m = 0` and `m = k + 1` being the extensions.
fn cycle_subseries(y: &[f64], period: usize, window: usize) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n + 2 * period];
    let mut weights = Vec::new();
    for j in 0..period {
        let sub: Vec<f64> = y.iter().skip(j).step_by(period).copied().collect();
        let k = sub.len();
        let smooth = loess_smooth(&sub, window);

        let before =
            loess_estimate(&sub, window, 0.0, 1, window.min(k), &mut weights).unwrap_or(smooth[0]);
        let left = (k + 1).saturating_sub(window).max(1);
        let after = loess_estimate(&sub, window, (k + 1) as f64, left, k, &mut weights)
            .unwrap_or(smooth[k - 1]);

        out[j] = before;
        for (m, &value) in smooth.iter().enumerate() {
            out[(m + 1) * period + j] = value;
        }
        out[(k + 1) * period + j] = after;
    }

    out
}

/// `stlfts`: moving averages of length `period`, `period` and 3, turning the
/// `n + 2 period` cycle-subseries values back into `n` values.
fn low_pass_averages(x: &[f64], period: usize) -> Vec<f64> {
    moving_average(&moving_average(&moving_average(x, period), period), 3)
}

/// `stlma`: the moving average of length `len`, `x.len() - len + 1` values.
fn moving_average(x: &[f64], len: usize) -> Vec<f64> {
    let count = x.len() - len + 1;
    let mut out = Vec::with_capacity(count);
    let mut sum: f64 = x[..len].iter().sum();
    out.push(sum / len as f64);
    for j in 1..count {
        sum = sum - x[j - 1] + x[j + len - 1];
        out.push(sum / len as f64);
    }

    out
}

/// `stless` with a jump of 1: the degree-1 LOESS of `y` with window `len`, evaluated at
/// every position. A position whose weights all vanish keeps its value.
fn loess_smooth(y: &[f64], len: usize) -> Vec<f64> {
    let n = y.len();
    if n < 2 {
        return y.to_vec();
    }

    let mut weights = Vec::new();
    if len >= n {
        return (1..=n)
            .map(|i| loess_estimate(y, len, i as f64, 1, n, &mut weights).unwrap_or(y[i - 1]))
            .collect();
    }

    let half = len.div_ceil(2);
    let (mut left, mut right) = (1, len);
    (1..=n)
        .map(|i| {
            if i > half && right != n {
                left += 1;
                right += 1;
            }

            loess_estimate(y, len, i as f64, left, right, &mut weights).unwrap_or(y[i - 1])
        })
        .collect()
}

/// `stlest`: the degree-1 LOESS estimate at position `xs` (1-based) from the values at
/// positions `left..=right`, with tricube weights over the window `len`.
///
/// Returns [`None`] when every weight is zero, the Fortran's `ok = .false.`.
fn loess_estimate(
    y: &[f64],
    len: usize,
    xs: f64,
    left: usize,
    right: usize,
    weights: &mut Vec<f64>,
) -> Option<f64> {
    let n = y.len();
    let range = (right - left) as f64;
    let mut h = (xs - left as f64).max(right as f64 - xs);
    if len > n {
        h += ((len - n) / 2) as f64;
    }

    let (h9, h1) = (0.999 * h, 0.001 * h);
    weights.clear();
    weights.resize(right - left + 1, 0.0);
    let mut total = 0.0;
    for (w, j) in weights.iter_mut().zip(left..=right) {
        let r = (j as f64 - xs).abs();
        if r <= h9 {
            *w = if r <= h1 {
                1.0
            } else {
                let u = r / h;
                let cube = 1.0 - u * u * u;
                cube * cube * cube
            };
            total += *w;
        }
    }

    if total <= 0.0 {
        return None;
    }

    for w in weights.iter_mut() {
        *w /= total;
    }

    if h > 0.0 {
        // Degree 1: tilt the weights towards a local line through the weighted centre.
        let centre: f64 = weights
            .iter()
            .zip(left..=right)
            .map(|(w, j)| w * j as f64)
            .sum();
        let spread: f64 = weights
            .iter()
            .zip(left..=right)
            .map(|(w, j)| w * (j as f64 - centre) * (j as f64 - centre))
            .sum();
        if crate::math::sqrt(spread) > 0.001 * range {
            let slope = (xs - centre) / spread;
            for (w, j) in weights.iter_mut().zip(left..=right) {
                *w *= slope * (j as f64 - centre) + 1.0;
            }
        }
    }

    Some(
        weights
            .iter()
            .zip(&y[left - 1..right])
            .map(|(w, v)| w * v)
            .sum(),
    )
}

/// The population variance.
fn variance(x: &[f64]) -> f64 {
    let mean = crate::math::mean(x);
    x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / x.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_windows_follow_statsmodels() {
        let w = Windows::new(12, 7);
        assert_eq!((w.seasonal, w.trend, w.low_pass), (7, 23, 13));
        let w = Windows::new(4, 7);
        assert_eq!((w.trend, w.low_pass), (9, 5));
        let w = Windows::new(7, 7);
        assert_eq!((w.trend, w.low_pass), (15, 9));
        // 1.5 * 11 / (1 - 1.5 / 7) is exactly 21, which statsmodels keeps.
        let w = Windows::new(11, 7);
        assert_eq!((w.trend, w.low_pass), (21, 13));
        assert_eq!(Windows::new(33, 7).trend, 63);
    }

    #[test]
    fn moving_average_matches_hand_computation() {
        assert_eq!(
            moving_average(&[1.0, 2.0, 3.0, 4.0, 6.0], 2),
            [1.5, 2.5, 3.5, 5.0]
        );
    }

    #[test]
    fn rejects_short_or_non_finite_input() {
        assert!(stl(&[1.0; 7], 4).is_none());
        assert!(stl(&[1.0; 8], 1).is_none());
        let mut y = [1.0; 12];
        y[3] = f64::NAN;
        assert!(stl(&y, 4).is_none());
    }

    #[test]
    fn constant_series_has_no_seasonality() {
        let parts = stl(&[0.1; 48], 4).unwrap();
        assert_eq!(parts.seasonal_strength(), 0.0);
        assert!(parts.trend.iter().all(|&t| t == 0.1));
    }
}
