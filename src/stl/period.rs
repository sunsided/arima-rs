//! Seasonal period candidates, by the rule of Nixtla's coreforecast
//! (`find_season_length`).
//!
//! The candidates are the lags in `[2, max_period]` at which the autocovariance of the first
//! differences peaks, ranked by it. A candidate is kept only if the series has a strong
//! seasonal pattern at that period: STL seasonal strength above
//! [`SEASONAL_STRENGTH_THRESHOLD`], the rule [`nsdiffs`](crate::nsdiffs) uses.

use super::{SEASONAL_STRENGTH_THRESHOLD, stl};
use alloc::vec::Vec;

/// A seasonal period candidate of [`find_period`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodCandidate {
    /// The season length.
    pub period: usize,
    /// The sum of the products of the first differences `period` steps apart.
    pub autocovariance: f64,
    /// The STL seasonal strength of the series at this period, above
    /// [`SEASONAL_STRENGTH_THRESHOLD`].
    pub strength: f64,
}

/// Ranks seasonal period candidates of `values` up to `max_period`, strongest first.
///
/// This follows coreforecast's `find_season_length`. It computes `c_k = sum_t dx_t dx_(t+k)`
/// for the first differences `dx` and each lag `k` in `[2, max_period]`; the products are
/// not demeaned or scaled, as coreforecast does it. The candidates are the lags where
/// `c_k` peaks (it is larger than at its neighbours within `[2, max_period]`), in
/// descending order of `c_k`, so the largest `c_k` comes first.
/// Only the 8 largest peaks are checked, and of those only candidates
/// whose STL seasonal strength is above [`SEASONAL_STRENGTH_THRESHOLD`] remain, which needs
/// two full seasons of data.
///
/// coreforecast returns only its single largest lag, and 0 if that lag fails the strength
/// check. When it finds a period, that period is the first candidate here too. Its STL
/// smooths the seasonal part with degree 0 and this crate's with degree 1, so strengths
/// near the threshold can differ.
///
/// The result is a heuristic and a ranking, not a test. R's `findfrequency`, coreforecast
/// and periodogram methods are heuristics, and R calls its own threshold arbitrary. No
/// established method detects several periods jointly. The candidates are integer periods,
/// and with a moderate `max_period` long periods such as 8766 on hourly data stay out of
/// reach.
///
/// Returns no candidate if a value is `NaN` or infinite, or fewer than 4 values are given.
///
/// # Examples
///
/// ```
/// use backshift::find_period;
///
/// let y: Vec<f64> = (0..120)
///     .map(|t| [3.0, 1.0, -2.0, 0.5, -1.0, -1.5, 0.0][t % 7] + 0.05 * t as f64)
///     .collect();
/// let candidates = find_period(&y, 30);
/// assert_eq!(candidates[0].period, 7);
/// ```
#[must_use]
pub fn find_period(values: &[f64], max_period: usize) -> Vec<PeriodCandidate> {
    if values.len() < 4 || !crate::math::all_finite(values) {
        return Vec::new();
    }

    let differences: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
    let largest = max_period.min(differences.len() - 1);
    // Index `k` holds the lag-`k` product sum; lags 0 and 1 are never candidates.
    let covariance: Vec<f64> = (0..=largest)
        .map(|k| lagged_products(&differences, k))
        .collect();

    let mut peaks: Vec<usize> = (2..=largest).filter(|&k| is_peak(&covariance, k)).collect();
    peaks.sort_by(|&a, &b| covariance[b].total_cmp(&covariance[a]));

    peaks
        .into_iter()
        .take(MAX_CHECKED_PEAKS)
        .filter_map(|period| {
            let strength = stl(values, period)?.seasonal_strength();
            (strength > SEASONAL_STRENGTH_THRESHOLD).then_some(PeriodCandidate {
                period,
                autocovariance: covariance[period],
                strength,
            })
        })
        .collect()
}

/// How many of the largest autocovariance peaks [`find_period`] checks with a full STL.
///
/// shortcut: noise has hundreds of peaks and each STL costs about `n period`, which made
/// 2000 values take 40 s; raise it if real series need weaker peaks ranked further down.
const MAX_CHECKED_PEAKS: usize = 8;

/// `sum_t x_t x_(t+k)`.
fn lagged_products(x: &[f64], k: usize) -> f64 {
    x.iter().zip(&x[k..]).map(|(a, b)| a * b).sum()
}

/// Whether `c[k]` is larger than its neighbours among lags `2..c.len()`; the first and the
/// last lag have only one.
fn is_peak(c: &[f64], k: usize) -> bool {
    let left = k == 2 || c[k] > c[k - 1];
    let right = c.get(k + 1).is_none_or(|&next| c[k] > next);
    left && right
}
