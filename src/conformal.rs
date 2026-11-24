//! Conformal prediction intervals for streaming forecasts, with no heap.
//!
//! [`Conformal`] keeps, per horizon `h`, a rolling buffer of the absolute errors of past
//! `h`-step forecasts. The half-width of an interval is an empirical quantile of that
//! buffer. Unlike the Gaussian intervals of
//! [`Filter::forecast_interval_into`](crate::Filter::forecast_interval_into), it does not
//! assume Gaussian innovations, so it adapts to skewed or heavy-tailed noise without a
//! change to the model.
//!
//! The error of an `h`-step forecast made at time `t` is known only at `t + h`. So each
//! time step first scores the forecasts made `1, ..., H` steps earlier, then records the
//! new ones (the delayed feedback of Wang & Hyndman 2024, arXiv:2410.13115). The
//! quantile rule is that of split conformal prediction (the `ceil((n + 1) level)`-th
//! smallest of `n` scores) with one rolling calibration set per horizon, the "MSCP"
//! method of Sabashvili 2026 (arXiv:2601.18509).
//!
//! On one time series the scores are not exchangeable, so no finite-sample coverage
//! guarantee holds. The papers above report coverage near the target in practice.
//!
//! ```mermaid
//! flowchart LR
//!     Y["new value y_t"] -->|"observe: |y_t - forecast|"| E["error buffer per h"]
//!     Y -->|"Filter::update"| F["Filter"]
//!     F -->|"forecast_into"| M["mean"]
//!     M -->|record| P["pending forecasts"]
//!     P -->|"next observe"| E
//!     E -->|"interval_into"| I["mean ± quantile"]
//! ```

/// Conformal intervals for forecasts up to `H` steps ahead, calibrated on the last `N`
/// errors per horizon. It lives inline with no heap.
///
/// The size is about `8 (H² + H N)` bytes. [`Conformal::interval_into`] uses about
/// `8 N` more bytes of stack.
///
/// # Examples
///
/// ```
/// use backshift::{ArimaSpec, Conformal, Filter};
///
/// let spec = ArimaSpec { ar: &[0.6], ..ArimaSpec::default() };
/// let mut filter = Filter::<1>::new(&spec).unwrap();
/// let mut conformal = Conformal::<3, 64>::new();
/// let mut mean = [0.0; 3];
/// for t in 0..200 {
///     let y = ((t * 7919) % 13) as f64 / 6.0 - 1.0;
///     conformal.observe(y); // score the forecasts made for this time
///     filter.update(y);
///     filter.forecast_into(&mut mean, &mut [0.0; 3]);
///     conformal.record(&mean); // remember them until their values arrive
/// }
///
/// let (mut lower, mut upper) = ([0.0; 3], [0.0; 3]);
/// conformal.interval_into(0.9, &mean, &mut lower, &mut upper);
/// assert!(lower[0] < mean[0] && mean[0] < upper[0]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Conformal<const H: usize, const N: usize> {
    /// `pending[i][h]` is the forecast for `h + 1` steps ahead made `i` observations ago;
    /// `NaN` where none was recorded.
    pending: [[f64; H]; H],
    /// The rolling buffer of absolute errors of each horizon.
    errors: [[f64; N]; H],
    /// The number of errors stored per horizon, at most `N`.
    count: [usize; H],
    /// The slot per horizon that the next error overwrites.
    next: [usize; H],
}

impl<const H: usize, const N: usize> Conformal<H, N> {
    /// An empty calibration: every interval is infinite until errors arrive.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: [[f64::NAN; H]; H],
            errors: [[0.0; N]; H],
            count: [0; H],
            next: [0; H],
        }
    }

    /// Takes the next observed value: scores each forecast recorded `1, ..., H` steps
    /// before for this time, and moves forward one step.
    ///
    /// Call it once per time step, before [`Conformal::record`]. A `NaN` value scores
    /// nothing but still moves forward, like a missing value in the filter.
    pub fn observe(&mut self, y: f64) {
        for h in 0..H {
            let forecast = self.pending[h][h];
            if !y.is_nan() && !forecast.is_nan() {
                self.push(h, (y - forecast).abs());
            }
        }

        self.pending.copy_within(..H.saturating_sub(1), 1);
        if let Some(newest) = self.pending.first_mut() {
            newest.fill(f64::NAN);
        }
    }

    /// Records the forecasts made now, `forecast[h]` for `h + 1` steps ahead, to score
    /// them when their values arrive. Extra entries beyond `H` are ignored; missing ones
    /// stay unscored.
    pub fn record(&mut self, forecast: &[f64]) {
        if let Some(newest) = self.pending.first_mut() {
            for (slot, &value) in newest.iter_mut().zip(forecast) {
                *slot = value;
            }
        }
    }

    /// The interval half-width for `h + 1` steps ahead at probability `level`: the
    /// `ceil(level (n + 1))`-th smallest of the `n` stored errors.
    ///
    /// Infinite while fewer than `level (n + 1) - 1` errors are stored (for example
    /// below 9 errors at `level = 0.9`), and for `level = 1`. `NaN` for a `level` outside
    /// `[0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics if `h >= H`.
    #[must_use]
    pub fn half_width(&self, h: usize, level: f64) -> f64 {
        if !(0.0..=1.0).contains(&level) {
            return f64::NAN;
        }

        let n = self.count[h];
        let rank = rank(level, n);
        if rank > n {
            return f64::INFINITY;
        }

        let mut sorted = self.errors[h];
        let sorted = &mut sorted[..n];
        sorted.sort_unstable_by(f64::total_cmp);
        rank.checked_sub(1).map_or(0.0, |index| sorted[index])
    }

    /// Writes conformal intervals around `mean` (the forecasts of
    /// [`Filter::forecast_into`](crate::Filter::forecast_into)) at probability `level`:
    /// `mean[h] ± half_width(h, level)`. Horizons beyond `H` get infinite bounds.
    ///
    /// # Panics
    ///
    /// Panics if `mean`, `lower` and `upper` do not all have the same length.
    pub fn interval_into(&self, level: f64, mean: &[f64], lower: &mut [f64], upper: &mut [f64]) {
        assert!(
            mean.len() == lower.len() && mean.len() == upper.len(),
            "mean, lower and upper must have the same length"
        );
        for (h, ((m, lo), up)) in mean.iter().zip(lower).zip(upper).enumerate() {
            let width = if h < H {
                self.half_width(h, level)
            } else {
                f64::INFINITY
            };
            *lo = m - width;
            *up = m + width;
        }
    }

    /// The number of errors stored for `h + 1` steps ahead, at most `N`.
    ///
    /// # Panics
    ///
    /// Panics if `h >= H`.
    #[must_use]
    pub fn calibration_len(&self, h: usize) -> usize {
        self.count[h]
    }

    /// Stores an error of horizon `h`, overwriting the oldest once `N` are stored.
    fn push(&mut self, h: usize, error: f64) {
        if N == 0 {
            return;
        }

        self.errors[h][self.next[h]] = error;
        self.next[h] = (self.next[h] + 1) % N;
        self.count[h] = (self.count[h] + 1).min(N);
    }
}

impl<const H: usize, const N: usize> Default for Conformal<H, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// The rank `ceil(level (n + 1))` of the split-conformal quantile, at least 0.
fn rank(level: f64, n: usize) -> usize {
    // `n + 1` is far below 2^53, so it is exact as an `f64`.
    let target = level * (n + 1) as f64;
    // `target` is finite and non-negative; truncation then rounding up gives the ceiling.
    let floor = target as usize;
    floor + usize::from((floor as f64) < target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_each_horizon_when_its_value_arrives() {
        let mut conformal = Conformal::<2, 8>::new();
        conformal.observe(0.0);
        conformal.record(&[1.0, 2.0]); // made at t = 0 for t = 1 and t = 2
        conformal.observe(1.5); // 1-step error 0.5
        conformal.record(&[1.0, 1.0]); // made at t = 1
        conformal.observe(4.0); // 1-step error 3.0 (from t = 1), 2-step error 2.0 (from t = 0)
        assert_eq!(conformal.calibration_len(0), 2);
        assert_eq!(conformal.calibration_len(1), 1);
        assert_eq!(conformal.errors[0][..2], [0.5, 3.0]);
        assert_eq!(conformal.errors[1][0], 2.0);
    }

    #[test]
    fn uses_the_split_conformal_rank() {
        let mut conformal = Conformal::<1, 16>::new();
        for e in 1..=9 {
            conformal.record(&[0.0]);
            conformal.observe(f64::from(e));
        }

        // n = 9 at level 0.9: rank ceil(9) = 9, the largest error.
        assert_eq!(conformal.half_width(0, 0.9), 9.0);
        // n = 9 at level 0.95: rank ceil(9.5) = 10 > 9, so no finite bound yet.
        assert_eq!(conformal.half_width(0, 0.95), f64::INFINITY);
        assert_eq!(conformal.half_width(0, 0.5), 5.0);
        assert!(conformal.half_width(0, 1.5).is_nan());
    }

    #[test]
    fn forgets_the_oldest_error() {
        let mut conformal = Conformal::<1, 2>::new();
        for e in [100.0, 1.0, 2.0] {
            conformal.record(&[0.0]);
            conformal.observe(e);
        }

        assert_eq!(conformal.calibration_len(0), 2);
        assert_eq!(conformal.half_width(0, 0.6), 2.0);
    }

    #[test]
    fn missing_values_score_nothing() {
        let mut conformal = Conformal::<1, 4>::new();
        conformal.record(&[0.0]);
        conformal.observe(f64::NAN);
        assert_eq!(conformal.calibration_len(0), 0);
    }
}
