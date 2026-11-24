//! Forecasts of intermittent demand, series that are zero most of the time: Croston's method
//! with the corrections of Syntetos and Boylan, and the method of Teunter, Syntetos and
//! Babai (TSB). Both are heap-free filters with `O(1)` work per value.
//!
//! Croston (1972) smooths the sizes of the non-zero values and the intervals between them
//! separately, each by simple exponential smoothing that starts from its first value, and
//! forecasts their ratio for every horizon, as R's `forecast::croston`. The Syntetos–Boylan
//! approximation (SBA) multiplies that by `1 - alpha / 2` and the Shale–Boylan–Johnston
//! correction (SBJ) by `1 - alpha / (2 - alpha)`, which remove most of Croston's upward
//! bias. TSB (2011) smooths the probability that a period has demand in every period, and
//! the size only when demand occurs, so its forecast decays through long gaps, as an item
//! goes obsolete.

mod grid;

use crate::filter::FilterError;
use crate::optimize;
use grid::{LANES, Lanes};

/// Which bias correction a [`CrostonFilter`] applies to Croston's ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CrostonVariant {
    /// Croston's ratio of smoothed size to smoothed interval, as R's `croston`.
    #[default]
    Classic,
    /// The Syntetos–Boylan approximation: the ratio times `1 - alpha / 2`.
    Sba,
    /// The Shale–Boylan–Johnston correction: the ratio times `1 - alpha / (2 - alpha)`.
    Sbj,
}

/// Croston's method on a stream of demand values, with no heap.
///
/// # Examples
///
/// ```
/// use backshift::{CrostonFilter, CrostonVariant};
///
/// let mut filter = CrostonFilter::new(0.1, CrostonVariant::Classic).unwrap();
/// filter.warm_up(&[0.0, 0.0, 6.0, 0.0, 4.0]);
/// // Sizes 6 then 4 smooth to 5.8; intervals 3 then 2 smooth to 2.9.
/// assert!((filter.forecast() - 2.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrostonFilter {
    /// The smoothing parameter of both the sizes and the intervals.
    alpha: f64,
    /// The bias correction.
    variant: CrostonVariant,
    /// The smoothed size of the non-zero values.
    size: f64,
    /// The smoothed interval between them, in periods.
    interval: f64,
    /// Periods since the last non-zero value (or since the start).
    since: f64,
    /// How many non-zero values were seen.
    demands: usize,
    /// Whether any value was seen.
    started: bool,
}

impl CrostonFilter {
    /// A filter with smoothing parameter `alpha` (R's default is 0.1) and the bias
    /// correction `variant`, before any value.
    ///
    /// # Errors
    ///
    /// [`FilterError::InvalidParameter`] unless `0 < alpha <= 1`.
    pub fn new(alpha: f64, variant: CrostonVariant) -> Result<Self, FilterError> {
        if !(alpha > 0.0 && alpha <= 1.0) {
            return Err(FilterError::InvalidParameter);
        }

        Ok(Self {
            alpha,
            variant,
            size: 0.0,
            interval: 0.0,
            since: 0.0,
            demands: 0,
            started: false,
        })
    }

    /// Feeds one period's demand and returns its error against the forecast before it
    /// (`NaN` before the first value).
    ///
    /// A positive value smooths the size and the interval since the last one; zero only
    /// lengthens the interval. A negative or non-finite value counts as missing: its error
    /// is `NaN` and, as for zero, time moves on without demand.
    pub fn update(&mut self, y: f64) -> f64 {
        let before = self.forecast();
        self.started = true;
        self.since += 1.0;
        if !(y.is_finite() && y >= 0.0) {
            return f64::NAN;
        }

        if y > 0.0 {
            if self.demands == 0 {
                self.size = y;
                self.interval = self.since;
            } else {
                self.size += self.alpha * (y - self.size);
                self.interval += self.alpha * (self.since - self.interval);
            }
            self.demands += 1;
            self.since = 0.0;
        }

        y - before
    }

    /// Feeds every value of `history` in order.
    pub fn warm_up(&mut self, history: &[f64]) {
        for &y in history {
            self.update(y);
        }
    }

    /// The filter whose `alpha` minimises the mean squared one-step error over `history`,
    /// warmed up on it, as the "optimised" Croston of foresight and statsforecast (with one
    /// `alpha` for sizes and intervals, as R).
    ///
    /// `alpha` is searched in `[0.01, 1]` on a grid refined by Brent's method, without
    /// a heap.
    ///
    /// # Errors
    ///
    /// [`FilterError::InvalidParameter`] if `history` has fewer than two non-zero values, so
    /// no error can be scored.
    pub fn optimised(history: &[f64], variant: CrostonVariant) -> Result<Self, FilterError> {
        let demands = history
            .iter()
            .filter(|&&y| y.is_finite() && y > 0.0)
            .count();
        if demands < 2 {
            return Err(FilterError::InvalidParameter);
        }

        let mse = |alpha: f64| {
            Self::new(alpha, variant).map_or(f64::INFINITY, |mut filter| {
                mean_squared_error(history, |y| filter.update(y))
            })
        };
        let alpha = minimise(
            |alphas| grid::croston(history, variant, alphas),
            mse,
            ALPHA_RANGE,
        );
        let mut filter = Self::new(alpha, variant)?;
        filter.warm_up(history);
        Ok(filter)
    }

    /// The smoothing parameter.
    #[must_use]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// The forecast for every horizon: the smoothed size over the smoothed interval, times
    /// the variant's correction. It is 0 before any non-zero value and `NaN` before any
    /// value, as in R.
    #[must_use]
    pub fn forecast(&self) -> f64 {
        if !self.started {
            return f64::NAN;
        }

        if self.demands == 0 {
            return 0.0;
        }

        let correction = match self.variant {
            CrostonVariant::Classic => 1.0,
            CrostonVariant::Sba => 1.0 - self.alpha / 2.0,
            CrostonVariant::Sbj => 1.0 - self.alpha / (2.0 - self.alpha),
        };
        self.size / self.interval * correction
    }
}

/// The method of Teunter, Syntetos and Babai (TSB) on a stream of demand values, with no
/// heap.
///
/// The probability that a period has demand is smoothed every period with
/// `alpha_probability`, the size of the non-zero values only when one occurs with
/// `alpha_size`; both start from their first value, as statsforecast's `TSB`. The forecast
/// for every horizon is their product.
///
/// # Examples
///
/// ```
/// use backshift::TsbFilter;
///
/// let mut filter = TsbFilter::new(0.5, 0.2).unwrap();
/// filter.warm_up(&[0.0, 3.0, 0.0, 0.0, 5.0]);
/// // Probability 0.3024 (from 0, 1, 0, 0, 1), size 4 (from 3 and 5).
/// assert!((filter.forecast() - 1.2096).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsbFilter {
    /// The smoothing parameter of the sizes.
    alpha_size: f64,
    /// The smoothing parameter of the demand probability.
    alpha_probability: f64,
    /// The smoothed size of the non-zero values.
    size: f64,
    /// The smoothed probability of demand in a period.
    probability: f64,
    /// Whether a non-zero value was seen.
    demanded: bool,
    /// Whether any value was seen.
    started: bool,
}

impl TsbFilter {
    /// A filter with the smoothing parameters of the sizes and of the demand probability,
    /// before any value.
    ///
    /// # Errors
    ///
    /// [`FilterError::InvalidParameter`] unless both lie in `(0, 1]`.
    pub fn new(alpha_size: f64, alpha_probability: f64) -> Result<Self, FilterError> {
        let valid = |a: f64| a > 0.0 && a <= 1.0;
        if !(valid(alpha_size) && valid(alpha_probability)) {
            return Err(FilterError::InvalidParameter);
        }

        Ok(Self {
            alpha_size,
            alpha_probability,
            size: 0.0,
            probability: 0.0,
            demanded: false,
            started: false,
        })
    }

    /// Feeds one period's demand and returns its error against the forecast before it
    /// (`NaN` before the first value).
    ///
    /// A negative or non-finite value counts as missing: its error is `NaN`, and nothing is
    /// smoothed.
    pub fn update(&mut self, y: f64) -> f64 {
        let before = self.forecast();
        if !(y.is_finite() && y >= 0.0) {
            return f64::NAN;
        }

        let occurred = if y > 0.0 { 1.0 } else { 0.0 };
        if self.started {
            self.probability += self.alpha_probability * (occurred - self.probability);
        } else {
            self.probability = occurred;
            self.started = true;
        }

        if y > 0.0 {
            if self.demanded {
                self.size += self.alpha_size * (y - self.size);
            } else {
                self.size = y;
                self.demanded = true;
            }
        }

        y - before
    }

    /// Feeds every value of `history` in order.
    pub fn warm_up(&mut self, history: &[f64]) {
        for &y in history {
            self.update(y);
        }
    }

    /// The filter whose smoothing parameters minimise the mean squared one-step error over
    /// `history`, warmed up on it, without a heap.
    ///
    /// A coarse grid over both parameters picks a start; searches on each in turn (a grid
    /// refined by Brent's method) improve it.
    ///
    /// # Errors
    ///
    /// [`FilterError::InvalidParameter`] if `history` has fewer than two non-zero values.
    pub fn optimised(history: &[f64]) -> Result<Self, FilterError> {
        let demands = history
            .iter()
            .filter(|&&y| y.is_finite() && y > 0.0)
            .count();
        if demands < 2 {
            return Err(FilterError::InvalidParameter);
        }

        let mse = |size: f64, probability: f64| {
            Self::new(size, probability).map_or(f64::INFINITY, |mut filter| {
                mean_squared_error(history, |y| filter.update(y))
            })
        };
        let (mut size, mut probability) = (TSB_GRID[0], TSB_GRID[0]);
        let mut best = f64::INFINITY;
        for &a in &TSB_GRID {
            for &b in &TSB_GRID {
                let value = mse(a, b);
                if value < best {
                    (best, size, probability) = (value, a, b);
                }
            }
        }

        for _ in 0..COORDINATE_ROUNDS {
            size = minimise(
                |sizes| grid::tsb(history, sizes, &[probability; LANES]),
                |a| mse(a, probability),
                ALPHA_RANGE,
            );
            probability = minimise(
                |probabilities| grid::tsb(history, &[size; LANES], probabilities),
                |b| mse(size, b),
                ALPHA_RANGE,
            );
        }

        let mut filter = Self::new(size, probability)?;
        filter.warm_up(history);
        Ok(filter)
    }

    /// The smoothing parameters of the sizes and of the demand probability.
    #[must_use]
    pub const fn alphas(&self) -> (f64, f64) {
        (self.alpha_size, self.alpha_probability)
    }

    /// The forecast for every horizon, the smoothed probability times the smoothed size: 0
    /// before any non-zero value, `NaN` before any value.
    #[must_use]
    pub fn forecast(&self) -> f64 {
        if !self.started {
            return f64::NAN;
        }

        self.probability * self.size
    }
}

/// The range the optimised filters search their smoothing parameters in.
const ALPHA_RANGE: (f64, f64) = (0.01, 1.0);

/// The grid of both smoothing parameters that [`TsbFilter::optimised`] scores first.
const TSB_GRID: [f64; 6] = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0];

/// How often [`TsbFilter::optimised`] alternates between its two parameters.
const COORDINATE_ROUNDS: usize = 3;

/// The mean squared one-step error over `history` of a filter's `update`, skipping the
/// errors that are `NaN` (before the first forecast, and for missing values).
fn mean_squared_error(history: &[f64], mut update: impl FnMut(f64) -> f64) -> f64 {
    let (mut sum, mut count) = (0.0, 0.0);
    for &y in history {
        let error = update(y);
        if !error.is_nan() {
            sum += error * error;
            count += 1.0;
        }
    }

    if count > 0.0 {
        sum / count
    } else {
        f64::INFINITY
    }
}

/// How many evenly spaced points [`minimise`] scores before it refines the best one; a
/// multiple of [`LANES`].
const GRID_POINTS: usize = 100;

/// The minimiser on `[low, high]` of the error `f`: the best of [`GRID_POINTS`] evenly
/// spaced points, scored [`LANES`] at a time by `lanes`, refined by Brent's method within one
/// grid step of it.
///
/// The in-sample error of these methods is not unimodal in their parameters, so a local
/// search alone can stop in a poor local minimum; the grid finds the right basin first.
fn minimise(
    lanes: impl Fn(&Lanes) -> Lanes,
    f: impl Fn(f64) -> f64,
    (low, high): (f64, f64),
) -> f64 {
    let step = (high - low) / grid_index(GRID_POINTS - 1);
    let mut best = (f64::INFINITY, low);
    for first in (0..GRID_POINTS).step_by(LANES) {
        let points: Lanes = core::array::from_fn(|j| low + step * grid_index(first + j));
        for (&value, &x) in lanes(&points).iter().zip(&points) {
            if value < best.0 {
                best = (value, x);
            }
        }
    }

    let (f_best, x_best) = best;
    let bracket = ((x_best - step).max(low), (x_best + step).min(high));
    let (x, fx) = optimize::brent(&f, bracket, 1e-8);
    if fx < f_best { x } else { x_best }
}

/// A grid index as a float; exact, since the grid has [`GRID_POINTS`] points.
#[expect(
    clippy::cast_precision_loss,
    reason = "grid indices stay below GRID_POINTS = 100, where the conversion is exact"
)]
const fn grid_index(index: usize) -> f64 {
    index as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrections_scale_the_classic_forecast() {
        let history = [0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 6.0];
        let forecast = |variant| {
            let mut filter = CrostonFilter::new(0.2, variant).unwrap();
            filter.warm_up(&history);
            filter.forecast()
        };
        let classic = forecast(CrostonVariant::Classic);
        assert!((forecast(CrostonVariant::Sba) - classic * 0.9).abs() < 1e-12);
        assert!((forecast(CrostonVariant::Sbj) - classic * (1.0 - 0.2 / 1.8)).abs() < 1e-12);
    }

    #[test]
    fn missing_and_invalid_values() {
        assert_eq!(
            CrostonFilter::new(0.0, CrostonVariant::Classic),
            Err(FilterError::InvalidParameter)
        );
        assert_eq!(TsbFilter::new(0.1, 1.5), Err(FilterError::InvalidParameter));
        let mut filter = TsbFilter::new(0.5, 0.5).unwrap();
        assert!(filter.forecast().is_nan());
        assert!(filter.update(-1.0).is_nan());
        // No forecast exists before the first value, so its error is `NaN`.
        assert!(filter.update(0.0).is_nan());
        assert_eq!(filter.forecast(), 0.0);
    }

    #[test]
    fn optimised_croston_beats_every_grid_point() {
        let history = [
            0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 4.0, 0.0, 0.0, 3.0,
        ];
        let best = CrostonFilter::optimised(&history, CrostonVariant::Classic).unwrap();
        let mse = |alpha: f64| {
            let mut filter = CrostonFilter::new(alpha, CrostonVariant::Classic).unwrap();
            mean_squared_error(&history, |y| filter.update(y))
        };
        let alpha = best.alpha();
        for i in 1..=100 {
            let grid = f64::from(i) / 100.0;
            assert!(mse(alpha) <= mse(grid) + 1e-12, "alpha {alpha} vs {grid}");
        }
        assert_eq!(
            CrostonFilter::optimised(&[0.0, 2.0], CrostonVariant::Classic),
            Err(FilterError::InvalidParameter)
        );
    }

    #[test]
    fn optimised_tsb_beats_its_grid() {
        let history = [
            0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 4.0, 0.0, 0.0, 3.0,
        ];
        let best = TsbFilter::optimised(&history).unwrap();
        let mse = |a: f64, b: f64| {
            let mut filter = TsbFilter::new(a, b).unwrap();
            mean_squared_error(&history, |y| filter.update(y))
        };
        let (a, b) = best.alphas();
        for grid in [0.05, 0.1, 0.2, 0.4, 0.7, 1.0] {
            assert!(mse(a, b) <= mse(grid, b) + 1e-12 && mse(a, b) <= mse(a, grid) + 1e-12);
        }
    }
}
