//! Backtesting: R's `tsCV` rolling-origin errors for any forecaster, and the choice of a
//! forecasting family by its out-of-sample error.
//!
//! [`select_family`] scores each family on the last few blocks of `horizon` values, each
//! forecast from the values before it, then refits the best one on the whole series:
//!
//! ```mermaid
//! flowchart LR
//!     Y["series"] --> O["origins: n - k h, k = origins..1"]
//!     O --> F["fit and forecast each family"]
//!     F --> S["mean MASE per family"]
//!     S --> W["lowest score"]
//!     W --> R["refit on the series and forecast"]
//! ```

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::accuracy::accuracy;
use crate::ets::EtsFilter;
use crate::filter::Filter;
use crate::fit::ets::MAX_PERIOD;
use crate::fit::{AutoArima, AutoEts, FitError, Theta};
use crate::math;
use crate::theta::ThetaFilter;

/// The state capacity of the ARIMA filter that forecasts a selected model. It holds R's
/// default search space (`p, q <= 5`, `P, Q <= 2`, `d <= 2`, `D <= 1`) up to period 24,
/// where the largest model has `3 m + 8 = 80` states.
const ARIMA_CAPACITY: usize = 80;

/// The longest season the Theta forecaster holds, a daily series' year.
const THETA_CAPACITY: usize = 366;

/// A forecasting family [`select_family`] compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// An automatic (seasonal) ARIMA search, as [`AutoArima`] with its defaults.
    Arima,
    /// An automatic exponential smoothing search, as [`AutoEts`] with its defaults.
    Ets,
    /// The Theta method, as [`Theta`].
    Theta,
    /// The last value, repeated.
    Naive,
    /// The value one season earlier; only for a period above 1.
    SeasonalNaive,
}

impl Family {
    /// Every family, in the order that breaks ties between equal scores.
    pub const ALL: [Family; 5] = [
        Family::Arima,
        Family::Ets,
        Family::Theta,
        Family::Naive,
        Family::SeasonalNaive,
    ];
}

/// The outcome of [`select_family`].
#[derive(Debug, Clone, PartialEq)]
pub struct FamilySelection {
    /// The family with the lowest mean MASE.
    pub family: Family,
    /// Each family of [`Family::ALL`] with its mean MASE over the origins; `NaN` where the
    /// family could not fit or forecast at some origin, or does not apply.
    pub scores: [(Family, f64); 5],
    /// The forecast of the next `horizon` values by the selected family, refitted on the
    /// whole series.
    pub forecast: Vec<f64>,
}

/// The forecast errors from every origin of `y`, as R's
/// `tsCV(y, forecastfunction, h = horizon, initial = initial)`.
///
/// The result is row-major with `horizon` errors per value. Row `t` holds `y[t + j] -
/// forecast_j` for `j = 1..=horizon`, where `forecaster(&y[..=t], horizon)` returns the
/// forecasts of the next `horizon` values. A row is `NaN` before `initial`, at the last
/// value, and where `forecaster` returns `None` (R's failed forecast); an error is `NaN`
/// past the end of `y`.
///
/// # Panics
///
/// Panics if `y.len() * horizon` overflows, or `forecaster` returns fewer than `horizon`
/// values.
///
/// # Examples
///
/// ```
/// use backshift::cross_validation_errors;
///
/// let y = [1.0, 2.0, 4.0, 7.0];
/// let naive = |history: &[f64], h: usize| history.last().map(|&last| vec![last; h]);
/// let errors = cross_validation_errors(&y, 2, 1, naive);
/// assert_eq!(&errors[2..4], &[2.0, 5.0]);
/// assert!(errors[0].is_nan() && errors[5].is_nan());
/// ```
pub fn cross_validation_errors<F>(
    y: &[f64],
    horizon: usize,
    initial: usize,
    mut forecaster: F,
) -> Vec<f64>
where
    F: FnMut(&[f64], usize) -> Option<Vec<f64>>,
{
    let n = y.len();
    let cells = n
        .checked_mul(horizon)
        .expect("y.len() * horizon must not overflow");
    let mut errors = vec![f64::NAN; cells];
    for t in initial..n.saturating_sub(1) {
        let Some(forecast) = forecaster(&y[..=t], horizon) else {
            continue;
        };
        assert!(
            forecast.len() >= horizon,
            "the forecaster must return horizon values"
        );

        let row = &mut errors[t * horizon..(t + 1) * horizon];
        for ((error, actual), predicted) in row.iter_mut().zip(&y[t + 1..]).zip(&forecast) {
            *error = actual - predicted;
        }
    }

    errors
}

/// Picks the forecasting family with the lowest out-of-sample error on `y`, and forecasts
/// the next `horizon` values with it.
///
/// Each family of [`Family::ALL`] forecasts the last `origins` blocks of `horizon` values,
/// each from all the values before it, and scores by the mean of the blocks' MASE (from
/// [`accuracy`](crate::accuracy()) with season length `period`). A family counts only if it
/// fits and forecasts finite values at every origin. The family with the lowest score is
/// fitted on all of `y`. A `period` above 1 makes the searches seasonal; ETS models a
/// season up to 24 long, and a seasonal ARIMA model needs at most 80 states (any default
/// model up to period 24).
///
/// The searches run at every origin, so the cost is about `origins + 1` automatic ARIMA and
/// ETS fits.
///
/// # Errors
///
/// - [`FitError::Unsupported`] if `horizon` or `origins` is 0.
/// - [`FitError::TooShort`] if `y` holds fewer than `origins * horizon + 2` values.
/// - [`FitError::Degenerate`] if no family fits and forecasts at every origin.
///
/// # Examples
///
/// ```
/// use backshift::{Family, select_family};
///
/// let y: Vec<f64> = (0..40).map(|t| 10.0 + (t as f64 * 0.7).sin() + 0.3 * t as f64).collect();
/// let choice = select_family(&y, 1, 4, 2).unwrap();
/// assert_eq!(choice.forecast.len(), 4);
/// assert!(choice.scores.iter().any(|&(family, _)| family == choice.family));
/// ```
pub fn select_family(
    y: &[f64],
    period: usize,
    horizon: usize,
    origins: usize,
) -> Result<FamilySelection, FitError> {
    if horizon == 0 || origins == 0 {
        return Err(FitError::Unsupported);
    }

    let needed = origins.saturating_mul(horizon).saturating_add(2);
    if y.len() < needed {
        return Err(FitError::TooShort {
            needed,
            present: y.len(),
        });
    }

    let scores = Family::ALL.map(|family| (family, score(family, y, period, horizon, origins)));
    let (family, _) = scores
        .iter()
        .filter(|(_, score)| score.is_finite())
        .fold(
            None,
            |best: Option<(Family, f64)>, &(family, score)| match best {
                Some((_, lowest)) if lowest <= score => best,
                _ => Some((family, score)),
            },
        )
        .ok_or(FitError::Degenerate)?;
    let forecast = forecast(family, y, period, horizon).ok_or(FitError::Degenerate)?;

    Ok(FamilySelection {
        family,
        scores,
        forecast,
    })
}

/// The mean MASE of `family` over the last `origins` blocks of `horizon` values of `y`;
/// `NaN` if it fails at any origin. The caller ensures `y` is longer than the blocks.
fn score(family: Family, y: &[f64], period: usize, horizon: usize, origins: usize) -> f64 {
    let mut total = 0.0;
    for k in (1..=origins).rev() {
        let end = y.len() - k * horizon;
        let (training, rest) = y.split_at(end);
        let Some(predicted) = forecast(family, training, period, horizon) else {
            return f64::NAN;
        };

        let mase = accuracy(&predicted, &rest[..horizon], training, period).mase;
        if !mase.is_finite() {
            return f64::NAN;
        }

        total += mase;
    }

    total / count_as_f64(origins)
}

/// The finite forecasts of the next `horizon` values of `y` by `family`, fitted on `y`;
/// `None` if it cannot fit `y` or does not apply.
fn forecast(family: Family, y: &[f64], period: usize, horizon: usize) -> Option<Vec<f64>> {
    let seasonal = period > 1;
    let mut mean = vec![0.0; horizon];
    let mut variance = vec![0.0; horizon];
    match family {
        Family::Arima => {
            let search = if seasonal {
                AutoArima::new().seasonal(period)
            } else {
                AutoArima::new()
            };
            let fit = search.fit(y).ok()?;
            let mut filter = Box::new(Filter::<ARIMA_CAPACITY>::new(&fit.fitted.spec()).ok()?);
            filter.warm_up(y);
            filter.forecast_into(&mut mean, &mut variance);
        }
        Family::Ets => {
            let fit = AutoEts::new().seasonal(period).fit(y).ok()?;
            let filter = EtsFilter::<MAX_PERIOD>::new(&fit.spec()).ok()?;
            // R's means: with a multiplicative season, the expectations of the future values.
            let mut scratch = vec![0.0; filter.forecast_scratch_len()];
            filter.forecast_with(&mut scratch, &mut mean, &mut variance);
        }
        Family::Theta => {
            let fit = Theta::new().seasonal(period).fit(y).ok()?;
            let filter = Box::new(ThetaFilter::<THETA_CAPACITY>::new(&fit.spec()).ok()?);
            filter.forecast_into(&mut mean, &mut variance);
        }
        Family::Naive => mean.fill(*y.last()?),
        Family::SeasonalNaive => {
            let cycle = y.get(y.len().checked_sub(period)?..).filter(|_| seasonal)?;
            for (value, last) in mean.iter_mut().zip(cycle.iter().cycle()) {
                *value = *last;
            }
        }
    }

    math::all_finite(&mean).then_some(mean)
}

/// A count as a float; exact up to 2^53, far beyond any number of origins.
#[expect(
    clippy::cast_precision_loss,
    reason = "origin counts stay far below 2^53, where the conversion is exact"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}
