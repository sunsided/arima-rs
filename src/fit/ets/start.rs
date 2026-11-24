//! Starting states for an ETS fit whose initial states are optimised, as R's
//! `forecast:::initstate` computes them.
//!
//! The seasonal states come from a classical decomposition (`stats::decompose`), additive or
//! multiplicative, or from a linear trend with one Fourier pair when the series holds fewer
//! than three seasons. The level and slope come from a least-squares line through the first
//! values of the seasonally adjusted series.

use super::super::hannan_rissanen::least_squares;
use super::layout::count_as_f64;
use crate::ets::Form;
use crate::fourier::{fourier_columns, fourier_row};
use alloc::vec;
use alloc::vec::Vec;

/// The starting free states: the level, the slope with a trend, then the first `m - 1`
/// seasonal states, most recent first (the last follows from the others).
///
/// `y` must be finite, positive for a multiplicative season, and hold more than `m` values
/// for a seasonal model (`m > 1`).
pub(super) fn start_states(y: &[f64], trend: bool, m: usize, season: Form) -> Vec<f64> {
    let (seasonal, adjusted) = if m > 1 {
        seasonal_start(y, m, season)
    } else {
        (Vec::new(), y.to_vec())
    };

    let mut states = level_and_slope(&adjusted, trend, m);
    states.extend(seasonal);
    states
}

/// The first `m - 1` seasonal states (most recent first) and the seasonally adjusted series.
fn seasonal_start(y: &[f64], m: usize, season: Form) -> (Vec<f64>, Vec<f64>) {
    let effects = seasonal_effects(y, m, season);
    // R keeps `rev(seasonal[2:m])`: the effects of positions m, m - 1, ..., 2.
    let mut first: Vec<f64> = effects[1..m].iter().rev().copied().collect();
    if season == Form::Additive {
        let adjusted = y.iter().zip(&effects).map(|(v, s)| v - s).collect();
        return (first, adjusted);
    }

    // R floors the factors at 0.01 and rescales them when they sum above `m`.
    for value in &mut first {
        *value = value.max(0.01);
    }
    let sum: f64 = first.iter().sum();
    if sum > count_as_f64(m) {
        let scale: f64 = first.iter().map(|v| v + 0.01).sum();
        for value in &mut first {
            *value /= scale;
        }
    }

    let adjusted = y
        .iter()
        .zip(&effects)
        .map(|(v, s)| v / s.max(0.01))
        .collect();
    (first, adjusted)
}

/// The seasonal effect of every value: an offset for an additive season, a factor for a
/// multiplicative one.
fn seasonal_effects(y: &[f64], m: usize, season: Form) -> Vec<f64> {
    if y.len() < 3 * m {
        return line_residuals(y, m, season);
    }

    let figure = decomposition_figure(y, m, season);
    (0..y.len()).map(|t| figure[t % m]).collect()
}

/// `stats::decompose(y, type)$figure`: the mean detrended value of each position in the
/// season, centred to sum to zero (additive) or to average one (multiplicative). The trend is
/// a centred moving average of order `m` (`2 x m` for even `m`).
fn decomposition_figure(y: &[f64], m: usize, season: Form) -> Vec<f64> {
    let weights = moving_average_weights(m);
    let half = weights.len() / 2;
    let mut sums = vec![0.0; m];
    let mut counts = vec![0_usize; m];
    for t in half..y.len().saturating_sub(half) {
        let trend: f64 = weights.iter().zip(&y[t - half..]).map(|(w, v)| w * v).sum();
        sums[t % m] += match season {
            Form::Additive => y[t] - trend,
            Form::Multiplicative => y[t] / trend,
        };
        counts[t % m] += 1;
    }

    let mut figure: Vec<f64> = sums
        .iter()
        .zip(&counts)
        .map(|(sum, &count)| sum / count_as_f64(count))
        .collect();
    let mean = figure.iter().sum::<f64>() / count_as_f64(m);
    for value in &mut figure {
        match season {
            Form::Additive => *value -= mean,
            Form::Multiplicative => *value /= mean,
        }
    }

    figure
}

/// The filter of `stats::decompose`: `m` equal weights for odd `m`, and for even `m` `m + 1`
/// weights with halves at both ends.
fn moving_average_weights(m: usize) -> Vec<f64> {
    let scale = 1.0 / count_as_f64(m);
    if m % 2 == 1 {
        return vec![scale; m];
    }

    let mut weights = vec![scale; m + 1];
    weights[0] = scale / 2.0;
    weights[m] = scale / 2.0;
    weights
}

/// For fewer than three seasons: `y` against the line `a + b t` of
/// `tslm(y ~ trend + fourier(y, 1))`, the seasonal part R takes when it cannot decompose the
/// series. Differences for an additive season, ratios for a multiplicative one.
fn line_residuals(y: &[f64], m: usize, season: Form) -> Vec<f64> {
    let period = count_as_f64(m);
    let columns = fourier_columns(period, 1);
    let rows: Vec<Vec<f64>> = (1..=y.len())
        .map(|t| {
            let mut row = vec![1.0, count_as_f64(t)];
            let mut terms = vec![0.0; columns];
            // `2 <= m` and `k = 1`, so the Fourier terms are valid.
            if fourier_row(period, 1, u64::try_from(t).unwrap_or(u64::MAX), &mut terms).is_ok() {
                row.extend(terms);
            }
            row
        })
        .collect();
    let coefficients = least_squares(&rows, y).unwrap_or_else(|| vec![0.0; columns + 2]);

    (1..=y.len())
        .zip(y)
        .map(|(t, v)| {
            let line = coefficients[0] + coefficients[1] * count_as_f64(t);
            match season {
                Form::Additive => v - line,
                Form::Multiplicative => v / line,
            }
        })
        .collect()
}

/// The level and, with a trend, the slope of a least-squares line through the first
/// `max(10, 2 m)` values of the seasonally adjusted series, as `initstate` (additive trend).
fn level_and_slope(adjusted: &[f64], trend: bool, m: usize) -> Vec<f64> {
    let n = adjusted.len().min(10.max(2 * m));
    let first = &adjusted[..n];
    if !trend {
        return vec![first.iter().sum::<f64>() / count_as_f64(n)];
    }

    let (intercept, slope) = line(first);
    // R nudges the pair apart when the first forecast `l + b` is almost exactly 0.
    if (intercept + slope).abs() < 1e-8 {
        return vec![intercept * 1.001, slope * 0.999];
    }

    vec![intercept, slope]
}

/// The least-squares line `a + b t` through `values` at `t = 1, ..., n`.
fn line(values: &[f64]) -> (f64, f64) {
    let n = count_as_f64(values.len());
    let t_mean = f64::midpoint(n, 1.0);
    let y_mean = values.iter().sum::<f64>() / n;
    let (sxy, sxx) = values
        .iter()
        .enumerate()
        .fold((0.0, 0.0), |(sxy, sxx), (i, v)| {
            let dt = count_as_f64(i + 1) - t_mean;
            (sxy + dt * (v - y_mean), sxx + dt * dt)
        });
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    (y_mean - slope * t_mean, slope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposition_recovers_a_pure_pattern() {
        // A level of 10 with the pattern (3, -1, -2) repeating, which already sums to zero.
        let y: Vec<f64> = (0..12).map(|t| 10.0 + [3.0, -1.0, -2.0][t % 3]).collect();
        let figure = decomposition_figure(&y, 3, Form::Additive);
        for (value, expected) in figure.iter().zip([3.0, -1.0, -2.0]) {
            assert!((value - expected).abs() < 1e-12, "{figure:?}");
        }

        // The level from the adjusted series, then the effects of positions 3 and 2.
        let states = start_states(&y, false, 3, Form::Additive);
        assert!((states[0] - 10.0).abs() < 1e-12);
        assert!((states[1] + 2.0).abs() < 1e-12 && (states[2] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn multiplicative_decomposition_recovers_factors() {
        let factors = [1.2, 0.9, 0.9];
        let y: Vec<f64> = (0..12).map(|t| 50.0 * factors[t % 3]).collect();
        let figure = decomposition_figure(&y, 3, Form::Multiplicative);
        for (value, expected) in figure.iter().zip(factors) {
            assert!((value - expected).abs() < 1e-12, "{figure:?}");
        }
    }

    #[test]
    fn line_fits_exact_data() {
        let (a, b) = line(&[3.0, 5.0, 7.0, 9.0]);
        assert!((a - 1.0).abs() < 1e-12 && (b - 2.0).abs() < 1e-12);
    }
}
