//! The classical decomposition of `stats::decompose`, which R's `ets` starts from and R's
//! `thetaf` deseasonalises with.

use crate::ets::Form;
use alloc::vec;
use alloc::vec::Vec;

/// `stats::decompose(y, type)$figure`: the mean detrended value of each position in the
/// season, centred to sum to zero (additive) or to average one (multiplicative). The trend is
/// a centred moving average of order `m` (`2 x m` for even `m`).
pub(crate) fn classical_figure(y: &[f64], m: usize, season: Form) -> Vec<f64> {
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

/// A count as a float. Counts here are series lengths and periods, far below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "series lengths and periods stay far below 2^53, where the conversion is exact"
)]
fn count_as_f64(count: usize) -> f64 {
    count as f64
}
