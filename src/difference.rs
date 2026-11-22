//! Differencing, the "I" in ARIMA, and its inverse for forecasts.
//!
//! An ARIMA(p, d, q) model is an ARMA(p, q) model of the series differenced
//! `d` times. [`difference`] produces that series. [`integrate`] maps values
//! on the differenced scale, such as forecasts, back to the original scale.

use alloc::vec::Vec;

/// Differences `x` `d` times: each pass replaces the series with `x[t + 1] - x[t]`.
///
/// The result has `x.len() - d` values, or none when `d >= x.len()`. `d = 0`
/// returns a copy. The passes run in the same order as `numpy.diff(x, n=d)`,
/// so the results match it bit for bit.
///
/// # Examples
///
/// ```
/// use backshift::difference;
///
/// assert_eq!(difference(&[1.0, 4.0, 9.0, 16.0], 1), [3.0, 5.0, 7.0]);
/// assert_eq!(difference(&[1.0, 4.0, 9.0, 16.0], 2), [2.0, 2.0]);
/// ```
#[must_use]
pub fn difference(x: &[f64], d: usize) -> Vec<f64> {
    let mut values = x.to_vec();
    for _ in 0..d {
        difference_once(&mut values);
    }

    values
}

/// Undoes [`difference`]: turns values on the `d`-times differenced scale back into
/// values that continue `history` on the original scale.
///
/// `history` holds the original series up to the point where `differenced` starts;
/// only its last `d` values are used. For example, with `d = 1` the result is
/// `history.last() + cumsum(differenced)`.
///
/// Returns [`None`] if `history` has fewer than `d` values.
///
/// # Examples
///
/// ```
/// use backshift::{difference, integrate};
///
/// let y = [1.0, 4.0, 9.0, 16.0, 25.0];
/// let tail = &difference(&y, 2)[1..]; // the differences that produce y[3..]
/// assert_eq!(integrate(&y[..3], tail, 2), Some(vec![16.0, 25.0]));
/// ```
#[must_use]
pub fn integrate(history: &[f64], differenced: &[f64], d: usize) -> Option<Vec<f64>> {
    let history = history.get(history.len().checked_sub(d)?..)?;
    let anchors = anchors(history);

    let mut values = differenced.to_vec();
    for anchor in anchors.into_iter().rev() {
        cumulative_sum_from(&mut values, anchor);
    }

    Some(values)
}

/// Replaces `values` with its first differences, shortening it by one (or leaving it empty).
fn difference_once(values: &mut Vec<f64>) {
    for i in 1..values.len() {
        values[i - 1] = values[i] - values[i - 1];
    }

    values.pop();
}

/// The last value of `history` differenced `k` times, for `k = 0..history.len()`.
///
/// These are the starting points that [`integrate`] adds back, one per level.
fn anchors(history: &[f64]) -> Vec<f64> {
    let mut level = history.to_vec();
    let mut anchors = Vec::with_capacity(history.len());
    while let Some(&last) = level.last() {
        anchors.push(last);
        difference_once(&mut level);
    }

    anchors
}

/// Replaces `values` with the running sum `start + values[0] + ... + values[i]`.
fn cumulative_sum_from(values: &mut [f64], start: f64) {
    let mut sum = start;
    for value in values {
        sum += *value;
        *value = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[test]
    fn difference_shortens_by_d() {
        let x = [1.0, 4.0, 9.0, 16.0, 25.0];
        assert_eq!(difference(&x, 0), x);
        assert_eq!(difference(&x, 1), [3.0, 5.0, 7.0, 9.0]);
        assert_eq!(difference(&x, 2), [2.0, 2.0, 2.0]);
        assert!(difference(&x, 5).is_empty());
        assert!(difference(&x, 9).is_empty());
        assert!(difference(&[], 1).is_empty());
    }

    #[test]
    fn integrate_continues_history() {
        let y = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        assert_eq!(integrate(&y[..4], &[9.0, 11.0], 1), Some(vec![25.0, 36.0]));
        assert_eq!(integrate(&y[..4], &[2.0, 2.0], 2), Some(vec![25.0, 36.0]));
        assert_eq!(integrate(&y[..4], &[7.0], 0), Some(vec![7.0]));
    }

    #[test]
    fn integrate_needs_d_values_of_history() {
        assert_eq!(integrate(&[1.0], &[1.0], 2), None);
        assert_eq!(integrate(&[], &[1.0], 1), None);
        assert_eq!(integrate(&[1.0, 2.0], &[], 2), Some(vec![]));
    }
}
