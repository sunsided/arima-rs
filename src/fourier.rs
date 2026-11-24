//! Fourier regressors: sine and cosine pairs that model a seasonal pattern as a regression.
//!
//! A seasonal pattern of period `m` is approximated by `K` harmonics,
//! `sum_(i=1..K) a_i sin(2 pi i t / m) + b_i cos(2 pi i t / m)`. Fed into
//! [`Arima::fit_exog`](crate::Arima::fit_exog) and [`Filter::update_exog`](crate::Filter::update_exog),
//! this is a regression with ARIMA errors, or dynamic harmonic regression (Hyndman &
//! Athanasopoulos, FPP3 §10.5 and §12.1). Unlike seasonal ARIMA, it allows long and
//! non-integer periods (365.25, 8766) and several periods at once, and the model's state
//! does not grow with the period.
//!
//! The columns follow R's `forecast::fourier()`: `sin(1), cos(1), sin(2), cos(2), ...`, and
//! the sine at the Nyquist frequency (`2K = m`), which is zero at every integer `t`, is left
//! out. statsmodels' `deterministic.Fourier` has the same order but keeps that column.
//!
//! Each term is computed from `t mod m`, not by a rotation recurrence, so the values do not
//! drift over millions of samples on a device.

use crate::math;
use core::error::Error;
use core::f64::consts::TAU;
use core::fmt::{Display, Formatter};

/// Why Fourier regressors could not be generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourierError {
    /// The number of harmonics `K` is 0, the period is not finite, or `2K > period`.
    InvalidTerms,
    /// The output row does not have one entry per column.
    RowLength {
        /// The number of columns, [`fourier_columns`].
        needed: usize,
        /// The length of the given row.
        got: usize,
    },
}

impl Display for FourierError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            FourierError::InvalidTerms => {
                write!(f, "need 1 <= K and 2K <= period, with a finite period")
            }
            FourierError::RowLength { needed, got } => {
                write!(f, "a row needs {needed} entries, got {got}")
            }
        }
    }
}

impl Error for FourierError {}

/// The number of regressor columns of `k` harmonics at `period`: `2k`, or `2k - 1` when
/// `2k == period`, because the Nyquist sine is left out.
///
/// The period is compared exactly, so only a period that is exactly `2k` drops the column.
#[must_use]
pub fn fourier_columns(period: f64, k: usize) -> usize {
    2 * k - usize::from(is_nyquist(period, k))
}

/// Writes the Fourier regressors of `k` harmonics at `period` for time `t` into `row`.
///
/// `row` gets `sin(2 pi i t / period), cos(2 pi i t / period)` for `i = 1, ..., k`, without
/// the Nyquist sine (see [`fourier_columns`]). `t` is the sample index; R's `fourier()` starts
/// at `t = 1`, statsmodels' `Fourier` at `t = 0`. The function needs no heap, so a device can
/// build the regressor row of each new value as it arrives.
///
/// # Errors
///
/// - [`FourierError::InvalidTerms`] unless `k >= 1`, `period` is finite and `2k <= period`.
/// - [`FourierError::RowLength`] if `row` does not hold [`fourier_columns`] values.
///
/// # Examples
///
/// ```
/// use backshift::fourier_row;
///
/// // A weekly pattern on daily data with two harmonics.
/// let mut row = [0.0; 4];
/// fourier_row(7.0, 2, 3, &mut row).unwrap();
/// let angle = core::f64::consts::TAU * 3.0 / 7.0;
/// assert!((row[0] - angle.sin()).abs() < 1e-15);
/// assert!((row[3] - (2.0 * angle).cos()).abs() < 1e-15);
/// ```
pub fn fourier_row(period: f64, k: usize, t: u64, row: &mut [f64]) -> Result<(), FourierError> {
    let needed = validate(period, k)?;
    if row.len() != needed {
        return Err(FourierError::RowLength {
            needed,
            got: row.len(),
        });
    }

    // `t` is exact as an `f64` below 2^53 samples, and `%` on floats is exact.
    let phase = t as f64 % period;
    let mut slots = row.iter_mut();
    for i in 1..=k {
        // Reduce `i t` modulo the period too, so the angle stays below 2 pi.
        let angle = TAU * ((i as f64 * phase) % period) / period;
        if !(i == k && is_nyquist(period, k)) {
            *slots.next().expect("one slot per column: checked above") = math::sin(angle);
        }

        *slots.next().expect("one slot per column: checked above") = math::cos(angle);
    }

    Ok(())
}

/// The column count of `k` harmonics at `period`, or [`FourierError::InvalidTerms`]
/// unless `k >= 1`, `period` is finite and `2k <= period`.
fn validate(period: f64, k: usize) -> Result<usize, FourierError> {
    if k == 0 || !period.is_finite() || 2.0 * k as f64 > period {
        return Err(FourierError::InvalidTerms);
    }

    Ok(fourier_columns(period, k))
}

/// Whether the `k`-th harmonic sits at the Nyquist frequency of `period`.
fn is_nyquist(period: f64, k: usize) -> bool {
    2.0 * k as f64 == period
}

/// Fourier regressors for several seasonal periods, `rows` rows from time `t0`, row-major.
///
/// Each `(period, k)` of `terms` adds the [`fourier_columns`] of [`fourier_row`], in order,
/// so daily, weekly and yearly patterns on hourly data are
/// `&[(24.0, 10), (168.0, 5), (8766.0, 10)]`. Pass the result to
/// [`Arima::fit_exog`](crate::Arima::fit_exog) with `columns` the sum of the column counts,
/// and the rows from `t0 = n` on to the forecasts.
///
/// # Errors
///
/// [`FourierError::InvalidTerms`] if `terms` is empty or a pair of it is invalid (see
/// [`fourier_row`]).
///
/// # Panics
///
/// Panics if `rows` times the column count overflows `usize`, as an allocation that size would.
///
/// # Examples
///
/// ```
/// use backshift::fourier;
///
/// let x = fourier(&[(7.0, 3), (4.0, 2)], 0, 10).unwrap();
/// assert_eq!(x.len(), 10 * (6 + 3)); // period 4 drops its Nyquist sine
/// ```
#[cfg(feature = "alloc")]
pub fn fourier(
    terms: &[(f64, usize)],
    t0: u64,
    rows: usize,
) -> Result<alloc::vec::Vec<f64>, FourierError> {
    if terms.is_empty() {
        return Err(FourierError::InvalidTerms);
    }

    let widths = terms
        .iter()
        .map(|&(period, k)| validate(period, k))
        .collect::<Result<alloc::vec::Vec<_>, _>>()?;

    let columns: usize = widths.iter().sum();
    let len = rows
        .checked_mul(columns)
        .expect("the regressor matrix fits in memory");
    let mut out = alloc::vec![0.0; len];
    for (row, t) in out.chunks_exact_mut(columns).zip(t0..) {
        let mut rest = row;
        for (&(period, k), &width) in terms.iter().zip(&widths) {
            let (part, tail) = rest.split_at_mut(width);
            fourier_row(period, k, t, part)?;
            rest = tail;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_the_nyquist_sine() {
        assert_eq!(fourier_columns(4.0, 2), 3);
        assert_eq!(fourier_columns(4.5, 2), 4);
        let mut row = [0.0; 3];
        fourier_row(4.0, 2, 1, &mut row).unwrap();
        // sin(pi/2), cos(pi/2), cos(pi).
        assert!((row[0] - 1.0).abs() < 1e-15);
        assert!(row[1].abs() < 1e-15);
        assert!((row[2] + 1.0).abs() < 1e-15);
    }

    #[test]
    fn rejects_bad_terms_and_rows() {
        let mut row = [0.0; 4];
        assert_eq!(
            fourier_row(3.0, 2, 0, &mut row),
            Err(FourierError::InvalidTerms)
        );
        assert_eq!(
            fourier_row(7.0, 0, 0, &mut row),
            Err(FourierError::InvalidTerms)
        );
        assert_eq!(
            fourier_row(f64::NAN, 1, 0, &mut row),
            Err(FourierError::InvalidTerms)
        );
        assert_eq!(
            fourier_row(7.0, 1, 0, &mut row),
            Err(FourierError::RowLength { needed: 2, got: 4 })
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn rejects_empty_terms() {
        assert_eq!(fourier(&[], 0, 5), Err(FourierError::InvalidTerms));
    }

    #[test]
    fn does_not_drift_over_many_periods() {
        // 10^12 samples later, at a non-integer period, the row repeats exactly.
        let (mut early, mut late) = ([0.0; 6], [0.0; 6]);
        fourier_row(365.25, 3, 17, &mut early).unwrap();
        // 4 * 365.25 = 1461 samples is a whole number of cycles.
        fourier_row(365.25, 3, 17 + 1461 * 684_462_696, &mut late).unwrap();
        for (a, b) in early.iter().zip(&late) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}
