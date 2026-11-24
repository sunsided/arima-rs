//! Regressors for a piecewise-linear trend, the trend of Prophet (Taylor and Letham 2018).
//!
//! Prophet's trend `k t + m + sum_j delta_j (t - s_j)^+` changes its slope by `delta_j` at
//! each changepoint `s_j`. With the hinge columns `(t - s_j)^+` as regressors, plus `t`
//! itself, a regression with ARIMA errors ([`Arima::fit_exog`](crate::Arima::fit_exog))
//! estimates the slope changes as coefficients; Fourier terms ([`fourier_row`]) and holiday
//! dummies complete the recipe. Prophet regularises the changes with a sparse prior and
//! places 25 changepoints in the first 80% of the history; a maximum-likelihood fit has no
//! prior, so it wants fewer, chosen by the information criteria.
//!
//! [`fourier_row`]: crate::fourier_row

/// Writes the trend regressors of time `t` into `row`: `t` itself, then the hinge
/// `max(0, t - s)` of each changepoint `s` of `knots`, in order. The function needs no heap,
/// so a device can build the row of each new value as it arrives.
///
/// # Panics
///
/// Panics if `row` does not hold `knots.len() + 1` values.
///
/// # Examples
///
/// ```
/// use backshift::trend_row;
///
/// let mut row = [0.0; 3];
/// trend_row(50, &[20, 80], &mut row);
/// assert_eq!(row, [50.0, 30.0, 0.0]);
/// ```
#[expect(
    clippy::cast_precision_loss,
    reason = "sample indices stay far below 2^53, where the conversion is exact"
)]
pub fn trend_row(t: u64, knots: &[u64], row: &mut [f64]) {
    assert_eq!(
        row.len(),
        knots.len() + 1,
        "row must hold one value for t and one per knot"
    );
    let (first, hinges) = row.split_at_mut(1);
    first[0] = t as f64;
    for (value, &knot) in hinges.iter_mut().zip(knots) {
        *value = t.saturating_sub(knot) as f64;
    }
}

/// The trend regressors of [`trend_row`] for `rows` consecutive times from `t0`, row-major
/// with `knots.len() + 1` columns. Pass them to
/// [`Arima::fit_exog`](crate::Arima::fit_exog) with `t0 = 0`, and the rows from `t0 = n` on
/// to the forecasts: the trend continues at its last slope.
///
/// # Panics
///
/// Panics if `rows` times the column count overflows `usize`, as an allocation that size
/// would.
///
/// # Examples
///
/// ```
/// use backshift::trend_regressors;
///
/// let x = trend_regressors(&[2], 0, 4);
/// assert_eq!(x, [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 1.0]);
/// ```
#[cfg(feature = "alloc")]
#[must_use]
pub fn trend_regressors(knots: &[u64], t0: u64, rows: usize) -> alloc::vec::Vec<f64> {
    let columns = knots.len() + 1;
    let len = rows
        .checked_mul(columns)
        .expect("the regressor matrix fits in memory");
    let mut out = alloc::vec![0.0; len];
    for (row, t) in out.chunks_exact_mut(columns).zip(t0..) {
        trend_row(t, knots, row);
    }

    out
}
