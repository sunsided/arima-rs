//! Lag polynomials of seasonal ARIMA models, expanded into caller-owned buffers.
//!
//! A seasonal ARIMA(p, d, q)(P, D, Q, s) model multiplies a non-seasonal and a seasonal
//! polynomial for each part. These helpers expand the products into plain coefficient
//! vectors, so the Kalman recursion only ever sees an ARMA model and a differencing
//! polynomial. None of them allocate.

/// Writes the coefficients of the product of a non-seasonal and a seasonal lag polynomial.
///
/// With `sign = -1.0` (AR convention) the product is
/// `(1 - sum a_i B^i)(1 - sum b_j B^(j s)) = 1 - sum c_k B^k`, so `c = a + b_s - a b_s`.
/// With `sign = 1.0` (MA convention) it is `(1 + sum a_i B^i)(1 + sum b_j B^(j s)) = 1 + sum c_k B^k`,
/// so `c = a + b_s + a b_s`. Here `b_s` places `b_j` at lag `j s`.
///
/// `out` receives `c_1, ..., c_n` with `n = a.len() + b.len() * s`, and is zeroed first.
/// Returns `n`. Requires `out.len() >= n`.
pub(crate) fn seasonal_product(
    a: &[f64],
    b: &[f64],
    period: usize,
    sign: f64,
    out: &mut [f64],
) -> usize {
    let n = a.len() + b.len() * period;
    out[..n].fill(0.0);
    out[..a.len()].copy_from_slice(a);
    for (j, &bj) in b.iter().enumerate() {
        let lag = (j + 1) * period;
        out[lag - 1] += bj;
        for (i, &ai) in a.iter().enumerate() {
            out[lag + i] += sign * ai * bj;
        }
    }

    n
}

/// Writes the coefficients of the differencing polynomial `(1 - B)^d (1 - B^s)^D`.
///
/// `out` receives `delta_0 = 1, delta_1, ..., delta_k` with `k = d + D s`, so the
/// differenced series is `w_t = sum_j delta_j y_(t-j)`. Returns `k`. Requires
/// `out.len() >= k + 1`.
pub(crate) fn differencing(d: usize, seasonal_d: usize, period: usize, out: &mut [f64]) -> usize {
    let k = d + seasonal_d * period;
    out[..=k].fill(0.0);
    out[0] = 1.0;

    let mut degree = 0;
    for lag in core::iter::repeat_n(1, d).chain(core::iter::repeat_n(period, seasonal_d)) {
        // Multiply by (1 - B^lag) in place, from the top down.
        degree += lag;
        for i in (lag..=degree).rev() {
            out[i] -= out[i - lag];
        }
    }

    k
}

/// `sum_j delta_j y_(t-j)` for `delta = [1, delta_1, ..., delta_k]`, where `current = y_t`
/// and `lags` yields `y_(t-1), y_(t-2), ...`.
///
/// The filter and the likelihood both difference through this one function, so the
/// two paths add the same terms in the same order and agree bit for bit.
pub(crate) fn apply(delta: &[f64], current: f64, lags: impl IntoIterator<Item = f64>) -> f64 {
    delta[1..]
        .iter()
        .zip(lags)
        .fold(current, |sum, (d, y)| sum + d * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar_product_matches_hand_expansion() {
        // (1 - 0.5 B)(1 - 0.3 B^4) = 1 - 0.5 B - 0.3 B^4 + 0.15 B^5.
        let mut out = [9.0; 6];
        let n = seasonal_product(&[0.5], &[0.3], 4, -1.0, &mut out);
        assert_eq!(n, 5);
        assert_eq!(&out[..5], &[0.5, 0.0, 0.0, 0.3, -0.15]);
    }

    #[test]
    fn ma_product_matches_hand_expansion() {
        // (1 + 0.4 B)(1 + 0.6 B^2 + 0.2 B^4) = 1 + 0.4 B + 0.6 B^2 + 0.24 B^3 + 0.2 B^4 + 0.08 B^5.
        let mut out = [0.0; 5];
        let n = seasonal_product(&[0.4], &[0.6, 0.2], 2, 1.0, &mut out);
        assert_eq!(n, 5);
        let expected = [0.4, 0.6, 0.24, 0.2, 0.08];
        for (a, e) in out.iter().zip(expected) {
            assert!((a - e).abs() < 1e-15, "{out:?}");
        }
    }

    #[test]
    fn differencing_polynomials() {
        let mut out = [0.0; 8];
        assert_eq!(differencing(0, 0, 4, &mut out), 0);
        assert_eq!(out[0], 1.0);

        assert_eq!(differencing(2, 0, 4, &mut out), 2);
        assert_eq!(&out[..3], &[1.0, -2.0, 1.0]);

        // (1 - B)(1 - B^4) = 1 - B - B^4 + B^5.
        assert_eq!(differencing(1, 1, 4, &mut out), 5);
        assert_eq!(&out[..6], &[1.0, -1.0, 0.0, 0.0, -1.0, 1.0]);
    }

    #[test]
    fn apply_differences_a_window() {
        let delta = [1.0, -1.0, 0.0, 0.0, -1.0, 1.0];
        // y_t = 10, lags y_(t-1..t-5) = 7, 3, 2, 6, 4: 10 - 7 - 6 + 4 = 1.
        assert_eq!(apply(&delta, 10.0, [7.0, 3.0, 2.0, 6.0, 4.0]), 1.0);
    }
}
