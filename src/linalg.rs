//! Small dense linear algebra on caller-owned, row-major slices, without allocation.
//!
//! The VAR models need it for covariance matrices of a handful of series: Cholesky
//! factorisation, triangular solves and the log-determinant. Every function works on an
//! `n x n` block with row stride `stride >= n`, so the same code runs on a `Vec` and on a
//! fixed `[[f64; K]; K]`. Each function checks the dimensions once, then works on whole row
//! slices with iterators, so the inner loops carry no bounds checks.

use crate::math;

/// The dot product of two equally long slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Checks that `a` holds an `n x n` block with row stride `stride`.
///
/// # Panics
///
/// Panics if `stride < n` or `a` is shorter than the block.
fn check(a: &[f64], n: usize, stride: usize) {
    assert!(
        stride >= n && (n == 0 || a.len() >= (n - 1) * stride + n),
        "the block must fit in the slice"
    );
}

/// Factorises the symmetric positive definite block `a` in place into its lower Cholesky
/// factor `L` (`a = L L'`); the upper triangle is set to zero.
///
/// Returns `false`, leaving `a` partly overwritten, if `a` is not positive definite (a pivot
/// is not positive and finite).
///
/// # Panics
///
/// Panics if `a` does not hold an `n x n` block with row stride `stride`.
pub(crate) fn cholesky(a: &mut [f64], n: usize, stride: usize) -> bool {
    check(a, n, stride);
    for j in 0..n {
        // Row `j` and the rows below it, borrowed apart.
        let split = ((j + 1) * stride).min(a.len());
        let (upper, lower) = a.split_at_mut(split);
        let row_j = &mut upper[j * stride..j * stride + n];
        let diagonal = row_j[j] - dot(&row_j[..j], &row_j[..j]);
        if !(diagonal > 0.0 && diagonal.is_finite()) {
            return false;
        }

        let pivot = math::sqrt(diagonal);
        row_j[j] = pivot;
        row_j[j + 1..].fill(0.0);
        let left = &row_j[..j];
        for row_i in lower.chunks_mut(stride).take(n - j - 1) {
            let row_i = &mut row_i[..n];
            row_i[j] = (row_i[j] - dot(&row_i[..j], left)) / pivot;
        }
    }

    true
}

/// Solves `L x = b` in place for the lower triangular `l` (from [`cholesky`]).
///
/// # Panics
///
/// Panics if `l` does not hold an `n x n` block with row stride `stride`, or `b` holds fewer
/// than `n` values.
pub(crate) fn solve_lower(l: &[f64], n: usize, stride: usize, b: &mut [f64]) {
    check(l, n, stride);
    let b = &mut b[..n];
    for (i, row) in l.chunks(stride).take(n).enumerate() {
        let (solved, rest) = b.split_at_mut(i);
        rest[0] = (rest[0] - dot(&row[..i], solved)) / row[i];
    }
}

/// Solves `L' x = b` in place for the lower triangular `l` (from [`cholesky`]).
///
/// # Panics
///
/// Panics if `l` does not hold an `n x n` block with row stride `stride`, or `b` holds fewer
/// than `n` values.
#[cfg(any(feature = "alloc", test))]
pub(crate) fn solve_upper_transposed(l: &[f64], n: usize, stride: usize, b: &mut [f64]) {
    check(l, n, stride);
    let b = &mut b[..n];
    for i in (0..n).rev() {
        // Column `i` of `L` below the diagonal, against the values already solved.
        let below: f64 = (i + 1..n).map(|j| l[j * stride + i] * b[j]).sum();
        b[i] = (b[i] - below) / l[i * stride + i];
    }
}

/// `ln det(a) = 2 sum ln L_ii` from the Cholesky factor `l` of `a`.
///
/// # Panics
///
/// Panics if `l` does not hold an `n x n` block with row stride `stride`.
#[cfg(any(feature = "alloc", test))]
pub(crate) fn log_det(l: &[f64], n: usize, stride: usize) -> f64 {
    check(l, n, stride);
    2.0 * l
        .chunks(stride)
        .take(n)
        .enumerate()
        .map(|(i, row)| math::ln(row[i]))
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_system() {
        // a = [[4, 2], [2, 3]]: L = [[2, 0], [1, sqrt 2]], det 8.
        let mut a = [4.0, 2.0, 2.0, 3.0];
        assert!(cholesky(&mut a, 2, 2));
        assert!((a[3] - 2.0_f64.sqrt()).abs() < 1e-15 && a[1] == 0.0);
        assert!((log_det(&a, 2, 2) - 8.0_f64.ln()).abs() < 1e-14);

        // a x = (6, 5) has x = (1, 1).
        let mut b = [6.0, 5.0];
        solve_lower(&a, 2, 2, &mut b);
        solve_upper_transposed(&a, 2, 2, &mut b);
        assert!((b[0] - 1.0).abs() < 1e-14 && (b[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn works_with_a_wider_stride() {
        // The same matrix inside a 3-wide row layout.
        let mut a = [4.0, 2.0, 9.0, 2.0, 3.0, 9.0];
        assert!(cholesky(&mut a, 2, 3));
        assert!((a[4] - 2.0_f64.sqrt()).abs() < 1e-15 && a[2] == 9.0);
    }

    #[test]
    fn rejects_indefinite_matrices() {
        let mut a = [1.0, 2.0, 2.0, 1.0];
        assert!(!cholesky(&mut a, 2, 2));
    }
}
