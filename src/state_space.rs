//! The Kalman recursion for an ARMA model in state-space form, without allocation.
//!
//! An ARMA(p, q) model with `r = max(p, q + 1)` states is written in Harvey's form,
//! the same one statsmodels SARIMAX uses:
//!
//! - state transition `T`: the AR coefficients in the first column, ones on the superdiagonal;
//! - selection `R = [1, theta_1, ..., theta_(r-1)]`, so the state noise covariance is `sigma2 R R'`;
//! - observation `y_t = a_t[0]`, with no measurement noise.
//!
//! Every function here works on caller-owned slices. A covariance matrix is stored
//! row-major with a row `stride >= r`, so the same code runs on a runtime-sized
//! `Vec` (stride `r`) and on a fixed `[[f64; R]; R]` (stride `R`). Only the top-left
//! `r x r` block is read or written.
//!
//! ```mermaid
//! flowchart LR
//!     A["predicted a, P"] -->|"update(y)"| B["filtered a, P"]
//!     B -->|predict| A
//!     A -->|"read a[0], P[0][0]"| C["innovation / forecast"]
//! ```

/// The system matrices of an ARMA model, padded to the state dimension `r`.
#[derive(Clone, Copy)]
pub(crate) struct System<'a> {
    /// The first column of `T`: `phi_1, ..., phi_p`, then zeros. Length `r`.
    pub(crate) phi: &'a [f64],
    /// The selection vector `[1, theta_1, ..., theta_q]`, then zeros. Length `r`.
    pub(crate) selection: &'a [f64],
    /// The innovation variance. Positive and finite.
    pub(crate) sigma2: f64,
}

/// The one-step-ahead prediction error for one observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Innovation {
    /// The observation minus its prediction, `y_t - E[y_t | y_1, ..., y_(t-1)]`.
    ///
    /// `NaN` when the observation was missing (`NaN`), as in statsmodels.
    pub value: f64,
    /// The variance of the prediction error. Always positive.
    pub variance: f64,
}

impl System<'_> {
    /// The state dimension `r`.
    fn dim(&self) -> usize {
        self.phi.len()
    }

    /// The entry `(i, j)` of the state noise covariance `sigma2 R R'`.
    fn noise(&self, i: usize, j: usize) -> f64 {
        self.sigma2 * self.selection[i] * self.selection[j]
    }
}

/// Moves the state forward one step: `a = T a`, `P = T P T' + sigma2 R R'`.
///
/// `row0` is scratch of length `>= r`. Only the upper triangle of `P` is computed,
/// then mirrored, so `P` stays exactly symmetric.
pub(crate) fn predict(
    sys: System<'_>,
    a: &mut [f64],
    p: &mut [f64],
    stride: usize,
    row0: &mut [f64],
) {
    let r = sys.dim();
    transition_vector(sys, a);

    // Row 0 of the old P; by symmetry it is also column 0.
    row0[..r].copy_from_slice(&p[..r]);
    let p00 = row0[0];
    let last = r - 1;

    // (T P T')_ij with T = [phi | shift]
    //   = phi_i phi_j P00 + phi_i P0(j+1) + phi_j P0(i+1) + P(i+1)(j+1) + noise_ij,
    // where every index past the last state contributes zero. Every term reads row 0
    // or row i + 1 >= i, which this loop has not overwritten yet. The last row and column
    // are split off so the inner loop has no branches.
    for i in 0..last {
        let (phi_i, next_i) = (sys.phi[i], row0[i + 1]);
        let (sel_i, row, below) = (sys.sigma2 * sys.selection[i], i * stride, (i + 1) * stride);
        for j in i..last {
            let phi_j = sys.phi[j];
            p[row + j] = phi_i * phi_j * p00
                + phi_i * row0[j + 1]
                + phi_j * next_i
                + p[below + j + 1]
                + sel_i * sys.selection[j];
        }

        let phi_last = sys.phi[last];
        p[row + last] = phi_i * phi_last * p00 + phi_last * next_i + sel_i * sys.selection[last];
    }

    let phi_last = sys.phi[last];
    p[last * stride + last] = phi_last * phi_last * p00 + sys.noise(last, last);

    mirror_upper(p, r, stride);
}

/// Multiplies the state vector by the transition matrix in place: `a = T a`.
pub(crate) fn transition_vector(sys: System<'_>, a: &mut [f64]) {
    let r = sys.dim();
    let a0 = a[0];
    for i in 0..r {
        let next = if i + 1 < r { a[i + 1] } else { 0.0 };
        a[i] = sys.phi[i] * a0 + next;
    }
}

/// Conditions the state on the observation `y` and returns its innovation.
///
/// `gain` is scratch of length `>= r`. A `NaN` observation counts as missing: the
/// state is left unchanged and the innovation value is `NaN`.
pub(crate) fn update(
    r: usize,
    a: &mut [f64],
    p: &mut [f64],
    stride: usize,
    y: f64,
    gain: &mut [f64],
) -> Innovation {
    let variance = p[0];
    let value = y - a[0];
    if y.is_nan() {
        return Innovation {
            value: f64::NAN,
            variance,
        };
    }

    // P Z' with Z = [1, 0, ...] is column 0, which equals row 0.
    gain[..r].copy_from_slice(&p[..r]);
    for i in 0..r {
        a[i] += gain[i] * value / variance;
        // The update is symmetric, so compute the upper triangle and mirror it.
        for j in i..r {
            p[i * stride + j] -= gain[i] * gain[j] / variance;
        }
    }

    mirror_upper(p, r, stride);

    Innovation { value, variance }
}

/// Scratch matrices for [`stationary_covariance`], each `r x r` with the same stride as `P`.
pub(crate) struct LyapunovScratch<'a> {
    /// Holds `T^(2^k)`.
    pub(crate) power: &'a mut [f64],
    /// Holds intermediate products.
    pub(crate) product: &'a mut [f64],
    /// Holds intermediate products.
    pub(crate) term: &'a mut [f64],
}

/// The most doubling steps [`stationary_covariance`] takes: enough for `T^(2^64)`.
const MAX_DOUBLINGS: usize = 64;

/// Writes the unconditional state covariance into `p`: the solution of `P = T P T' + sigma2 R R'`.
///
/// Uses the doubling algorithm: `P_(k+1) = P_k + A_k P_k A_k'` and `A_(k+1) = A_k^2`,
/// starting from `P_0 = sigma2 R R'` and `A_0 = T`. Returns `false` if the model is not
/// stationary, so the iteration does not converge or the result is not a valid
/// covariance (`P[0][0]` not positive and finite).
pub(crate) fn stationary_covariance(
    sys: System<'_>,
    p: &mut [f64],
    stride: usize,
    scratch: LyapunovScratch<'_>,
) -> bool {
    let r = sys.dim();
    let LyapunovScratch {
        power,
        product,
        term,
    } = scratch;

    for i in 0..r {
        for j in 0..r {
            p[i * stride + j] = sys.noise(i, j);
            power[i * stride + j] = transition(sys, i, j);
        }
    }

    for _ in 0..MAX_DOUBLINGS {
        // term = A P A', symmetric because P is.
        multiply(product, power, p, r, stride);
        multiply_transposed_symmetric(term, product, power, r, stride);

        let mut change: f64 = 0.0;
        let mut size: f64 = 0.0;
        for i in 0..r {
            for j in 0..r {
                let k = i * stride + j;
                p[k] += term[k];
                change = change.max(term[k].abs());
                size = size.max(p[k].abs());
            }
        }

        if !size.is_finite() {
            return false;
        }

        if change <= f64::EPSILON * size {
            mirror_upper(p, r, stride);
            return p[0].is_finite() && p[0] > 0.0;
        }

        // A = A A
        multiply(product, power, power, r, stride);
        for i in 0..r {
            power[i * stride..i * stride + r].copy_from_slice(&product[i * stride..i * stride + r]);
        }
    }

    false
}

/// Entry `(i, j)` of the transition matrix `T`.
fn transition(sys: System<'_>, i: usize, j: usize) -> f64 {
    if j == 0 {
        sys.phi[i]
    } else if j == i + 1 {
        1.0
    } else {
        0.0
    }
}

/// Writes `a b` into `out`. All are `r x r` with row `stride`.
///
/// Runs in `i, k, j` order so the inner loop walks rows of `b` contiguously, and skips
/// zero entries of `a`: the early powers of `T` are mostly zeros.
fn multiply(out: &mut [f64], a: &[f64], b: &[f64], r: usize, stride: usize) {
    for i in 0..r {
        let row = &mut out[i * stride..i * stride + r];
        row.fill(0.0);
        for k in 0..r {
            let a_ik = a[i * stride + k];
            if a_ik == 0.0 {
                continue;
            }

            for (o, b_kj) in row.iter_mut().zip(&b[k * stride..k * stride + r]) {
                *o += a_ik * b_kj;
            }
        }
    }
}

/// Writes `a b'` into `out` when the result is known to be symmetric: only the upper
/// triangle is computed, as row-by-row dot products, then mirrored.
fn multiply_transposed_symmetric(out: &mut [f64], a: &[f64], b: &[f64], r: usize, stride: usize) {
    for i in 0..r {
        let a_row = &a[i * stride..i * stride + r];
        for j in i..r {
            let b_row = &b[j * stride..j * stride + r];
            out[i * stride + j] = a_row.iter().zip(b_row).map(|(x, y)| x * y).sum();
        }
    }

    mirror_upper(out, r, stride);
}

/// Copies the upper triangle of the `r x r` matrix `p` onto its lower triangle.
fn mirror_upper(p: &mut [f64], r: usize, stride: usize) {
    for i in 0..r {
        for j in i + 1..r {
            p[j * stride + i] = p[i * stride + j];
        }
    }
}
