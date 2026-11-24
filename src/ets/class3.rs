//! Forecast variances of the ETS models with multiplicative errors and season (R's class 3:
//! ETS(M,N,M), ETS(M,A,M), ETS(M,Ad,M)), as `forecast:::class3` computes them.
//!
//! The variance has no closed form here. R propagates the mean `M_h` of the product of the
//! level-and-trend states with the seasonal states, a `t x m` matrix for `t` level and trend
//! states, and the covariance `V_h` of its vectorised form, a `q x q` matrix with `q = t m`,
//! through Kronecker products of the transition and gain matrices. This module does the
//! same on caller-owned scratch, so it runs without a heap: [`scratch_len`] values.
//!
//! R's blocks ignore the damping in the level: it uses `F_1 = [[1, 1], [0, phi]]`,
//! `G_1 = [[alpha, alpha], [beta, beta]]` and the observation weights `(1, 1)`. The damped
//! model updates the level from `l + phi b`, so this module uses `F_1 = [[1, phi], [0, phi]]`,
//! `G_1 = [[alpha, alpha phi], [beta, beta phi]]` and the weights `(1, phi)`. Without damping
//! (`phi = 1`) the two agree; for ETS(M,Ad,M) R's means and variances are off.

use super::Smoothing;

/// The scratch [`moments`] needs for period `m`: seven `q x q` matrices and three
/// `q`-vectors, `q = t m` with `t = 2` for a model with a trend and 1 without.
#[must_use]
pub(crate) const fn scratch_len(trend: bool, m: usize) -> usize {
    let q = states(trend) * m;
    7 * q * q + 3 * q
}

/// The number of level and trend states: 2 with a trend, 1 without.
const fn states(trend: bool) -> usize {
    if trend { 2 } else { 1 }
}

/// Writes R's class-3 forecast means and variances for horizons `1..=mean.len()`.
///
/// The means are the expectations of the future values, `H_1 M_h H_2'`: with a
/// multiplicative season the expected product of level and season exceeds the product of
/// their forecasts by a term in `sigma2`, which R includes. `seasonal` holds the `m` seasonal
/// states, most recent first. `scratch` holds at least [`scratch_len`] values.
///
/// # Panics
///
/// Panics if `scratch` is shorter than [`scratch_len`], `seasonal` is empty, or `mean` and
/// `variance` have different lengths.
#[expect(
    clippy::too_many_arguments,
    reason = "the states, the scratch and both outputs are separate borrows"
)]
pub(crate) fn moments(
    smoothing: &Smoothing,
    level: f64,
    slope: f64,
    seasonal: &[f64],
    sigma2: f64,
    scratch: &mut [f64],
    mean: &mut [f64],
    variance: &mut [f64],
) {
    let m = seasonal.len();
    let t = states(smoothing.trend);
    let q = t * m;
    let (matrices, rest) = scratch[..scratch_len(smoothing.trend, m)].split_at_mut(7 * q * q);
    let (f21, rest_matrices) = matrices.split_at_mut(q * q);
    let (g21, rest_matrices) = rest_matrices.split_at_mut(q * q);
    let (gains, rest_matrices) = rest_matrices.split_at_mut(q * q);
    let (vh, rest_matrices) = rest_matrices.split_at_mut(q * q);
    let (outer, rest_matrices) = rest_matrices.split_at_mut(q * q);
    let (product, accumulator) = rest_matrices.split_at_mut(q * q);
    let (mh, rest) = rest.split_at_mut(q);
    let (next_mh, column) = rest.split_at_mut(q);

    let small = Small::of(smoothing, m);
    small.kronecker_into(f21, gains, g21);
    vh.fill(0.0);

    // `M_0` is the outer product of the level and trend states with the seasonal states,
    // stored column-major: entry (i, j) at `j t + i`.
    let lead = [level, slope];
    for (j, &season) in seasonal.iter().enumerate() {
        for (i, &state) in lead.iter().take(t).enumerate() {
            mh[j * t + i] = state * season;
        }
    }

    for (mu_out, value) in mean.iter_mut().zip(variance.iter_mut()) {
        // `H_1 M H_2'` and `H_21 V H_21'`: the last seasonal column, weighted by `H_1`.
        let mu: f64 = (0..t).map(|i| small.h1[i] * mh[(m - 1) * t + i]).sum();
        *mu_out = mu;
        let mut block = 0.0;
        for i in 0..t {
            for j in 0..t {
                let index = ((m - 1) * t + i) * q + (m - 1) * t + j;
                block += small.h1[i] * small.h1[j] * vh[index];
            }
        }
        *value = (1.0 + sigma2) * block + sigma2 * mu * mu;

        // `W = V + v v'` for the term with `K`, then `V` for the next horizon.
        for (i, row) in outer.chunks_exact_mut(q).enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = vh[i * q + j] + mh[i] * mh[j];
            }
        }

        let next = Step {
            f21,
            g21,
            k: gains,
            sigma2,
            q,
        };
        next.covariance(vh, outer, product, accumulator, column);
        small.mean(sigma2, mh, next_mh);
    }
}

/// The small blocks the Kronecker products are built from: `F_1`, `G_1` (`t x t`) and the
/// seasonal shift `F_2`, gain `G_2` (`m x m`), as R's `class3`.
struct Small {
    /// The number of level and trend states.
    t: usize,
    /// The period.
    m: usize,
    /// `F_1`, row-major `t x t`: `[1]`, or `[[1, phi], [0, phi]]` with a trend.
    f1: [f64; 4],
    /// `G_1`, row-major `t x t`: `[alpha]`, or `[[alpha, alpha phi], [beta, beta phi]]`.
    g1: [f64; 4],
    /// `H_1`, the weights of the level and trend states in the forecast: `[1]`, or
    /// `[1, phi]` with a trend.
    h1: [f64; 2],
    /// The seasonal smoothing parameter, the only non-zero entry of `G_2` (row 0, column
    /// `m - 1`).
    gamma: f64,
}

impl Small {
    /// The blocks of `smoothing` at period `m`.
    fn of(smoothing: &Smoothing, m: usize) -> Self {
        let t = states(smoothing.trend);
        let (f1, g1, h1) = if t == 2 {
            let Smoothing {
                alpha, beta, phi, ..
            } = *smoothing;
            (
                [1.0, phi, 0.0, phi],
                [alpha, alpha * phi, beta, beta * phi],
                [1.0, phi],
            )
        } else {
            (
                [1.0, 0.0, 0.0, 0.0],
                [smoothing.alpha, 0.0, 0.0, 0.0],
                [1.0, 0.0],
            )
        };

        Self {
            t,
            m,
            f1,
            g1,
            h1,
            gamma: smoothing.gamma,
        }
    }

    /// `F_2[a][b]`: a cyclic shift, with row 0 taking the last state and row `a > 0`
    /// taking state `a - 1`.
    fn f2(&self, a: usize, b: usize) -> f64 {
        let source = if a == 0 { self.m - 1 } else { a - 1 };
        if b == source { 1.0 } else { 0.0 }
    }

    /// `G_2[a][b]`: `gamma` at row 0, column `m - 1`, and 0 elsewhere.
    fn g2(&self, a: usize, b: usize) -> f64 {
        if a == 0 && b == self.m - 1 {
            self.gamma
        } else {
            0.0
        }
    }

    /// Writes `F_21 = F_2 (x) F_1`, `K = G_2 (x) F_1 + F_2 (x) G_1` and `G_21 = G_2 (x) G_1`,
    /// each `q x q` row-major. Entry `(a t + i, b t + j)` of `A (x) B` is `A[a][b] B[i][j]`.
    fn kronecker_into(&self, f21: &mut [f64], k: &mut [f64], g21: &mut [f64]) {
        let (t, q) = (self.t, self.t * self.m);
        for a in 0..self.m {
            for b in 0..self.m {
                let (f2, g2) = (self.f2(a, b), self.g2(a, b));
                for i in 0..t {
                    for j in 0..t {
                        let index = (a * t + i) * q + b * t + j;
                        let (f1, g1) = (self.f1[i * 2 + j], self.g1[i * 2 + j]);
                        f21[index] = f2 * f1;
                        g21[index] = g2 * g1;
                        k[index] = g2 * f1 + f2 * g1;
                    }
                }
            }
        }
    }

    /// The next mean `M = F_1 M F_2' + sigma2 G_1 M G_2'`, in place in `mh` (column-major
    /// `t x m`), using `next` as scratch.
    fn mean(&self, sigma2: f64, mh: &mut [f64], next: &mut [f64]) {
        let (t, m) = (self.t, self.m);
        for b in 0..m {
            for i in 0..t {
                let mut value = 0.0;
                for a in 0..m {
                    let (f2, g2) = (self.f2(b, a), self.g2(b, a));
                    if f2 == 0.0 && g2 == 0.0 {
                        continue;
                    }

                    for j in 0..t {
                        let state = mh[a * t + j];
                        value += self.f1[i * 2 + j] * state * f2;
                        value += sigma2 * self.g1[i * 2 + j] * state * g2;
                    }
                }
                next[b * t + i] = value;
            }
        }

        mh.copy_from_slice(&next[..t * m]);
    }
}

/// One step of the covariance recursion over the precomputed Kronecker products.
struct Step<'a> {
    /// `F_2 (x) F_1`.
    f21: &'a [f64],
    /// `G_2 (x) G_1`.
    g21: &'a [f64],
    /// `G_2 (x) F_1 + F_2 (x) G_1`.
    k: &'a [f64],
    /// The innovation variance.
    sigma2: f64,
    /// The dimension `q`.
    q: usize,
}

impl Step<'_> {
    /// The next covariance, in place in `vh`:
    /// `F V F' + s (F V G' + G V F' + K W K' + s G (3 V + 2 v v') G')` with `F = F_21`,
    /// `G = G_21`, `s = sigma2` and `W = V + v v'` already in `w`.
    fn covariance(
        &self,
        vh: &mut [f64],
        w: &mut [f64],
        product: &mut [f64],
        sum: &mut [f64],
        column: &mut [f64],
    ) {
        let s = self.sigma2;
        sum.fill(0.0);
        self.sandwich(self.f21, vh, self.f21, 1.0, product, sum, column);
        self.sandwich(self.f21, vh, self.g21, s, product, sum, column);
        self.sandwich(self.g21, vh, self.f21, s, product, sum, column);
        self.sandwich(self.k, w, self.k, s, product, sum, column);

        // `3 V + 2 v v'` is `2 W + V`.
        for (cell, &v) in w.iter_mut().zip(vh.iter()) {
            *cell = 2.0 * *cell + v;
        }

        self.sandwich(self.g21, w, self.g21, s * s, product, sum, column);
        vh.copy_from_slice(sum);
    }

    /// Adds `scale A X B'` to `sum`, with `product` and `column` as scratch.
    #[expect(
        clippy::too_many_arguments,
        reason = "scratch buffers keep the recursion heap-free"
    )]
    fn sandwich(
        &self,
        a: &[f64],
        x: &[f64],
        b: &[f64],
        scale: f64,
        product: &mut [f64],
        sum: &mut [f64],
        column: &mut [f64],
    ) {
        let q = self.q;
        // `product = A X`, row by row.
        for i in 0..q {
            column.fill(0.0);
            for l in 0..q {
                let a_il = a[i * q + l];
                if a_il == 0.0 {
                    continue;
                }

                for (c, &x_lj) in column.iter_mut().zip(&x[l * q..(l + 1) * q]) {
                    *c += a_il * x_lj;
                }
            }
            product[i * q..(i + 1) * q].copy_from_slice(column);
        }

        // `sum += scale product B'`: entry (i, j) adds row i of `product` dot row j of `B`.
        for i in 0..q {
            let row = &product[i * q..(i + 1) * q];
            for j in 0..q {
                let dot: f64 = row
                    .iter()
                    .zip(&b[j * q..(j + 1) * q])
                    .map(|(p, b)| p * b)
                    .sum();
                sum[i * q + j] += scale * dot;
            }
        }
    }
}
