//! A small BFGS minimiser with central-difference gradients.
//!
//! Nocedal & Wright (2006), *Numerical Optimization*, Algorithm 6.1, with a
//! backtracking Armijo line search. The objective may return `+inf` (or `NaN`)
//! for points it cannot evaluate; the line search then steps back.
//!
//! It allocates nothing: the point is updated in place and every working vector,
//! including the inverse Hessian, comes from one caller-owned scratch slice.

use crate::buffer::take;
use crate::math::dot;

/// When to stop.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Options {
    /// The largest number of BFGS iterations.
    pub(crate) max_iterations: usize,
    /// Stop when every gradient entry is at most this in absolute value.
    pub(crate) gradient_tolerance: f64,
    /// Stop when an iteration improves the objective by at most this, relative to `max(1, |f|)`.
    pub(crate) value_tolerance: f64,
    /// Start from the inverse Hessian a previous [`minimize`] left in the scratch, instead
    /// of the identity: a related objective's curvature saves the first iterations.
    pub(crate) warm_hessian: bool,
}

impl Options {
    /// The default iteration limit.
    pub(crate) const DEFAULT_MAX_ITERATIONS: usize = 200;
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: Self::DEFAULT_MAX_ITERATIONS,
            gradient_tolerance: 1e-7,
            value_tolerance: 1e-13,
            warm_hessian: false,
        }
    }
}

/// The result of [`minimize`]; the best point found is left in its `x`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Minimum {
    /// The BFGS iterations taken.
    pub(crate) iterations: usize,
    /// Whether a tolerance was met, rather than the iteration limit or a failed line search.
    pub(crate) converged: bool,
}

/// The cube root of machine epsilon: the relative step that balances truncation and
/// rounding error in a central difference.
const STEP: f64 = 6.055_454_452_393_343e-6;

/// The Armijo sufficient-decrease constant.
const ARMIJO: f64 = 1e-4;

/// The most step halvings one line search tries.
const MAX_HALVINGS: usize = 50;

/// The scratch [`minimize`] needs for `n` parameters: the `n x n` inverse Hessian and
/// eight vectors of length `n`.
pub(crate) const fn scratch_len(n: usize) -> usize {
    n * n + 8 * n
}

/// The working vectors of one minimisation, carved out of the caller's scratch.
struct Buffers<'a> {
    /// The inverse Hessian approximation, `n x n` row-major.
    inverse_hessian: &'a mut [f64],
    /// The gradient at the current point.
    gradient: &'a mut [f64],
    /// The search direction.
    direction: &'a mut [f64],
    /// The line search's trial point; the accepted next point once it succeeds.
    candidate: &'a mut [f64],
    /// The gradient at the accepted next point.
    next_gradient: &'a mut [f64],
    /// The step `s = x_next - x`.
    s: &'a mut [f64],
    /// The gradient change `y = g_next - g`.
    y: &'a mut [f64],
    /// `H y`, scratch for the BFGS update.
    hy: &'a mut [f64],
    /// The probe point of the finite-difference gradient.
    point: &'a mut [f64],
}

impl<'a> Buffers<'a> {
    /// Splits `scratch` into the buffers for `n` parameters.
    ///
    /// # Panics
    ///
    /// Panics if `scratch` holds fewer than [`scratch_len`] values.
    fn new(mut scratch: &'a mut [f64], n: usize) -> Self {
        Self {
            inverse_hessian: take(&mut scratch, n * n),
            gradient: take(&mut scratch, n),
            direction: take(&mut scratch, n),
            candidate: take(&mut scratch, n),
            next_gradient: take(&mut scratch, n),
            s: take(&mut scratch, n),
            y: take(&mut scratch, n),
            hy: take(&mut scratch, n),
            point: take(&mut scratch, n),
        }
    }
}

/// Minimises `f`, starting from and overwriting `x`.
///
/// # Panics
///
/// Panics if `scratch` holds fewer than [`scratch_len`]`(x.len())` values.
pub(crate) fn minimize(
    mut f: impl FnMut(&[f64]) -> f64,
    x: &mut [f64],
    scratch: &mut [f64],
    options: Options,
) -> Minimum {
    let n = x.len();
    let mut value = f(x);
    if n == 0 || !value.is_finite() {
        let converged = n == 0 && value.is_finite();
        return Minimum {
            iterations: 0,
            converged,
        };
    }

    let b = Buffers::new(scratch, n);
    gradient_at(&mut f, x, value, b.point, b.gradient);
    if !options.warm_hessian {
        identity(b.inverse_hessian, n);
    }
    let mut fresh = !options.warm_hessian;
    for iteration in 0..options.max_iterations {
        if max_abs(b.gradient) <= options.gradient_tolerance {
            return Minimum {
                iterations: iteration,
                converged: true,
            };
        }

        descent_direction(b.inverse_hessian, b.gradient, b.direction);
        if dot(b.gradient, b.direction) >= 0.0 {
            identity(b.inverse_hessian, n);
            fresh = true;
            for (d, g) in b.direction.iter_mut().zip(b.gradient.iter()) {
                *d = -g;
            }
        }

        let Some(next_value) = line_search(&mut f, x, value, b.gradient, b.direction, b.candidate)
        else {
            if fresh {
                return Minimum {
                    iterations: iteration,
                    converged: false,
                };
            }

            // The curvature model went stale; retry from steepest descent.
            identity(b.inverse_hessian, n);
            fresh = true;
            continue;
        };

        gradient_at(&mut f, b.candidate, next_value, b.point, b.next_gradient);
        difference_into(b.s, b.candidate, x);
        difference_into(b.y, b.next_gradient, b.gradient);
        let improvement = value - next_value;
        if fresh {
            scale_initial(b.inverse_hessian, n, b.s, b.y);
        }

        fresh = !bfgs_update(b.inverse_hessian, b.s, b.y, b.hy) && fresh;
        x.copy_from_slice(b.candidate);
        value = next_value;
        b.gradient.copy_from_slice(b.next_gradient);
        if improvement <= options.value_tolerance * value.abs().max(1.0) {
            return Minimum {
                iterations: iteration + 1,
                converged: true,
            };
        }
    }

    let converged = max_abs(b.gradient) <= options.gradient_tolerance;
    Minimum {
        iterations: options.max_iterations,
        converged,
    }
}

/// Writes the central-difference gradient of `f` at `x`, where `f(x) = value`, into
/// `gradient`, using `point` as the probe.
///
/// Falls back to a one-sided difference where one side cannot be evaluated, and to
/// zero where neither can.
fn gradient_at(
    f: &mut impl FnMut(&[f64]) -> f64,
    x: &[f64],
    value: f64,
    point: &mut [f64],
    gradient: &mut [f64],
) {
    point.copy_from_slice(x);
    for (i, g) in gradient.iter_mut().enumerate() {
        let h = STEP * x[i].abs().max(1.0);
        point[i] = x[i] + h;
        let up = f(point);
        point[i] = x[i] - h;
        let down = f(point);
        point[i] = x[i];
        *g = match (up.is_finite(), down.is_finite()) {
            (true, true) => (up - down) / (2.0 * h),
            (true, false) => (up - value) / h,
            (false, true) => (value - down) / h,
            (false, false) => 0.0,
        };
    }
}

/// Backtracks from a full step until the Armijo condition holds.
///
/// Leaves the accepted point in `candidate` and returns its value, or [`None`] if no
/// step decreases `f` enough.
fn line_search(
    f: &mut impl FnMut(&[f64]) -> f64,
    x: &[f64],
    value: f64,
    gradient: &[f64],
    direction: &[f64],
    candidate: &mut [f64],
) -> Option<f64> {
    let slope = dot(gradient, direction);
    let mut step = 1.0;
    for _ in 0..MAX_HALVINGS {
        for ((c, a), d) in candidate.iter_mut().zip(x).zip(direction) {
            *c = a + step * d;
        }

        let candidate_value = f(candidate);
        if candidate_value.is_finite() && candidate_value <= value + ARMIJO * step * slope {
            return Some(candidate_value);
        }

        step *= 0.5;
    }

    None
}

/// Writes `-H g` into `direction`.
fn descent_direction(inverse_hessian: &[f64], gradient: &[f64], direction: &mut [f64]) {
    let n = gradient.len();
    for (d, row) in direction.iter_mut().zip(inverse_hessian.chunks_exact(n)) {
        *d = -dot(row, gradient);
    }
}

/// Writes `a - b` into `out`.
fn difference_into(out: &mut [f64], a: &[f64], b: &[f64]) {
    for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
        *o = a - b;
    }
}

/// Scales the identity to `(s'y / y'y) I` before the first update, so the first step has the right size.
fn scale_initial(inverse_hessian: &mut [f64], n: usize, s: &[f64], y: &[f64]) {
    let (sy, yy) = (dot(s, y), dot(y, y));
    if sy > 0.0 && yy > 0.0 {
        for (i, row) in inverse_hessian.chunks_exact_mut(n).enumerate() {
            row.fill(0.0);
            row[i] = sy / yy;
        }
    }
}

/// The BFGS update `H = (I - rho s y') H (I - rho y s') + rho s s'` with `rho = 1 / s'y`,
/// with `hy` as scratch for `H y`.
///
/// Skipped, returning `false`, when the curvature `s'y` is not clearly positive.
fn bfgs_update(inverse_hessian: &mut [f64], s: &[f64], y: &[f64], hy: &mut [f64]) -> bool {
    let sy = dot(s, y);
    if sy <= 1e-12 * crate::math::sqrt(dot(s, s) * dot(y, y)) {
        return false;
    }

    let n = s.len();
    let rho = 1.0 / sy;
    for (h, row) in hy.iter_mut().zip(inverse_hessian.chunks_exact(n)) {
        *h = dot(row, y);
    }

    let yhy = dot(y, hy);
    for (i, row) in inverse_hessian.chunks_exact_mut(n).enumerate() {
        for (j, h) in row.iter_mut().enumerate() {
            *h += -rho * (hy[i] * s[j] + s[i] * hy[j]) + (rho * rho * yhy + rho) * s[i] * s[j];
        }
    }

    true
}

/// Sets the `n x n` matrix to the identity.
fn identity(matrix: &mut [f64], n: usize) {
    matrix.fill(0.0);
    for i in 0..n {
        matrix[i * n + i] = 1.0;
    }
}

/// The largest absolute entry.
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    /// Minimises `f` from `x0` with fresh scratch and returns the point and the result.
    fn run(f: impl FnMut(&[f64]) -> f64, x0: &[f64]) -> (std::vec::Vec<f64>, Minimum) {
        let mut x = x0.to_vec();
        let mut scratch = vec![0.0; scratch_len(x.len())];
        let minimum = minimize(f, &mut x, &mut scratch, Options::default());
        (x, minimum)
    }

    #[test]
    fn minimises_rosenbrock() {
        let rosenbrock = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let (x, minimum) = run(rosenbrock, &[-1.2, 1.0]);
        assert!(minimum.converged, "{minimum:?}");
        assert!(
            (x[0] - 1.0).abs() < 1e-5 && (x[1] - 1.0).abs() < 1e-5,
            "{x:?}"
        );
    }

    #[test]
    fn steps_back_from_infinite_regions() {
        // The minimum at 2 sits next to a region the objective cannot evaluate.
        let f = |x: &[f64]| {
            if x[0] > 3.0 {
                f64::INFINITY
            } else {
                (x[0] - 2.0).powi(2)
            }
        };
        let (x, _) = run(f, &[0.0]);
        assert!((x[0] - 2.0).abs() < 1e-6, "{x:?}");
    }

    #[test]
    fn no_parameters() {
        let (x, minimum) = run(|_| 4.0, &[]);
        assert!(minimum.converged);
        assert!(x.is_empty());
    }
}
