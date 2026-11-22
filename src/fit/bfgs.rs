//! A small BFGS minimiser with central-difference gradients.
//!
//! Nocedal & Wright (2006), *Numerical Optimization*, Algorithm 6.1, with a
//! backtracking Armijo line search. The objective may return `+inf` (or `NaN`)
//! for points it cannot evaluate; the line search then steps back.

use alloc::vec;
use alloc::vec::Vec;

/// When to stop.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Options {
    /// The largest number of BFGS iterations.
    pub(crate) max_iterations: usize,
    /// Stop when every gradient entry is at most this in absolute value.
    pub(crate) gradient_tolerance: f64,
    /// Stop when an iteration improves the objective by at most this, relative to `max(1, |f|)`.
    pub(crate) value_tolerance: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            gradient_tolerance: 1e-7,
            value_tolerance: 1e-13,
        }
    }
}

/// The result of [`minimize`].
#[derive(Debug, Clone)]
pub(crate) struct Minimum {
    /// The best point found.
    pub(crate) x: Vec<f64>,
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

/// Minimises `f` from `x0`.
pub(crate) fn minimize(
    mut f: impl FnMut(&[f64]) -> f64,
    x0: Vec<f64>,
    options: Options,
) -> Minimum {
    let n = x0.len();
    let mut x = x0;
    let mut value = f(&x);
    if n == 0 || !value.is_finite() {
        let converged = n == 0 && value.is_finite();
        return Minimum {
            x,
            iterations: 0,
            converged,
        };
    }

    let mut gradient = gradient_at(&mut f, &x, value);
    let mut inverse_hessian = identity(n);
    let mut fresh = true;

    for iteration in 0..options.max_iterations {
        if max_abs(&gradient) <= options.gradient_tolerance {
            return Minimum {
                x,
                iterations: iteration,
                converged: true,
            };
        }

        let mut direction = descent_direction(&inverse_hessian, &gradient);
        if dot(&gradient, &direction) >= 0.0 {
            inverse_hessian = identity(n);
            fresh = true;
            direction = gradient.iter().map(|g| -g).collect();
        }

        let Some((next, next_value)) = line_search(&mut f, &x, value, &gradient, &direction) else {
            if fresh {
                return Minimum {
                    x,
                    iterations: iteration,
                    converged: false,
                };
            }

            // The curvature model went stale; retry from steepest descent.
            inverse_hessian = identity(n);
            fresh = true;
            continue;
        };

        let next_gradient = gradient_at(&mut f, &next, next_value);
        let s: Vec<f64> = next.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = next_gradient
            .iter()
            .zip(&gradient)
            .map(|(a, b)| a - b)
            .collect();
        let improvement = value - next_value;

        if fresh {
            scale_initial(&mut inverse_hessian, &s, &y);
        }

        fresh = !bfgs_update(&mut inverse_hessian, &s, &y) && fresh;
        x = next;
        value = next_value;
        gradient = next_gradient;

        if improvement <= options.value_tolerance * value.abs().max(1.0) {
            return Minimum {
                x,
                iterations: iteration + 1,
                converged: true,
            };
        }
    }

    let converged = max_abs(&gradient) <= options.gradient_tolerance;
    Minimum {
        x,
        iterations: options.max_iterations,
        converged,
    }
}

/// The central-difference gradient of `f` at `x`, where `f(x) = value`.
///
/// Falls back to a one-sided difference where one side cannot be evaluated, and to
/// zero where neither can.
fn gradient_at(f: &mut impl FnMut(&[f64]) -> f64, x: &[f64], value: f64) -> Vec<f64> {
    let mut point = x.to_vec();
    (0..x.len())
        .map(|i| {
            let h = STEP * x[i].abs().max(1.0);
            point[i] = x[i] + h;
            let up = f(&point);
            point[i] = x[i] - h;
            let down = f(&point);
            point[i] = x[i];

            match (up.is_finite(), down.is_finite()) {
                (true, true) => (up - down) / (2.0 * h),
                (true, false) => (up - value) / h,
                (false, true) => (value - down) / h,
                (false, false) => 0.0,
            }
        })
        .collect()
}

/// Backtracks from a full step until the Armijo condition holds.
///
/// Returns the new point and its value, or [`None`] if no step decreases `f` enough.
fn line_search(
    f: &mut impl FnMut(&[f64]) -> f64,
    x: &[f64],
    value: f64,
    gradient: &[f64],
    direction: &[f64],
) -> Option<(Vec<f64>, f64)> {
    let slope = dot(gradient, direction);
    let mut step = 1.0;
    for _ in 0..MAX_HALVINGS {
        let candidate: Vec<f64> = x.iter().zip(direction).map(|(a, d)| a + step * d).collect();
        let candidate_value = f(&candidate);
        if candidate_value.is_finite() && candidate_value <= value + ARMIJO * step * slope {
            return Some((candidate, candidate_value));
        }

        step *= 0.5;
    }

    None
}

/// `-H g`.
fn descent_direction(inverse_hessian: &[Vec<f64>], gradient: &[f64]) -> Vec<f64> {
    inverse_hessian
        .iter()
        .map(|row| -dot(row, gradient))
        .collect()
}

/// Scales the identity to `(s'y / y'y) I` before the first update, so the first step has the right size.
fn scale_initial(inverse_hessian: &mut [Vec<f64>], s: &[f64], y: &[f64]) {
    let (sy, yy) = (dot(s, y), dot(y, y));
    if sy > 0.0 && yy > 0.0 {
        for (i, row) in inverse_hessian.iter_mut().enumerate() {
            row.fill(0.0);
            row[i] = sy / yy;
        }
    }
}

/// The BFGS update `H = (I - rho s y') H (I - rho y s') + rho s s'` with `rho = 1 / s'y`.
///
/// Skipped, returning `false`, when the curvature `s'y` is not clearly positive.
fn bfgs_update(inverse_hessian: &mut [Vec<f64>], s: &[f64], y: &[f64]) -> bool {
    let sy = dot(s, y);
    if sy <= 1e-12 * crate::math::sqrt(dot(s, s) * dot(y, y)) {
        return false;
    }

    let rho = 1.0 / sy;
    let hy: Vec<f64> = inverse_hessian.iter().map(|row| dot(row, y)).collect();
    let yhy = dot(y, &hy);
    for (i, row) in inverse_hessian.iter_mut().enumerate() {
        for (j, h) in row.iter_mut().enumerate() {
            *h += -rho * (hy[i] * s[j] + s[i] * hy[j]) + (rho * rho * yhy + rho) * s[i] * s[j];
        }
    }

    true
}

/// The `n x n` identity matrix.
fn identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect()
}

/// The dot product of two equally long slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// The largest absolute entry.
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimises_rosenbrock() {
        let rosenbrock = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let minimum = minimize(rosenbrock, vec![-1.2, 1.0], Options::default());
        assert!(minimum.converged, "{minimum:?}");
        assert!(
            (minimum.x[0] - 1.0).abs() < 1e-5 && (minimum.x[1] - 1.0).abs() < 1e-5,
            "{minimum:?}"
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
        let minimum = minimize(f, vec![0.0], Options::default());
        assert!((minimum.x[0] - 2.0).abs() < 1e-6, "{minimum:?}");
    }

    #[test]
    fn no_parameters() {
        let minimum = minimize(|_| 4.0, vec![], Options::default());
        assert!(minimum.converged);
        assert!(minimum.x.is_empty());
    }
}
