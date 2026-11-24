//! A derivative-free minimiser for the small smoothing-parameter searches: Brent's method
//! in one dimension, heap-free.

/// `sqrt(f64::EPSILON)`, the finest relative tolerance a minimiser can reach.
const SQRT_EPSILON: f64 = 1.490_116_119_384_765_6e-8;

/// `(3 - sqrt(5)) / 2`, the golden-section fraction.
const GOLDEN: f64 = 0.381_966_011_250_105_1;

/// The minimiser of `f` on `[low, high]` and its value, by Brent's method (Brent 1973,
/// `localmin`): golden-section steps, replaced by parabolic interpolation where it is safe.
/// It stops when the bracket is within `tolerance` (relative, plus the same absolute) of the
/// best point, and returns the best point it evaluated; for a unimodal `f` that is the
/// global minimum. The tolerance is at least `sqrt(f64::EPSILON)`: below it, the values
/// near a minimum differ by rounding only, and the parabolic steps fail.
///
/// An infinite value of `f` marks a point to avoid; the search then takes golden-section
/// steps away from it.
pub(crate) fn brent(f: impl Fn(f64) -> f64, (low, high): (f64, f64), tolerance: f64) -> (f64, f64) {
    let tolerance = tolerance.max(SQRT_EPSILON);
    let (mut a, mut b) = (low, high);
    let mut x = a + GOLDEN * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    let (mut fw, mut fv) = (fx, fx);
    let (mut d, mut e) = (0.0_f64, 0.0_f64);
    loop {
        let middle = 0.5 * (a + b);
        let tol1 = tolerance * x.abs() + tolerance;
        let tol2 = 2.0 * tol1;
        if (x - middle).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }

        let parabola = (e.abs() > tol1)
            .then(|| parabolic_step(x, w, v, fx, fw, fv))
            .filter(|&(p, q)| p.abs() < (0.5 * q * e).abs() && p > q * (a - x) && p < q * (b - x));
        if let Some((p, q)) = parabola {
            e = d;
            d = p / q;
            let u = x + d;
            if u - a < tol2 || b - u < tol2 {
                d = tol1.copysign(middle - x);
            }
        } else {
            e = if x < middle { b - x } else { a - x };
            d = GOLDEN * e;
        }

        let u = x + if d.abs() >= tol1 { d } else { tol1.copysign(d) };
        let fu = f(u);
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            (v, fv, w, fw, x, fx) = (w, fw, x, fx, u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, fv, w, fw) = (w, fw, u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }

    (x, fx)
}

/// The step `p / q` to the vertex of the parabola through `(x, fx)`, `(w, fw)` and
/// `(v, fv)`, with `q >= 0`.
fn parabolic_step(x: f64, w: f64, v: f64, fx: f64, fw: f64, fv: f64) -> (f64, f64) {
    let r = (x - w) * (fx - fv);
    let q = (x - v) * (fx - fw);
    let p = (x - v) * q - (x - w) * r;
    let q = 2.0 * (q - r);
    if q > 0.0 { (-p, q) } else { (p, -q) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brent_finds_a_parabola_minimum_in_few_steps() {
        let calls = core::cell::Cell::new(0);
        let (x, fx) = brent(
            |x| {
                calls.set(calls.get() + 1);
                (x - 0.3) * (x - 0.3) + 1.0
            },
            (0.0, 1.0),
            0.0,
        );
        assert!((x - 0.3).abs() < 1e-8 && (fx - 1.0).abs() < 1e-15);
        assert!(calls.get() <= 8, "{} calls", calls.get());
    }

    #[test]
    fn brent_avoids_infinite_regions() {
        let (x, _) = brent(
            |x| {
                if x < 0.5 {
                    f64::INFINITY
                } else {
                    (x - 0.7).abs()
                }
            },
            (0.0, 1.0),
            1e-9,
        );
        assert!((x - 0.7).abs() < 1e-6);
    }
}
