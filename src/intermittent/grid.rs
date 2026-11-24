//! Scoring many smoothing parameters in one pass over the history.
//!
//! The optimised filters score a grid of parameters, each by its mean squared one-step
//! error. Run one filter per grid point and every point repeats the same branches over the
//! history. Here the state that differs between points (the smoothed size, interval or
//! probability) sits in arrays of [`LANES`], while the branches that depend only on the
//! data run once, so the per-point arithmetic is a straight loop the compiler vectorises.
//! Each lane does the same operations in the same order as the filter's `update`, so its
//! error equals the filter's bit for bit.

use super::CrostonVariant;

/// How many parameters one pass scores; the stack holds a few arrays of this many values.
pub(super) const LANES: usize = 20;

/// The parameters of one pass.
pub(super) type Lanes = [f64; LANES];

/// The running sums of squared errors of the lanes, and how many errors each holds.
///
/// Once a filter scores, its forecast is finite (an interval is at least 1, a probability
/// and a size are averages of finite values), so no error is `NaN` and every lane counts the
/// same errors. The loop has no branch, which lets the compiler vectorise it.
#[derive(Debug, Clone, Copy)]
struct Accumulator {
    /// The sums of squared errors.
    sums: Lanes,
    /// The number of errors in every lane.
    count: f64,
}

impl Accumulator {
    /// No errors yet.
    const fn new() -> Self {
        Self {
            sums: [0.0; LANES],
            count: 0.0,
        }
    }

    /// Adds the squared error `y - before[i]` to each lane.
    fn add(&mut self, y: f64, before: &Lanes) {
        for (sum, &forecast) in self.sums.iter_mut().zip(before) {
            let error = y - forecast;
            *sum += error * error;
        }
        self.count += 1.0;
    }

    /// Each lane's mean squared error; infinite with no errors, as `mean_squared_error`.
    fn means(&self) -> Lanes {
        if self.count > 0.0 {
            self.sums.map(|sum| sum / self.count)
        } else {
            [f64::INFINITY; LANES]
        }
    }
}

/// The mean squared one-step error of Croston's method over `history` for each smoothing
/// parameter of `alphas`, as [`CrostonFilter`](super::CrostonFilter) scores one.
pub(super) fn croston(history: &[f64], variant: CrostonVariant, alphas: &Lanes) -> Lanes {
    let correction: Lanes = alphas.map(|alpha| match variant {
        CrostonVariant::Classic => 1.0,
        CrostonVariant::Sba => 1.0 - alpha / 2.0,
        CrostonVariant::Sbj => 1.0 - alpha / (2.0 - alpha),
    });
    let (mut size, mut interval) = ([0.0; LANES], [0.0; LANES]);
    let mut forecast = [0.0; LANES];
    let mut errors = Accumulator::new();
    let (mut started, mut since, mut demands) = (false, 0.0, 0_usize);
    for &y in history {
        let valid = y.is_finite() && y >= 0.0;
        if started && valid {
            errors.add(y, &forecast);
        }

        started = true;
        since += 1.0;
        if !(valid && y > 0.0) {
            continue;
        }

        if demands == 0 {
            size = [y; LANES];
            interval = [since; LANES];
        } else {
            for ((s, i), &alpha) in size.iter_mut().zip(&mut interval).zip(alphas) {
                *s += alpha * (y - *s);
                *i += alpha * (since - *i);
            }
        }

        demands += 1;
        since = 0.0;
        for (((f, &s), &i), &c) in forecast
            .iter_mut()
            .zip(&size)
            .zip(&interval)
            .zip(&correction)
        {
            *f = s / i * c;
        }
    }

    errors.means()
}

/// The mean squared one-step error of TSB over `history` for each lane's pair of the size
/// parameter `sizes[i]` and the probability parameter `probabilities[i]`, as
/// [`TsbFilter`](super::TsbFilter) scores one.
pub(super) fn tsb(history: &[f64], sizes: &Lanes, probabilities: &Lanes) -> Lanes {
    let (mut size, mut probability) = ([0.0; LANES], [0.0; LANES]);
    let mut before = [0.0; LANES];
    let mut errors = Accumulator::new();
    let (mut started, mut demanded) = (false, false);
    for &y in history {
        if !(y.is_finite() && y >= 0.0) {
            continue;
        }

        if started {
            for ((b, &p), &s) in before.iter_mut().zip(&probability).zip(&size) {
                *b = p * s;
            }
            errors.add(y, &before);
        }

        let occurred = if y > 0.0 { 1.0 } else { 0.0 };
        if started {
            for (p, &alpha) in probability.iter_mut().zip(probabilities) {
                *p += alpha * (occurred - *p);
            }
        } else {
            probability = [occurred; LANES];
            started = true;
        }

        if y > 0.0 {
            if demanded {
                for (s, &alpha) in size.iter_mut().zip(sizes) {
                    *s += alpha * (y - *s);
                }
            } else {
                size = [y; LANES];
                demanded = true;
            }
        }
    }

    errors.means()
}

#[cfg(test)]
mod tests {
    use super::super::{CrostonFilter, TsbFilter, mean_squared_error};
    use super::*;

    /// A history with gaps, a missing value and a negative one.
    const HISTORY: [f64; 16] = [
        0.0,
        3.0,
        0.0,
        f64::NAN,
        5.0,
        0.0,
        2.0,
        0.0,
        -1.0,
        0.0,
        6.0,
        1.0,
        0.0,
        4.0,
        0.0,
        3.0,
    ];

    /// Lane parameters across `(0, 1]`.
    fn alphas() -> Lanes {
        core::array::from_fn(|i| 0.01 + 0.99 * (i as f64) / ((LANES - 1) as f64))
    }

    #[test]
    fn croston_lanes_equal_the_filter_bit_for_bit() {
        for variant in [
            CrostonVariant::Classic,
            CrostonVariant::Sba,
            CrostonVariant::Sbj,
        ] {
            let lanes = croston(&HISTORY, variant, &alphas());
            for (&alpha, &value) in alphas().iter().zip(&lanes) {
                let mut filter = CrostonFilter::new(alpha, variant).unwrap();
                let expected = mean_squared_error(&HISTORY, |y| filter.update(y));
                assert_eq!(value.to_bits(), expected.to_bits(), "alpha {alpha}");
            }
        }
    }

    #[test]
    fn tsb_lanes_equal_the_filter_bit_for_bit() {
        let probabilities = alphas().map(|a| 1.0 - 0.9 * a);
        let lanes = tsb(&HISTORY, &alphas(), &probabilities);
        for ((&a, &b), &value) in alphas().iter().zip(&probabilities).zip(&lanes) {
            let mut filter = TsbFilter::new(a, b).unwrap();
            let expected = mean_squared_error(&HISTORY, |y| filter.update(y));
            assert_eq!(value.to_bits(), expected.to_bits(), "alphas {a}, {b}");
        }
    }
}
