//! Bayesian online changepoint detection (Adams and MacKay 2007) with a Normal-Gamma model
//! and a fixed number of run-length hypotheses.
//!
//! The values come in segments. A new segment starts before each value with probability
//! `hazard`. Within a segment, the values are normal with an unknown mean and precision
//! drawn from the [`NormalGamma`] prior, so a change shows as a new mean, a new scale, or
//! both. After each value, the detector holds the posterior probability that the current
//! segment holds exactly the last `r` values, for each run length `r >= 1`.
//!
//! The exact posterior has one hypothesis per value seen. [`Bocpd`] keeps `R`: when it needs
//! room, it merges the two oldest, adding their probabilities and keeping the statistics of
//! the more probable one. This is an approximation: the `R - 1` youngest run lengths, the
//! ones a recent change makes probable, keep their exact segment statistics, but the merged
//! tail scores new values with only one of its segments, and every probability is
//! normalised against it. With `R` above the number of values, the posterior is exact.

use crate::filter::FilterError;
use crate::math;
use crate::special::ln_gamma;

/// The Normal-Gamma prior of a segment's mean `mu` and precision `lambda`:
/// `lambda ~ Gamma(shape, rate)` and `mu | lambda ~ N(mean, 1 / (precision lambda))`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalGamma {
    /// The prior mean of the segment mean.
    pub mean: f64,
    /// How many values' worth of weight the prior mean has, `> 0`.
    pub precision: f64,
    /// The shape of the precision's Gamma prior, `>= 0.5`.
    pub shape: f64,
    /// The rate of the precision's Gamma prior, `> 0`; the prior mean of the precision is
    /// `shape / rate`.
    pub rate: f64,
}

impl NormalGamma {
    /// A prior for standardised innovations: mean 0 with the weight of one value, and a
    /// precision of 1 on average (shape 1, rate 1).
    pub const STANDARD: Self = Self {
        mean: 0.0,
        precision: 1.0,
        shape: 1.0,
        rate: 1.0,
    };

    /// Whether every value is finite and in its range.
    fn is_valid(&self) -> bool {
        let finite = [self.mean, self.precision, self.shape, self.rate]
            .iter()
            .all(|v| v.is_finite());
        finite && self.precision > 0.0 && self.shape >= 0.5 && self.rate > 0.0
    }
}

/// One run-length hypothesis: its log-probability and the posterior of its segment.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Run {
    /// The log of its posterior probability.
    log_weight: f64,
    /// How many values the segment holds.
    length: usize,
    /// The posterior Normal-Gamma parameters of the segment.
    posterior: NormalGamma,
}

impl Run {
    /// A segment with no values yet, at log-probability `log_weight`.
    const fn fresh(prior: NormalGamma, log_weight: f64) -> Self {
        Self {
            log_weight,
            length: 0,
            posterior: prior,
        }
    }

    /// The log-density of `x` under the posterior predictive, a Student-t with `2 shape`
    /// degrees of freedom, location `mean` and squared scale
    /// `rate (precision + 1) / (shape precision)`.
    fn log_predictive(&self, x: f64) -> f64 {
        let NormalGamma {
            mean,
            precision,
            shape,
            rate,
        } = self.posterior;
        let scale2 = rate * (precision + 1.0) / (shape * precision);
        let degrees = 2.0 * shape;
        let d = x - mean;
        ln_gamma(shape + 0.5)
            - ln_gamma(shape)
            - 0.5 * math::ln(degrees * core::f64::consts::PI * scale2)
            - (shape + 0.5) * math::ln(1.0 + d * d / (degrees * scale2))
    }

    /// Adds `x` to the segment: the conjugate update of the posterior.
    fn observe(&mut self, x: f64) {
        let p = &mut self.posterior;
        let d = x - p.mean;
        p.rate += p.precision * d * d / (2.0 * (p.precision + 1.0));
        p.mean += d / (p.precision + 1.0);
        p.precision += 1.0;
        p.shape += 0.5;
        self.length = self.length.saturating_add(1);
    }
}

/// Bayesian online changepoint detection with at most `R` run-length hypotheses, stored
/// inline with no heap; `O(R)` work per value.
///
/// Feed it standardised innovations. [`Bocpd::run_length`] drops to a few values soon after
/// a change, and [`Bocpd::probability_within`] gives the probability that the current
/// segment started within the last `w` values.
///
/// # Examples
///
/// ```
/// use backshift::{Bocpd, NormalGamma};
///
/// let mut detector = Bocpd::<32>::new(1.0 / 200.0, NormalGamma::STANDARD).unwrap();
/// for t in 0..200 {
///     detector.update(((t * 37 % 11) as f64 - 5.0) / 3.0);
/// }
/// assert!(detector.run_length() > 30);
/// for _ in 0..8 {
///     detector.update(6.0);
/// }
/// assert!(detector.run_length() <= 8);
/// assert!(detector.probability_within(8) > 0.9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bocpd<const R: usize> {
    /// `ln hazard`, the log-probability that a segment starts before a value.
    log_hazard: f64,
    /// `ln(1 - hazard)`.
    log_survival: f64,
    /// The prior of a new segment.
    prior: NormalGamma,
    /// The hypotheses, youngest first; the first `count` are in use.
    runs: [Run; R],
    /// How many hypotheses are in use.
    count: usize,
}

impl<const R: usize> Bocpd<R> {
    /// A detector where a segment starts before each value with probability `hazard`, the
    /// inverse of the expected segment length, and segments follow `prior`.
    ///
    /// # Errors
    ///
    /// - [`FilterError::InvalidParameter`] if `hazard` is not in `(0, 1)`, or a value of
    ///   `prior` is not finite or out of its range.
    /// - [`FilterError::Capacity`] if `R < 2`: merging needs two hypotheses.
    pub fn new(hazard: f64, prior: NormalGamma) -> Result<Self, FilterError> {
        if !(hazard > 0.0 && hazard < 1.0 && prior.is_valid()) {
            return Err(FilterError::InvalidParameter);
        }

        if R < 2 {
            return Err(FilterError::Capacity {
                needed: 2,
                capacity: R,
            });
        }

        Ok(Self {
            log_hazard: math::ln(hazard),
            log_survival: math::ln(1.0 - hazard),
            prior,
            runs: [Run::fresh(prior, 0.0); R],
            count: 0,
        })
    }

    /// Feeds one value and updates the run-length posterior. A value that is not finite,
    /// or whose square is not, is skipped.
    pub fn update(&mut self, x: f64) {
        if !(x * x).is_finite() {
            return;
        }

        self.start_segment();
        for run in &mut self.runs[..self.count] {
            run.log_weight += run.log_predictive(x);
            run.observe(x);
        }

        self.normalise();
    }

    /// The most probable run length, the number of values in the current segment; the
    /// shorter one on a tie, and 0 before any value.
    #[must_use]
    pub fn run_length(&self) -> usize {
        self.runs[..self.count]
            .iter()
            .fold(None, |best: Option<&Run>, run| match best {
                Some(b) if b.log_weight >= run.log_weight => best,
                _ => Some(run),
            })
            .map_or(0, |run| run.length)
    }

    /// The probability that the current segment holds at most `w` values: that it started
    /// within the last `w`. Once hypotheses merge, it counts the merged oldest one at its
    /// kept length.
    #[must_use]
    pub fn probability_within(&self, w: usize) -> f64 {
        self.runs[..self.count]
            .iter()
            .filter(|run| run.length <= w)
            .map(|run| math::exp(run.log_weight))
            .sum()
    }

    /// Forgets every value, as after [`Bocpd::new`].
    pub const fn reset(&mut self) {
        self.count = 0;
    }

    /// Ages every hypothesis by the survival probability and adds a new segment, youngest
    /// first, merging the two oldest when there is no room.
    fn start_segment(&mut self) {
        if self.count == R {
            self.merge_oldest();
        }

        let log_weight = if self.count == 0 {
            0.0
        } else {
            self.log_hazard
        };
        self.runs.copy_within(..self.count, 1);
        for run in &mut self.runs[1..=self.count] {
            run.log_weight += self.log_survival;
        }

        self.runs[0] = Run::fresh(self.prior, log_weight);
        self.count += 1;
    }

    /// Merges the two oldest hypotheses into one: their probabilities add, and the more
    /// probable one keeps its statistics. The caller ensures `count == R >= 2`.
    fn merge_oldest(&mut self) {
        let (older, newer) = (self.runs[R - 1], self.runs[R - 2]);
        let kept = if newer.log_weight >= older.log_weight {
            newer
        } else {
            older
        };
        self.runs[R - 2] = Run {
            log_weight: log_add(older.log_weight, newer.log_weight),
            ..kept
        };
        self.count = R - 1;
    }

    /// Scales the probabilities to sum to 1, in log space.
    fn normalise(&mut self) {
        let runs = &mut self.runs[..self.count];
        let max = runs
            .iter()
            .map(|run| run.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);
        let total = max
            + math::ln(
                runs.iter()
                    .map(|run| math::exp(run.log_weight - max))
                    .sum::<f64>(),
            );
        for run in runs {
            run.log_weight -= total;
        }
    }
}

/// `ln(e^a + e^b)` without overflow.
fn log_add(a: f64, b: f64) -> f64 {
    let (high, low) = if a >= b { (a, b) } else { (b, a) };
    if low == f64::NEG_INFINITY {
        high
    } else {
        high + math::ln(1.0 + math::exp(low - high))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_value_starts_a_segment_with_certainty() {
        let mut detector = Bocpd::<4>::new(0.1, NormalGamma::STANDARD).unwrap();
        assert_eq!(detector.run_length(), 0);
        detector.update(0.5);
        assert_eq!(detector.run_length(), 1);
        assert!((detector.probability_within(1) - 1.0).abs() < 1e-15);

        detector.update(f64::NAN);
        detector.update(1e300);
        assert_eq!(detector.count, 1);
    }

    #[test]
    fn keeps_at_most_r_hypotheses_and_unit_mass() {
        let mut detector = Bocpd::<3>::new(0.2, NormalGamma::STANDARD).unwrap();
        for t in 0..10 {
            detector.update(f64::from(t % 3) - 1.0);
            assert!(detector.count <= 3);
            assert!((detector.probability_within(usize::MAX) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_invalid_parameters() {
        let prior = NormalGamma::STANDARD;
        for hazard in [0.0, 1.0, f64::NAN] {
            assert_eq!(
                Bocpd::<4>::new(hazard, prior),
                Err(FilterError::InvalidParameter)
            );
        }
        let bad = NormalGamma {
            shape: 0.25,
            ..prior
        };
        assert_eq!(
            Bocpd::<4>::new(0.1, bad),
            Err(FilterError::InvalidParameter)
        );
        assert_eq!(
            Bocpd::<1>::new(0.1, prior),
            Err(FilterError::Capacity {
                needed: 2,
                capacity: 1
            })
        );
    }
}
