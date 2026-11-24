//! Page's two-sided tabular CUSUM.

use crate::filter::FilterError;

/// The direction of a shift that [`Cusum`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shift {
    /// The mean went up.
    Up,
    /// The mean went down.
    Down,
}

/// Page's two-sided cumulative sum on standardised values `z`:
///
/// ```text
/// S+ = max(0, S+ + z - k)        S- = max(0, S- - z - k)
/// ```
///
/// It alarms when a sum goes above the threshold `h`, then starts again from zero. The
/// allowance `k` is half the shift to detect, in standard deviations. With `k = 0.5`,
/// `h = 4` it alarms about every 168 values on standard normal noise, and about 8 values
/// after a shift of one standard deviation; `h = 5` gives about 465 and 10 (Montgomery,
/// *Introduction to Statistical Quality Control*, table 9.3).
///
/// # Examples
///
/// ```
/// use backshift::{Cusum, Shift};
///
/// let mut cusum = Cusum::new(0.5, 4.0).unwrap();
/// assert_eq!(cusum.update(0.3), None);
/// let alarm = (0..10).find_map(|_| cusum.update(2.0));
/// assert_eq!(alarm, Some(Shift::Up));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cusum {
    /// The allowance `k`.
    allowance: f64,
    /// The threshold `h`.
    threshold: f64,
    /// The upper sum `S+`.
    upper: f64,
    /// The lower sum `S-`.
    lower: f64,
}

impl Cusum {
    /// A CUSUM with allowance `k >= 0` and threshold `h > 0`, both in standard deviations.
    ///
    /// # Errors
    ///
    /// [`FilterError::InvalidParameter`] if `k` is negative or `h` is not positive, or either
    /// is not finite.
    pub fn new(allowance: f64, threshold: f64) -> Result<Self, FilterError> {
        let valid =
            allowance.is_finite() && threshold.is_finite() && allowance >= 0.0 && threshold > 0.0;
        if !valid {
            return Err(FilterError::InvalidParameter);
        }

        Ok(Self {
            allowance,
            threshold,
            upper: 0.0,
            lower: 0.0,
        })
    }

    /// Feeds one standardised value; returns the direction of a shift when a sum goes above
    /// the threshold, and then resets both sums. A value that is not finite is skipped.
    pub fn update(&mut self, z: f64) -> Option<Shift> {
        if !z.is_finite() {
            return None;
        }

        self.upper = (self.upper + z - self.allowance).max(0.0);
        self.lower = (self.lower - z - self.allowance).max(0.0);
        let shift = if self.upper > self.threshold {
            Shift::Up
        } else if self.lower > self.threshold {
            Shift::Down
        } else {
            return None;
        };

        self.reset();
        Some(shift)
    }

    /// Sets both sums to zero, as after an alarm.
    pub const fn reset(&mut self) {
        self.upper = 0.0;
        self.lower = 0.0;
    }

    /// The upper sum `S+`, the evidence for an upward shift.
    #[must_use]
    pub const fn upper(&self) -> f64 {
        self.upper
    }

    /// The lower sum `S-`, the evidence for a downward shift.
    #[must_use]
    pub const fn lower(&self) -> f64 {
        self.lower
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_alarms_above_the_threshold_and_resets() {
        let mut cusum = Cusum::new(0.5, 2.0).unwrap();
        assert_eq!(cusum.update(1.5), None);
        assert_eq!(cusum.upper(), 1.0);
        assert_eq!(cusum.lower(), 0.0);
        // Exactly at the threshold is not above it.
        assert_eq!(cusum.update(1.5), None);
        assert_eq!(cusum.upper(), 2.0);
        assert_eq!(cusum.update(f64::NAN), None);
        assert_eq!(cusum.update(0.6), Some(Shift::Up));
        assert_eq!((cusum.upper(), cusum.lower()), (0.0, 0.0));

        let down: Option<Shift> = [-2.0, -2.0].iter().find_map(|&z| cusum.update(z));
        assert_eq!(down, Some(Shift::Down));
    }

    #[test]
    fn rejects_invalid_parameters() {
        for (k, h) in [
            (-0.1, 4.0),
            (0.5, 0.0),
            (f64::NAN, 4.0),
            (0.5, f64::INFINITY),
        ] {
            assert_eq!(Cusum::new(k, h), Err(FilterError::InvalidParameter));
        }
    }
}
