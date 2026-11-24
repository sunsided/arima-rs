//! The ETS state recursion on caller-owned seasonal states, shared by the heap-free filter
//! and the fit. It follows R's `etscalc` and `forecast.ets` (class 1) in `forecast` 8.23.
//!
//! The `m` seasonal states live in a ring: `seasonal[(head + j) % m]` is the state `j`
//! places older than the most recent one, which sits at `head`. An update overwrites the
//! oldest state and moves `head` onto it, so it costs `O(1)` instead of shifting `m`
//! values as R does.

/// Below this modulus R's `etscalc` treats a divisor as zero and uses [`HUGE`] instead.
const TOLERANCE: f64 = 1e-10;

/// R's stand-in (`HUGEN`) for the quotient by a divisor of about zero.
const HUGE: f64 = 1e10;

/// The smoothing parameters of an ETS model, as the recursion takes them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Smoothing {
    /// The level smoothing parameter.
    pub(crate) alpha: f64,
    /// The slope smoothing parameter; 0 without a trend.
    pub(crate) beta: f64,
    /// The seasonal smoothing parameter; unused without seasonal states.
    pub(crate) gamma: f64,
    /// The damping parameter; 1 without damping.
    pub(crate) phi: f64,
    /// Whether the model has a trend (and so a slope state).
    pub(crate) trend: bool,
    /// Whether the season multiplies the level and trend instead of adding to them.
    pub(crate) multiplicative_season: bool,
}

impl Smoothing {
    /// The one-step forecast `(l + phi b) + s_m`, or `(l + phi b) s_m` with a multiplicative
    /// season, from the current states.
    ///
    /// `seasonal` is the ring of seasonal states with the most recent at `head`, or empty.
    pub(crate) fn forecast(&self, level: f64, slope: f64, seasonal: &[f64], head: usize) -> f64 {
        let trend = if self.trend { self.phi * slope } else { 0.0 };
        let base = level + trend;
        match oldest(seasonal, head) {
            None => base,
            Some(i) => self.season(base, seasonal[i]),
        }
    }

    /// `base` with the seasonal state `state` applied: added, or multiplied.
    fn season(&self, base: f64, state: f64) -> f64 {
        if self.multiplicative_season {
            base * state
        } else {
            base + state
        }
    }

    /// `y` with the seasonal state `state` taken out: subtracted, or divided (R's `HUGE` for
    /// a divisor of about zero).
    fn deseason(&self, y: f64, state: f64) -> f64 {
        if !self.multiplicative_season {
            return y - state;
        }

        if state.abs() < TOLERANCE {
            HUGE
        } else {
            y / state
        }
    }

    /// Moves the states on with the observed value `y`, as R's `etscalc` `update`.
    ///
    /// `seasonal` is the ring of seasonal states with the most recent at `head`, or empty.
    /// `y` must be finite; a missing value is fed as its forecast, which moves the states on
    /// unchanged.
    pub(crate) fn update(
        &self,
        level: &mut f64,
        slope: &mut f64,
        seasonal: &mut [f64],
        head: &mut usize,
        y: f64,
    ) {
        let damped_slope = if self.trend { self.phi * *slope } else { 0.0 };
        let base = *level + damped_slope;
        let oldest_index = oldest(seasonal, *head);
        let adjusted = oldest_index.map_or(y, |i| self.deseason(y, seasonal[i]));
        let new_level = base + self.alpha * (adjusted - base);
        if self.trend {
            *slope = damped_slope + self.beta / self.alpha * (new_level - *level - damped_slope);
        }

        if let Some(i) = oldest_index {
            // The seasonal state learns from `y` with the level and trend taken out.
            let observed = self.deseason(y, base);
            seasonal[i] += self.gamma * (observed - seasonal[i]);
            *head = i;
        }

        *level = new_level;
    }

    /// Writes the forecast means `(l + phi_h b) + s` (or times `s` with a multiplicative
    /// season) for horizons `1..=mean.len()`, where `phi_h = phi + ... + phi^h` and `s` is
    /// the seasonal state of that horizon.
    pub(crate) fn forecast_means(
        &self,
        level: f64,
        slope: f64,
        seasonal: &[f64],
        head: usize,
        mean: &mut [f64],
    ) {
        let m = seasonal.len();
        let mut damping = 0.0;
        let mut power = 1.0;
        for (h, value) in mean.iter_mut().enumerate() {
            power *= self.phi;
            damping += power;
            let trend = if self.trend { damping * slope } else { 0.0 };
            // The state for horizon `h + 1` is `m - 1 - h % m` places older than the newest.
            *value = if m == 0 {
                level + trend
            } else {
                self.season(level + trend, seasonal[(head + m - 1 - h % m) % m])
            };
        }
    }

    /// Writes R's class-1 forecast variances `sigma2 (1 + c_1^2 + ... + c_(h-1)^2)` for
    /// horizons `1..=variance.len()`, where `c_j = alpha + beta phi_j + gamma [m divides j]`.
    pub(crate) fn forecast_variances(&self, m: usize, sigma2: f64, variance: &mut [f64]) {
        let mut damping = 0.0;
        let mut power = 1.0;
        let mut sum = 0.0;
        for (h, value) in variance.iter_mut().enumerate() {
            *value = sigma2 * (1.0 + sum);
            let j = h + 1;
            power *= self.phi;
            damping += power;
            let season = if m > 0 && j % m == 0 { self.gamma } else { 0.0 };
            let c = self.alpha + self.beta * damping + season;
            sum += c * c;
        }
    }
}

impl Smoothing {
    /// Turns the class-1 variances in `variance` into R's class-2 ones, for a multiplicative
    /// error with no or an additive season: `(1 + sigma2) theta_h - mu_h^2`, with
    /// `theta_1 = mu_1^2` and `theta_h = mu_h^2 + sigma2 (c_1^2 theta_(h-1) + ... +
    /// c_(h-1)^2 theta_1)`.
    ///
    /// `mean` holds the forecast means `mu_h`. `variance` is overwritten: it holds `theta`
    /// while the recursion runs.
    pub(crate) fn relative_variances(
        &self,
        m: usize,
        sigma2: f64,
        mean: &[f64],
        variance: &mut [f64],
    ) {
        for h in 0..variance.len() {
            let mut sum = 0.0;
            let (mut damping, mut power) = (0.0, 1.0);
            for i in 1..=h {
                power *= self.phi;
                damping += power;
                let season = if m > 0 && i % m == 0 { self.gamma } else { 0.0 };
                let c = self.alpha + self.beta * damping + season;
                sum += c * c * variance[h - i];
            }
            variance[h] = mean[h] * mean[h] + sigma2 * sum;
        }

        for (value, mu) in variance.iter_mut().zip(mean) {
            *value = (1.0 + sigma2) * *value - mu * mu;
        }
    }
}

/// The index of the oldest seasonal state in a ring with the newest at `head`, or [`None`]
/// without seasonal states.
fn oldest(seasonal: &[f64], head: usize) -> Option<usize> {
    let m = seasonal.len();
    (m > 0).then(|| (head + m - 1) % m)
}

/// Writes the ring `seasonal` with the newest state at `head` into `out`, most recent first.
#[cfg(any(feature = "alloc", test))]
pub(crate) fn unroll(seasonal: &[f64], head: usize, out: &mut [f64]) {
    let m = seasonal.len();
    for (j, value) in out.iter_mut().enumerate().take(m) {
        *value = seasonal[(head + j) % m];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Holt-Winters with an additive trend and season of period 2.
    const HOLT_WINTERS: Smoothing = Smoothing {
        alpha: 0.5,
        beta: 0.2,
        gamma: 0.1,
        phi: 1.0,
        trend: true,
        multiplicative_season: false,
    };

    #[test]
    fn a_perfect_forecast_leaves_the_states_on_their_path() {
        let (mut level, mut slope, mut seasonal, mut head) = (10.0, 1.0, [2.0, -2.0], 0);
        let y = HOLT_WINTERS.forecast(level, slope, &seasonal, head);
        assert_eq!(y, 9.0);
        HOLT_WINTERS.update(&mut level, &mut slope, &mut seasonal, &mut head, y);
        let mut ordered = [0.0; 2];
        unroll(&seasonal, head, &mut ordered);
        assert_eq!((level, slope, ordered), (11.0, 1.0, [-2.0, 2.0]));
    }

    #[test]
    fn variances_add_the_seasonal_gain_every_period() {
        let mut variance = [0.0; 3];
        HOLT_WINTERS.forecast_variances(2, 1.0, &mut variance);
        // c_1 = 0.5 + 0.2 = 0.7, c_2 = 0.5 + 0.4 + 0.1 = 1.0.
        assert_eq!(variance[0], 1.0);
        assert!((variance[1] - 1.49).abs() < 1e-12);
        assert!((variance[2] - 2.49).abs() < 1e-12);
    }
}
