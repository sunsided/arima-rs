//! The ETS state recursion on caller-owned seasonal states, shared by the heap-free filter
//! and the fit. It follows R's `etscalc` and `forecast.ets` (class 1) in `forecast` 8.23.
//!
//! The `m` seasonal states live in a ring: `seasonal[(head + j) % m]` is the state `j`
//! places older than the most recent one, which sits at `head`. An update overwrites the
//! oldest state and moves `head` onto it, so it costs `O(1)` instead of shifting `m`
//! values as R does.

/// The smoothing parameters of an additive ETS model, as the recursion takes them.
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
}

impl Smoothing {
    /// The one-step forecast `l + phi b + s_m` from the current states.
    ///
    /// `seasonal` is the ring of seasonal states with the most recent at `head`, or empty.
    pub(crate) fn forecast(&self, level: f64, slope: f64, seasonal: &[f64], head: usize) -> f64 {
        let trend = if self.trend { self.phi * slope } else { 0.0 };
        level + trend + oldest(seasonal, head).map_or(0.0, |i| seasonal[i])
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
        let oldest_state = oldest_index.map_or(0.0, |i| seasonal[i]);
        let new_level = base + self.alpha * (y - oldest_state - base);
        if self.trend {
            *slope = damped_slope + self.beta / self.alpha * (new_level - *level - damped_slope);
        }

        if let Some(i) = oldest_index {
            seasonal[i] = oldest_state + self.gamma * (y - base - oldest_state);
            *head = i;
        }

        *level = new_level;
    }

    /// Writes the forecast means `l + phi_h b + s` for horizons `1..=mean.len()`, where
    /// `phi_h = phi + ... + phi^h` and `s` is the seasonal state of that horizon.
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
            let season = if m == 0 {
                0.0
            } else {
                seasonal[(head + m - 1 - h % m) % m]
            };
            *value = level + trend + season;
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
