//! Checks the empirical coverage of conformal intervals on simulated ARMA series.
//!
//! There is no statsmodels counterpart, so the tests measure coverage: the share of
//! values inside their interval. The noise is Gaussian, skewed (a centred exponential) or
//! heavy-tailed (Student's t with 3 degrees of freedom), all with unit variance.

use backshift::{ArimaSpec, Conformal, Filter};

/// The coverage target.
const LEVEL: f64 = 0.9;

/// The forecast horizons.
const HORIZON: usize = 3;

/// The calibration window per horizon.
const WINDOW: usize = 250;

/// The steps simulated after the calibration windows are full.
const STEPS: usize = 4000;

/// A small xorshift generator, so the tests need no extra dependency.
struct Rng(u64);

impl Rng {
    /// A uniform value in `(0, 1)`.
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    /// A standard normal value (Box–Muller).
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// The noise distributions, each with mean 0 and variance 1.
#[derive(Debug, Clone, Copy)]
enum Noise {
    /// Standard normal.
    Gaussian,
    /// `Exp(1) - 1`: skewness 2.
    Skewed,
    /// Student's t with 3 degrees of freedom, divided by `sqrt(3)`.
    HeavyTailed,
}

impl Noise {
    /// One draw.
    fn draw(self, rng: &mut Rng) -> f64 {
        match self {
            Noise::Gaussian => rng.normal(),
            Noise::Skewed => -rng.uniform().ln() - 1.0,
            Noise::HeavyTailed => {
                let chi2: f64 = (0..3).map(|_| rng.normal().powi(2)).sum();
                rng.normal() / (chi2 / 3.0).sqrt() / 3.0_f64.sqrt()
            }
        }
    }
}

/// The conformal and Gaussian intervals made at one time step, one entry per horizon.
struct Bounds {
    /// The conformal lower bounds.
    lower: [f64; HORIZON],
    /// The conformal upper bounds.
    upper: [f64; HORIZON],
    /// The Gaussian lower bounds.
    g_lower: [f64; HORIZON],
    /// The Gaussian upper bounds.
    g_upper: [f64; HORIZON],
}

/// The coverage per horizon of conformal and of Gaussian intervals on an ARMA(1, 1)
/// series with the given noise, filtered with its true coefficients.
fn coverage(noise: Noise, seed: u64) -> ([f64; HORIZON], [f64; HORIZON]) {
    let spec = ArimaSpec {
        ar: &[0.6],
        ma: &[0.3],
        ..ArimaSpec::default()
    };
    let mut filter = Filter::<2>::new(&spec).unwrap();
    let mut conformal = Conformal::<HORIZON, WINDOW>::new();
    let mut rng = Rng(seed);
    let (mut y, mut e) = (0.0, 0.0);

    // Each entry holds the intervals made at one time step.
    let mut pending: Vec<Bounds> = Vec::new();
    let (mut hits, mut gaussian_hits) = ([0_usize; HORIZON], [0_usize; HORIZON]);
    let warm_up = WINDOW + HORIZON + 50;
    for t in 0..warm_up + STEPS {
        let next = noise.draw(&mut rng);
        y = 0.6 * y + next + 0.3 * e;
        e = next;

        if t >= warm_up + HORIZON {
            for h in 0..HORIZON {
                let made = &pending[pending.len() - 1 - h];
                hits[h] += usize::from(made.lower[h] <= y && y <= made.upper[h]);
                gaussian_hits[h] += usize::from(made.g_lower[h] <= y && y <= made.g_upper[h]);
            }
        }

        conformal.observe(y);
        filter.update(y);
        let (mut mean, mut lower, mut upper) = ([0.0; HORIZON], [0.0; HORIZON], [0.0; HORIZON]);
        let (mut g_lower, mut g_upper) = ([0.0; HORIZON], [0.0; HORIZON]);
        filter.forecast_interval_into(LEVEL, &mut mean, &mut g_lower, &mut g_upper);
        conformal.record(&mean);
        conformal.interval_into(LEVEL, &mean, &mut lower, &mut upper);
        pending.push(Bounds {
            lower,
            upper,
            g_lower,
            g_upper,
        });
    }

    let scored = (STEPS - HORIZON) as f64;
    (
        hits.map(|h| h as f64 / scored),
        gaussian_hits.map(|h| h as f64 / scored),
    )
}

#[test]
fn conformal_intervals_reach_their_coverage() {
    for (seed, noise) in [
        (1, Noise::Gaussian),
        (2, Noise::Skewed),
        (3, Noise::HeavyTailed),
    ] {
        let (conformal, gaussian) = coverage(noise, seed);
        println!("{noise:?}: conformal {conformal:.3?}, Gaussian {gaussian:.3?}");
        for (h, c) in conformal.iter().enumerate() {
            assert!((c - LEVEL).abs() < 0.03, "{noise:?} h = {}: {c}", h + 1);
        }
    }
}
