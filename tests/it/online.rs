//! Compares the online AR predictor with the exact filter on series whose dynamics change.
//!
//! The settings follow Anava et al. (2013), section 5: an ARMA(5, 2) with uniform noise on
//! `[-0.5, 0.5]` (variance 1/12, the floor of any one-step squared error), whose AR
//! coefficients either drift linearly or switch halfway.

use backshift::{Arima, ArimaSpec, Filter, OnlineAr, OnlineArOptions, RefitOptions};

/// The length of each series.
const LENGTH: usize = 10_000;

/// The values used to fit the exact model before streaming starts.
const TRAIN: usize = 1_000;

/// How often the refitted filter re-estimates, and on how many recent values.
const REFIT_EVERY: usize = 500;

/// The AR coefficients of the first regime.
const START: [f64; 5] = [0.6, -0.5, 0.4, -0.4, 0.3];

/// The AR coefficients of the second regime.
const END: [f64; 5] = [-0.4, -0.5, 0.4, 0.4, 0.1];

/// How the AR coefficients change over time.
#[derive(Debug, Clone, Copy)]
enum Change {
    /// Linearly from [`START`] to [`END`] (setting 2, MA `[0.32, -0.2]`).
    Drift,
    /// From [`START`] to [`END`] halfway, with MA `[0.3, -0.2]` then `[-0.3, 0.2]` (setting 3).
    Switch,
}

/// A simulated series with the given change, from the given seed.
fn series(change: Change, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
    };
    let (mut y, mut e) = (vec![0.0; LENGTH], vec![0.0; LENGTH]);
    for t in 0..LENGTH {
        let share = t as f64 / LENGTH as f64;
        let (alpha, beta): ([f64; 5], [f64; 2]) = match change {
            Change::Drift => (
                core::array::from_fn(|i| START[i] * (1.0 - share) + END[i] * share),
                [0.32, -0.2],
            ),
            Change::Switch if t < LENGTH / 2 => (START, [0.3, -0.2]),
            Change::Switch => (END, [-0.3, 0.2]),
        };
        e[t] = uniform();
        y[t] = e[t];
        for (i, a) in alpha.iter().enumerate() {
            y[t] += a * t.checked_sub(i + 1).map_or(0.0, |s| y[s]);
        }
        for (i, b) in beta.iter().enumerate() {
            y[t] += b * t.checked_sub(i + 1).map_or(0.0, |s| e[s]);
        }
    }

    y
}

/// The mean squared one-step error of an online AR(10) over `y[TRAIN..]`, after it has
/// also seen `y[..TRAIN]`.
fn online_mse(y: &[f64], options: OnlineArOptions) -> f64 {
    let mut ar = OnlineAr::<10>::new(options).unwrap();
    for &value in &y[..TRAIN] {
        ar.update(value);
    }

    mean_square(y[TRAIN..].iter().map(|&value| ar.update(value)))
}

/// The mean squared one-step error of the exact ARMA(5, 2) filter fitted on `y[..TRAIN]`,
/// refitted every [`REFIT_EVERY`] values on the latest ones if `refit`.
fn filter_mse(y: &[f64], refit: bool) -> f64 {
    let fitted = Arima::new(5, 0, 2).mean(false).fit(&y[..TRAIN]).unwrap();
    let (mut ar, mut ma, mut sigma2) = (fitted.ar.clone(), fitted.ma.clone(), fitted.sigma2);
    let spec_of = |ar: &[f64], ma: &[f64], sigma2: f64| ArimaSpec {
        ar: ar.to_vec().leak(),
        ma: ma.to_vec().leak(),
        sigma2,
        ..ArimaSpec::default()
    };
    let options = RefitOptions::default();
    let shape = spec_of(&ar, &ma, sigma2);
    let mut coefficients = vec![0.0; shape.coefficient_len()];
    let mut scratch = vec![0.0; shape.refit_scratch_len(REFIT_EVERY, options)];
    let mut filter = Filter::<6>::new(&shape).unwrap();
    filter.warm_up(&y[..TRAIN]);

    let mut errors = Vec::with_capacity(y.len() - TRAIN);
    for t in TRAIN..y.len() {
        errors.push(filter.update(y[t]).unwrap().value);
        if refit && (t + 1 - TRAIN) % REFIT_EVERY == 0 {
            let window = &y[t + 1 - REFIT_EVERY..=t];
            let current = spec_of(&ar, &ma, sigma2);
            if let Ok(new) = current.refit(window, options, &mut coefficients, &mut scratch) {
                (ar, ma, sigma2) = (new.spec.ar.to_vec(), new.spec.ma.to_vec(), new.spec.sigma2);
                filter = Filter::<6>::new(&spec_of(&ar, &ma, sigma2)).unwrap();
                filter.warm_up(window);
            }
        }
    }

    mean_square(errors.into_iter())
}

/// The mean of the squares.
fn mean_square(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0_usize), |(s, n), v| (s + v * v, n + 1));
    sum / count as f64
}

/// The noise variance, `1 / 12`: no predictor's mean squared error can go below it.
const FLOOR: f64 = 1.0 / 12.0;

/// The mean squared one-step errors of the online predictor (default options), the stale
/// exact filter and the periodically refitted one.
fn errors(change: Change, seed: u64) -> (f64, f64, f64) {
    let y = series(change, seed);
    let errors = (
        online_mse(&y, OnlineArOptions::default()),
        filter_mse(&y, false),
        filter_mse(&y, true),
    );
    println!(
        "{change:?} seed {seed:#x}: online {:.4}, stale {:.4}, refit {:.4}",
        errors.0, errors.1, errors.2
    );
    errors
}

#[test]
fn online_ar_follows_drifting_coefficients() {
    // The defaults were tuned on the first seed; the second one checks they carry over.
    for seed in [0x9E37_79B9_7F4A_7C15, 0x2545_F491_4F6C_DD1D] {
        let (online, stale, refit) = errors(Change::Drift, seed);
        assert!(online < 1.1 * FLOOR, "online {online}");
        assert!(online < 0.7 * stale, "online {online}, stale {stale}");
        // A refit every 500 values follows a slow drift about as well.
        assert!(online < 1.1 * refit, "online {online}, refit {refit}");
    }
}

#[test]
fn online_ar_recovers_from_a_switch() {
    for seed in [0x9E37_79B9_7F4A_7C15, 0x2545_F491_4F6C_DD1D] {
        let (online, stale, refit) = errors(Change::Switch, seed);
        assert!(online < 1.2 * FLOOR, "online {online}");
        assert!(online < 0.3 * stale, "online {online}, stale {stale}");
        // The refit lags the switch by up to a window; the online predictor does not.
        assert!(online < refit, "online {online}, refit {refit}");
    }
}
