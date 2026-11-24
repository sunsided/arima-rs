//! Checks the ETS models against R's `forecast::ets` in [`crate::r_ets_data`].
//!
//! The filter runs at R's fitted parameters and initial states, so its residuals, states,
//! likelihood and forecasts must match R's closely; fits are compared on the likelihood
//! they reach, since R's Nelder–Mead and this crate's BFGS stop at different points.

use crate::r_ets_data::{ALL, AUTO_ADDITIVE, EtsCase};
use backshift::{AutoEts, Ets, EtsFilter, EtsSpec, Trend};

/// Relative tolerance for values that only depend on the recursion.
const TIGHT: f64 = 1e-9;

/// The seasonal capacity of the filters in these tests.
const CAPACITY: usize = 12;

/// Asserts `|a - b| <= tolerance * max(1, |b|)`.
fn assert_close(a: f64, b: f64, tolerance: f64, what: &str) {
    assert!(
        (a - b).abs() <= tolerance * b.abs().max(1.0),
        "{what}: {a} vs R's {b}"
    );
}

/// The trend of R's model string, with R's fitted parameters.
fn trend(case: &EtsCase) -> Trend {
    let [_, beta, _, phi] = case.parameters;
    match &case.model[1..] {
        m if m.starts_with("Ad") => Trend::Damped { beta, phi },
        m if m.starts_with('A') => Trend::Additive { beta },
        _ => Trend::None,
    }
}

/// The number of level and slope states before the seasonal ones.
fn leading_states(case: &EtsCase) -> usize {
    if matches!(trend(case), Trend::None) {
        1
    } else {
        2
    }
}

/// R's model at its fitted parameters and the given states.
fn spec<'a>(case: &EtsCase, states: &'a [f64], sigma2: f64) -> EtsSpec<'a> {
    let lead = leading_states(case);
    let [alpha, _, gamma, _] = case.parameters;
    EtsSpec {
        alpha,
        trend: trend(case),
        gamma: if gamma.is_nan() { 0.0 } else { gamma },
        level: states[0],
        slope: if lead == 2 { states[1] } else { 0.0 },
        seasonal: &states[lead..],
        sigma2,
    }
}

/// Whether R's `forecast:::class1` misplaces `gamma` for this model.
///
/// It sets `G[3]`, the third state, to `gamma`. That is the first seasonal state only when the
/// model has a slope state; without one it is the second, so R adds `gamma^2` at horizons
/// `j = m - 1 (mod m)` instead of `j = 0 (mod m)`.
fn r_misplaces_gamma(case: &EtsCase) -> bool {
    case.period > 1 && matches!(trend(case), Trend::None)
}

#[test]
fn filter_reproduces_r_at_its_parameters() {
    for case in ALL {
        let mut filter =
            EtsFilter::<CAPACITY>::new(&spec(case, case.initial_states, case.sigma2)).unwrap();
        for (&y, &residual) in case.values.iter().zip(case.residuals) {
            let innovation = filter.update(y);
            assert_close(innovation.value, residual, TIGHT, case.model);
        }

        let end = EtsFilter::<CAPACITY>::new(&spec(case, case.final_states, case.sigma2)).unwrap();
        assert_close(filter.level(), end.level(), TIGHT, case.model);
    }
}

#[test]
fn forecasts_and_intervals_match_r() {
    for case in ALL {
        let filter =
            EtsFilter::<CAPACITY>::new(&spec(case, case.final_states, case.sigma2)).unwrap();
        let h = case.mean.len();
        let (mut mean, mut lower, mut upper) = (vec![0.0; h], vec![0.0; h], vec![0.0; h]);
        filter.forecast_interval_into(0.95, &mut mean, &mut lower, &mut upper);
        for i in 0..h {
            assert_close(mean[i], case.mean[i], TIGHT, case.model);
        }

        // R's class-1 variance puts `gamma` on the wrong state for seasonal models without a
        // trend; `seasonal_variance_without_trend_matches_simulation` checks those instead.
        if r_misplaces_gamma(case) {
            continue;
        }

        for i in 0..h {
            assert_close(lower[i], case.intervals[i][2], TIGHT, case.model);
            assert_close(upper[i], case.intervals[i][3], TIGHT, case.model);
        }

        filter.forecast_interval_into(0.8, &mut mean, &mut lower, &mut upper);
        for i in 0..h {
            assert_close(lower[i], case.intervals[i][0], TIGHT, case.model);
            assert_close(upper[i], case.intervals[i][1], TIGHT, case.model);
        }
    }
}

/// Standard normal draws from a fixed linear congruential generator (Box–Muller).
fn normals(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 53 bits as a float in (0, 1).
        (f64::from(u32::try_from(state >> 43).expect("21 bits")) * 2_f64.powi(32)
            + f64::from(u32::try_from((state >> 11) & 0xFFFF_FFFF).expect("32 bits"))
            + 0.5)
            / 2_f64.powi(53)
    };
    move || (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()
}

#[test]
fn seasonal_variance_without_trend_matches_simulation() {
    // ETS(A,N,A) with a large gamma, where the correct variance at horizon 4 (1.27) and R's
    // (1.82, see `r_misplaces_gamma`) are far apart.
    let seasonal = [1.0, -1.0, 2.0, -2.0];
    let spec = EtsSpec {
        alpha: 0.3,
        gamma: 0.5,
        level: 10.0,
        seasonal: &seasonal,
        ..EtsSpec::default()
    };
    let start = EtsFilter::<4>::new(&spec).unwrap();
    let (mut mean, mut variance) = ([0.0; 6], [0.0; 6]);
    start.forecast_into(&mut mean, &mut variance);
    assert_close(variance[3], 1.27, 1e-12, "formula at h = 4");

    let paths = 200_000;
    let mut draw = normals(7);
    let mut sums = [0.0; 6];
    for _ in 0..paths {
        let mut filter = start;
        let mut probe = [0.0; 1];
        for (h, sum) in sums.iter_mut().enumerate() {
            filter.forecast_into(&mut probe, &mut [0.0]);
            let y = probe[0] + draw();
            let error = y - mean[h];
            *sum += error * error;
            filter.update(y);
        }
    }

    for (h, sum) in sums.iter().enumerate() {
        let empirical = sum / f64::from(paths);
        assert_close(empirical, variance[h], 0.02, "simulated variance");
    }
}

/// The model of `case` as this crate builds it.
fn model(case: &EtsCase) -> Ets {
    let model = match trend(case) {
        Trend::None => Ets::new(),
        Trend::Additive { .. } => Ets::new().trend(),
        Trend::Damped { .. } => Ets::new().damped(),
    };
    model.seasonal(case.period)
}

/// R's log-likelihood with its dropped constants restored: `n (ln(2 pi) + 1 - ln n) / 2`
/// less than R's.
fn full_loglik(case: &EtsCase) -> f64 {
    let n = f64::from(u32::try_from(case.values.len()).expect("short series"));
    case.loglik - 0.5 * n * ((std::f64::consts::TAU).ln() + 1.0 - n.ln())
}

#[test]
fn fits_reach_rs_likelihood() {
    for case in ALL {
        let fit = model(case).fit(case.values).unwrap();
        let r = full_loglik(case);
        assert!(
            fit.loglike >= r - 1e-4,
            "{}: {} vs R's {r}",
            case.model,
            fit.loglike
        );
        // The criteria differ from R's by the same constant as the likelihood.
        let offset = 2.0 * (r - case.loglik);
        if (fit.loglike - r).abs() < 1e-6 {
            assert_close(fit.aicc(), case.criteria[1] - offset, 1e-6, "AICc");
        }
    }
}

#[test]
fn automatic_search_picks_rs_model() {
    for (case, expected) in ALL.iter().zip(AUTO_ADDITIVE) {
        let best = AutoEts::new()
            .seasonal(case.period)
            .fit(case.values)
            .unwrap();
        let label = best.label().replace(['(', ')', ','], "").replace("ETS", "");
        assert_eq!(&label, expected, "{} series", case.model);
    }
}
