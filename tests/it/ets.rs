//! Checks the ETS models against R's `forecast::ets` in [`crate::r_ets_data`].
//!
//! The filter runs at R's fitted parameters and initial states, so its residuals, states,
//! likelihood and forecasts must match R's closely; fits are compared on the likelihood
//! they reach, since R's Nelder–Mead and this crate's BFGS stop at different points.

use crate::r_ets_data::{ALL, AUTO_ADDITIVE, AUTO_DEFAULT, EtsCase};
use backshift::{AutoEts, Ets, EtsFilter, EtsSpec, Form, Trend};

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

/// The form R's model string gives a component: `M` is multiplicative, `A` or `N` additive.
fn form(letter: u8) -> Form {
    if letter == b'M' {
        Form::Multiplicative
    } else {
        Form::Additive
    }
}

/// Whether R's model has a multiplicative error.
fn multiplicative_error(case: &EtsCase) -> bool {
    case.model.starts_with('M')
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
        error: form(case.model.as_bytes()[0]),
        season: form(*case.model.as_bytes().last().expect("model string")),
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
    case.period > 1 && matches!(trend(case), Trend::None) && !case.model.ends_with('M')
}

/// Whether R's `forecast:::class3` ignores the damping in the level: it uses
/// `F_1 = [[1, 1], [0, phi]]` and the weights `(1, 1)` where the damped model needs `(1, phi)`
/// in both, so its ETS(M,Ad,M) means and variances are off.
fn r_ignores_damping(case: &EtsCase) -> bool {
    case.model == "MAdM"
}

#[test]
fn filter_reproduces_r_at_its_parameters() {
    for case in ALL {
        let mut filter =
            EtsFilter::<CAPACITY>::new(&spec(case, case.initial_states, case.sigma2)).unwrap();
        for (&y, &residual) in case.values.iter().zip(case.residuals) {
            let innovation = filter.update(y);
            // R reports relative errors for a multiplicative error: `(y - f) / f`.
            let value = if multiplicative_error(case) {
                innovation.value / (y - innovation.value)
            } else {
                innovation.value
            };
            assert_close(value, residual, TIGHT, case.model);
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
        let mut scratch = vec![0.0; filter.forecast_scratch_len()];
        filter.forecast_interval_with(&mut scratch, 0.95, &mut mean, &mut lower, &mut upper);
        // R's class-3 means are off for a damped trend; `damped_class3_matches_simulation`
        // checks that model instead.
        if r_ignores_damping(case) {
            continue;
        }

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

        filter.forecast_interval_with(&mut scratch, 0.8, &mut mean, &mut lower, &mut upper);
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

/// Checks the forecast means and variances of `spec` against `paths` simulated futures of
/// the model itself, within `tolerance` relative to each variance.
fn matches_simulation(spec: &EtsSpec<'_>, paths: u32, tolerance: f64) {
    let start = EtsFilter::<CAPACITY>::new(spec).unwrap();
    let h = 6;
    let (mut mean, mut variance) = (vec![0.0; h], vec![0.0; h]);
    let mut scratch = vec![0.0; start.forecast_scratch_len()];
    start.forecast_with(&mut scratch, &mut mean, &mut variance);

    let mut draw = normals(7);
    let (mut sums, mut squares) = (vec![0.0; h], vec![0.0; h]);
    let sigma = spec.sigma2.sqrt();
    for _ in 0..paths {
        let mut filter = start;
        for (sum, square) in sums.iter_mut().zip(&mut squares) {
            let (mut point, mut ignored) = ([0.0; 1], [0.0; 1]);
            filter.forecast_into(&mut point, &mut ignored);
            let shock = sigma * draw();
            let y = match spec.error {
                Form::Additive => point[0] + shock,
                Form::Multiplicative => point[0] * (1.0 + shock),
            };
            *sum += y;
            *square += y * y;
            filter.update(y);
        }
    }

    let n = f64::from(paths);
    for i in 0..h {
        let empirical_mean = sums[i] / n;
        let empirical_variance = squares[i] / n - empirical_mean * empirical_mean;
        let spread = variance[i].sqrt();
        assert!(
            (empirical_mean - mean[i]).abs() < 4.0 * spread / n.sqrt(),
            "mean at h = {}: {empirical_mean} vs {}",
            i + 1,
            mean[i]
        );
        assert_close(
            empirical_variance,
            variance[i],
            tolerance,
            "simulated variance",
        );
    }
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
    let filter = EtsFilter::<4>::new(&spec).unwrap();
    let (mut mean, mut variance) = ([0.0; 6], [0.0; 6]);
    filter.forecast_into(&mut mean, &mut variance);
    assert_close(variance[3], 1.27, 1e-12, "formula at h = 4");
    matches_simulation(&spec, 200_000, 0.02);
}

#[test]
fn relative_variance_without_trend_matches_simulation() {
    // ETS(M,N,A): R's class-2 variance inherits the misplaced `gamma` of class 1.
    let seasonal = [3.0, -2.0, 4.0, -5.0];
    let spec = EtsSpec {
        error: Form::Multiplicative,
        alpha: 0.3,
        gamma: 0.5,
        level: 100.0,
        seasonal: &seasonal,
        sigma2: 0.01,
        ..EtsSpec::default()
    };
    matches_simulation(&spec, 200_000, 0.02);
}

#[test]
fn damped_class3_matches_simulation() {
    // ETS(M,Ad,M): R's class 3 drops `phi` from the level update (`r_ignores_damping`).
    let seasonal = [1.1, 0.95, 1.05, 0.9];
    let spec = EtsSpec {
        error: Form::Multiplicative,
        alpha: 0.4,
        trend: Trend::Damped {
            beta: 0.1,
            phi: 0.85,
        },
        season: Form::Multiplicative,
        gamma: 0.3,
        level: 100.0,
        slope: 5.0,
        seasonal: &seasonal,
        sigma2: 0.004,
    };
    matches_simulation(&spec, 200_000, 0.02);
}

/// The model of `case` as this crate builds it.
fn model(case: &EtsCase) -> Ets {
    let model = match trend(case) {
        Trend::None => Ets::new(),
        Trend::Additive { .. } => Ets::new().trend(),
        Trend::Damped { .. } => Ets::new().damped(),
    };
    model
        .error(form(case.model.as_bytes()[0]))
        .seasonal(case.period)
        .season(form(*case.model.as_bytes().last().expect("model string")))
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

/// R's model string of a fitted label: `ETS(M,Ad,M)` becomes `MAdM`.
fn r_string(label: &str) -> String {
    label.replace(['(', ')', ','], "").replace("ETS", "")
}

/// Checks that `search` picks R's choice on every series, or a model that scores at least as
/// well by `AICc`: this crate fits each candidate at least as well as R, so a better fit of
/// another model can overtake R's choice, as it should.
fn matches_or_beats_r(search: impl Fn(&EtsCase) -> AutoEts, choices: &[&str]) {
    for (case, expected) in ALL.iter().zip(choices) {
        let best = search(case).fit(case.values).unwrap();
        let ours = r_string(best.label());
        if &ours == expected {
            continue;
        }

        let r_choice = refit(case, expected);
        assert!(
            best.aicc() <= r_choice + 1e-6,
            "{} series: ours {ours} ({}) vs R's {expected} ({r_choice})",
            case.model,
            best.aicc()
        );
    }
}

#[test]
fn additive_search_matches_or_beats_rs_choice() {
    matches_or_beats_r(
        |case| AutoEts::new().seasonal(case.period).additive_only(true),
        AUTO_ADDITIVE,
    );
}

#[test]
fn default_search_matches_or_beats_rs_choice() {
    matches_or_beats_r(|case| AutoEts::new().seasonal(case.period), AUTO_DEFAULT);
}

/// This crate's `AICc` of R's model string `model` on the series of `case`.
fn refit(case: &EtsCase, model: &str) -> f64 {
    let bytes = model.as_bytes();
    let base = match &model[1..] {
        m if m.starts_with("Ad") => Ets::new().damped(),
        m if m.starts_with('A') => Ets::new().trend(),
        _ => Ets::new(),
    };
    let season = bytes[bytes.len() - 1];
    let model = base.error(form(bytes[0]));
    let model = if season == b'N' {
        model
    } else {
        model.seasonal(case.period).season(form(season))
    };
    model
        .fit(case.values)
        .map_or(f64::INFINITY, |fit| fit.aicc())
}
