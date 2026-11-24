//! Checks the accuracy measures and rolling-origin errors against R's `accuracy` and
//! `tsCV` in [`crate::r_accuracy_data`], and the family selection on simulated series.

use crate::r_accuracy_data::{ACCURACY, CROSS_VALIDATION};
use backshift::{Family, accuracy, cross_validation_errors, select_family};

/// A deterministic monthly series: trend, a strong season and small noise.
fn seasonal_series(n: usize) -> Vec<f64> {
    let mut state = 7_u64;
    (0..n)
        .map(|t| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let t = t as f64;
            50.0 + 0.3 * t + 10.0 * (core::f64::consts::TAU * t / 12.0).sin() + noise
        })
        .collect()
}

#[test]
fn accuracy_matches_r() {
    let case = &ACCURACY;
    let scores = accuracy(case.forecast, case.test, case.training, case.period);
    let ours = [
        scores.me,
        scores.rmse,
        scores.mae,
        scores.mpe,
        scores.mape,
        scores.mase,
    ];
    for (ours, r) in ours.iter().zip(case.metrics) {
        assert!(
            (ours - r).abs() < 1e-12 * r.abs().max(1.0),
            "{ours} vs R {r}"
        );
    }
}

#[test]
fn cross_validation_matches_tscv() {
    let case = &CROSS_VALIDATION;
    let naive = |history: &[f64], h: usize| history.last().map(|&last| vec![last; h]);
    let errors = cross_validation_errors(case.values, 3, 10, naive);
    assert_eq!(errors.len(), case.errors.len());
    for (t, (ours, r)) in errors.iter().zip(case.errors).enumerate() {
        assert_eq!(ours.is_nan(), r.is_nan(), "NaN pattern differs at cell {t}");
        if r.is_finite() {
            assert!((ours - r).abs() < 1e-12, "cell {t}: {ours} vs R {r}");
        }
    }
}

#[test]
fn failed_forecasts_leave_nan_rows() {
    let y = [1.0, 2.0, 3.0, 4.0];
    let errors = cross_validation_errors(&y, 1, 0, |history: &[f64], _| {
        (history.len() != 2).then(|| vec![0.0])
    });
    assert_eq!(&errors[..1], &[2.0]);
    assert!(errors[1].is_nan() && errors[3].is_nan());
    assert_eq!(errors[2], 4.0);
}

#[test]
fn selection_picks_the_lowest_score_and_refits() {
    let y = seasonal_series(72);
    let choice = select_family(&y, 12, 6, 2).unwrap();

    // The winner has the lowest finite score.
    let best = choice
        .scores
        .iter()
        .filter(|(_, score)| score.is_finite())
        .map(|&(_, score)| score)
        .fold(f64::INFINITY, f64::min);
    let winner = choice
        .scores
        .iter()
        .find(|(f, _)| *f == choice.family)
        .unwrap()
        .1;
    assert_eq!(winner, best);

    // A strong season makes the plain naive forecast lose to a seasonal family.
    assert_ne!(choice.family, Family::Naive);

    // The seasonal naive score recomputed by hand from the two origins.
    let manual: f64 = [12, 6]
        .iter()
        .map(|&back| {
            let (training, rest) = y.split_at(y.len() - back);
            let forecast = &training[training.len() - 12..][..6];
            accuracy(forecast, &rest[..6], training, 12).mase
        })
        .sum::<f64>()
        / 2.0;
    let seasonal_naive = choice.scores[4];
    assert_eq!(seasonal_naive.0, Family::SeasonalNaive);
    assert!((seasonal_naive.1 - manual).abs() < 1e-12);
    assert_eq!(choice.forecast.len(), 6);
    assert!(choice.forecast.iter().all(|v| v.is_finite()));
}

#[test]
fn selection_rejects_bad_input() {
    use backshift::FitError;

    let y = seasonal_series(20);
    assert_eq!(select_family(&y, 1, 0, 2), Err(FitError::Unsupported));
    assert_eq!(select_family(&y, 1, 3, 0), Err(FitError::Unsupported));
    assert_eq!(
        select_family(&y, 1, 10, 2),
        Err(FitError::TooShort {
            needed: 22,
            present: 20
        })
    );
}
