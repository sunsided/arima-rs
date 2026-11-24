//! Checks the Theta method against R's `forecast::thetaf` in [`crate::r_theta_data`].

use crate::r_theta_data::{ALL, ThetaCase};
use backshift::{Theta, ThetaFilter, ThetaSpec};

/// R's forecast state for `case`.
fn r_spec(case: &ThetaCase) -> ThetaSpec<'_> {
    ThetaSpec {
        alpha: case.alpha,
        drift: case.drift,
        level: case.level,
        count: case.values.len(),
        sigma2: case.sigma2,
        seasonal: case.factors,
    }
}

/// Asserts `|a - b| <= tolerance * max(1, |b|)`.
fn assert_close(a: f64, b: f64, tolerance: f64, what: &str) {
    assert!(
        (a - b).abs() <= tolerance * b.abs().max(1.0),
        "{what}: {a} vs R's {b}"
    );
}

#[test]
fn filter_reproduces_thetaf_at_rs_state() {
    for case in ALL {
        let filter = ThetaFilter::<12>::new(&r_spec(case)).unwrap();
        let h = case.mean.len();
        for (level, (lo, up)) in [(0.8, (0, 1)), (0.95, (2, 3))] {
            let (mut mean, mut lower, mut upper) = (vec![0.0; h], vec![0.0; h], vec![0.0; h]);
            filter.forecast_interval_into(level, &mut mean, &mut lower, &mut upper);
            for i in 0..h {
                assert_close(mean[i], case.mean[i], 1e-9, "mean");
                assert_close(lower[i], case.intervals[i][lo], 1e-9, "lower");
                assert_close(upper[i], case.intervals[i][up], 1e-9, "upper");
            }
        }
    }
}

#[test]
fn fit_makes_rs_seasonal_decision_and_forecasts() {
    for case in ALL {
        let fit = Theta::new().seasonal(case.period).fit(case.values).unwrap();
        assert_eq!(
            !fit.factors.is_empty(),
            case.seasonal,
            "period {}",
            case.period
        );
        assert_close(fit.drift, case.drift, 1e-9, "drift");
        for (ours, r) in fit.factors.iter().zip(case.factors) {
            assert_close(*ours, *r, 1e-9, "seasonal index");
        }

        // The smoothing's optimum can sit slightly off R's Nelder-Mead stop.
        let filter = ThetaFilter::<12>::new(&fit.spec()).unwrap();
        let mut mean = vec![0.0; case.mean.len()];
        filter.forecast_into(&mut mean, &mut vec![0.0; case.mean.len()]);
        for (ours, r) in mean.iter().zip(case.mean) {
            assert_close(*ours, *r, 5e-3, "forecast");
        }
    }
}

#[test]
fn streaming_continues_the_level() {
    // One more value, exactly at the forecast, keeps the level on the forecast path.
    let case = ALL[0];
    let mut filter = ThetaFilter::<1>::new(&r_spec(case)).unwrap();
    let (mut mean, mut variance) = ([0.0; 2], [0.0; 2]);
    filter.forecast_into(&mut mean, &mut variance);
    let innovation = filter.update(mean[0]);
    assert!(innovation.value.abs() < 1e-12);
    assert!(filter.update(f64::NAN).value.is_nan());
}
