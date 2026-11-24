//! Checks the stepwise auto-ARIMA search against pmdarima's choices in [`crate::auto_data`].

use crate::auto_data::ALL;
use backshift::AutoArima;

#[test]
fn stepwise_search_matches_pmdarima() {
    for (index, case) in ALL.iter().enumerate() {
        let search = AutoArima::new().d(case.d).seasonal_d(case.seasonal_d);
        let search = if case.period > 1 {
            search.seasonal(case.period)
        } else {
            search
        };
        let best = search.fit(case.values).unwrap();
        let fitted = &best.fitted;
        let orders = (
            fitted.ar.len(),
            fitted.ma.len(),
            fitted.seasonal_ar.len(),
            fitted.seasonal_ma.len(),
        );

        assert_eq!(orders, case.orders, "case {index}: {fitted:?}");
        // pmdarima also tries a constant when d + D = 2; R and this crate do not.
        if case.d + case.seasonal_d <= 1 {
            assert_eq!(fitted.with_mean, case.constant, "case {index}: constant");
        }
    }
}

#[test]
fn exhaustive_search_is_at_least_as_good_as_stepwise() {
    for case in ALL.iter().filter(|c| c.period == 0).take(4) {
        let search = AutoArima::new().d(case.d).max_orders(2, 2, 0, 0);
        let stepwise = search.fit(case.values).unwrap();
        let exhaustive = search.exhaustive().fit(case.values).unwrap();
        assert!(exhaustive.criterion <= stepwise.criterion + 1e-9);
        assert!(exhaustive.candidates >= stepwise.candidates);
    }
}

#[test]
fn chooses_the_generating_differencing_orders() {
    // (case, d, D) where the generating model leaves no doubt. The I1 series is generated
    // with d = 1, but R's `ndiffs` (KPSS with R's lag) picks d = 2 for it, and so does this
    // crate; see `ndiffs_matches_r_on_the_pmdarima_series`.
    let expected = [
        ("WHITE", 0, 0),
        ("I1", 2, 0),
        ("DRIFT", 1, 0),
        ("AIRLINE12", 1, 1),
    ];
    for (name, d, seasonal_d) in expected {
        let case = ALL
            .iter()
            .zip(crate::auto_data::NAMES)
            .find(|(_, n)| **n == name)
            .unwrap()
            .0;
        let search = if case.period > 1 {
            AutoArima::new().seasonal(case.period)
        } else {
            AutoArima::new()
        };
        let fitted = search.fit(case.values).unwrap().fitted;
        assert_eq!((fitted.d, fitted.seasonal_d), (d, seasonal_d), "{name}");
    }
}

#[test]
fn ndiffs_matches_r_on_the_pmdarima_series() {
    // `ndiffs` follows R's `forecast::ndiffs`, whose KPSS lag `trunc(3 sqrt(n) / 13)` differs
    // from pmdarima's `trunc(4 (n / 100)^(1/4))`. On these series the two disagree twice
    // (ARMA11 and I1); this crate agrees with R on all of them.
    let r = crate::r_data::PMDARIMA_SERIES_NDIFFS;
    let mut disagreements = 0;
    for ((name, case), &expected) in crate::auto_data::NAMES.iter().zip(ALL).zip(r) {
        assert_eq!(backshift::ndiffs(case.values, 0.05, 2), expected, "{name}");
        disagreements += usize::from(case.ndiffs != expected);
    }

    assert_eq!(disagreements, 2);
}

/// A random walk of `n` steps from an LCG seeded with `seed`.
fn random_walk(seed: u32, n: usize) -> Vec<f64> {
    let mut state = seed;
    let mut level = 0.0;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            level += f64::from(state >> 8) / f64::from(1_u32 << 24) - 0.5;
            level
        })
        .collect()
}

#[test]
fn stepwise_drops_the_constant_with_the_null_model() {
    // Once the null model without drift wins, R searches on without drift. With these
    // small bounds that reaches every model, so stepwise must find the exhaustive best.
    for seed in 0..40 {
        let y = random_walk(seed, 150);
        let search = AutoArima::new().d(1).max_orders(1, 1, 0, 0);
        let stepwise = search.fit(&y).unwrap().fitted.aicc();
        let exhaustive = search.exhaustive().fit(&y).unwrap().fitted.aicc();
        assert!(
            stepwise <= exhaustive + 1e-9,
            "seed {seed}: stepwise {stepwise}, exhaustive {exhaustive}"
        );
    }
}
