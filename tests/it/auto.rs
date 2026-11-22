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
    // (case, d, D) where the generating model leaves no doubt.
    let expected = [
        ("WHITE", 0, 0),
        ("I1", 1, 0),
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
fn ndiffs_matches_pmdarima() {
    for (name, case) in crate::auto_data::NAMES.iter().zip(ALL) {
        assert_eq!(
            backshift::ndiffs(case.values, 0.05, 2),
            case.ndiffs,
            "{name}"
        );
    }
}
