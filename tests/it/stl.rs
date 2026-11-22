//! Checks the STL decomposition against statsmodels in [`crate::stl_data`].

use crate::stl_data::ALL;
use backshift::stl;

#[test]
fn stl_matches_statsmodels() {
    for case in ALL {
        let parts = stl(case.values, case.period).unwrap();
        for (what, actual, expected) in [
            ("seasonal", &parts.seasonal, case.seasonal),
            ("trend", &parts.trend, case.trend),
        ] {
            assert_eq!(actual.len(), expected.len());
            for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                    "period {} {what}[{t}]: got {a}, statsmodels {e}",
                    case.period
                );
            }
        }

        for (t, r) in parts.remainder.iter().enumerate() {
            assert!((r - (case.values[t] - parts.seasonal[t] - parts.trend[t])).abs() < 1e-12);
        }
    }
}
