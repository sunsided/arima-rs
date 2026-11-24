//! Checks the MSTL decomposition against statsmodels.

use crate::mstl_data::ALL;
use backshift::mstl;

/// Absolute tolerance against statsmodels, as for STL.
const TOLERANCE: f64 = 1e-9;

/// Asserts two equally long slices agree within [`TOLERANCE`].
fn assert_close(what: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (t, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= TOLERANCE, "{what}[{t}]: {a} vs {e}");
    }
}

#[test]
fn matches_statsmodels() {
    for case in ALL {
        let parts = mstl(case.values, case.periods, case.iterate).unwrap();
        assert_eq!(parts.seasonal.len(), case.kept);
        let n = case.values.len();
        for (i, component) in parts.seasonal.iter().enumerate() {
            let what = format!("periods {:?} seasonal {i}", case.periods);
            assert_close(&what, component, &case.seasonal[i * n..(i + 1) * n]);
        }

        assert_close("trend", &parts.trend, case.trend);
        assert_close("remainder", &parts.remainder, case.remainder);
    }
}

#[test]
fn rejects_what_statsmodels_cannot_decompose() {
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert!(mstl(&y, &[4], 2).is_none(), "4 is half the length");
    assert!(mstl(&y, &[1], 2).is_none());
    assert!(mstl(&y, &[2], 0).is_none());
}
