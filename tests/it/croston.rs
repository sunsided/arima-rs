//! Checks Croston's method against R's `forecast::croston` in [`crate::r_croston_data`].

use crate::r_croston_data::ALL;
use backshift::{CrostonFilter, CrostonVariant};

#[test]
fn streaming_matches_every_fitted_value() {
    // R recomputes each fitted value from the values before it; the filter carries the
    // same smoothing forward one value at a time.
    for case in ALL {
        let mut filter = CrostonFilter::new(case.alpha, CrostonVariant::Classic).unwrap();
        for (t, &fitted) in case.fitted.iter().enumerate() {
            let ours = filter.forecast();
            assert!(
                (ours.is_nan() && fitted.is_nan()) || (ours - fitted).abs() < 1e-12,
                "t = {t}: {ours} vs R's {fitted}"
            );
            filter.update(case.values[t]);
        }
        assert!((filter.forecast() - case.forecast).abs() < 1e-12);
    }
}
