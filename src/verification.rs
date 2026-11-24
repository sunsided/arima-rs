//! Kani proofs for the heap-free device path: run with `cargo kani`.
//!
//! The filters run on microcontrollers, where a panic is a reset. These proofs check
//! for every model shape within small bounds (and every period) that building a filter
//! either succeeds or returns an error, and never panics or overflows. Coefficient
//! values are fixed: the proofs are about indexing and sizes, which the values do not
//! affect, and symbolic floating point would make them intractable.

use crate::filter::{ArimaSpec, Seasonal};
use crate::fourier::fourier_columns;
use crate::polynomial;

/// Coefficients to slice the model's parts from; their values do not matter here.
const COEFFICIENTS: [f64; 2] = [0.3, -0.2];

/// A slice of up to two coefficients, its length chosen by Kani.
fn part() -> &'static [f64] {
    let len: usize = kani::any();
    kani::assume(len <= COEFFICIENTS.len());
    &COEFFICIENTS[..len]
}

/// A model of any orders up to two, differencing up to two, and any period.
fn any_spec() -> ArimaSpec<'static> {
    let (d, seasonal_d): (usize, usize) = (kani::any(), kani::any());
    kani::assume(d <= 2 && seasonal_d <= 2);
    ArimaSpec {
        ar: part(),
        d,
        ma: part(),
        seasonal: Seasonal {
            ar: part(),
            d: seasonal_d,
            ma: part(),
            period: kani::any(),
        },
        ..ArimaSpec::default()
    }
}

/// What `Filter::new` does before the starting covariance: a valid spec that passes the
/// capacity check expands into the filter's arrays without going out of bounds.
#[kani::proof]
#[kani::unwind(10)]
fn filter_expansion_stays_in_bounds() {
    const R: usize = 8;
    let spec = any_spec();
    if spec.validate().is_err() || spec.states() > R {
        return;
    }

    let (mut phi, mut selection, mut delta) = ([0.0; R], [0.0; R], [0.0; R]);
    spec.expand_arma_into(&mut phi, &mut selection);
    polynomial::differencing(spec.d, spec.seasonal.d, spec.seasonal.period, &mut delta);
}

/// The differencing polynomial starts with 1 and vanishes at `B = 1`: differencing
/// removes a constant level.
#[kani::proof]
#[kani::unwind(10)]
fn differencing_removes_a_level() {
    let (d, seasonal_d, period): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    kani::assume(d <= 2 && seasonal_d <= 1 && (1..=4).contains(&period));
    let mut delta = [0.0; 7];
    let k = polynomial::differencing(d, seasonal_d, period, &mut delta);
    assert_eq!(k, d + seasonal_d * period);
    assert_eq!(delta[0], 1.0);
    let sum: f64 = delta[..=k].iter().sum();
    assert!(k == 0 || sum == 0.0);
}

/// `fourier_columns` never overflows: at most `2k` columns for any `k` and period.
#[kani::proof]
fn fourier_columns_never_overflow() {
    let (period, k): (f64, usize) = (kani::any(), kani::any());
    let columns = fourier_columns(period, k);
    assert!(columns <= k.saturating_mul(2));
}
