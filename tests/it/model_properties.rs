//! Property tests of the model layer: the heap-free filter against the runtime-sized
//! likelihood, fits against their own reported likelihood, and no panic on any input.

use backshift::{Arima, ArimaSpec, AutoArima, Filter, Seasonal, innovations, loglike};
use proptest::prelude::*;

/// The state capacity of the filters below; enough for every model the strategies build.
const CAPACITY: usize = 32;

/// Coefficients of a stationary AR polynomial (or, negated, an invertible MA one) from
/// partial autocorrelations in `(-0.9, 0.9)`, by the Durbin–Levinson step-up.
fn step_up(pacf: &[f64]) -> Vec<f64> {
    let mut phi: Vec<f64> = Vec::with_capacity(pacf.len());
    for (k, &r) in pacf.iter().enumerate() {
        let previous = phi.clone();
        for j in 0..k {
            phi[j] = previous[j] - r * previous[k - 1 - j];
        }
        phi.push(r);
    }
    phi
}

/// Up to two partial autocorrelations, bounded away from the unit circle.
fn pacf() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-0.9..0.9_f64, 0..=2)
}

/// A seasonal ARIMA model with stationary AR and invertible MA parts.
#[derive(Debug, Clone)]
struct Model {
    ar: Vec<f64>,
    d: usize,
    ma: Vec<f64>,
    seasonal_ar: Vec<f64>,
    seasonal_d: usize,
    seasonal_ma: Vec<f64>,
    period: usize,
    mean: f64,
}

impl Model {
    fn spec(&self) -> ArimaSpec<'_> {
        ArimaSpec {
            ar: &self.ar,
            d: self.d,
            ma: &self.ma,
            seasonal: Seasonal {
                ar: &self.seasonal_ar,
                d: self.seasonal_d,
                ma: &self.seasonal_ma,
                period: self.period,
            },
            mean: self.mean,
            ..ArimaSpec::default()
        }
    }
}

fn model() -> impl Strategy<Value = Model> {
    let parts = (pacf(), pacf(), prop::collection::vec(-0.9..0.9_f64, 0..=1));
    let seasonal_ma = prop::collection::vec(-0.9..0.9_f64, 0..=1);
    (
        parts,
        seasonal_ma,
        0..=1_usize,
        0..=1_usize,
        2..=6_usize,
        -5.0..5.0_f64,
    )
        .prop_map(
            |((ar, ma, seasonal_ar), seasonal_ma, d, seasonal_d, period, mean)| Model {
                ar: step_up(&ar),
                d,
                ma: step_up(&ma).iter().map(|v| -v).collect(),
                seasonal_ar,
                seasonal_d,
                seasonal_ma,
                period,
                mean,
            },
        )
}

/// A series of 10 to 80 values, some of them missing.
fn series() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(
        prop_oneof![9 => -10.0..10.0_f64, 1 => Just(f64::NAN)],
        10..80,
    )
}

/// `|a - b| <= 1e-9 max(1, |b|)`, with `NaN` equal to `NaN`.
fn close(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * b.abs().max(1.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The heap-free filter and the runtime-sized likelihood share the recursion but not
    /// the buffers: they must give the same innovations for every model and series.
    #[test]
    fn filter_matches_innovations(model in model(), y in series()) {
        let spec = model.spec();
        let expected = innovations(&y, &spec).unwrap();
        let mut filter = Filter::<CAPACITY>::new(&spec).unwrap();
        let streamed: Vec<_> = y.iter().filter_map(|&v| filter.update(v)).collect();

        prop_assert_eq!(streamed.len(), expected.len());
        for (s, e) in streamed.iter().zip(&expected) {
            prop_assert!(close(s.value, e.value), "{s:?} vs {e:?}");
            prop_assert!(close(s.variance, e.variance), "{s:?} vs {e:?}");
            prop_assert!(s.variance >= 1.0 - 1e-12, "variance below sigma2: {s:?}");
        }
    }

    /// Once the last `k = d + D s` values are present, every forecast is finite and its
    /// variance positive, whatever came before (missing values included).
    #[test]
    fn forecasts_are_finite_after_warm_up(model in model(), y in series()) {
        let spec = model.spec();
        let mut filter = Filter::<CAPACITY>::new(&spec).unwrap();
        filter.warm_up(&y);
        let (mut mean, mut variance) = ([0.0; 8], [0.0; 8]);
        filter.forecast_into(&mut mean, &mut variance);
        let k = spec.d + spec.seasonal.d * spec.seasonal.period;
        if y.len() >= k && y[y.len() - k..].iter().all(|v| v.is_finite()) {
            prop_assert!(mean.iter().all(|m| m.is_finite()), "{mean:?}");
            prop_assert!(variance.iter().all(|v| v.is_finite() && *v > 0.0), "{variance:?}");
        }
    }

    /// A fit reports the log-likelihood that `loglike` gives at its estimates, and its
    /// estimates make a valid filter.
    #[test]
    fn fits_are_consistent(model in model(), noise in prop::collection::vec(-2.0..2.0_f64, 60..120)) {
        // Simulate from the model's filter-free ARMA recursion on uniform noise.
        let (ar, ma) = (&model.ar, &model.ma);
        let mut y = vec![0.0; noise.len()];
        for t in 0..y.len() {
            let past: f64 = ar.iter().enumerate().filter(|(i, _)| t > *i).map(|(i, a)| a * y[t - 1 - i]).sum();
            let shocks: f64 = ma.iter().enumerate().filter(|(i, _)| t > *i).map(|(i, m)| m * noise[t - 1 - i]).sum();
            y[t] = past + noise[t] + shocks;
        }

        let Ok(fitted) = Arima::new(ar.len(), 0, ma.len()).fit(&y) else { return Ok(()); };
        let spec = fitted.spec();
        let direct = loglike(&y, &spec).unwrap();
        prop_assert!(close(fitted.loglike, direct), "{} vs {direct}", fitted.loglike);
        prop_assert!(Filter::<CAPACITY>::new(&spec).is_ok());
    }

    /// The automatic search returns a model or an error, never a panic, on any short input.
    #[test]
    fn auto_arima_never_panics(
        y in prop::collection::vec(prop_oneof![8 => -100.0..100.0_f64, 1 => Just(f64::NAN), 1 => Just(0.0)], 0..40),
        seasonal in any::<bool>(),
    ) {
        let search = AutoArima::new().max_order(3);
        let search = if seasonal { search.seasonal(4) } else { search };
        let _ = search.fit(&y);
    }
}
