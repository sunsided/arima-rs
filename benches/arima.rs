//! Criterion benchmarks of the hot paths: one likelihood evaluation, a fit, an
//! auto-ARIMA search, and streaming updates through the heap-free filter.
//!
//! Run with `cargo bench`. `ci/compare` compares against other crates instead.

use backshift::{Arima, ArimaSpec, AutoArima, Filter, Seasonal, loglike};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

/// Standard normal draws from a fixed linear congruential generator (Box–Muller).
fn normals(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    (0..n)
        .map(|_| (-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos())
        .collect()
}

/// An airline-model series of `n` values.
fn airline(n: usize) -> Vec<f64> {
    let e = normals(n, 3);
    let mut y = vec![0.0; n];
    for t in 0..n {
        let lag = |k: usize| if t >= k { e[t - k] } else { 0.0 };
        let past = |k: usize| if t >= k { y[t - k] } else { 0.0 };
        y[t] = e[t] - 0.4 * lag(1) - 0.6 * lag(12) + 0.24 * lag(13) + past(1) + past(12) - past(13);
    }
    y
}

/// The airline model at fixed parameters.
fn airline_spec() -> ArimaSpec<'static> {
    ArimaSpec {
        d: 1,
        ma: &[-0.4],
        seasonal: Seasonal {
            ar: &[],
            d: 1,
            ma: &[-0.6],
            period: 12,
        },
        ..ArimaSpec::default()
    }
}

/// All benchmarks.
fn benches(c: &mut Criterion) {
    let y = airline(144);
    let spec = airline_spec();

    c.bench_function("loglike airline n=144", |b| {
        b.iter(|| loglike(black_box(&y), &spec))
    });
    c.bench_function("fit airline n=144", |b| {
        b.iter(|| Arima::new(0, 1, 1).seasonal(0, 1, 1, 12).fit(black_box(&y)))
    });
    c.bench_function("auto-arima airline n=144", |b| {
        b.iter(|| {
            AutoArima::new()
                .seasonal(12)
                .d(1)
                .seasonal_d(1)
                .fit(black_box(&y))
        })
    });
    c.bench_function("filter update airline", |b| {
        let mut filter = Filter::<27>::new(&spec).expect("fits");
        let mut t = 0;
        b.iter(|| {
            t = (t + 1) % y.len();
            filter.update(black_box(y[t]))
        });
    });
}

criterion_group!(arima, benches);
criterion_main!(arima);
