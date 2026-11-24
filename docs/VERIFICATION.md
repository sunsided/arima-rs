# Verification

Beyond the reference comparisons in [`CONFORMANCE.md`](CONFORMANCE.md), three tools check the
code itself. None of them runs in CI: Kani and cargo-mutants take minutes to hours.

## Property tests

`cargo test` runs them with the rest. `tests/it/properties.rs` checks the statistics
(autocorrelation bounds, estimator identities, differencing round trips) and
`tests/it/model_properties.rs` the model layer:

- the heap-free `Filter` gives the same innovations as `innovations` for random seasonal
  models and series with gaps (two implementations of one recursion);
- forecasts stay finite once the last `d + D s` values are present;
- a fit reports the log-likelihood `loglike` gives at its estimates, and its estimates make
  a valid filter;
- `AutoArima` returns a model or an error, never a panic, on short and degenerate input.

## Kani

`src/verification.rs` holds proofs for the device path, where a panic is a reset. Run them
one at a time; CBMC's memory grows fast with the bounds:

```sh
cargo kani --lib --harness filter_expansion_stays_in_bounds   # ~7 min
cargo kani --lib --harness differencing_removes_a_level
cargo kani --lib --harness fourier_columns_never_overflow
```

They prove that a valid model of any orders up to two and any period either fails the
capacity check or expands into the filter's arrays in bounds, that the differencing
polynomial removes a level, and that `fourier_columns` never overflows. The first and the
last each found an overflow that is now fixed. A proof of `Conformal` did not finish even at
one horizon and two slots: CBMC cannot sort symbolic floats at any useful size.

## Mutation testing

[cargo-mutants](https://mutants.rs) changes the code one operator at a time and reports
the changes no test notices. On the core numeric files it is slow (about a minute per mutant
per job) and each job builds its own copy, so cap its memory:

```sh
systemd-run --user --scope -p MemoryMax=16G \
  cargo mutants -j 2 -f src/filter.rs -f src/state_space.rs -f src/likelihood.rs \
    -f src/fit/objective.rs -f src/fit/bfgs.rs -f src/diagnostics.rs -f src/conformal.rs
```

Add `--iterate` to skip mutants a previous run caught.

The first runs (about 390 mutants) left none alive in the filter, the state-space recursion,
the likelihood or the optimiser. Four gaps in the statistical tests now have tests. These
survivors are equivalent, so no test can catch them:

| Mutant | Why it changes nothing |
|---|---|
| `jarque_bera`: `m3 += …` to `-=` | the statistic uses the skewness squared |
| `kpss`: `partial += e` to `-=` | only `partial²` is used |
| `kpss_residuals`: `v - level` to `v + level` in the trend fit | adds `level · Σ dt`, and `Σ dt = 0` |
| `ndiffs`: `>` to `>=` against the critical value | differs only on exact equality |
| `jarque_bera`: `n < 2` to `n == 2` | the constant-series check already rejects `n < 2` |

## Miri

The crate is `#![forbid(unsafe_code)]` and its only dependency on the device path is `libm`,
so Miri has little to find; a run of the heap-free modules' unit tests did not finish in 25
minutes and was not repeated.
