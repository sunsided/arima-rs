//! Integration tests for the public API.
//!
//! - [`r_conformance`] compares fits, forecasts, diagnostics, Fourier terms and the automatic
//!   search against R 4.4 and `forecast`.
//! - [`reference`] compares results against statsmodels reference vectors.
//! - [`arma`] compares the ARMA filter and likelihood against statsmodels.
//! - [`auto`] compares the stepwise auto-ARIMA search against pmdarima.
//! - [`conformal`] measures the coverage of conformal intervals on simulated series.
//! - [`fourier`] compares the Fourier regressors against statsmodels.
//! - [`fixed`] compares fits with fixed parts against statsmodels' `fix_params`.
//! - [`fit`] compares Hannan–Rissanen and maximum-likelihood fits against statsmodels.
//! - [`stl`] compares the STL decomposition against statsmodels.
//! - [`mstl`] compares the MSTL decomposition against statsmodels.
//! - [`period`] compares the seasonal period candidates against coreforecast.
//! - [`steady`] compares the filter's steady-state mode against statsmodels.
//! - [`online`] compares the online AR predictor with the exact filter under changing dynamics.
//! - [`pmm2`] compares PMM2 with maximum likelihood on series with skewed innovations.
//! - [`properties`] checks mathematical invariants on random series with proptest.
//! - [`croston`] compares Croston's method against R's `croston`, fitted value by fitted value.
//! - [`theta`] compares the Theta method against R's `thetaf`.
//! - [`var`] compares VAR and VARX fits, forecasts and lag selection against statsmodels.
//! - [`backtest`] compares the accuracy measures and rolling-origin errors against R's
//!   `accuracy` and `tsCV`, and checks the family selection.
//! - [`garch`] compares GARCH(1,1) against R's `tseries::garch`, and its fit against the
//!   likelihood's optimum.
//! - [`model_properties`] checks the filter, fits and search on random models and series.

mod arma;
mod arma_data;
mod auto;
mod auto_data;
mod backtest;
mod conformal;
mod croston;
mod ets;
mod fit;
mod fixed;
mod fixed_data;
mod fourier;
mod fourier_data;
mod garch;
mod model_properties;
mod mstl;
mod mstl_data;
mod online;
mod period;
mod period_data;
mod pmm2;
mod properties;
mod r_accuracy_data;
mod r_conformance;
mod r_croston_data;
mod r_data;
mod r_ets_data;
mod r_garch_data;
mod r_theta_data;
mod reference;
mod reference_data;
mod steady;
mod steady_data;
mod stl;
mod stl_data;
mod theta;
mod var;
mod var_data;
