//! Integration tests for the public API.
//!
//! - [`reference`] compares results against statsmodels reference vectors.
//! - [`arma`] compares the ARMA filter and likelihood against statsmodels.
//! - [`auto`] compares the stepwise auto-ARIMA search against pmdarima.
//! - [`conformal`] measures the coverage of conformal intervals on simulated series.
//! - [`fourier`] compares the Fourier regressors against statsmodels.
//! - [`fit`] compares Hannan–Rissanen and maximum-likelihood fits against statsmodels.
//! - [`stl`] compares the STL decomposition against statsmodels.
//! - [`mstl`] compares the MSTL decomposition against statsmodels.
//! - [`period`] compares the seasonal period candidates against coreforecast.
//! - [`steady`] compares the filter's steady-state mode against statsmodels.
//! - [`online`] compares the online AR predictor with the exact filter under changing dynamics.
//! - [`pmm2`] compares PMM2 with maximum likelihood on series with skewed innovations.
//! - [`properties`] checks mathematical invariants on random series with proptest.

mod arma;
mod arma_data;
mod auto;
mod auto_data;
mod conformal;
mod fit;
mod fourier;
mod fourier_data;
mod mstl;
mod mstl_data;
mod online;
mod period;
mod period_data;
mod pmm2;
mod properties;
mod reference;
mod reference_data;
mod steady;
mod steady_data;
mod stl;
mod stl_data;
