//! Integration tests for the public API.
//!
//! - [`reference`] compares results against statsmodels reference vectors.
//! - [`arma`] compares the ARMA filter and likelihood against statsmodels.
//! - [`auto`] compares the stepwise auto-ARIMA search against pmdarima.
//! - [`fit`] compares Hannan–Rissanen and maximum-likelihood fits against statsmodels.
//! - [`stl`] compares the STL decomposition against statsmodels.
//! - [`properties`] checks mathematical invariants on random series with proptest.

mod arma;
mod arma_data;
mod auto;
mod auto_data;
mod fit;
mod properties;
mod reference;
mod reference_data;
mod stl;
mod stl_data;
