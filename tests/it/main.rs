//! Integration tests for the public API.
//!
//! - [`reference`] compares results against statsmodels reference vectors.
//! - [`arma`] compares the ARMA filter and likelihood against statsmodels.
//! - [`fit`] compares Hannan–Rissanen and maximum-likelihood fits against statsmodels.
//! - [`properties`] checks mathematical invariants on random series with proptest.

mod arma;
mod arma_data;
mod fit;
mod properties;
mod reference;
mod reference_data;
