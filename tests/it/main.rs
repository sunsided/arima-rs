//! Integration tests for the public API.
//!
//! - [`reference`] compares results against statsmodels reference vectors.
//! - [`arma`] compares the ARMA filter and likelihood against statsmodels.
//! - [`properties`] checks mathematical invariants on random series with proptest.

mod arma;
mod arma_data;
mod properties;
mod reference;
mod reference_data;
