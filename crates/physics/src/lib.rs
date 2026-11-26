//! UBC Solar's physics models for simulating a solar-powered vehicle.
//!
//! This crate is pure Rust. The Python bindings (the `physics_rs` extension module used by the
//! `ubc-solar-physics` Python package) live in the separate `physics-py` crate.

pub mod environment;
pub mod models;
