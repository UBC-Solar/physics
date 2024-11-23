use chrono::{Datelike, NaiveDateTime, Timelike};
use numpy::ndarray::{s, Array, Array2, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis};
use numpy::{PyArray, PyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod environment;
pub mod models;
use crate::environment::gis::gis::rust_closest_gis_indices_loop;
use crate::environment::meteorology::meteorology::{rust_calculate_array_ghi_times, rust_closest_weather_indices_loop, rust_weather_in_time, rust_closest_timestamp_indices};
use crate::models::battery::battery::update_battery_array;

fn constrain_speeds(speed_limits: ArrayViewD<f64>,  speeds: ArrayViewD<f64>, tick: i32) -> Vec<f64> {
    let mut distance: f64 = 0.0;
    static KMH_TO_MS: f64 = 1.0 / 3.6;

    let ret: Vec<f64> = speeds.iter().map(| speed: &f64 | {
        let speed_limit: f64 = speed_limits[distance.floor() as usize];
        let vehicle_speed: f64 =f64::min(speed_limit, *speed);
        distance += vehicle_speed * KMH_TO_MS * tick as f64;
        vehicle_speed
    }).collect();

    return ret
}

/// A Python module implemented in Rust. The name of this function is the Rust module name!
#[pymodule]
#[pyo3(name = "core")]
fn rust_simulation(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
        #[pyo3(name = "constrain_speeds")]
        fn constrain_speeds_py<'py>(py: Python<'py>, x: PyReadwriteArrayDyn<'py, f64>, y: PyReadwriteArrayDyn<'py, f64>, z: i32) -> &'py PyArrayDyn<f64> {
            let x = x.as_array();
            let y = y.as_array();
            let result = constrain_speeds(x, y, z);
            return PyArray::from_vec(py, result).to_dyn();
    }

    #[pyfn(m)]
    #[pyo3(name = "calculate_array_ghi_times")]
    fn calculate_array_ghi_times<'py>(
        py: Python<'py>,
        python_local_times: PyReadwriteArrayDyn<'py, u64>,
    ) -> (&'py PyArrayDyn<f64>, &'py PyArrayDyn<f64>) {
        let local_times = python_local_times.as_array();
        let (day_of_year_out, local_time_out) = rust_calculate_array_ghi_times(local_times);
        let py_day_out = PyArray::from_vec(py, day_of_year_out).to_dyn();
        let py_time_out = PyArray::from_vec(py, local_time_out).to_dyn();
        (py_day_out, py_time_out)
    }

    #[pyfn(m)]
    #[pyo3(name = "closest_gis_indices_loop")]
    fn closest_gis_indices_loop<'py>(
        py: Python<'py>,
        python_cumulative_distances: PyReadwriteArrayDyn<'py, f64>,
        python_average_distances: PyReadwriteArrayDyn<'py, f64>,
    ) -> &'py PyArrayDyn<i64> {
        let average_distances = python_average_distances.as_array();
        let cumulative_distances = python_cumulative_distances.as_array();
        let result = rust_closest_gis_indices_loop(cumulative_distances, average_distances);
        let py_result = PyArray::from_vec(py, result).to_dyn();
        py_result
    }

    #[pyfn(m)]
    #[pyo3(name = "closest_weather_indices_loop")]
    fn closest_weather_indices_loop<'py>(
        py: Python<'py>,
        python_cumulative_distances: PyReadwriteArrayDyn<'py, f64>,
        python_average_distances: PyReadwriteArrayDyn<'py, f64>,
    ) -> &'py PyArrayDyn<i64> {
        let average_distances = python_average_distances.as_array();
        let cumulative_distances = python_cumulative_distances.as_array();
        let result = rust_closest_weather_indices_loop(cumulative_distances, average_distances);
        let py_result = PyArray::from_vec(py, result).to_dyn();
        py_result
    }

    #[pyfn(m)]
    #[pyo3(name = "weather_in_time")]
    fn weather_in_time<'py>(
        py: Python<'py>,
        python_unix_timestamps: PyReadwriteArrayDyn<'py, i64>,
        python_indices: PyReadwriteArrayDyn<'py, i64>,
        python_weather_forecast: PyReadwriteArrayDyn<'py, f64>,
        index: u8
    ) -> &'py PyArrayDyn<f64> {
        let unix_timestamps = python_unix_timestamps.as_array();
        let indices = python_indices.as_array();
        let weather_forecast = python_weather_forecast.as_array();
        let mut result = rust_weather_in_time(unix_timestamps, indices, weather_forecast, index);
        let py_result = PyArray::from_array(py, &mut result).to_dyn();
        py_result
    }

    #[pyfn(m)]
    #[pyo3(name = "update_battery_array")]
    fn update_battery_array_py<'py>(
        py: Python<'py>,
        python_delta_energy_array: PyReadwriteArrayDyn<'py, f64>,
        time_step: f64,
        initial_state_of_charge: f64,
        initial_polarization_potential: f64,
        polarization_resistance: f64,
        python_internal_resistance_coeffs: PyReadwriteArrayDyn<'py, f64>,
        python_open_circuit_voltage_coeffs: PyReadwriteArrayDyn<'py, f64>,
        time_constant: f64,
        nominal_charge_capacity: f64,
    ) -> (&'py PyArrayDyn<f64>, &'py PyArrayDyn<f64>) {
        let delta_energy_array = python_delta_energy_array.as_array();
        let internal_resistance_coeffs = python_internal_resistance_coeffs.as_array();
        let open_circuit_voltage_coeffs = python_open_circuit_voltage_coeffs.as_array();
        let (soc_array, voltage_array): (Vec<f64>, Vec<f64>) = update_battery_array(
            delta_energy_array,
            time_step,
            initial_state_of_charge,
            initial_polarization_potential,
            polarization_resistance,
            internal_resistance_coeffs,
            open_circuit_voltage_coeffs,
            time_constant,
            nominal_charge_capacity,
        );
        let py_soc_array = PyArray::from_vec(py, soc_array).to_dyn();
        let py_voltage_array = PyArray::from_vec(py, voltage_array).to_dyn();
        (py_soc_array, py_voltage_array)
    }

    Ok(())
}