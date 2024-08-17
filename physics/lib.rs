use chrono::{Datelike, NaiveDateTime, Timelike};
use numpy::ndarray::{s, Array, Array2, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis};
use numpy::{PyArray, PyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod environment;
pub mod models;
use crate::environment::gis::gis::rust_closest_gis_indices_loop;
use crate::environment::solar_calculations::solar_calculations::rust_calculate_array_ghi_times;
use crate::environment::weather_forecasts::weather_forecasts::{rust_closest_weather_indices_loop, rust_weather_in_time, rust_closest_timestamp_indices};

/// A Python module implemented in Rust. The name of this function is the Rust module name!
#[pymodule]
#[pyo3(name = "core")]
fn rust_simulation(_py: Python, m: &PyModule) -> PyResult<()> {
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

    Ok(())
}