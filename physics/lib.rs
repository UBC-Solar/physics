use chrono::{Datelike, NaiveDateTime, Timelike};
use numpy::ndarray::{s, Array, Array2, ArrayViewD, ArrayViewMut2, ArrayViewMut3, Axis};
use numpy::{PyArray, PyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod environment;
pub mod models;
use crate::environment::gis::gis::rust_closest_gis_indices_loop;
use crate::environment::meteorology::meteorology::{rust_calculate_array_ghi_times, rust_closest_weather_indices_loop, rust_weather_in_time, rust_closest_timestamp_indices};

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

    Ok(())
}