use numpy::ndarray::ArrayViewD;
use numpy::{PyArray, PyArrayDyn, PyReadwriteArrayDyn, PyReadwriteArray1, PyReadonlyArray1, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod environment;
pub mod models;
use crate::environment::gis::gis::{rust_closest_gis_indices_loop, get_driving_speeds, calculate_speeds_and_position};
use crate::environment::meteorology::meteorology::{rust_calculate_array_ghi_times, rust_closest_weather_indices_loop, rust_weather_in_time};
use crate::models::battery::battery::update_battery_state;

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
#[pyo3(name = "physics_rs")]
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
    #[pyo3(name = "update_battery_state")]
    fn update_battery_state_py<'py>(
        py: Python<'py>,
        python_energy_or_current_array: PyReadwriteArray1<'py, f64>,
        time_step: f64,
        initial_state_of_charge: f64,
        initial_polarization_potential: f64,
        python_internal_resistance_lookup: PyReadwriteArray1<'py, f64>,
        python_open_circuit_voltage_lookup: PyReadwriteArray1<'py, f64>,
        python_polarization_resistance_lookup: PyReadwriteArray1<'py, f64>,
        python_polarization_capacitance_lookup: PyReadwriteArray1<'py, f64>,
        nominal_charge_capacity: f64,
        is_power: bool,
        quantization_step: f64,
        min_soc: f64,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>) {
        let energy_or_current_array = python_energy_or_current_array.as_array();
        let internal_resistance_lookup = python_internal_resistance_lookup.as_array();
        let open_circuit_voltage_lookup = python_open_circuit_voltage_lookup.as_array();
        let polarization_resistance_lookup = python_polarization_resistance_lookup.as_array();
        let polarization_capacitance_lookup = python_polarization_capacitance_lookup.as_array();
        let (soc_array, voltage_array): (Vec<f64>, Vec<f64>) = update_battery_state(
            energy_or_current_array,
            time_step,
            initial_state_of_charge,
            initial_polarization_potential,
            internal_resistance_lookup,
            open_circuit_voltage_lookup,
            polarization_resistance_lookup,
            polarization_capacitance_lookup,
            nominal_charge_capacity,
            is_power,
            quantization_step,
            min_soc
        );
        let py_soc_array = PyArray::from_vec(py, soc_array);
        let py_voltage_array = PyArray::from_vec(py, voltage_array);
        (py_soc_array, py_voltage_array)
    }

    #[pyfn(m)]
    #[pyo3(name = "calculate_speeds_and_position")]
    fn calculate_speeds_and_position_py<'py>(
        py: Python<'py>,
        speeds_kmh_py: PyReadwriteArray1<'py, f64>,
        path_distances_py: PyReadwriteArray1<'py, f64>,
        track_speeds_py: PyReadwriteArray1<'py, f64>,
        simulation_dt: u32,
    ) -> (&'py PyArray1<usize>, &'py PyArray1<f64>) {
        let speeds_kmh = speeds_kmh_py.as_array();
        let path_distances = path_distances_py.as_array();
        let track_speeds = track_speeds_py.as_array();
        let (gis_indices, actual_speeds_kmh): (Vec<usize>, Vec<f64>) = calculate_speeds_and_position(
            speeds_kmh,
            path_distances,
            track_speeds,
            simulation_dt,
        );
        let gis_indices_py = PyArray::from_vec(py, gis_indices);
        let actual_speeds_kmh_py = PyArray::from_vec(py, actual_speeds_kmh);
        (gis_indices_py, actual_speeds_kmh_py)
    }

    #[pyfn(m)]
    #[pyo3(name = "get_driving_speeds")]
    fn py_get_driving_speeds<'py>(
        py: Python<'py>,
        py_average_speeds: PyReadonlyArray1<'py, f64>,            // Average speeds in m/s
        simulation_dt: i64,                                       // Time step in seconds
        py_driving_allowed_boolean: PyReadonlyArray1<'py, bool>,  // Simulation-time boolean array
        track_length: f64,                                        // Track length in meters
        idle_time: i64                                            // Time to idle in seconds
    ) -> PyResult<&'py PyArray1<f64>> {
        let average_speeds = py_average_speeds.as_array();
        let driving_allowed_boolean = py_driving_allowed_boolean.as_array();

        match get_driving_speeds(
            average_speeds,
            simulation_dt,
            driving_allowed_boolean,
            track_length,
            idle_time
        ) {
            Ok(driving_speeds) => Ok(PyArray1::from_vec(py, driving_speeds)),
            Err(error) => Err(pyo3::exceptions::PyValueError::new_err(error))
        }
    }

    Ok(())
}