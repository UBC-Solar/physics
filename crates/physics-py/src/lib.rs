//! Python bindings for UBC Solar's physics models (the `ubc-solar-physics` crate).
//!
//! Each function converts NumPy arrays to array views, calls into `physics`, and converts the
//! results back. Keep logic out of this crate: anything beyond conversion belongs in `physics`.
//! Keep `physics_rs/__init__.pyi` in sync with the signatures here.

use pyo3::prelude::*;

/// Compiled physics routines used by the `physics` Python package.
#[pymodule]
mod _physics_rs {
    use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray3, ToPyArray};
    use physics::environment::{gis, meteorology};
    use physics::models::battery;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyfunction]
    fn constrain_speeds<'py>(
        py: Python<'py>,
        speed_limits: PyReadonlyArray1<'py, f64>,
        speeds: PyReadonlyArray1<'py, f64>,
        tick: i32,
    ) -> Bound<'py, PyArray1<f64>> {
        let result = gis::constrain_speeds(speed_limits.as_array(), speeds.as_array(), tick);
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    fn calculate_array_ghi_times<'py>(
        py: Python<'py>,
        python_local_times: PyReadonlyArray1<'py, u64>,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (day_of_year, local_time) =
            meteorology::calculate_array_ghi_times(python_local_times.as_array());
        (
            PyArray1::from_vec(py, day_of_year),
            PyArray1::from_vec(py, local_time),
        )
    }

    #[pyfunction]
    fn closest_gis_indices_loop<'py>(
        py: Python<'py>,
        python_cumulative_distances: PyReadonlyArray1<'py, f64>,
        python_average_distances: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let result = gis::closest_gis_indices(
            python_cumulative_distances.as_array(),
            python_average_distances.as_array(),
        );
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    fn closest_weather_indices_loop<'py>(
        py: Python<'py>,
        python_cumulative_distances: PyReadonlyArray1<'py, f64>,
        python_average_distances: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let result = meteorology::closest_weather_indices(
            python_cumulative_distances.as_array(),
            python_average_distances.as_array(),
        );
        PyArray1::from_vec(py, result)
    }

    #[pyfunction]
    fn weather_in_time<'py>(
        py: Python<'py>,
        python_unix_timestamps: PyReadonlyArray1<'py, i64>,
        python_indices: PyReadonlyArray1<'py, i64>,
        python_weather_forecast: PyReadonlyArray3<'py, f64>,
        index: u8,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let weather = meteorology::weather_in_time(
            python_unix_timestamps.as_array(),
            python_indices.as_array(),
            python_weather_forecast.as_array(),
            index as usize,
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(weather.to_pyarray(py))
    }

    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn update_battery_state<'py>(
        py: Python<'py>,
        python_energy_or_current_array: PyReadonlyArray1<'py, f64>,
        time_step: f64,
        initial_state_of_charge: f64,
        initial_polarization_potential: f64,
        python_internal_resistance_lookup: PyReadonlyArray1<'py, f64>,
        python_open_circuit_voltage_lookup: PyReadonlyArray1<'py, f64>,
        python_polarization_resistance_lookup: PyReadonlyArray1<'py, f64>,
        python_polarization_capacitance_lookup: PyReadonlyArray1<'py, f64>,
        nominal_charge_capacity: f64,
        is_power: bool,
        quantization_step: f64,
        min_soc: f64,
    ) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (soc, voltage) = battery::update_battery_state(
            python_energy_or_current_array.as_array(),
            time_step,
            initial_state_of_charge,
            initial_polarization_potential,
            python_internal_resistance_lookup.as_array(),
            python_open_circuit_voltage_lookup.as_array(),
            python_polarization_resistance_lookup.as_array(),
            python_polarization_capacitance_lookup.as_array(),
            nominal_charge_capacity,
            is_power,
            quantization_step,
            min_soc,
        );
        (PyArray1::from_vec(py, soc), PyArray1::from_vec(py, voltage))
    }

    #[pyfunction]
    fn calculate_speeds_and_position<'py>(
        py: Python<'py>,
        speeds_kmh_py: PyReadonlyArray1<'py, f64>,
        path_distances_py: PyReadonlyArray1<'py, f64>,
        track_speeds_py: PyReadonlyArray1<'py, f64>,
        simulation_dt: u32,
    ) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<f64>>) {
        let (gis_indices, actual_speeds_kmh) = gis::calculate_speeds_and_position(
            speeds_kmh_py.as_array(),
            path_distances_py.as_array(),
            track_speeds_py.as_array(),
            simulation_dt,
        );
        (
            PyArray1::from_vec(py, gis_indices),
            PyArray1::from_vec(py, actual_speeds_kmh),
        )
    }

    #[pyfunction]
    fn get_driving_speeds<'py>(
        py: Python<'py>,
        py_average_speeds: PyReadonlyArray1<'py, f64>,
        simulation_dt: i64,
        py_driving_allowed_boolean: PyReadonlyArray1<'py, bool>,
        track_length: f64,
        idle_time: i64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let driving_speeds = gis::get_driving_speeds(
            py_average_speeds.as_array(),
            simulation_dt,
            py_driving_allowed_boolean.as_array(),
            track_length,
            idle_time,
        )
        .map_err(PyValueError::new_err)?;
        Ok(PyArray1::from_vec(py, driving_speeds))
    }
}
