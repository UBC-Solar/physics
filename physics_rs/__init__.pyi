import numpy as np


def constrain_speeds(speed_limits: np.ndarray, speeds: np.ndarray, tick: int) -> np.ndarray:
    """
    Constrains the vehicle speeds based on the speed limits and computes new speeds.

    :param speed_limits: Array of speed limits (km/h) for each point.
    :param speeds: Array of vehicle speeds (km/h).
    :param tick: The time step (in some unit, e.g., seconds or ticks).

    :return: A NumPy array of constrained vehicle speeds (km/h).
    """
    ...


def calculate_array_ghi_times(python_local_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the array of GHI times based on local times.

    :param python_local_times: Array of local times (UNIX timestamps or other format).

    :return: A tuple of two NumPy arrays:
        - Day of year (f64)
        - Local time (f64)
    """
    ...


def closest_gis_indices_loop(python_cumulative_distances: np.ndarray,
                             python_average_distances: np.ndarray) -> np.ndarray:
    """
    Finds the closest GIS indices based on cumulative and average distances.

    :param python_cumulative_distances: Array of cumulative distances.
    :param python_average_distances: Array of average distances.

    :return: A NumPy array of indices (i64) corresponding to the closest GIS indices.
    """
    ...


def closest_weather_indices_loop(
        python_cumulative_distances: np.ndarray,
        python_average_distances: np.ndarray
) -> np.ndarray:
    """
    Finds the closest weather indices based on cumulative and average distances.

    :param python_cumulative_distances: Array of cumulative distances.
    :param python_average_distances: Array of average distances.

    :return: A NumPy array of indices (i64) corresponding to the closest weather indices.
    """
    ...


def weather_in_time(
        python_unix_timestamps: np.ndarray,
        python_indices: np.ndarray,
        python_weather_forecast: np.ndarray,
        index: int
) -> np.ndarray:
    """
    Retrieves the weather forecast at specific times for given indices.

    :param python_unix_timestamps: Array of UNIX timestamps.
    :param python_indices: Array of indices to look up.
    :param python_weather_forecast: Array of weather forecasts.
    :param index: A specific index to look up in the weather forecast.

    :return: A NumPy array of weather values (f64) at the specified times and indices.
    """
    ...


def update_battery_state(
        python_energy_or_current_array: np.ndarray,
        time_step: float,
        initial_state_of_charge: float,
        initial_polarization_potential: float,
        python_internal_resistance_lookup: np.ndarray,
        python_open_circuit_voltage_lookup: np.ndarray,
        python_polarization_resistance_lookup: np.ndarray,
        python_polarization_capacitance_lookup: np.ndarray,
        nominal_charge_capacity: float,
        is_power: bool,
        quantization_step: float,
        min_soc: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Updates the battery state (SOC and terminal voltage) based on energy/current input.

    :param python_energy_or_current_array: Array of energy or current input values (f64).
    :param time_step: Time step (f64).
    :param initial_state_of_charge: Initial state of charge (f64).
    :param initial_polarization_potential: Initial polarization potential (f64).
    :param python_internal_resistance_lookup: Array of internal resistance values (f64).
    :param python_open_circuit_voltage_lookup: Array of open-circuit voltage values (f64).
    :param python_polarization_resistance_lookup: Array of polarization resistance values (f64).
    :param python_polarization_capacitance_lookup: Array of polarization capacitance values (f64).
    :param nominal_charge_capacity: Nominal charge capacity (f64).
    :param is_power: Boolean flag to indicate if the input is power (`True`) or current (`False`).
    :param quantization_step: The step size used to quantize the SOC (f64).
    :param min_soc: The minimum SOC used when computing the lookup tables

    :return: A tuple containing:
        - An array of updated SOC values (f64).
        - An array of updated terminal voltage values (f64).
    """
    ...
