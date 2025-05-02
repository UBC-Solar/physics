import core
import numpy as np
import pytest


def test_calculate_driving_speeds_simple():
    """
    Test a simple case of the driving speeds with no rounding
    """
    average_speeds = np.array([10, 14, 30, 0, 10, 20, 30], dtype=np.float64)
    track_length = 30
    simulation_dt = 1

    idle_time = 3
    driving_allowed = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    driving_speeds = core.get_driving_speeds(average_speeds, simulation_dt, driving_allowed, track_length, idle_time)

    assert np.all(np.equal(driving_speeds, [0, 10, 10, 10, 14, 14, 0, 0, 0, 0, 0, 30, 0, 0, 0, 10, 10, 10, 20, 20, 30]))


def test_calculate_driving_speeds_fp():
    """
    Test the driving speeds with floating point rounding going on
    :return:
    """
    average_speeds = np.array([9.9, 30.7, 15.2, 15.3], dtype=np.float64)
    track_length = 30.6
    simulation_dt = 1

    idle_time = 3
    driving_allowed = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    driving_speeds = core.get_driving_speeds(average_speeds, simulation_dt, driving_allowed, track_length, idle_time)

    assert np.allclose(driving_speeds, np.array([0, 0, 9.9, 9.9, 9.9, 9.9, 30.7, 15.2, 15.2, 15.2, 15.3, 15.3]), atol=1e-3)


def test_calculate_driving_speeds_fp_with_tick():
    """
    Test the driving speeds with floating point rounding going on and tick != 1.
    """
    average_speeds = np.array([10.2, 30.7, 15.2, 15.3], dtype=np.float64)
    track_length = 30.6
    simulation_dt = 2

    idle_time = 3
    driving_allowed = np.array([0, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    driving_speeds = core.get_driving_speeds(average_speeds, simulation_dt, driving_allowed, track_length, idle_time)

    assert np.allclose(driving_speeds, np.array([0, 0, 10.2, 10.2, 30.7, 15.2, 15.2, 15.3]), atol=1e-3)
    pass


def test_failure():
    """
    Test that nothing fails when too many average speeds are provided, and that ValueError is raised when
    not enough laps are provided.
    """
    average_speeds_too_many = np.array([10, 14, 30, 0, 10, 20, 30, 10, 10], dtype=np.float64)
    average_speeds_too_few = np.array([10, 14, 30, 0, 10, 20], dtype=np.float64)
    track_length = 30
    simulation_dt = 1
    idle_time = 3
    driving_allowed = np.array([0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

    with pytest.raises(ValueError):
        core.get_driving_speeds(average_speeds_too_few, simulation_dt, driving_allowed, track_length, idle_time)

    driving_speeds = core.get_driving_speeds(average_speeds_too_many, simulation_dt, driving_allowed, track_length, idle_time)

    assert np.all(np.equal(driving_speeds, [0, 10, 10, 10, 14, 14, 0, 0, 0, 0, 0, 30, 0, 0, 0, 10, 10, 10, 20, 20, 30]))
