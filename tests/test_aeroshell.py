import pytest
import numpy as np
from physics.models.aeroshell.aeroshell import Aeroshell

# create a basic regression test for the Aeroshell class
# create a fixture to initialise the model

@pytest.fixture
def aeroshell_motor():
    return Aeroshell(drag_lookup={0: 23.41,
                                  18: 39.73,
                                  36: 101.51,
                                  54: 208.35,
                                  72: 316.84,
                                  90: 411.29,
                                  108: 352.76,
                                  126: 270.13,
                                  144: 94.67,
                                  162: 36.43,
                                  180: 23.58
                                  }, down_lookup={
        0: 63.84,
        18: 57.48,
        36: 98.06,
        54: 147.61,
        72: 203.43,
        90: 457.57,
        108: 378.23,
        126: 269.12,
        144: 29.26,
        162: 32.03,
        180: 36.64
    })


def test_calculate_drag_force(aeroshell_motor):
    # Define deterministic inputs for the calculate dragforce method

    wind_attack_angles = np.array([0.0, 18.0, 36.0])
    wind_speeds = np.full_like(wind_attack_angles, 16.67)
    required_speed_ms = np.zeros_like(wind_speeds)

    drag_force = aeroshell_motor.calculate_drag(wind_speeds, wind_attack_angles, required_speed_ms)

    expected = np.array([23.41842819, 39.7443038, 101.54654616])
    assert np.allclose(drag_force, expected, atol=1e-3)


def test_calculate_down_force(aeroshell_motor):
    # Define deterministic inputs for the calculate downforce method

    wind_attack_angles = np.array([0.0, 18.0, 36.0])
    wind_speeds = np.full_like(wind_attack_angles, 16.67)
    required_speed_ms = np.zeros_like(wind_speeds)

    down_force = aeroshell_motor.calculate_down(wind_speeds, wind_attack_angles, required_speed_ms)
    expected = np.array([63.84, 57.48, 98.06])
    assert np.allclose(down_force, expected, atol=1e-1)
