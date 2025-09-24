import pytest
import numpy as np
from physics.models.aeroshell.aeroshell import Aeroshell
from physics.models.constants import AIR_DENSITY


# create a basic regression test for the Aeroshell class
# create a fixture to initialise the model
@pytest.fixture
def aeroshell_motor():
    return Aeroshell(drag_lookup={0: 0.137601477, 18: 0.2335363083, 36: 0.5965968882, 54: 1.224448936, 72: 1.861868971,
                                  90: 2.38148208, 108: 2.073196244, 126: 1.587471653, 144: 0.5564901716,
                                  162: 0.2141437734, 180: 0.1386601712},

                     down_lookup={

                         0: 0.37526598, 18: 0.3378390168, 36: 0.576439927, 54: 0.8675973423, 72: 1.19551954,
                         90: 2.683269654,
                         108: 2.223002744, 126: 1.581662338, 144: 0.17190782, 162: 0.1882638387, 180: 0.2153506426

                     })

def test_calculate_drag_force(aeroshell_motor):
    # Define deterministic inputs for the calculate drag_force method

    wind_attack_angles = np.array([0.0, 18.0, 36.0])
    wind_speeds = np.zeros_like(wind_attack_angles)
    required_speed_ms = np.full_like(wind_attack_angles, 16.67)
    drag_force = aeroshell_motor.calculate_drag(wind_speeds, wind_attack_angles, required_speed_ms)
    expected = np.array([23.42, 39.75, 101.54])
    assert np.allclose(drag_force, expected, atol=1e-2)


def test_calculate_down_force(aeroshell_motor):
    # Define deterministic inputs for the calculate downforce method

    wind_attack_angles = np.array([0.0, 18.0, 36.0])
    wind_speeds = np.full_like(wind_attack_angles, 16.67)
    required_speed_ms = np.zeros_like(wind_speeds)
    down_force = aeroshell_motor.calculate_down(wind_speeds, wind_attack_angles, required_speed_ms)
    expected = np.array([63.84, 57.48, 98.06])
    assert np.allclose(down_force, expected, atol=1e-1)