import pytest
import numpy as np

from physics.models.aeroshell.aeroshell import Aeroshell


#create a basic regression test for the Aeroshell class


#create a fixture to initialise the model

@pytest.fixture
def aeroshell_motor():
    return Aeroshell()






def test_calculate_drag_force(aeroshell_motor):
    # Define deterministic inputs for the calculate dragforce method


    wind_attack_angles = np.array([0.0, 18.0, 36.0])
    wind_speeds = np.full_like(wind_attack_angles, 16.67)
    required_speed_ms = np.zeros_like(wind_speeds)

    drag_force = aeroshell_motor.calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, "drag")

    expected = np.array([ 23.41842819,  39.7443038 , 101.54654616])
    assert np.allclose(drag_force, expected, atol=1e-3)

def test_calculate_down_force(aeroshell_motor):
    # Define deterministic inputs for the calculate downforce method

    wind_attack_angles = np.array([0.0, 18.0, 36.0])
    wind_speeds = np.full_like(wind_attack_angles, 16.67)
    required_speed_ms = np.zeros_like(wind_speeds)

    down_force = aeroshell_motor.calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, "down")
    expected = np.array([63.84 ,57.48,98.06])
    assert np.allclose(down_force, expected, atol=1e-1)






