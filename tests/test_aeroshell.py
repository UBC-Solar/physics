import pytest
import physics_rs
import numpy as np

from physics.models.motor import BasicMotor
from physics.models.motor import Aeroshell

from physics.models.motor import basic_motor


#create a basic regression test for the Aeroshell class


#create a fixture to initialise the model

@pytest.fixture
def aeroshell_motor():
    return Aeroshell()



def test_calculate_drag_force(aeroshell_motor):
    # Define deterministic inputs for the calculate dragforce method

    #wind_speeds, wind_attack_angles, required_speed_ms


    wind_attack_angles = np.array([0.0, 18.0, 36.0])
    wind_speeds = np.full_like(wind_attack_angles, 16.67)
    required_speed_ms = np.zeros_like(wind_speeds)

    drag_force = aeroshell_motor.calculate_drag_force(wind_speeds, wind_attack_angles, required_speed_ms)

    expected = np.array([ 23.41842819,  39.7443038 , 101.54654616])
    #expected = np.array([63.84 ,57.48,98.06])


    assert np.allclose(drag_force, expected, atol=1e-3)






