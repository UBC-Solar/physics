import pytest
import physics_rs
import numpy as np

from physics.models import BasicMotor
from physics.models.motor import basic_motor


#create a basic regression test for the BasicMotor class


#create a fixture to initialise the motor model

@pytest.fixture
def basic_motor():
    return BasicMotor(vehicle_mass=350,
        road_friction=0.012,
        tire_radius=0.2032,
        vehicle_frontal_area= 25,
        drag_coefficient=0.11609
    )


#required_speed_kmh, gradients, wind_speeds, tick, **kwargs


def test_calculate_energy_in_(basic_motor):
    # Define deterministic inputs for the calculate_energy_in method

    # required_speed_kmh = np.array([0, 35, 45, 78 ], dtype=float)#40
    # gradients = np.array([0, 0, 0, 0], dtype=float)
    # winds = np.array([1, 1, 1, 1], dtype=float)
    # tick = 1.0
    #


    required_speed_kmh =  40.0
    gradients = 0.0
    winds = 1.0
    tick = 1.0
    energies = basic_motor.calculate_energy_in(required_speed_kmh, gradients, winds, tick)


    expected =
    assert np.allclose(energies, expected, atol=1e-3)






