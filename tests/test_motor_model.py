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
        vehicle_frontal_area= 1.1853,
        drag_coefficient=0.11609
    )


#required_speed_kmh, gradients, wind_speeds, tick, **kwargs


def test_calculate_energy_in_(basic_motor):
    # Define deterministic inputs for the calculate_energy_in method

    required_speed_kmh = np.linspace(0.0, 40.0, num=10)

    gradients = np.zeros_like(required_speed_kmh)
    winds = np.full_like(required_speed_kmh, 1)
    tick = 1.0


    energies = basic_motor.calculate_energy_in(required_speed_kmh, gradients, winds, tick)

    expected = np.array([0, 875.37511917, 1712.88660841 ,2532.60601288, 3347.36667064,
     4165.06559618,4990.09286635 ,5579.38323421, 5946.16366471, 6069.18142423])

    assert np.allclose(energies, expected, atol=1e-3)






