import pytest
import numpy as np
from physics.models import BasicMotor
from physics.models.aeroshell.aeroshell import Aeroshell
from physics.models.constants import AIR_DENSITY


# create a basic regression test for the BasicMotor class
# create a fixture to initialise the motor model

@pytest.fixture
def basic_motor():
    return BasicMotor(vehicle_mass=350,
                      road_friction=0.012,
                      tire_radius=0.2032,

                      )


# calculates aerodynamic forces that are provided to the basicmotor
aero_motor = Aeroshell(
    drag_lookup={0: 0.137601477, 18: 0.2335363083, 36: 0.5965968882, 54: 1.224448936, 72: 1.861868971,
                 90: 2.38148208, 108: 2.073196244, 126: 1.587471653, 144: 0.5564901716,
                 162: 0.2141437734, 180: 0.1386601712},

    down_lookup={

        0: 0.37526598, 18: 0.3378390168, 36: 0.576439927, 54: 0.8675973423, 72: 1.19551954, 90: 2.683269654,
        108: 2.223002744, 126: 1.581662338, 144: 0.17190782, 162: 0.1882638387, 180: 0.2153506426

    })

wind_attack_angles = np.array([0.0, 18.0, 36.0, 54.0, 72.0, 90.0, 108.0, 126.0, 144.0, 162.0])
wind_speeds = np.full_like(wind_attack_angles, 16.67)
required_speed_ms = np.zeros_like(wind_speeds)

drag_force = aero_motor.calculate_drag(wind_speeds, wind_attack_angles, required_speed_ms)
down_force = aero_motor.calculate_down(wind_speeds, wind_attack_angles, required_speed_ms)


def test_calculate_energy_in_(basic_motor):
    # Define deterministic inputs for the calculate_energy_in method

    required_speed_kmh = np.linspace(0.0, 40.0, num=10)  # even out so that acceleration force is not impacted

    gradients = np.zeros_like(required_speed_kmh)
    winds = np.full_like(required_speed_kmh, 1)
    tick = 1.0

    energies = basic_motor.calculate_energy_in(required_speed_kmh, gradients, drag_force, down_force, tick)

    expected = np.array([0, 945.28684212, 2027.43705448, 3281.1128736, 4092.09697594
                            , 4174.85805775, 3712.97377444, 3571.64065035, 5091.54144206, 5840.36657648])

    assert np.allclose(energies, expected, atol=1e-3)
