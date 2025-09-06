import pytest
import physics_rs
import numpy as np
from physics.models import BasicMotor
from physics.models.motor import basic_motor
from physics.models.aeroshell.aeroshell import Aeroshell


# create a basic regression test for the BasicMotor class
# create a fixture to initialise the motor model

@pytest.fixture
def basic_motor():
    return BasicMotor(vehicle_mass=350,
                      road_friction=0.012,
                      tire_radius=0.2032,

                      )


aero_motor = Aeroshell(drag_lookup={0: 23.41,
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

    expected = np.array([0., 948.99456941, 2115.57457112, 3810.27889217, 5892.03839313,
                         7759.88635444, 7194.56938554, 7170.5217449, 6357.62768734, 6112.87374071])

    assert np.allclose(energies, expected, atol=1e-3)
