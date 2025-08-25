import math
import numpy as np
from numpy.typing import NDArray
from physics.models.motor.base_motor import BaseMotor
from physics.models.constants import AIR_DENSITY

class Aeroshell(BaseMotor):
    def __init__(self):
        super().__init__()
        # self.air_density = AIR_DENSITY
        # self.vehicle_frontal_area = vehicle_frontal_area
        # self.drag_coefficient = drag_coefficient

    @staticmethod
    def calculate_drag_force(wind_speeds, wind_attack_angles, required_speed_ms):
        """
                Calculate the force of drag acting in the direction opposite the movement of the car at every tick.

                :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                :returns: (float[N]) the drag force in Newtons at every tick of the race
                :rtype: np.ndarray

        """
        # Lookup table mapping wind angle to drag values for a wind speed of 60 km/hr. Comes from CFD simulation in the google drive.
        angle_to_unscaled_drag = {
            0: 23.41,
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
        }

        drag_forces = np.zeros_like(wind_speeds, dtype=float)
        rounded_attack_angles = np.round(wind_attack_angles / 18) * 18
        unscaled_wind_drag = np.array(list(map(lambda x: angle_to_unscaled_drag[x], rounded_attack_angles)))

        # data from lookup table corresponds to wind speed of 16.667 m/s
        direction = np.sign(wind_speeds)
        wind_drag = direction * unscaled_wind_drag * (wind_speeds ** 2) / (16.667 ** 2)
        car_drag = angle_to_unscaled_drag[0] * (required_speed_ms ** 2) / (16.667 ** 2)
        drag_forces = wind_drag + car_drag

        return drag_forces





wind_attack_angles = np.array([0.0, 18.0, 36.0])
wind_speeds = np.full_like(wind_attack_angles, 16.67)
required_speed_ms = np.zeros_like(wind_speeds)

drag_force = Aeroshell.calculate_drag_force(wind_speeds, wind_attack_angles, required_speed_ms)
print(drag_force)




