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
        #direction = np.sign(wind_speeds)
        wind_drag = unscaled_wind_drag * (wind_speeds ** 2) / (16.667 ** 2)
        car_drag = angle_to_unscaled_drag[0] * (required_speed_ms ** 2) / (16.667 ** 2)
        drag_forces = wind_drag + car_drag

        return drag_forces





class AeroshellWithDownForce(Aeroshell):

    def __init__(self):
        super().__init__()

    @staticmethod
    def calculate_down_force(wind_speeds, wind_attack_angles, required_speed_ms ):
        """
                        Calculate the negative lift force i.e. drag acting against the normal force at every tick.

                        :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                        :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                        :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                        :returns: (float[N]) the down force in Newtons at every tick of the race
                        :rtype: np.ndarray

                """
        # Lookup table mapping wind angle to lift values for a wind speed of 60 km/hr. Comes from CFD simulation in the google drive. Positive sign convention as it is directed towards the ground, and is taken as the negative of lift
        angle_to_lift = {
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
        }

        down_forces = np.zeros_like(wind_speeds, dtype=float)
        rounded_attack_angles = np.round(wind_attack_angles / 18) * 18
        unscaled_wind = np.array(list(map(lambda x: angle_to_lift[x], rounded_attack_angles)))

        # data from lookup table corresponds to wind speed of 16.667 m/s
        direction = np.sign(wind_speeds)
        wind_down_force = direction * unscaled_wind * (wind_speeds ** 2) / (16.667 ** 2)
        car_down_force = angle_to_lift[0] * (required_speed_ms ** 2) / (16.667 ** 2)
        down_forces = wind_down_force + car_down_force

        return down_forces




