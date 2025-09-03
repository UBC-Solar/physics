import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


class Aeroshell():
    """
        The Aeroshell class is used to calculate two aerodynamic forces: Drag and down force

        Drag force refers to the resistive force that affects the vehicles. This is considered by both the wind and motion of the car
        Down force refers to negative lift force i.e. acting against the normal force at every tick

                        :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                        :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                        :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                        :returns: (float[N]) the drag or down force in Newtons at every tick of the race
                        :rtype: np.ndarray

                ""

        The look up table data comes from a CFD carried out by the Aeroshell team - https://docs.google.com/spreadsheets/d/1D1ydUj-6aG-gBzlq2zTr8WgdxVIGQ9PqUZPuLfDphcg/edit?usp=sharing

    """
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
    angle_to_down_force = {
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


    def __init__(self):
        pass

    def calculate_aero_force(self, wind_speeds, wind_attack_angles, required_speed_ms, force_type):
        """
                Calculate the force of drag acting in the direction opposite the movement of the car at every tick.

                :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                :param str force_type: (str) if force to be calculated is drag or down force
                :returns: (float[N]) the drag force in Newtons at every tick of the race
                :rtype: np.ndarray

        """
        # Lookup table mapping wind angle to drag values for a wind speed of 60 km/hr. Comes from CFD simulation in the google drive.

        if force_type == "drag":
            table = Aeroshell.angle_to_unscaled_drag
        elif force_type == "down":
            table = Aeroshell.angle_to_down_force
        else:
            raise ValueError("force_type must be drag or down")


        direction = np.sign(wind_speeds)
        angles = np.array(list(table.keys()))
        values = np.array(list(table.values()))
        func = make_interp_spline(angles, values, k=3)
        force_ref = func(wind_attack_angles)
        wind_drag = direction * force_ref * (wind_speeds ** 2) / (16.667 ** 2)
        car_drag = table[0] * (required_speed_ms ** 2) / (16.667 ** 2)
        drag_forces = wind_drag + car_drag


        return drag_forces











