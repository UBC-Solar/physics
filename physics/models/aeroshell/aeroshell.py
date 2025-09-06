
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


class Aeroshell:
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

    def __init__(self, drag_lookup, down_lookup):
        self.drag_lookup = drag_lookup
        self.down_lookup = down_lookup

    @staticmethod
    def calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, lookup_table):
        """
                Calculate the force of drag acting in the direction opposite the movement of the car at every tick.

                :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                :param str force_type: (str) if force to be calculated is drag or down force
                :returns: (float[N]) the drag force in Newtons at every tick of the race
                :rtype: np.ndarray

        """

        direction = np.sign(wind_speeds)
        angles = np.array(list(lookup_table.keys()))
        values = np.array(list(lookup_table.values()))
        func = make_interp_spline(angles, values, k=3)
        force_ref = func(wind_attack_angles)
        wind_drag = direction * force_ref * (wind_speeds ** 2) / (16.667 ** 2)
        car_drag = lookup_table[0] * (required_speed_ms ** 2) / (16.667 ** 2)
        drag_forces = wind_drag + car_drag

        return drag_forces

    def calculate_drag(self, wind_speeds, wind_attack_angles, required_speed_ms):
        return Aeroshell.calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, self.drag_lookup)

    def calculate_down(self, wind_speeds, wind_attack_angles, required_speed_ms):
        return Aeroshell.calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, self.down_lookup)
