import numpy as np
from scipy.interpolate import make_interp_spline


class Aeroshell:
    """
        The Aeroshell class is used to calculate two aerodynamic forces: Drag and down force

        Drag force refers to the resistive force that affects the vehicles. This is considered by both the wind and motion of the car
        Down force refers to negative lift force i.e. acting against the normal force at every tick
                        :param dict drag_lookup: refers to a look up table that usually consists of data from a CFD carried out by the Aeroshell team
                        :param dict down_lookup: similar look up table consisting of down force to angle references.
                        :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                        :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                        :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                        :returns: (float[N]) the drag or down force in Newtons at every tick of the race
                        :rtype: np.ndarray

                ""
    """

    def __init__(self, drag_lookup: dict[float, float], down_lookup: dict[float, float]):
        self.drag_lookup = drag_lookup
        self.down_lookup = down_lookup

    @staticmethod
    def calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, lookup_table):
        """
                Calculates aerodynamic forces - drag and down  the force of drag acting in the direction opposite the movement of the car at every tick.

                :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                :param dict lookup_table: (float:float) specifies if force to be calculated is drag or down force
                :returns: (float[N]) the aerodynamic force in Newtons at every tick of the race
                :rtype: np.ndarray

        """
        wind_speed_ref = 16.667  # reference speed for force calculations
        direction = np.sign(wind_speeds)
        angles = np.array(list(lookup_table.keys()))
        values = np.array(list(lookup_table.values()))
        func = make_interp_spline(angles, values, k=3)
        force_ref = func(wind_attack_angles)
        wind_force = direction * force_ref * (wind_speeds ** 2) / (wind_speed_ref ** 2)
        car_force = lookup_table[0] * (required_speed_ms ** 2) / (wind_speed_ref ** 2)
        net_forces = wind_force + car_force

        return net_forces

    def calculate_drag(self, wind_speeds, wind_attack_angles, required_speed_ms):
        """
                      Specifically calculates the force of drag acting in the direction opposite the movement of the car at every tick.

                      :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                      :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                      :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                      :returns: (float[N]) the aerodynamic force in Newtons at every tick of the race
                      :rtype: np.ndarray
        """
        return Aeroshell.calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, self.drag_lookup)

    def calculate_down(self, wind_speeds, wind_attack_angles, required_speed_ms):
        """
                      Specifically calculates the down force - negative lift force acting on the vehicle
                      :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
                      :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                      :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                      :returns: (float[N]) the aerodynamic force in Newtons at every tick of the race
                      :rtype: np.ndarray
        """
        return Aeroshell.calculate_aero_force(wind_speeds, wind_attack_angles, required_speed_ms, self.down_lookup)
