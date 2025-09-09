import numpy as np
from scipy.interpolate import make_interp_spline
from numpy.typing import NDArray


class Aeroshell:


#just need to keep the interpolation stuff in the constructor
    def __init__(self, drag_lookup: dict[float, float], down_lookup: dict[float, float], wind_reference_speed):

        self.drag_lookup = drag_lookup #look up table (corresponds angle to force) that usually consists of data from a CFD carried out by the Aeroshell team
        self.down_lookup = down_lookup #similar look up table consisting of down force to angle references.
        self.wind_reference_speed = wind_reference_speed #reference speed of the wind in m/s

        drag_angles = np.array(list(drag_lookup.keys()))  # keys in the values of angles from the look_up table
        drag_values = np.array(list(drag_lookup.values()))  # keys in the values of corresponding forces computed by the CFD from the look_up table
        self.angle_to_drag_force = make_interp_spline(drag_angles, drag_values,k=3)  # interpolation function to estimate values of angles
        #similar procedure for down force
        down_angles = np.array(list(down_lookup.keys()))
        down_values = np.array(list(down_lookup.values()))
        self.angle_to_down_force =make_interp_spline(down_angles, down_values,k=3)


    def calculate_aero_force(self, interpolation_function, wind_speeds: NDArray, wind_attack_angles: NDArray, required_speed_ms:NDArray, lookup_table:dict[float, float]):
        """
                Calculates aerodynamic forces - drag and down.
                In general, aerodynamic forces are described by:
                F = 1/2 * coefficient * density* area * (velocity)^2
                :param interpolation_function: refers to the interpolation function for drag or down force calculations
                :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where < 0 means against the direction of the vehicle
                :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                :param dict lookup_table: (float:float) specifies if force to be calculated is drag or down force. These values are from a CFD carried out by the Aeroshell team
                :returns: (float[N]) the aerodynamic force in Newtons at every tick of the race
                :rtype: np.ndarray
        """

        direction = np.sign(wind_speeds)  # refers to the direction of wind (tailwind vs headwind), this is used to compute directional drag
        interp_angles = interpolation_function(wind_attack_angles)  # interpolated angles
        wind_force = direction * interp_angles * (wind_speeds ** 2) / (self.wind_reference_speed ** 2) #scaled relative to the square of wind speed, also accounts for direction of the wind
        car_force = lookup_table[0] * (required_speed_ms ** 2) / (self.wind_reference_speed ** 2)  #aerodynamic force scaled relative to reference force from the given lookup table and velocity squared
        net_forces = wind_force + car_force

        return net_forces

    def calculate_drag(self, wind_speeds:NDArray, wind_attack_angles:NDArray, required_speed_ms:NDArray):
        """
                      Specifically calculates the force of drag acting in the direction opposite the movement of the car at every tick.
                      :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where < 0 means against the direction of the vehicle
                      :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                      :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                      :returns: (float[N]) the drag force in Newtons at every tick of the race
                      :rtype: np.ndarray
        """

        return self.calculate_aero_force(self.angle_to_drag_force, wind_speeds, wind_attack_angles, required_speed_ms, self.drag_lookup)

    def calculate_down(self, wind_speeds: NDArray, wind_attack_angles:NDArray,  required_speed_ms:NDArray):
        """
                      Specifically calculates the down force - negative lift force acting on the vehicle
                      :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where < 0 means against the direction of the vehicle
                      :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                      :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                      :returns: (float[N]) the down force in Newtons at every tick of the race
                      :rtype: np.ndarray
        """

        return self.calculate_aero_force(self.angle_to_down_force, wind_speeds, wind_attack_angles, required_speed_ms, self.down_lookup)