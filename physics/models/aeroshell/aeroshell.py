import numpy as np
from scipy.interpolate import make_interp_spline
from numpy.typing import NDArray
from physics.models.constants import AIR_DENSITY


class Aeroshell:

    def __init__(self, drag_lookup: dict[float, float], down_lookup: dict[float, float], wind_reference_speed, density):


        self.drag_lookup = drag_lookup #look up table (corresponds angle to CdA) that usually consists of data from a CFD carried out by the Aeroshell team
        self.down_lookup = down_lookup #similar look up table consisting of ClA(coefficients) to angle references.
        self.wind_reference_speed = wind_reference_speed #reference speed of the wind in m/s
        self.density = AIR_DENSITY

        drag_angles = np.array(list(drag_lookup.keys()))  # keys in the values of angles from the look_up table
        self.drag_coefficients = np.array(list(drag_lookup.values()))  # keys in the values of corresponding coefficients computed by the CFD from the look_up table
        self.angle_to_drag_coefficient = make_interp_spline(drag_angles, self.drag_coefficients,k=3)  # interpolation function to estimate values

        #similar procedure for down force
        down_angles = np.array(list(down_lookup.keys()))
        down_values = np.array(list(down_lookup.values()))
        self.angle_to_down_coefficient =make_interp_spline(down_angles, down_values,k=3)


    def calculate_aero_force(self, density, interpolation_function, wind_speeds: NDArray, wind_attack_angles: NDArray, required_speed_ms:NDArray, lookup_table:dict[float, float]):
        """
                Calculates aerodynamic forces - drag and down.
                In general, aerodynamic forces are described by:
                F = 1/2 * coefficient * density* area * (velocity)^2
                :param density: refers to the air density
                :param interpolation_function: refers to the interpolation function for drag or down force calculations
                :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where < 0 means against the direction of the vehicle
                :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                :param dict lookup_table: (float:float) specifies if force to be calculated is drag or down force. These values are from a CFD carried out by the Aeroshell team
                :returns: (float[N]) the aerodynamic force in Newtons at every tick of the race
                :rtype: np.ndarray
        """
        interp_coefficients = (interpolation_function(wind_attack_angles))        #interpolation maps angles to coefficients
        relative_speed = wind_speeds + required_speed_ms
        net_forces = 0.5 * density * interp_coefficients * (relative_speed **2)

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

        drag_force =  np.cos(np.radians(wind_attack_angles)) * self.calculate_aero_force(self.density, self.angle_to_drag_coefficient, wind_speeds, wind_attack_angles, required_speed_ms, self.drag_lookup)
        return drag_force

    def calculate_down(self, wind_speeds: NDArray, wind_attack_angles:NDArray,  required_speed_ms:NDArray):
        """
                      Specifically calculates the down force - negative lift force acting on the vehicle
                      :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where < 0 means against the direction of the vehicle
                      :param np.ndarray wind_attack_angles: (float[N]) The attack angle of the wind for a given moment
                      :param np.ndarray required_speed_ms: (float[N]) required speed array in m/s
                      :returns: (float[N]) the down force in Newtons at every tick of the race
                      :rtype: np.ndarray
        """

        down_force = self.calculate_aero_force(self.density, self.angle_to_down_coefficient, wind_speeds, wind_attack_angles, required_speed_ms, self.down_lookup)
        return down_force