import math
import numpy as np

from physics.models.motor.base_motor import BaseMotor
from physics.models.constants import ACCELERATION_G, AIR_DENSITY


class BasicMotor(BaseMotor):
    def __init__(self, vehicle_mass, road_friction, tire_radius, vehicle_frontal_area, drag_coefficient):
        super().__init__()

        # Instantaneous voltage supplied by the battery to the motor controller
        self.dc_v = 0

        # Instantaneous current supplied by the battery to the motor controller
        self.dc_i = 0

        self.input_power = 0
        self.vehicle_mass = vehicle_mass
        self.acceleration_g = ACCELERATION_G
        self.road_friction = road_friction
        self.tire_radius = tire_radius

        self.air_density = AIR_DENSITY
        self.vehicle_frontal_area = vehicle_frontal_area
        self.drag_coefficient = drag_coefficient

        self.friction_force = (self.vehicle_mass * self.acceleration_g * self.road_friction)

        self.e_mc = 0.98  # motor controller efficiency, subject to change
        self.e_m = 0.9  # motor efficiency, subject to change

        # print("torque experienced by motor: {} Nm".format(self.constant_torque))

    @staticmethod
    def calculate_motor_efficiency(motor_angular_speed, motor_output_energy, tick):
        """

        Calculates a NumPy array of motor efficiency from NumPy array of operating angular speeds and NumPy array
            of output power. Based on data obtained from NGM SC-M150 Datasheet and modelling done in MATLAB

        r squared value: 0.873

        :param np.ndarray motor_angular_speed: (float[N]) angular speed motor operates in rad/s
        :param np.ndarray motor_output_energy: (float[N]) energy motor outputs to the wheel in J
        :param float tick: length of 1 update cycle in seconds
        :returns e_m: (float[N]) efficiency of the motor
        :rtype: np.ndarray

        """

        # Power = Energy / Time
        motor_output_power = motor_output_energy * tick
        rads_rpm_conversion_factor = 30 / math.pi

        revolutions_per_minute = motor_angular_speed * rads_rpm_conversion_factor

        e_m = calculate_motor_efficiency(motor_output_power, revolutions_per_minute)

        e_m[e_m < 0.7382] = 0.7382
        e_m[e_m > 1] = 1

        return e_m

    @staticmethod
    def calculate_motor_controller_efficiency(motor_angular_speed, motor_output_energy, tick):
        """

        Calculates a NumPy array of motor controller efficiency from NumPy array of operating angular speeds and
        NumPy array of output power. Based on data obtained from the WaveSculptor Motor Controller Datasheet efficiency
        curve for a 90 V DC Bus and modelling done in MATLAB.

        r squared value: 0.7431

        :param np.ndarray motor_angular_speed: (float[N]) angular speed motor operates in rad/s
        :param np.ndarray motor_output_energy: (float[N]) energy motor outputs to the wheel in J
        :param float tick: length of 1 update cycle in seconds
        :returns e_mc (float[N]) efficiency of the motor controller
        :rtype: np.ndarray

        """

        # Ignore nan warning. Set nan value to 0
        np.seterr(divide='ignore', invalid='ignore')

        # Power = Energy / Time
        motor_output_power = motor_output_energy / tick

        # Torque = Power / Angular Speed
        motor_torque_array = np.nan_to_num(motor_output_power / motor_angular_speed)

        np.seterr(divide='warn', invalid='warn')

        e_mc = calculate_motor_controller_efficiency(motor_angular_speed, motor_torque_array)

        e_mc[e_mc < 0.9] = 0.9
        e_mc[e_mc > 1] = 1

        return e_mc

    def calculate_energy_in(self, required_speed_kmh, gradients, wind_speeds, tick):
        """

        Create a function which takes in array of elevation, array of wind speed, required
            speed, returns the consumed energy.

        :param np.ndarray required_speed_kmh: (float[N]) required speed array in km/h
        :param np.ndarray gradients: (float[N]) gradient at parts of the road
        :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
        :param float tick: length of 1 update cycle in seconds
        :returns: (float[N]) energy expended by the motor at every tick
        :rtype: np.ndarray

        """
        required_speed_ms = required_speed_kmh / 3.6

        acceleration_ms2 = np.clip(np.gradient(required_speed_ms), a_min=0, a_max=None)
        acceleration_force = acceleration_ms2 * self.vehicle_mass

        required_angular_speed_rads = required_speed_ms / self.tire_radius

        drag_forces = 0.5 * self.air_density * (
                (required_speed_ms + wind_speeds) ** 2) * self.drag_coefficient * self.vehicle_frontal_area

        angles = np.arctan(gradients)
        g_forces = self.vehicle_mass * self.acceleration_g * np.sin(angles)

        road_friction_array = self.road_friction * self.vehicle_mass * self.acceleration_g * np.cos(angles)

        net_force = road_friction_array + drag_forces + g_forces + acceleration_force

        motor_output_energies = required_angular_speed_rads * net_force * self.tire_radius * tick
        motor_output_energies = np.clip(motor_output_energies, a_min=0, a_max=None)

        e_m = self.calculate_motor_efficiency(required_angular_speed_rads, motor_output_energies, tick)
        e_mc = self.calculate_motor_controller_efficiency(required_angular_speed_rads, motor_output_energies, tick)

        motor_controller_input_energies = motor_output_energies / (e_m * e_mc)

        # Filter out and replace negative energy consumption as 0
        motor_controller_input_energies = np.where(motor_controller_input_energies > 0,
                                                   motor_controller_input_energies, 0)

        return motor_controller_input_energies

    def __str__(self):
        return (f"Tire radius: {self.tire_radius}m\n"
                f"Rolling resistance coefficient: {self.road_friction}\n"
                f"Vehicle mass: {self.vehicle_mass}kg\n"
                f"Acceleration of gravity: {self.acceleration_g}m/s^2\n"
                f"Motor controller efficiency: {self.e_mc}%\n"
                f"Motor efficiency: {self.e_m}%\n")


def calculate_motor_efficiency(motor_output_power, revolutions_per_minute):
    return 0.7382 - (6.281e-5 * motor_output_power) + (6.708e-4 * revolutions_per_minute) \
        - (2.89e-8 * motor_output_power ** 2) + (2.416e-7 * motor_output_power * revolutions_per_minute) \
        - (8.672e-7 * revolutions_per_minute ** 2) + (5.653e-12 * motor_output_power ** 3) \
        - (1.74e-11 * motor_output_power ** 2 * revolutions_per_minute) \
        - (7.322e-11 * motor_output_power * revolutions_per_minute ** 2) \
        + (3.263e-10 * revolutions_per_minute ** 3)


def calculate_motor_controller_efficiency(motor_angular_speed, motor_torque_array):
    return 0.7694 + (0.007818 * motor_angular_speed) + (0.007043 * motor_torque_array) \
        - (1.658e-4 * motor_angular_speed ** 2) - (1.806e-5 * motor_torque_array * motor_angular_speed) \
        - (1.909e-4 * motor_torque_array ** 2) + (1.602e-6 * motor_angular_speed ** 3) \
        + (4.236e-7 * motor_angular_speed ** 2 * motor_torque_array) \
        - (2.306e-7 * motor_angular_speed * motor_torque_array ** 2) \
        + (2.122e-06 * motor_torque_array ** 3) - (5.701e-09 * motor_angular_speed ** 4) \
        - (2.054e-9 * motor_angular_speed ** 3 * motor_torque_array) \
        - (3.126e-10 * motor_angular_speed ** 2 * motor_torque_array ** 2) \
        + (1.708e-09 * motor_angular_speed * motor_torque_array ** 3) \
        - (8.094e-09 * motor_torque_array ** 4)
