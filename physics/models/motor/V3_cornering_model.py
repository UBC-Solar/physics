from physics.models.motor import BasicMotor
import numpy as np
from physics.models.steering import SteeringModel

class V3CorneringModel(BasicMotor):
    def __init__(self, left: bool, front: bool, **kwargs):
        super().__init__(**kwargs)
        self.tire_stiffness = 50000 #arbitrary value for now
        self.trackwidth = 1.5 #meters, arbitrary
        self.front_distance = 2 #meters, distance from front tire patch to center of gravity
        self.back_distance = 2 #meters, distance from back tire patch to center of gravity
        self.front_weight = 0.5 #fraction of total weight on front axle (back axle fraction would be 1 - this quantity)
        self.left_side = left
        self.front_side = front
        self.COG_height = 0.5 #meters, arbitrary value for now

    def get_slip (self, IMU_lateral_acceleration, ):
        """
        A function to estimate the slip angle of a tire.

        Param 1: lateral acceleration of the CoG from the IMU
        Param 2:
        """
        cornering_force = IMU_lateral_acceleration * self.vehicle_mass


    def calculate_tire_patch_load(self, x1, y1, x2, y2, x3, y3):
        """
        Function to find the load on a given tire patch with considerations to cornering and weight distribution

        Returns: weight on chosen tire patch in kilograms
        """
        total_load = self.vehicle_mass

        if (self.front_side == True):
            weight_fraction = self.front_weight
        else:
            weight_fraction = 1 - self.front_weight

        static_tire_load = total_load * weight_fraction / 2

        total_LLT = self.vehicle_mass * self.COG_height / self.trackwidth

        if (self.left_side == True):

        else:

    def calculate_power_loss(self, IMU_lateral_acceleration, x1, y1, x2, y2, x3, y3, forward_velocity):
        """
        Function to calculate power loss of 1 tire patch

        Last Param: Forward component of vehicle velocity relative to direction of the car.
        Other Parameters here are really only used to call necessary functions, and are defined for said called functions. See specific function descriptions for more details.
        """

        power = np.tan( get_slip(IMU_lateral_acceleration) ) * forward_velocity *

        return power
