from physics.models.motor import BasicMotor
import numpy as np

class V3CorneringModel(BasicMotor):
    def __init__(self, left: bool, front: bool, **kwargs):
        super().__init__(**kwargs)
        self.tire_stiffness = 50000 # arbitrary value for now
        self.trackwidth = 1.5 # meters, arbitrary
        self.front_distance = 2 # meters, distance from front tire patch to center of gravity
        self.back_distance = 2 # meters, distance from back tire patch to center of gravity
        self.front_weight_fraction = 0.5 # fraction of total weight on front axle (back axle fraction would be 1 - this quantity)
        self.left_side = left
        self.front_side = front
        self.COG_height = 0.5 # meters, arbitrary value for now

    def get_cornering_force(self, IMU_lateral_acceleration):
        """
        Function to calculate lateral force pulled in a corner.

        Param: lateral acceleration of the CoG from the IMU
        """
        return IMU_lateral_acceleration * self.vehicle_mass

    def get_slip (self, IMU_lateral_acceleration):
        """
        A function to estimate the slip angle of a tire using an estimated tire stiffness coefficient.

        Param: lateral acceleration of the CoG from the IMU
        Returns: slip angle in radians
        """
        slip_angle = self.get_cornering_force(IMU_lateral_acceleration) / self.tire_stiffness

        return slip_angle


    def calculate_tire_patch_load(self, x1, y1, x2, y2, x3, y3):
        """
        Function to find the load on a given tire patch with considerations to cornering and weight distribution

        Param 1:
        Returns: weight on chosen tire patch in kilograms
        """
        total_load = self.vehicle_mass

        if self.front_side:
            weight_fraction = self.front_weight_fraction
        else:
            weight_fraction = 1 - self.front_weight_fraction

        static_tire_load = total_load * weight_fraction / 2
        total_LLT = self.vehicle_mass * self.COG_height / self.trackwidth

        if (self.left_side == True):

        else:

    def calculate_power_loss(self, IMU_lateral_acceleration, x1, y1, x2, y2, x3, y3, forward_velocity):
        """
        Function to calculate power loss of 1 tire patch

        Last Param: Forward component of vehicle velocity relative to the direction the car is pointing in.
        Other Parameters here are really only used to call necessary functions, and are defined for said called functions. See specific function descriptions for more details.
        """
        power = np.tan( self.get_slip(IMU_lateral_acceleration) ) * forward_velocity * self.get_cornering_force(IMU_lateral_acceleration)

        return power