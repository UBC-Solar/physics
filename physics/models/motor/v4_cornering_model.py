from physics.models.motor import AdvancedMotor
from physics.models.steering import SteeringModel
import numpy as np

class V4CorneringModel(AdvancedMotor):
    car_mass = 160 # In kg, arbitrary value for now



    GRAVITY = 9.81  # N / kg
    def __init__(self, **kwargs):
        self.car_mass = 160 # kg
        self.track_width = 1.5 # In m, arbitrary value for now
        self.front_distance = 2  # meters, distance from front tire patch to center of gravity
        self.back_distance = 2  # meters, distance from back tire patch to center of gravity
        self.tires = {
            "FL": {
                "static_load": 30, # kg
            },
            "FR": {
                "static_load": 40, # kg
            },
            "BL": {
                "static_load": 40, # kg
            },
            "BR": {
                "static_load": 50, # kg
            }
        }


    def tireLoads (self, imu_lateral_acceleration, imu_vertical_acceleration):
        """
        Get dynamic tire loads of each tire (assumes IMU is oriented in non-stupid way)

        :param imu_lateral_acceleration in m^2/s, right is positive
        :param imu_vertical_acceleration in m^2/s, up is positive
        :return: list of dynamic tire loads
        """
        corner_force_in_Gs = imu_lateral_acceleration / self.GRAVITY
        apparent_mass = self.car_mass + self.car_mass(1 + imu_vertical_acceleration/GRAVITY)
        LLT = corner_force_in_Gs * apparent_mass * self.COG_height * self.track_width
        front_weight_transferred = (self.tires["FR"]["static_load"] + self.tires["FL"]["static_load"]) / self.car_mass * LLT
        back_weight_transferred = (self.tires["BR"]["static_load"] + self.tires["BL"]["static_load"]) / self.car_mass * LLT

        FL_load = self.tires["FL"]["static_load"] + front_weight_transferred
        FR_load = self.tires["FR"]["static_load"] - front_weight_transferred
        BR_load = self.tires["BR"]["static_load"] + back_weight_transferred
        BL_load = self.tires["BL"]["static_load"] - back_weight_transferred

        return FL_load, FR_load, BR_load, BL_load

    def SlipAngles (self, yolk_angle, yaw_rate, velocity):
        """

        :param yolk_angle: from sensor, clockwise is positive
        :param yaw_rate: from IMU, clockwise is positive
        :param velocity: given from GPS+odometry+IMU estimations
        :return:
        """
        FL_angle, FR_angle = SteeringModel.calculate_steer_angles(yoke_angle)

        FL_dif = - np.arctan( (yaw_rate*self.front_distance) / (velocity-self.track_width*yaw_rate))
        FR_dif = -np.arctan( (yaw_rate*self.front_distance) / (velocity+self.track_width*yaw_rate))
        BL_dif = -np.arctan( ())
        BR_dif


        FL_slip = FL_angle + FL_dif
        FR_slip = FR_angle + FR_dif
        BL_slip = BL_dif
        BR_slip = BR_dif

        return

    def getPowerLoss (self, velocity, imu_lateral_acceleration, imu_vertical_acceleration):
        FL_load, FR_load, BR_load, BL_load = self.tireLoads(imu_lateral_acceleration, imu_vertical_acceleration)
        FL_slip, FR_slip, BR_slip, BL_slip = self.slipAngles()

        power_FL = FL_load * imu_lateral_acceleration * velocity * np.tan(FL_slip)
        power_FR = FR_load * imu_lateral_acceleration * velocity * np.tan(FR_slip)
        power_BR = BR_load * imu_lateral_acceleration * velocity * np.tan(BR_slip)
        power_BL = BL_load * imu_lateral_acceleration * velocity * np.tan(BL_slip)

        return power_FL + power_FR + power_BR + power_BL
