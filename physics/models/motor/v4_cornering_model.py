from physics.models.motor import AdvancedMotor
import numpy as np

class V4CorneringModel(AdvancedMotor):
    def __init__(self, **kwargs):
        self.car_mass = 100 # In kg, arbitrary value for now
        self.track_width = 1.5 # In m, arbitrary value for now
        self.front_distance = 2  # meters, distance from front tire patch to center of gravity
        self.back_distance = 2  # meters, distance from back tire patch to center of gravity

    def getCorneringForce (self, IMU_lateral_acceleration):
        return IMU_lateral_acceleration * self.car_mass

    def getTravelDirection (self, ):
        pass

    def getSlipAngle (self, heading_angle, tire_angle):
        return np.abs(heading_angle - tire_angle)

    def getPowerLoss (self, velocity, slip_angle, cornering_force):
        return velocity * np.tan(slip_angle) * cornering_force
