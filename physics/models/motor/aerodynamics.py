import math
import numpy as np
from numpy.typing import NDArray
from physics.models.motor.base_motor import BaseMotor
from physics.models.constants import ACCELERATION_G, AIR_DENSITY

class AerodynamicMotor(BaseMotor):
    def __init__(self):
        super().__init__()

        self.vehicle_mass = vehicle_mass
        self.acceleration_g = ACCELERATION_G
        self.road_friction = road_friction
        self.tire_radius = tire_radius

        self.air_density = AIR_DENSITY
        self.vehicle_frontal_area = vehicle_frontal_area
        self.drag_coefficient = drag_coefficient

        self.friction_force = (self.vehicle_mass * self.acceleration_g * self.road_friction)