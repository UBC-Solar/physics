from physics.models.regen import BaseRegen
import numpy as np


class BasicRegen(BaseRegen):
    GRAVITY = 9.81
    EFFICIENCY = 0.5  # currently set to 50% but best case scenario is 60-70%

    def __init__(self, vehicle_mass):
        super().__init__()
        self.min_decel_mag = 0
        self.vehicle_mass = vehicle_mass
        self.kmh_to_mps = 0.278

    def get_regen_efficiency(self, speed_array):
        """
        Returns a numpy array of regen efficiency percentage based on the vehicle speed in m/s.

        :param speed_array: a numpy array of speeds in m/s
        :returns: numpy array of regen efficiency percentage
        """
        # Efficiency polynomial, more details can be found in regen_analysis folder located in data_analysis
        efficiency_poly = [0.022288416685942, 0.026545396753597]

        return np.polyval(efficiency_poly, speed_array)

    def calculate_produced_energy(self, speed_kmh, gis_route_elevations, min_regen_speed, max_power):
        """
        Returns a numpy array containing the energy produced by regen
        during each tick of the race based on the change in energy in that tick
        :param speed_kmh: an array containing the speeds at each tick
        :param gis_route_elevations: an array containing elevations on the route at each tick
        """
        # get the changes of energy from tick i to tick i + 1
        speed_ms = speed_kmh / 3.6  # Convert to m/s from km/h
        delta_kinetic_energy = np.diff((1 / 2) * self.vehicle_mass * pow(speed_ms, 2), append=[0])
        delta_potential_energy = np.diff(self.vehicle_mass * self.GRAVITY * gis_route_elevations, append=[0])

        # get the total change in energy at each tick
        delta_energy = delta_kinetic_energy + delta_potential_energy

        # create regen energy produced array
        # if delta_energy is negative, we regen that energy back at the set efficiency rate; else 0 energy regen
        efficiencies = self.get_regen_efficiency(speed_ms)
        produced_energy = np.where(delta_energy < 0, abs(delta_energy) * efficiencies, 0)

        # Regen does not occur below a certain speed
        produced_energy = np.where(speed_ms >= min_regen_speed, produced_energy, 0)

        # Regen power is capped by current limitations

        return np.clip(produced_energy, a_min=0, a_max=max_power)
