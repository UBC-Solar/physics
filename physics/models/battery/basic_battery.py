import numpy as np
from numpy.polynomial import Polynomial

from physics.models.battery.base_battery import BaseBattery


class BasicBattery(BaseBattery):
    """
    Class representing the DayBreak battery pack.

    Attributes:
        max_voltage (float): maximum voltage of the DayBreak battery pack (V)
        min_voltage (float): minimum voltage of the DayBreak battery pack (V)
        max_current_capacity (float): nominal capacity of the DayBreak battery pack (Ah)
        max_energy_capacity (float): nominal energy capacity of the DayBreak battery pack (Wh)

        state_of_charge (float): instantaneous battery state-of-charge (0.00 - 1.00)
        discharge_capacity (float): instantaneous amount of charge extracted from battery (Ah)
        voltage (float): instantaneous voltage of the battery (V)
        stored_energy (float): instantaneous energy stored in the battery (Wh)
    """

    def __init__(self, state_of_charge, max_voltage, min_voltage, max_current_capacity, max_energy_capacity):
        """

        Constructor for BasicBattery class.

        :param float state_of_charge: initial battery state of charge

        """

        # ----- DayBreak battery constants -----

        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.max_current_capacity = max_current_capacity
        self.max_energy_capacity = max_energy_capacity

        # ----- DayBreak battery equations -----

        self.calculate_voltage_from_discharge_capacity = calculate_voltage_from_discharge_capacity()

        self.calculate_energy_from_discharge_capacity = calculate_energy_from_discharge_capacity()

        self.calculate_soc_from_discharge_capacity = calculate_soc_from_discharge_capacity(self.max_current_capacity)

        self.calculate_discharge_capacity_from_soc = calculate_discharge_capacity_from_soc(self.max_current_capacity)

        self.calculate_discharge_capacity_from_energy = calculate_discharge_capacity_from_energy()

        # ----- DayBreak battery variables -----

        self.state_of_charge = state_of_charge

        # SOC -> discharge_capacity
        self.discharge_capacity = self.calculate_discharge_capacity_from_soc(self.state_of_charge)

        # discharge_capacity -> voltage
        self.voltage = self.calculate_voltage_from_discharge_capacity(self.discharge_capacity)

        # discharge_capacity -> energy
        self.stored_energy = self.max_energy_capacity - self.calculate_energy_from_discharge_capacity(
            self.discharge_capacity)

        # ----- DayBreak battery initialisation -----

        super().__init__(self.stored_energy, self.max_current_capacity, self.max_energy_capacity,
                         self.max_voltage, self.min_voltage, self.voltage, self.state_of_charge)

    def update_array(self, cumulative_energy_array):
        """
        Performs energy calculations with NumPy arrays

        :param cumulative_energy_array: a NumPy array containing the cumulative energy changes at each time step
        experienced by the battery

        :return: soc_array – a NumPy array containing the battery state of charge at each time step

        :return: voltage_array – a NumPy array containing the voltage of the battery at each time step

        :return: stored_energy_array– a NumPy array containing the energy stored in the battery at each time step

        """

        stored_energy_array = np.full_like(cumulative_energy_array, fill_value=self.stored_energy)
        stored_energy_array += cumulative_energy_array / 3600
        stored_energy_array = np.clip(stored_energy_array, a_min=0, a_max=self.max_energy_capacity)

        energy_discharged_array = np.full_like(cumulative_energy_array, fill_value=self.max_energy_capacity) - \
                                  stored_energy_array

        discharge_capacity_array = self.calculate_discharge_capacity_from_energy(energy_discharged_array)

        soc_array = self.calculate_soc_from_discharge_capacity(discharge_capacity_array)
        voltage_array = self.calculate_voltage_from_discharge_capacity(discharge_capacity_array)

        return soc_array, voltage_array, stored_energy_array

    def get_raw_soc(self, cumulative_energy_array):
        """

        Return the not truncated (SOC is allowed to go above 100% and below 0%) state of charge.

        :param np.ndarray cumulative_energy_array: a NumPy array containing the cumulative energy changes at each time step
        experienced by the battery

        :return: a NumPy array containing the battery state of charge at each time step
        :rtype: np.ndarray

        """

        stored_energy_array = np.full_like(cumulative_energy_array, fill_value=self.stored_energy)
        stored_energy_array += cumulative_energy_array / 3600

        energy_discharged_array = np.full_like(cumulative_energy_array, fill_value=self.max_energy_capacity) - stored_energy_array

        discharge_capacity_array = self.calculate_discharge_capacity_from_energy(energy_discharged_array)

        soc_array = self.calculate_soc_from_discharge_capacity(discharge_capacity_array)

        return soc_array


def calculate_voltage_from_discharge_capacity():
    return Polynomial([117.6, -0.858896])  # -0.8589x + 117.6


def calculate_energy_from_discharge_capacity():
    return Polynomial([0, 117.6, -0.429448])  # -0.4294x^2 + 117.6x


def calculate_soc_from_discharge_capacity(max_current_capacity):
    return Polynomial([1, -1 / max_current_capacity])


def calculate_discharge_capacity_from_soc(max_current_capacity):
    return Polynomial([max_current_capacity, -max_current_capacity])


def calculate_discharge_capacity_from_energy():
    return lambda x: 136.92 - np.sqrt(18747.06027 - 2.32857 * x)
