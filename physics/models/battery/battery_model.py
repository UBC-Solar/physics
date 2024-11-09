import numpy as np
from .battery_config import BatteryModelConfig



class BatteryModel:
    """
    Class representing the Thevenin equivalent battery model with modular parameters

    Attributes:
        max_voltage (float): maximum voltage of the BrightSide battery pack (V)
        min_voltage (float): minimum voltage of the BrightSide battery pack (V)
        max_current_capacity (float): nominal capacity of the BrightSide battery pack (Ah)
        max_energy_capacity (float): nominal energy capacity of the BrightSide battery pack (Wh)

        state_of_charge (float): instantaneous battery state-of-charge (0.00 - 1.00)
        discharge_capacity (float): instantaneous amount of charge extracted from battery (Ah)
        voltage (float): instantaneous voltage of the battery (V)
        stored_energy (float): instantaneous energy stored in the battery (Wh)
    """

    def __init__(self, battery_config: BatteryModelConfig, state_of_charge = 1):

        """
        Constructor for BrightSide battery class.

        :param float state_of_charge: initial battery state of charge
        """

        # ----- Load Config -----

        self.R_P = battery_config.R_P
        self.C_P = battery_config.C_P
        self.max_current_capacity = battery_config.max_current_capacity
        self.max_energy_capacity = battery_config.max_energy_capacity
        self.nominal_charge_capacity = battery_config.Q_total
        U_oc_coefficients = np.array(battery_config.Uoc_data)
        R_0_coefficients = np.array(battery_config.R_0_data)

        
        # ----- Initialize Parameters -----

        self.U_oc = lambda soc: np.polyval(U_oc_coefficients, soc)          # V
        self.R_0 = lambda soc: np.polyval(R_0_coefficients, soc) / 1000     # Ohms

        self.U_P = 0.0              # V
        self.U_L = 0.0              # V
        self.state_of_charge = state_of_charge

        self.max_voltage = U_oc_coefficients[-1]
        self.min_voltage = U_oc_coefficients[0]


        # calculated the charging and discharging currents
        self.discharge_current = lambda P, U_oc, U_P, R_0: ((U_oc - U_P) - np.sqrt(np.power((U_oc - U_P), 2) - 4 * R_0 * P)) / (2 * R_0)
        self.charge_current = lambda P, U_oc, U_P, R_0: (-(U_oc + U_P) + np.sqrt(np.power((U_oc + U_P), 2) + 4 * R_0 * P)) / (2 * R_0)


    def _evolve(self, power: float, T: float):
        soc = self.state_of_charge          # State of Charge (dimensionless, 0 < soc < 1)
        U_P = self.U_P                      # Polarization Potential (V)
        R_P = self.R_P                      # Polarization Resistance (Ohms)
        U_oc = self.U_oc(soc)               # Open-Circuit Potential (V)
        R_0 = self.R_0(soc)                 # Ohmic Resistance (Ohms)
        Q = self.nominal_charge_capacity    # Nominal Charge Capacity (C)
        t = self.R_P * self.C_P             # Characteristic Time (seconds)

        I = self.discharge_current(power, U_oc, U_P, R_0) if power <= 0 else self.charge_current(power, U_oc, U_P, R_0)  # Current (A)

        new_soc = soc + (I * T / Q)
        new_U_P = np.exp(-T / t) * U_P + I * R_P * (1 - np.exp(-T / t))

        self.state_of_charge = new_soc
        self.U_P = new_U_P
        self.U_L = U_oc + U_P + (I * R_0)

    def update_array(self, delta_energy_array, tick):
        """
        Performs energy calculations with NumPy arrays

        :param cumulative_energy_array: a NumPy array containing the cumulative energy changes at each time step
        experienced by the battery

        :return: soc_array – a NumPy array containing the battery state of charge at each time step

        :return: voltage_array – a NumPy array containing the voltage of the battery at each time step

        :return: stored_energy_array– a NumPy array containing the energy stored in the battery at each time step

        """
        soc = np.empty_like(delta_energy_array, dtype=float)
        voltage = np.empty_like(delta_energy_array, dtype=float)
        for i, energy in enumerate(delta_energy_array):
            self._evolve(energy, tick)
            soc[i] = self.state_of_charge
            voltage[i] = self.U_L

        return soc, voltage
