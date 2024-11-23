import numpy as np
import core
from scipy import optimize
from physics.models.battery.battery_config import BatteryModelConfig


class BatteryModel:
    """
    Class representing the Thevenin equivalent battery model with modular parameters.

    Attributes:
        R_P (float): Polarization resistance of the battery (Ohms).
        C_P (float): Polarization capacitance of the battery (Farads).
        max_current_capacity (float): Nominal capacity of the battery (Ah).
        max_energy_capacity (float): Maximum energy capacity of the battery (Wh).
        nominal_charge_capacity (float): Total charge capacity of the battery (Coulombs).
        state_of_charge (float): Current state of charge (dimensionless, 0.0 to 1.0).
        U_oc_coefficients (np.ndarray): Coefficients for the open-circuit voltage polynomial.
        R_0_coefficients (np.ndarray): Coefficients for the ohmic resistance polynomial.
        U_oc (callable): Function for open-circuit voltage as a function of state of charge (V).
        R_0 (callable): Function for ohmic resistance as a function of state of charge (Ohms).
        U_P (float): Current polarization potential (V).
        U_L (float): Current terminal voltage (V).
        tau (float): Time constant of the battery model (seconds).
    """

    def __init__(self, battery_config: BatteryModelConfig, state_of_charge=1):
        """
        Constructor for the BatteryModel class.

        :param BatteryModelConfig battery_config: Configuration object containing the battery's parameters and data.
        :param float state_of_charge: Initial state of charge of the battery (default is 1.0, fully charged).
        """

        # ----- Load Config -----

        self.R_P = battery_config.R_P
        self.C_P = battery_config.C_P
        self.max_current_capacity = battery_config.max_current_capacity
        self.max_energy_capacity = battery_config.max_energy_capacity
        self.nominal_charge_capacity = battery_config.Q_total
        Soc_data = battery_config.SOC_data
        Uoc_data = battery_config.Uoc_data
        R_0_data = battery_config.R_0_data

        # ----- Initialize Parameters -----
        def quintic_polynomial(x, x0, x1, x2, x3, x4):
            return np.polyval(np.array([x0, x1, x2, x3, x4]), x)

        self.U_oc_coefficients, _ = optimize.curve_fit(quintic_polynomial, Soc_data, Uoc_data)
        self.R_0_coefficients, _ = optimize.curve_fit(quintic_polynomial, Soc_data, R_0_data)
        self.U_oc = lambda soc: np.polyval(self.U_oc_coefficients, soc)  # V
        self.R_0 = lambda soc: np.polyval(self.R_0_coefficients, soc)  # Ohms

        self.U_P = 0.0  # V
        self.U_L = 0.0  # V
        self.state_of_charge = state_of_charge

        self.tau = self.R_P * self.C_P  # Characteristic Time (seconds)

        # calculated the charging and discharging currents
        self.discharge_current = lambda P, U_oc, U_P, R_0: ((U_oc - U_P) - np.sqrt(
            np.power((U_oc - U_P), 2) - 4 * R_0 * P)) / (2 * R_0)
        self.charge_current = lambda P, U_oc, U_P, R_0: (-(U_oc + U_P) + np.sqrt(
            np.power((U_oc + U_P), 2) + 4 * R_0 * P)) / (2 * R_0)

    def _evolve(self, power: float, tick: float) -> None:
        """
        Update the battery state given the power and time elapsed.

        :param float power: Power applied to the battery (W). Positive for charging, negative for discharging.
        :param float T: Time interval over which the power is applied (seconds).
        """
        soc = self.state_of_charge  # State of Charge (dimensionless, 0 < soc < 1)
        U_P = self.U_P  # Polarization Potential (V)
        R_P = self.R_P  # Polarization Resistance (Ohms)
        U_oc = self.U_oc(soc)  # Open-Circuit Potential (V)
        R_0 = self.R_0(soc)  # Ohmic Resistance (Ohms)
        Q = self.nominal_charge_capacity  # Nominal Charge Capacity (C)

        current = self.discharge_current(power, U_oc, U_P, R_0) if power <= 0 else self.charge_current(power, U_oc, U_P, R_0)  # Current (A)

        new_soc = soc + (current * tick / Q)
        new_U_P = np.exp(-tick / self.tau) * U_P + current * R_P * (1 - np.exp(-tick / self.tau))

        self.state_of_charge = new_soc
        self.U_P = new_U_P
        self.U_L = U_oc + U_P + (current * R_0)

    def update_array(self, delta_energy_array, tick, rust=True):
        """
        Compute the battery's state of charge, voltage, and stored energy over time.
        This function is a wrapper for the Rust-based and Python-based implementations.

        :param np.ndarray delta_energy_array: Array of energy changes (J) at each time step.
        :param float tick: Time interval for each step (seconds).
        :param bool rust: If True, use Rust-based calculations (default is True).

        :return: A tuple containing arrays for state-of-charge, voltage, and stored energy.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
        """

        if rust:
            return core.update_battery_array(
                delta_energy_array,
                tick,
                self.state_of_charge,
                self.U_P,
                self.R_P,
                self.R_0_coefficients,
                self.U_oc_coefficients,
                self.tau,
                self.nominal_charge_capacity
            )
        else:
            return self._update_array_py(delta_energy_array, tick)

    def _update_array_py(self, delta_energy_array, tick):
        """
        Perform energy calculations using Python (fallback method if Rust is disabled).

        :param np.ndarray delta_energy_array: Array of energy changes (J) at each time step.
        :param float tick: Time interval for each step (seconds).

        :return: A tuple containing arrays for state-of-charge and voltage.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        soc = np.empty_like(delta_energy_array, dtype=float)
        voltage = np.empty_like(delta_energy_array, dtype=float)
        for i, energy in enumerate(delta_energy_array):
            self._evolve(energy, tick)
            soc[i] = self.state_of_charge
            voltage[i] = self.U_L

        return soc, voltage
