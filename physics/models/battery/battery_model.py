import numpy as np
import physics_rs
from typing import Callable, TypeAlias, Protocol, runtime_checkable, Optional, cast
from numpy.typing import NDArray


SOCDependent: TypeAlias = Callable[[float | NDArray[float]], float | NDArray[float]]


@runtime_checkable
class EquivalentCircuitModelConfig(Protocol):
    """
    A specification for a configuration object which contains the requisite data to specify
    a `EquivalentCircuitBatteryModel`.
    """

    @property
    def get_Uoc(self) -> SOCDependent:
        """
        A map from an SOC to Uoc (open-circuit voltage).
        Should be compatible with non-vectorized and vectorized calls: float -> float or NDArray -> NDArray
        """
        ...

    @property
    def get_R_0(self) -> SOCDependent:
        """
        A map from an SOC to R_0 (internal resistance).
        Should be compatible with non-vectorized and vectorized calls: float -> float or NDArray -> NDArray
        """
        ...

    @property
    def get_R_P(self) -> SOCDependent:
        """
        A map from an SOC to R_P (polarization resistance).
        Should be compatible with non-vectorized and vectorized calls: float -> float or NDArray -> NDArray
        """
        ...

    @property
    def get_C_P(self) -> SOCDependent:
        """
        A map from an SOC to C_P (polarization capacitance).
        Should be compatible with non-vectorized and vectorized calls: float -> float or NDArray -> NDArray
        """
        ...

    @property
    def Q_total(self) -> float:
        """
        The total charge capacity of the battery pack, in Coulombs.
        """
        ...


class EquivalentCircuitBatteryModel:
    """
    A first-order Thevenin equivalent model of a lithium-ion battery pack
    """

    def __init__(self, battery_config: EquivalentCircuitModelConfig, state_of_charge: float = 1.0):
        """
        Constructor for the EquivalentCircuitBatteryModel class.

        :param BatteryModelConfig battery_config: Configuration object containing the battery's parameters and data.
        :param float state_of_charge: Initial state of charge of the battery (default is 1.0, fully charged).
        """

        # We initialize the active components as uncharged
        self._U_P = 0.0  # V
        self._U_L = 0.0  # V
        self._state_of_charge = state_of_charge
        self._nominal_charge_capacity = battery_config.Q_total

        # Now, the config contains methods to map SOC to each respective parameter.
        # We can't efficiently pass these functions to compiled libraries.
        # Instead, we will pre-compute the parameters as a function of SOC
        # to create fine lookup tables as a portable substitute for runtime computation.

        # Things are going to get a tiny bit messy here, so we will go through this carefully.
        # I'll write what each resulting map achieves below each code block.

        # Firstly, we're going to discretize SOC by making a range of SOC values in the range [-0.05, 1.1], because
        # sometimes we are marginally outside the range (0.0, 1.0].
        # We will quantize at about 4 digits of precision, so ~10,000 values
        self._min_soc = -0.05
        self._max_soc = 1.1
        self._num_indices = int((self._max_soc - self._min_soc) * 10000)
        SOC_values = np.linspace(self._min_soc, self._max_soc, self._num_indices, dtype=float)
        # maps: (discrete index) -> (SOC)

        # Now, we're going to create a map from an arbitrary SOC, to the index of the closest SOC
        # value in our quantized SOC range (`SOC_values`)
        self._quantization_step: float = (self._max_soc - self._min_soc) / self._num_indices
        self._soc_to_index = lambda _soc: int(
            max(0, min(self._num_indices - 1, (_soc - self._min_soc) // self._quantization_step))
        )
        # maps: (SOC) -> (discrete index)

        # Now, calculate the value of each parameter for each discrete SOC value using the injected `get_` functions
        self._U_oc_lookup: NDArray[float] = battery_config.get_Uoc(SOC_values)
        self._R_0_lookup: NDArray[float] = battery_config.get_R_0(SOC_values)
        self._R_P_lookup: NDArray[float] = battery_config.get_R_P(SOC_values)
        self._C_P_lookup: NDArray[float] = battery_config.get_C_P(SOC_values)
        # maps: (discrete index) -> (parameter)

        # Finally, combine the above maps to create a map from an arbitrary SOC to each battery parameter, using
        # the discrete lookup tables
        # These `cast` calls just promise to the type-checker that these will map floats to floats
        self._U_oc = cast(Callable[[float], float], lambda SOC: self._U_oc_lookup[self._soc_to_index(SOC)])
        self._R_0 = cast(Callable[[float], float], lambda SOC: self._R_0_lookup[self._soc_to_index(SOC)])
        self._R_P = cast(Callable[[float], float], lambda SOC: self._R_P_lookup[self._soc_to_index(SOC)])
        self._C_P = cast(Callable[[float], float], lambda SOC: self._C_P_lookup[self._soc_to_index(SOC)])
        # maps: ((SOC) -> (discrete index)) -> ((discrete index) -> (parameter))  |==>  (SOC) -> (parameter)

        self._tau: Callable[[float], float] = lambda soc: self._R_P(soc) * self._C_P(soc)  # Characteristic Time in s

    def update_array(
            self,
            tick: float,
            delta_energy_array: Optional[NDArray] = None,
            current_array: Optional[NDArray] = None,
            use_compiled: bool = True
    ) -> tuple[NDArray, NDArray]:
        """
        Compute the battery's state of charge and terminal voltage over time in response to a
        time series of energy/current draw from a load.

        Only ONE of `current_array` or `delta_energy_array` should be provided.

        Notes
        -----
        If both current and power are known, current should be provided.
        The model implementation requires current for calculations, so it must be derived from power if power
        was provided.
        Computing current from power relies on voltage, which is a model output, and therefore
        the derived current could be less accurate.

        :param NDArray delta_energy_array: Array of energy changes (J) at each time step.
        :param float tick: Time interval for each step (seconds).
        :param NDArray current_array: Array of current draw (positive sign convention) in Amperes at each time step.
        :param bool use_compiled: If `True`, use compiled binaries for calculations.
            Disable for better debugging.
        :return: A tuple containing arrays for state-of-charge and terminal voltage.
        :raises ValueError: If BOTH or NEITHER of `current_array` or `delta_energy_array` are provided.
        :rtype: tuple[NDArray, NDArray]
        """
        if (delta_energy_array is None) == (current_array is None):  # Enforce that only one should be provided
            raise ValueError("Exactly one of `delta_energy_array` or `current_array` "
                             "must be provided, not both or neither.")

        energy_or_current = delta_energy_array if delta_energy_array is not None else current_array

        if use_compiled:
            return physics_rs.update_battery_state(
                energy_or_current,
                tick,
                self._state_of_charge,
                self._U_P,
                self._R_0_lookup,
                self._U_oc_lookup,
                self._R_P_lookup,
                self._C_P_lookup,
                self._nominal_charge_capacity,
                current_array is None,  # Pass to the library if `energy_or_current` is current or power,
                self._quantization_step,
                self._min_soc
            )

        else:
            return self._update_array_py(energy_or_current, tick, current_array is None)

    def _update_array_py(self, energy_or_current, tick, is_power):
        """
        Perform energy calculations using Python (fallback method if Rust is disabled).

        :param NDArray energy_or_current: Array of energy changes (J) at each time step.
        :param float tick: Time interval for each step (seconds).

        :return: A tuple containing arrays for state-of-charge and voltage.
        """
        soc = np.empty_like(energy_or_current, dtype=float)
        voltage = np.empty_like(energy_or_current, dtype=float)

        for (i, value) in enumerate(energy_or_current):
            if is_power:
                # Use the last voltage to calculate current, or an absurdly large number if it is the first,
                # because we don't know voltage yet.
                # We will have a very small initial current, no matter what.
                # We shouldn't be starting to simulate when the battery is in an active state anyway,
                # so this should be an alright compromise.
                last_terminal_voltage = voltage[i - 1] if i - 1 >= 0 else 10000

                current: float = value / (tick * last_terminal_voltage)
            else:
                current = value

            self._evolve(current, tick)
            soc[i] = self._state_of_charge
            voltage[i] = self._U_L

        return soc, voltage

    def _evolve(self, current: float, tick: float) -> None:
        """
        Update the battery state given the current and time elapsed.

        :param float current: Current applied to the battery (A).
            Positive for charging, negative for discharging.
        :param float tick: Time interval over which the power is applied (seconds).
        """
        soc = self._state_of_charge  # State of Charge (dimensionless, 0 < soc < 1)
        U_P = self._U_P  # Polarization Potential (V)
        R_P = self._R_P(soc)  # Polarization Resistance (Ohms)
        U_oc = self._U_oc(soc)  # Open-Circuit Potential (V)
        R_0 = self._R_0(soc)  # Ohmic Resistance (Ohms)
        Q = self._nominal_charge_capacity  # Nominal Charge Capacity (C)
        tau = self._tau(soc)  # Time constant (s)

        new_soc = soc + (current * tick / Q)
        new_U_P = np.exp(-tick / tau) * U_P + current * R_P * (1 - np.exp(-tick / tau))

        self._state_of_charge = new_soc
        self._U_P = new_U_P
        self._U_L = U_oc + new_U_P + (current * R_0)
