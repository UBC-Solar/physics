from typing import Optional
import numpy as np
from abc import ABC, abstractmethod


class BaseMeteorology(ABC):
    def __init__(self):
        self._wind_speed: Optional[np.ndarray] = None
        self._wind_direction: Optional[np.ndarray] = None
        self._solar_irradiance: Optional[np.ndarray] = None
        self._weather_indices: Optional[np.ndarray] = None

    def _return_if_available(self, attr):
        if (value := getattr(self, attr)) is not None:
            return value
        else:
            raise UnboundLocalError(f"{attr} is not available!")

    @property
    def wind_speed(self) -> np.ndarray:
        """
        Return the wind speeds in m/s at every tick, if available.

        :return: ``ndarray`` of wind speeds in m/s at every tick
        :raises UnboundLocalError: if wind speeds are not available.
        """
        return self._return_if_available("_wind_speed")

    @property
    def wind_direction(self) -> np.ndarray:
        """
        Return the wind direction in degrees, following the meteorological convention, if available.

        :return: ``ndarray`` of wind directions in degrees at every tick.
        :raises UnboundLocalError: if wind directions are not available.
        """
        return self._return_if_available("_wind_direction")

    @property
    def solar_irradiance(self) -> np.ndarray:
        """
        Return the solar irradiance in W/m^2 every tick, if available.

        :return: ``ndarray`` of solar irradiances in W/m^2 at every tick
        :raises UnboundLocalError: if solar irradiances are not available.
        """
        return self._return_if_available("_solar_irradiance")

    @property
    def weather_indices(self) -> np.ndarray:
        """
        Return the weather indices at every tick, if available.

        :return: ``ndarray`` of weather indices at every tick
        :raises UnboundLocalError: if weather indices are not available.
        """
        return self._return_if_available("_weather_indices")

    @abstractmethod
    def spatially_localize(self, cumulative_distances: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def temporally_localize(self, unix_timestamps, start_time, tick) -> None:
        raise NotImplementedError

    @abstractmethod
    def calculate_solar_irradiances(self, coords, time_zones, local_times, elevations):
        raise NotImplementedError
