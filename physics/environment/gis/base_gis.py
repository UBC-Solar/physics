from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike, NDArray


class BaseGIS(ABC):
    @abstractmethod
    def calculate_closest_gis_indices(self, cumulative_distances) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_path_elevations(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_gradients(self, gis_indices) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_time_zones(self, gis_indices) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_path(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def calculate_current_heading_array(self) -> np.ndarray:
        raise NotImplementedError

    def calculate_driving_speeds(
            self,
            average_lap_speeds: ArrayLike,
            simulation_dt: int,
            driving_allowed: ArrayLike,
            idle_time: int
    ) -> NDArray[float]:
        raise NotImplementedError
