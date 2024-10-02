from physics.environment.meteorology.base_meteorology import BaseMeteorology
from physics.environment.gis.gis import calculate_path_distances
import numpy as np
import core
from typing import Optional


class IrradiantMeteorology(BaseMeteorology):
    """
    IrradiantMeteorology encapsulates meteorological data that includes
    solar irradiance data, but not cloud cover.

    """
    def __init__(self, race, weather_forecasts):
        self._race = race
        self._raw_weather_data = weather_forecasts

        self._time_dt: Optional[np.ndarray] = None
        self._ghi: Optional[np.ndarray] = None
        self._longitude: Optional[np.ndarray] = None
        self._latitude: Optional[np.ndarray] = None

        self.last_updated_time = self._raw_weather_data[0, 0, 0]

        super().__init__()

    def spatially_localize(self, cumulative_distances: np.ndarray, simplify_weather: bool = False) -> None:
        """

        :param np.ndarray cumulative_distances: NumPy Array representing cumulative distances theoretically achievable for a given input speed array
        :param bool simplify_weather: enable to only use a single weather coordinate (for track races without varying weather)

        """
        # If racing a track race, there is no need for distance calculations. We will return only the origin coordinate
        # This characterizes the weather at every point along the FSGP tracks
        # with the weather at a single coordinate on the track, which is great for reducing the API calls and is a
        # reasonable assumption to make for FSGP only.
        if simplify_weather:
            self._weather_indices = np.zeros_like(cumulative_distances, dtype=int)
            return

        # a list of all the coordinates that we have weather data for
        weather_coords = self._raw_weather_data[:, 0, 1:3]

        # distances between all the coordinates that we have weather data for
        weather_path_distances = calculate_path_distances(weather_coords)
        cumulative_weather_path_distances = np.cumsum(weather_path_distances)

        # makes every even-index element negative, this allows the use of np.diff() to calculate the sum of consecutive
        # elements
        cumulative_weather_path_distances[::2] *= -1

        # contains the average distance between two consecutive elements in the cumulative_weather_path_distances array
        average_distances = np.abs(np.diff(cumulative_weather_path_distances) / 2)

        self._weather_indices = core.closest_weather_indices_loop(cumulative_distances, average_distances)

    def temporally_localize(self, unix_timestamps, start_time, tick) -> None:
        """

        Takes in an array of indices of the weather_forecast array, and an array of timestamps. Uses those to figure out
        what the weather forecast is at each time step being simulated.

        we only have weather at discrete timestamps. The car however can be in any timestamp in
        between. Therefore, we must be able to choose the weather timestamp that is closest to the one that the car is in
        so that we can more accurately determine the weather experienced by the car at that timestamp.

        For example, imagine the car is at some coordinate (x,y) at timestamp 100. Imagine we know the weather forecast
        at (x,y) for five different timestamps: 0, 30, 60, 90, and 120. Which weather forecast should we
        choose? Clearly, we should choose the weather forecast at 90 since it is the closest to 100. That's what the
        below code is accomplishing.

        :param np.ndarray unix_timestamps: (int[N]) unix timestamps of the vehicle's journey
        :param int start_time: time since the start of the race that simulation is beginning
        :param int tick: length of a tick in seconds
        :returns: a SolcastEnvironment object with time_dt, latitude, longitude, wind_speed, wind_direction, and ghi.
        :rtype: SolcastEnvironment
        """
        forecasts_array = core.weather_in_time(unix_timestamps.astype(np.int64), self._weather_indices.astype(np.int64),
                                               self._raw_weather_data, 0)

        self._time_dt = forecasts_array[:, 0]
        self._latitude = forecasts_array[:, 1]
        self._longitude = forecasts_array[:, 2]
        self._wind_speed = forecasts_array[:, 3]
        self._wind_direction = forecasts_array[:, 4]
        self._solar_irradiance = forecasts_array[:, 5]

    def calculate_solar_irradiances(self, coords, time_zones, local_times, elevations):
        """
        Calculates the Global Horizontal Irradiance from the Sun, relative to a location
        on the Earth, for arrays of coordinates, times, elevations and weathers
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/calculation-of-solar-insolation
        Note: If local_times and time_zones are both unadjusted for Daylight Savings, the
                calculation will end up just the same

        :param np.ndarray coords: (float[N][lat, lng]) array of latitudes and longitudes
        :param np.ndarray time_zones: (int[N]) time zones at different locations in seconds relative to UTC
        :param np.ndarray local_times: (int[N]) unix time that the vehicle will be at each location. (Adjusted for Daylight Savings)
        :param np.ndarray elevations: (float[N]) elevation from sea level in m
        :returns: (float[N]) Global Horizontal Irradiance in W/m2
        :rtype: np.ndarray

        """
        return self.solar_irradiance


