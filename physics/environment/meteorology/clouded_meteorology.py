from physics.environment.meteorology.base_meteorology import BaseMeteorology
from physics.environment.gis.gis import calculate_path_distances
import numpy as np
from numba import jit
import core
from typing import Optional
import datetime


class CloudedMeteorology(BaseMeteorology):
    """
    CloudedMeteorology encapsulates meteorological data that includes
    cloud cover, but not solar irradiance (necessitating manual computation).
    """
    def __init__(self, race, weather_forecasts):
        super().__init__()

        self._latitude: Optional[np.ndarray] = None
        self._longitude: Optional[np.ndarray] = None
        self._unix_time: Optional[np.ndarray] = None
        self._cloud_cover: Optional[np.ndarray] = None

        self._race = race
        self._weather_forecast = weather_forecasts

        self.S_0 = 1367.0  # Solar Constant, 1367W/m^2

        self.last_updated_time = self._weather_forecast[0, 0, 2]

    def spatially_localize(self, cumulative_distances: np.ndarray, simplify_weather: bool = False) -> None:
        """

        IMPORTANT: we only have weather coordinates for a discrete set of coordinates. However, the car could be at any
        coordinate in between these available weather coordinates. We need to figure out what coordinate the car is at
        at each timestep and then we can figure out the full weather forecast at each timestep.

        For example, imagine the car is at some coordinate (10, 20). Further imagine that we have a week's worth of
        weather forecasts for the following five coordinates: (5, 4), (11, 19), (20, 30), (40, 30), (0, 60). Which
        set of weather forecasts should we choose? Well, we should choose the (11, 19) one since our coordinate
        (10, 20) is closest to (11, 19). This is what the following code is accomplishing. However, it is not dealing
        with the coordinates directly but rather is dealing with the distances between the coordinates.

        Furthermore, once we have chosen a week's worth of weather forecasts for a specific coordinate, we must isolate
        a single weather forecast depending on what time the car is at the coordinate (10, 20). That is the job of the
        `get_weather_forecast_in_time()` method.

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
        weather_coords = self._weather_forecast[:, 0, 0:2]

        # distances between all the coordinates that we have weather data for
        weather_path_distances = calculate_path_distances(weather_coords)
        cumulative_weather_path_distances = np.cumsum(weather_path_distances)

        # makes every even-index element negative, this allows the use of np.diff() to calculate the sum of consecutive
        # elements
        cumulative_weather_path_distances[::2] *= -1

        # contains the average distance between two consecutive elements in the cumulative_weather_path_distances array
        average_distances = np.abs(np.diff(cumulative_weather_path_distances) / 2)

        return core.closest_weather_indices_loop(cumulative_distances, average_distances)

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
        :param int tick: length of a tick in seconds
        :returns:
            - A NumPy array of size [N][9]
            - [9] (latitude, longitude, unix_time, timezone_offset, unix_time_corrected, wind_speed, wind_direction,
                        cloud_cover, precipitation, description):
        :rtype: np.ndarray

        """
        weather_data = core.weather_in_time(unix_timestamps.astype(np.int64), self._weather_indices.astype(np.int64), self._weather_forecast, 4)
        # roll_by_tick = int(3600 / tick) * (24 + start_hour - hour_from_unix_timestamp(weather_data[0, 2]))
        # weather_data = np.roll(weather_data, -roll_by_tick, 0)

        self._latitude = weather_data[:, 0]
        self._longitude = weather_data[:, 1]
        self._unix_time = weather_data[:, 2]
        self._wind_speed = weather_data[:, 5]
        self._wind_direction = weather_data[:, 6]
        self._cloud_cover = weather_data[:, 7]

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
        day_of_year, local_time = core.calculate_array_ghi_times(local_times)

        ghi = self._calculate_GHI(coords[:, 0], coords[:, 1], time_zones,
                                  day_of_year, local_time, elevations, self._cloud_cover)

        stationary_irradiance = self._calculate_angled_irradiance(coords[:, 0], coords[:, 1], time_zones, day_of_year,
                                                                  local_time, elevations, self._cloud_cover)

        # Use stationary irradiance when the car is not driving
        effective_irradiance = np.where(
            np.logical_not(self._race.driving_boolean),
            stationary_irradiance,
            ghi)

        return effective_irradiance

    @staticmethod
    def _calculate_hour_angle(time_zone_utc, day_of_year, local_time, longitude):
        """

        Calculates and returns the Hour Angle of the Sun in the sky.
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time
        Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
                calculation will end up just the same
        :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset.
        :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param np.ndarray local_time: The local time in hours from midnight. (Adjust for Daylight Savings)
        :param np.ndarray longitude: The longitude of a location on Earth
        :returns: The Hour Angle in degrees.
        :rtype: np.ndarray

        """

        lst = local_time_to_apparent_solar_time(time_zone_utc / 3600, day_of_year,
                                                        local_time, longitude)

        hour_angle = 15 * (lst - 12)

        return hour_angle

    def _calculate_elevation_angle(self, latitude, longitude, time_zone_utc, day_of_year,
                                   local_time):
        """

        Calculates the Elevation Angle of the Sun relative to a location on the Earth
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
        Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
                calculation will end up just the same

        :param np.ndarray latitude: The latitude of a location on Earth
        :param np.ndarray longitude: The longitude of a location on Earth
        :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset. For example, Vancouver has time_zone_utc = -7
        :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param np.ndarray  local_time: The local time in hours from midnight. (Adjust for Daylight Savings)
        :returns: The elevation angle in degrees
        :rtype: np.ndarray

        """

        # Negative declination angles: Northern Hemisphere winter
        # 0 declination angle : Equinoxes (March 22, Sept 22)
        # Positive declination angle: Northern Hemisphere summer
        declination_angle = calculate_declination_angle(day_of_year)

        # Negative hour angles: Morning
        # 0 hour angle : Solar noon
        # Positive hour angle: Afternoon
        hour_angle = self._calculate_hour_angle(time_zone_utc, day_of_year,
                                                local_time, longitude)
        # From: https://en.wikipedia.org/wiki/Hour_angle#:~:text=At%20solar%20noon%20the%20hour,times%201.5%20hours%20before%20noon).
        # "For example, at 10:30 AM local apparent time
        # the hour angle is −22.5° (15° per hour times 1.5 hours before noon)."

        # mathy part is delegated to a helper function to optimize for numba compilation
        return compute_elevation_angle_math(declination_angle, hour_angle, latitude)

    def _calculate_zenith_angle(self, latitude, longitude, time_zone_utc, day_of_year,
                                local_time):
        """

        Calculates the Zenith Angle of the Sun relative to a location on the Earth
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/azimuth-angle
        Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
                calculation will end up just the same

        :param latitude: The latitude of a location on Earth
        :param longitude: The longitude of a location on Earth
        :param time_zone_utc: The UTC time zone of your area in hours of UTC offset.
        :param day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param local_time: The local time in hours from midnight. (Adjust for Daylight Savings)
        :return: The zenith angle in degrees
        :rtype: float

        """

        elevation_angle = self._calculate_elevation_angle(latitude, longitude,
                                                          time_zone_utc, day_of_year, local_time)

        return 90 - elevation_angle

    def _calculate_azimuth_angle(self, latitude, longitude, time_zone_utc, day_of_year,
                                local_time):
        """

        Calculates the Azimuth Angle of the Sun relative to a location on the Earth.
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/azimuth-angle
        Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
                calculation will end up just the same

        :param latitude: The latitude of a location on Earth
        :param longitude: The longitude of a location on Earth
        :param time_zone_utc: The UTC time zone of your area in hours of UTC offset. For example, Vancouver has time_zone_utc = -7
        :param day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param local_time: The local time in hours from midnight. (Adjust for Daylight Savings)
        :returns: The azimuth angle in degrees
        :rtype: np.ndarray

        """

        declination_angle = calculate_declination_angle(day_of_year)
        hour_angle = self._calculate_hour_angle(time_zone_utc, day_of_year,
                                                local_time, longitude)

        term_1 = np.sin(np.radians(declination_angle)) * \
            np.sin(np.radians(latitude))

        term_2 = np.cos(np.radians(declination_angle)) * \
            np.sin(np.radians(latitude)) * \
            np.cos(np.radians(hour_angle))

        elevation_angle = self._calculate_elevation_angle(latitude, longitude,
                                                          time_zone_utc, day_of_year, local_time)

        term_3 = np.float_(term_1 - term_2) / \
            np.cos(np.radians(elevation_angle))

        if term_3 < -1:
            term_3 = -1
        elif term_3 > 1:
            term_3 = 1

        azimuth_angle = np.arcsin(term_3)

        return np.degrees(azimuth_angle)

    # ----- Calculation of sunrise and sunset times -----

    # ----- Calculation of modes of solar irradiance -----

    def _calculate_DNI(self, latitude, longitude, time_zone_utc, day_of_year,
                       local_time, elevation):
        """

        Calculates the Direct Normal Irradiance from the Sun, relative to a location
        on the Earth (clearsky)
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/calculation-of-solar-insolation
        Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
                calculation will end up just the same

        :param np.ndarray latitude: The latitude of a location on Earth
        :param np.ndarray longitude: The longitude of a location on Earth
        :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset.
        :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param np.ndarray local_time: The local time in hours from midnight. (Adjust for Daylight Savings)
        :param np.ndarray elevation: The local elevation of a location in metres
        :returns: The Direct Normal Irradiance in W/m2
        :rtype: np.ndarray

        """

        zenith_angle = self._calculate_zenith_angle(latitude, longitude,
                                                    time_zone_utc, day_of_year, local_time)
        a = 0.14

        # https://www.pveducation.org/pvcdrom/properties-of-sunlight/air-mass
        # air_mass = 1 / (math.cos(math.radians(zenith_angle)) + \
        #            0.50572*pow((96.07995 - zenith_angle), -1.6364))

        with np.errstate(invalid="ignore"):
            air_mass = np.float_(1) / (np.float_(np.cos(np.radians(zenith_angle)))
                                       + 0.50572*np.power((96.07995 - zenith_angle), -1.6364))

        with np.errstate(over="ignore"):
            DNI = self.S_0 * ((1 - a * elevation * 0.001) * np.power(0.7, np.power(air_mass, 0.678))
                                  + a * elevation * 0.001)

        return np.where(zenith_angle > 90, 0, DNI)

    def _calculate_DHI(self, latitude, longitude, time_zone_utc, day_of_year,
                       local_time, elevation):
        """

        Calculates the Diffuse Horizontal Irradiance from the Sun, relative to a location
        on the Earth (clearsky)
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/calculation-of-solar-insolation
        Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
                calculation will end up just the same

        :param np.ndarray latitude: The latitude of a location on Earth
        :param np.ndarray longitude: The longitude of a location on Earth
        :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset.
        :param np.ndarray np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param np.ndarray local_time: The local time in hours from midnight
        :param np.ndarray elevation: The local elevation of a location in metres
        :returns: The Diffuse Horizontal Irradiance in W/m2
        :rtype: np.ndarray

        """

        DNI = self._calculate_DNI(latitude, longitude, time_zone_utc, day_of_year,
                                  local_time, elevation)

        DHI = 0.1 * DNI

        return DHI

    def _calculate_GHI(self, latitude, longitude, time_zone_utc, day_of_year,
                       local_time, elevation, cloud_cover):
        """

        Calculates the Global Horizontal Irradiance from the Sun, relative to a location
        on the Earth
        https://www.pveducation.org/pvcdrom/properties-of-sunlight/calculation-of-solar-insolation
        Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
                calculation will end up just the same

        :param np.ndarray latitude: The latitude of a location on Earth
        :param np.ndarray longitude: The longitude of a location on Earth
        :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset, without including the effects of Daylight Savings Time. For example, Vancouver has time_zone_utc = -8 year-round.
        :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param np.ndarray local_time: The local time in hours from midnight.
        :param np.ndarray elevation: The local elevation of a location in metres
        :param np.ndarray cloud_cover: A NumPy array representing cloud cover as a percentage from 0 to 100
        :returns: The Global Horizontal Irradiance in W/m^2
        :rtype: np.ndarray

        """

        DHI = self._calculate_DHI(latitude, longitude, time_zone_utc, day_of_year,
                                  local_time, elevation)

        DNI = self._calculate_DNI(latitude, longitude, time_zone_utc, day_of_year,
                                  local_time, elevation)

        zenith_angle = self._calculate_zenith_angle(latitude, longitude,
                                                    time_zone_utc, day_of_year, local_time)

        GHI = DNI * np.cos(np.radians(zenith_angle)) + DHI

        return self._apply_cloud_cover(GHI=GHI, cloud_cover=cloud_cover)

    @staticmethod
    def _apply_cloud_cover(GHI, cloud_cover):
        """

        Applies a cloud cover model to the GHI data.

        Cloud cover adjustment follows the equation laid out here:
        http://www.shodor.org/os411/courses/_master/tools/calculators/solarrad/

        :param np.ndarray GHI: Global Horizontal Index in W/m^2
        :param np.ndarray cloud_cover: A NumPy array representing cloud cover as a percentage from 0 to 100

        :returns: GHI after considering cloud cover data
        :rtype: np.ndarray

        """

        assert np.logical_and(cloud_cover >= 0, cloud_cover <= 100).all()

        scaled_cloud_cover = cloud_cover / 100

        assert np.logical_and(scaled_cloud_cover >= 0,
                              scaled_cloud_cover <= 1).all()

        return GHI * (1 - (0.75 * np.power(scaled_cloud_cover, 3.4)))

    # ----- Calculation of modes of solar irradiance, but returning numpy arrays -----
    @staticmethod
    def _date_convert(date):
        """

        Convert a date into local time.

        :param datetime.datetime date: date to be converted
        :return: a date converted into local time.
        :rtype: int

        """

        return date.hour + (float(date.minute * 60 + date.second) / 3600)

    def _calculate_angled_irradiance(self, latitude, longitude, time_zone_utc, day_of_year,
                                     local_time, elevation, cloud_cover, array_angles=np.array([0, 15, 30, 45])):
        """

        Determine the direct and diffuse irradiance on an array which can be mounted at different angles.
        During stationary charging, the car can mount the array at different angles, resulting in a higher
        component of direct irradiance captured.

        Uses the GHI formula, GHI = DNI*cos(zenith)+DHI but with an 'effective zenith',
        the angle between the mounted panel's normal and the sun.

        :param np.ndarray latitude: The latitude of a location on Earth
        :param np.ndarray longitude: The longitude of a location on Earth
        :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset, without including the effects of Daylight Savings Time. For example, Vancouver has time_zone_utc = -8 year-round.
        :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
        :param np.ndarray local_time: The local time in hours from midnight.
        :param np.ndarray elevation: The local elevation of a location in metres
        :param np.ndarray cloud_cover: A NumPy array representing cloud cover as a percentage from 0 to 100
        :param np.ndarray array_angles: An array containing the discrete angles on which the array can be mounted
        :returns: The "effective Global Horizontal Irradiance" in W/m^2
        :rtype: np.ndarray

        """

        DHI = self._calculate_DHI(latitude, longitude, time_zone_utc, day_of_year,
                                  local_time, elevation)

        DNI = self._calculate_DNI(latitude, longitude, time_zone_utc, day_of_year,
                                  local_time, elevation)

        zenith_angle = self._calculate_zenith_angle(latitude, longitude,
                                                    time_zone_utc, day_of_year, local_time)

        # Calculate the absolute differences
        differences = np.abs(zenith_angle[:, np.newaxis] - array_angles)

        # Find the minimum difference for each element in zenith_angle
        effective_zenith = np.min(differences, axis=1)

        # Now effective_zenith contains the minimum absolute difference for each element in zenith_angle

        GHI = DNI * np.cos(np.radians(effective_zenith)) + DHI

        return self._apply_cloud_cover(GHI=GHI, cloud_cover=cloud_cover)


def local_time_to_apparent_solar_time(time_zone_utc, day_of_year, local_time,
                                      longitude):
    """

    Converts between the local time to the apparent solar time and returns the apparent
    solar time.
    https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time

    Note: If local time and time_zone_utc are both unadjusted for Daylight Savings, the
        calculation will end up just the same

    :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset.
    :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
    :param np.ndarray local_time: The local time in hours from midnight (Adjust for Daylight Savings)
    :param np.ndarray longitude: The longitude of a location on Earth
    :returns: The Apparent Solar Time of a location, in hours from midnight
    :rtype: np.ndarray

    """

    lstm = calculate_LSTM(time_zone_utc)
    eot = calculate_eot_correction(day_of_year)

    # local solar time
    lst = local_time + np.float_(longitude - lstm) / 15 + np.float_(eot) / 60

    return lst


@jit(nopython=True)
def calculate_LSTM(time_zone_utc):
    """

    Calculates and returns the LSTM, or Local Solar Time Meridian.
    https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time

    :param np.ndarray time_zone_utc: The UTC time zone of your area in hours of UTC offset.
    :returns: The Local Solar Time Meridian in degrees
    :rtype: np.ndarray

    """

    return 15 * time_zone_utc


@jit(nopython=True)
def calculate_eot_correction(day_of_year):
    """

    Approximates and returns the correction factor between the apparent
    solar time and the mean solar time

    :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
    :returns: The Equation of Time correction EoT in minutes, where apparent Solar Time = Mean Solar Time + EoT
    :rtype: np.ndarray

    """

    b = np.radians((np.float_(360) / 364) * (day_of_year - 81))

    eot = 9.87 * np.sin(2 * b) - 7.83 * np.cos(b) - 1.5 * np.sin(b)

    return eot


def get_day_of_year_map(date):
    """

    Extracts day, month, year, from datetime object

    :param datetime.date date: date to be decomposed

    """
    return get_day_of_year(date.day, date.month, date.year)


def get_day_of_year(day, month, year):
    """

    Calculates the day of the year, given the day, month and year.
    Day refers to a number representing the nth day of the year. So, Jan 1st will be the 1st day of the year

    :param int day: nth day of the year
    :param int month: month
    :param int year: year
    :returns: day of year
    :rtype: int

    """

    return (datetime.date(year, month, day) - datetime.date(year, 1, 1)).days + 1


@jit(nopython=True)
def calculate_declination_angle(day_of_year):
    """

    Calculates the Declination Angle of the Earth at a given day
    https://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle

    :param np.ndarray day_of_year: The number of the day of the current year, with January 1 being the first day of the year.
    :returns: The declination angle of the Earth relative to the Sun, in degrees
    :rtype: np.ndarray

    """

    declination_angle = -23.45 * np.cos(np.radians((np.float_(360) / 365) *
                                                   (day_of_year + 10)))

    return declination_angle


@jit(nopython=True)
def compute_elevation_angle_math(declination_angle, hour_angle, latitude):
    """

    Gets the two terms to calculate and return elevation angle, given the
    declination angle, hour angle, and latitude.

    This method separates the math part of the calculation from its caller
    method to optimize for numba compilation.

    :param np.ndarray latitude: array of latitudes
    :param np.ndarray declination_angle: The declination angle of the Earth relative to the Sun
    :param np.ndarray hour_angle: The hour angle of the sun in the sky
    :returns: The elevation angle in degrees
    :rtype: np.ndarray

    """

    term_1 = np.sin(np.radians(declination_angle)) * np.sin(np.radians(latitude))
    term_2 = np.cos(np.radians(declination_angle)) * np.cos(np.radians(latitude)) * np.cos(np.radians(hour_angle))
    elevation_angle = np.arcsin(term_1 + term_2)

    return np.degrees(elevation_angle)

