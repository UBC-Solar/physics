import logging
import math
import core
import numpy as np
import sys

from tqdm import tqdm
from xml.dom import minidom
from haversine import haversine, Unit
from physics.environment.gis.base_gis import BaseGIS


class GIS(BaseGIS):
    def __init__(self, route_data, origin_coord, current_coord=None):
        """

        Initialises a GIS (geographic location system) object. This object is responsible for getting the
        simulation's planned route from the Google Maps API and performing operations on the received data.

        Requires a map, ``route_data`` with certain keys.
            1. "path": an iterable of shape [N, 2] representing N coordinates in the form (latitude, longitude).
            2. "elevations": an iterable of shape [N] where each Nth element is the elevation, in meters, of the Nth path coordinate.
            3. "time_zones": an iterable of shape [N] where each Nth element is the UTC time zone offset of the Nth path coordinate.
            4. "num_unique_coords": the number of unique coordinates (that is, if the path is a single lap that has been tiled, how many path coordinates compose a single lap).

        :param route_data: map of data containing "path", "elevations", "time_zones", and "num_unique_coords".
        :param origin_coord: NumPy array containing the start coordinate (lat, long) of the planned travel route

        """
        self.path = route_data['path']
        self.launch_point = route_data['path'][0]
        self.path_elevations = route_data['elevations']
        self.path_time_zones = route_data['time_zones']
        self.num_unique_coords = route_data['num_unique_coords']

        if current_coord is not None:
            if not np.array_equal(current_coord, origin_coord):
                logging.warning("Current position is not origin position. Modifying path data.\n")

                # We need to find the closest coordinate along the path to the vehicle position
                current_coord_index = GIS._find_closest_coordinate_index(current_coord, self.path)

                # All coords before the current coordinate should be discarded
                self.path = self.path[current_coord_index:]
                self.path_elevations = self.path_elevations[current_coord_index:]
                self.path_time_zones = self.path_time_zones[current_coord_index:]

        self.path_distances = calculate_path_distances(self.path)
        self.path_length = np.cumsum(calculate_path_distances(self.path[:self.num_unique_coords]))[-1]
        self.path_gradients = calculate_path_gradients(self.path_elevations, self.path_distances)

    @staticmethod
    def process_KML_file(route_file):
        """

        Load the FSGP Track from a KML file exported from a Google Earth project.

        Ensure to follow guidelines enumerated in this directory's `README.md` when creating and
        loading new route files.

        :return: Array of N coordinates (latitude, longitude) in the shape [N][2].
        """
        with open(route_file) as f:
            data = minidom.parse(f)
            kml_coordinates = data.getElementsByTagName("coordinates")[0].childNodes[0].data
            coordinates: np.ndarray = np.array(parse_coordinates_from_kml(kml_coordinates))

            # Google Earth exports coordinates in order longitude, latitude, when we want the opposite
            return np.roll(coordinates, 1, axis=1)

    def calculate_closest_gis_indices(self, distances):
        """

        Takes in an array of point distances from starting point, returns a list of
        ``self.path`` indices of coordinates which have a distance from the starting point
        closest to the point distances.

        :param np.ndarray distances: (float[N]) array of distances, where cumulative_distances[x] > cumulative_distances[x-1]
        :returns: (float[N]) array of indices of path
        :rtype: np.ndarray

        """
        return core.closest_gis_indices_loop(distances, self.path_distances)

    @staticmethod
    def _python_calculate_closest_gis_indices(distances, path_distances):
        """

        Python implementation of rust core.closest_gis_indices_loop. See parent function for documentation details.

        """

        current_coordinate_index = 0
        result = []

        with tqdm(total=len(distances), file=sys.stdout, desc="Calculating closest GIS indices") as pbar:
            distance_travelled = 0
            for distance in np.nditer(distances):
                distance_travelled += distance

                while distance_travelled > path_distances[current_coordinate_index]:
                    distance_travelled -= path_distances[current_coordinate_index]
                    current_coordinate_index += 1

                    if current_coordinate_index >= len(path_distances) - 1:
                        current_coordinate_index = len(path_distances) - 1

                result.append(current_coordinate_index)
                pbar.update(1)

        return np.array(result)

    # ----- Getters -----
    def get_time_zones(self, gis_indices):
        """

        Takes in an array of path indices, returns the time zone at each index

        :param np.ndarray gis_indices: (float[N]) array of path indices
        :returns: (float[N]) array of time zones in seconds
        :rtype: np.ndarray

        """

        return self.path_time_zones[gis_indices]

    def get_gradients(self, gis_indices):
        """

        Takes in an array of path indices, returns the road gradient at each index

        :param np.ndarray gis_indices: (float[N]) array of path indices
        :returns: (float[N]) array of road gradients
        :rtype np.ndarray:

        """

        return self.path_gradients[gis_indices]

    def get_path(self):
        """
        Returns all N coordinates of the path in a NumPy array
        [N][latitude, longitude]

        :rtype: np.ndarray

        """

        return self.path

    def get_path_elevations(self):
        """

        Returns all N elevations of the path in a NumPy array
        [N][elevation]

        :rtype: np.ndarray

        """

        return self.path_elevations

    def get_path_distances(self):
        """

        Returns all N-1 distances of the path in a NumPy array
        [N-1][elevation]

        :rtype: np.ndarray

        """

        return self.path_distances

    def get_path_gradients(self):
        """

        Returns all N-1 gradients of a path in a NumPy array
        [N-1][gradient]

        :rtype: np.ndarray

        """

        return self.path_gradients

    # ----- Path calculation functions -----
    def calculate_path_min_max(self):
        logging.warning(f"Using deprecated function 'calculate_path_min_max()'!")
        min_lat, min_long = self.path.min(axis=0)
        max_lat, max_long = self.path.max(axis=0)
        return [min_long, min_lat, max_long, max_lat]

    def calculate_current_heading_array(self):
        """

        Calculates the bearing of the vehicle between consecutive points
        https://www.movable-type.co.uk/scripts/latlong.html

        :returns: array of bearings
        :rtype: np.ndarray

        """
        bearing_array = np.zeros(len(self.path))

        for index in range(0, len(self.path) - 1):
            coord_1 = np.radians(self.path[index])
            coord_2 = np.radians(self.path[index + 1])

            y = math.sin(coord_2[1] - coord_1[1]) \
                * math.cos(coord_2[0])

            x = math.cos(coord_1[0]) \
                * math.sin(coord_2[0]) \
                - math.sin(coord_1[0]) \
                * math.cos(coord_2[0]) \
                * math.cos(coord_2[1] - coord_1[1])

            theta = math.atan2(y, x)

            bearing_array[index] = ((theta * 180) / math.pi + 360) % 360

        bearing_array[-1] = bearing_array[-2]

        return bearing_array

    @staticmethod
    def _calculate_vector_square_magnitude(vector):
        """

        Calculate the square magnitude of an input vector. Must be one-dimensional.

        :param np.ndarray vector: NumPy array[N] representing a vector[N]
        :return: square magnitude of the input vector
        :rtype: float

        """

        return sum(i ** 2 for i in vector)

    @staticmethod
    def _find_closest_coordinate_index(current_coord, path):
        """

        Returns the closest coordinate to current_coord in path

        :param np.ndarray current_coord: A NumPy array[N] representing a N-dimensional vector
        :param np.ndarray path: A NumPy array[M][N] of M coordinates which should be N-dimensional vectors
        :returns: index of the closest coordinate.
        :rtype: int

        """

        to_current_coord_from_path = np.abs(path - current_coord)
        distances_from_current_coord = np.zeros(len(to_current_coord_from_path))
        for i in range(len(to_current_coord_from_path)):
            # As we just need the minimum, using square magnitude will save performance
            distances_from_current_coord[i] = GIS._calculate_vector_square_magnitude(to_current_coord_from_path[i])

        return distances_from_current_coord.argmin()


def calculate_path_distances(coords):
    """

    Obtain the distance between each coordinate by approximating the spline between them
    as a straight line, and use the Haversine formula (https://en.wikipedia.org/wiki/Haversine_formula)
    to calculate distance between coordinates on a sphere.

    :param np.ndarray coords: A NumPy array [n][latitude, longitude]
    :returns path_distances: a NumPy array [n-1][distances],
    :rtype: np.ndarray

    """

    coords_offset = np.roll(coords, (1, 1))
    path_distances = []
    for u, v in zip(coords, coords_offset):
        path_distances.append(haversine(u, v, unit=Unit.METERS))

    return np.array(path_distances)


def parse_coordinates_from_kml(coords_str: str) -> np.ndarray:
    """

    Parse a coordinates string from a XML (KML) file into a list of coordinates (2D vectors).
    Requires coordinates in the format "39.,41.,0  39.,40.,0" which will return [ [39., 41.], [39., 40.] ].

    :param coords_str: coordinates string from a XML (KML) file
    :return: list of 2D vectors representing coordinates
    :rtype: np.ndarray

    """

    def parse_coord(pair):
        coord = pair.split(',')
        coord.pop()
        coord = [float(value) for value in coord]
        return coord

    return list(map(parse_coord, coords_str.split()))


def calculate_path_gradients(elevations, distances):
    """

    Get the approximate gradients of every point on the path.

    Note:
        - gradient > 0 corresponds to uphill
        - gradient < 0 corresponds to downhill

    :param np.ndarray elevations: [N][elevations]
    :param np.ndarray distances: [N-1][distances]
    :returns gradients: [N-1][gradients]
    :rtype: np.ndarray

    """

    # subtract every next elevation with the previous elevation to
    # get the difference in elevation
    # [1 2 3 4 5]
    # [5 1 2 3 4] -
    # -------------
    #   [1 1 1 1]

    offset = np.roll(elevations, 1)
    delta_elevations = elevations - offset

    # Divide the difference in elevation to get the gradient
    # gradient > 0: uphill
    # gradient < 0: downhill
    with np.errstate(invalid='ignore'):
        gradients = delta_elevations / distances

    return np.nan_to_num(gradients, nan=0.)
