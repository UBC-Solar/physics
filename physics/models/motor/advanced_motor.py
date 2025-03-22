import numpy as np
from haversine import haversine, Unit
from numpy.typing import NDArray
from physics.models.motor import BasicMotor


class AdvancedMotor(BasicMotor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cornering_coefficient = 15  # tuned to Day 1 and 3 FSGP data

    def calculate_energy_in(self, required_speed_kmh, gradients, wind_speeds, tick, coords):
        """
        A function which takes in array of elevation, array of wind speed, required
            speed, returns the consumed energy.

        :param np.ndarray required_speed_kmh: (float[N]) required speed array in km/h
        :param np.ndarray gradients: (float[N]) gradient at parts of the road
        :param np.ndarray wind_speeds: (float[N]) speeds of wind in m/s, where > 0 means against the direction of the vehicle
        :param float tick: length of 1 update cycle in seconds
        :param np.ndarray coords: ([float[N,2]) The lat,lon coordinate  of the car at each tick
        :returns: (float[N]) energy expended by the motor at every tick
        :rtype: np.ndarray

        """
        net_force, required_angular_speed_rads = self.calculate_net_force(required_speed_kmh, wind_speeds, gradients)

        cornering_work = self.calculate_cornering_losses(required_speed_kmh, coords, tick)

        motor_output_energies = required_angular_speed_rads * net_force * self.tire_radius * tick + cornering_work
        motor_output_energies = np.clip(motor_output_energies, a_min=0, a_max=None)

        e_m = self.calculate_motor_efficiency(required_angular_speed_rads, motor_output_energies, tick)
        e_mc = self.calculate_motor_controller_efficiency(required_angular_speed_rads, motor_output_energies, tick)

        motor_controller_input_energies = motor_output_energies / (e_m * e_mc)

        # Filter out and replace negative energy consumption as 0
        motor_controller_input_energies = np.where(motor_controller_input_energies > 0,
                                                   motor_controller_input_energies, 0)

        return motor_controller_input_energies

    def calculate_cornering_losses(self, required_speed_kmh, coords, tick):
        """
        Calculate the energy losses due to cornering based on vehicle speed and trajectory.

        :param np.ndarray required_speed_kmh: (float[N]) Required speed array in km/h
        :param np.ndarray coords: (float[N, 2]) Array containing latitude and longitude coordinates of the car at each tick
        :param float tick: Length of one update cycle in seconds
        :returns: (float[N]) Energy loss due to cornering at each tick
        :rtype: np.ndarray
        """
        required_speed_ms = required_speed_kmh / 3.6
        cornering_radii = self.calculate_radii(coords)

        centripetal_lateral_force = self.vehicle_mass * (required_speed_ms ** 2) / cornering_radii
        centripetal_lateral_force = np.clip(centripetal_lateral_force, a_min=0, a_max=10000)

        slip_angles_degrees = self.get_slip_angle_for_tire_force(centripetal_lateral_force)
        slip_angles_radians = np.radians(slip_angles_degrees)
        slip_distances = np.tan(slip_angles_radians) * required_speed_ms * tick

        return slip_distances * centripetal_lateral_force * self.cornering_coefficient

    def calculate_radii(self, coords):
        """
        Calculate the cornering radii for a given set of waypoints.

        :param np.ndarray coords: (float[N, 2]) Array containing latitude and longitude coordinates of the car's path
        :returns: (float[N]) Array of cornering radii at each waypoint
        :rtype: np.ndarray
        """

        # pop off last coordinate if first and last coordinate are the same
        repeated_last_coordinate = False
        if np.array_equal(coords[0], coords[len(coords) - 1]):
            coords = coords[:-1]
            repeated_last_coordinate = True

        cornering_radii = np.empty(len(coords))
        for i in range(len(coords)):
            # if the next point or previous point is out of bounds, wrap the index around the array
            i2 = (i - 1) % len(coords)
            i3 = (i + 1) % len(coords)
            current_point = coords[i]
            previous_point = coords[i2]
            next_point = coords[i3]

            x1 = 0
            y1 = 0
            x2, y2 = self.calculate_meter_distance(current_point, previous_point)
            x3, y3 = self.calculate_meter_distance(current_point, next_point)
            cornering_radii[i] = self.radius_of_curvature(x1, y1, x2, y2, x3, y3)

        # If the last coordinate was removed, duplicate the first radius value to the end of the array
        if repeated_last_coordinate:
            cornering_radii = np.append(cornering_radii, cornering_radii[0])

        # ensure that super large radii are bounded by a large number, like 10000
        cornering_radii = np.where(np.isnan(cornering_radii), 10000, cornering_radii)
        cornering_radii = np.where(cornering_radii > 10000, 10000, cornering_radii)

        return cornering_radii

    def generate_slip_angle_lookup(self, min_degrees, max_degrees, num_elements):
        """
        Generate a lookup table of slip angles and corresponding tire forces using Pacejka's Magic Formula.

        https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/

        :param float min_degrees: Minimum slip angle in degrees
        :param float max_degrees: Maximum slip angle in degrees
        :param int num_elements: Number of discrete elements in the lookup table
        :returns: (float[num_elements], float[num_elements]) Arrays of slip angles (degrees) and corresponding tire forces (Newtons)
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        b = .25  # Stiffness
        c = 2.2  # Shape
        d = 2.75  # Peak
        e = 1.0  # Curvature

        fz = self.vehicle_mass * 9.81  # Newtons

        slip_angles = np.linspace(min_degrees, max_degrees, num_elements)
        tire_forces = fz * d * np.sin(
            c * np.arctan(b * slip_angles - e * (b * slip_angles - np.arctan(b * slip_angles))))

        return slip_angles, tire_forces

    def get_slip_angle_for_tire_force(self, desired_tire_force):
        slip_angles, tire_forces = self.generate_slip_angle_lookup(0, 70, 100000)

        # Use the numpy interpolation function to find slip angle for the given tire force
        estimated_slip_angle = np.interp(desired_tire_force, tire_forces, slip_angles)

        return estimated_slip_angle

    @staticmethod
    def calculate_meter_distance(coord1: NDArray, coord2: NDArray):
        """
        Calculate the x and y distance in meters between two latitude-longitude coordinates.

        :param tuple coord1: (float[2]) The (latitude, longitude) coordinates of the first point
        :param tuple coord2: (float[2]) The (latitude, longitude) coordinates of the second point
        :returns: (float[2]) The x (longitude) and y (latitude) distances in meters
        :rtype: tuple
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        # Base coordinate
        coord_base = (lat1, lon1)
        # Coordinate for latitude difference (keep longitude the same)
        coord_lat = (lat2, lon1)
        # Coordinate for longitude difference (keep latitude the same)
        coord_long = (lat1, lon2)

        # Calculate y distance (latitude difference)
        y_distance = haversine(coord_base, coord_lat, unit=Unit.METERS)
        # Calculate x distance (longitude difference)
        x_distance = haversine(coord_base, coord_long, unit=Unit.METERS)

        if lat2 < lat1:
            y_distance = -y_distance
        if lon2 < lon1:
            x_distance = -x_distance

        return x_distance, y_distance

    @staticmethod
    def radius_of_curvature(x1, y1, x2, y2, x3, y3):
        """
        Uses the circumcircle the radius of curvature of a circle passing through three points.

        :param float x1: X-coordinate of the first point
        :param float y1: Y-coordinate of the first point
        :param float x2: X-coordinate of the second point
        :param float y2: Y-coordinate of the second point
        :param float x3: X-coordinate of the third point
        :param float y3: Y-coordinate of the third point
        :returns: Radius of curvature of the circle passing through the three points
        :rtype: float
        """
        numerator = np.sqrt(
            ((x3 - x2) ** 2 + (y3 - y2) ** 2) *
            ((x1 - x3) ** 2 + (y1 - y3) ** 2) *
            ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        )

        denominator = 2 * abs(
            ((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
        )

        return numerator / denominator
