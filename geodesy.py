import numpy as np
from utilities import UnitConverterFactory


class GeoLocation:
    """ A geographical location specified by latitude and longitude."""

    def __init__(self, lat=0.0, lon=0.0):
        self._lat = lat
        self._lon = lon

    @property
    def latitude(self):
        return self._lat

    @latitude.setter
    def latitude(self, value):
        if not value.isnumeric():
            raise TypeError("Latitude must be a number")
        if -90 < value < 90:
            self._lat = value
        else:
            raise ValueError("Latitude value should be in the interval (-90, 90) degrees")

    @property
    def longitude(self):
        return self._lon

    @longitude.setter
    def longitude(self, value):
        if not value.isnumeric():
            raise TypeError("Longitude must be a number")
        if -180 <= value < 180:
            self._lon = value
        else:
            raise ValueError("Longitude value should be in the interval [-180, 180) degrees")


class GeoLocationFinder:
    """
    Given a geographical location, a distance, and an azimuth with respect to the North,
    calculate the geographical location of the point specified by the distance and the azimuth
    """

    RADIUS = 6371.393 * 1000

    def __init__(self, origin, distance, azimuth):
        self._origin = origin
        self._distance = distance  # in meters, range [0, 1000]
        self._azimuth = azimuth  # angle with respect to the North

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, target):
        if isinstance(target, GeoLocation):
            self._origin = target
        else:
            raise TypeError("Origin is GeoLocation type")

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        if not value.isnumeric():
            raise TypeError("Distance must be a number")
        if 0 <= value <= 1000:
            self._distance = value
        else:
            raise ValueError("Distance is in the range [0, 1000] meters")

    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        if not value.isnumeric():
            raise TypeError("Azimuth must be a number")
        if -180 <= value < 180:
            self._azimuth = value
        else:
            raise ValueError("Azimuth value should be in the interval [-180, 180) degrees")

    def get_target_location(self):
        deg_to_rad = UnitConverterFactory.create_converter("DegreeToRadian")
        rad_to_deg = UnitConverterFactory.create_converter("RadianToDegree")
        alpha = self._azimuth * deg_to_rad.get_factor()
        lat1 = self._origin.latitude
        lon1 = self._origin.longitude

        lon2 = lon1 + self._distance * np.sin(alpha) / \
               (self.RADIUS * np.cos(lat1 * deg_to_rad.get_factor())) * rad_to_deg.get_factor()
        lat2 = lat1 + self._distance * np.cos(alpha) / self.RADIUS * rad_to_deg.get_factor()
        return lat2, lon2


if __name__ == "__main__":
    p1 = GeoLocation(41.746396, -111.816586)
    location_finder = GeoLocationFinder(p1, 150, -45)
    lat, lon = location_finder.get_target_location()
    print(f"latitude: {lat}")
    print(f"longitude: {lon}")
