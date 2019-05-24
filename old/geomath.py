import numpy as np
import math


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.
    Taken from here: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas#29546836
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    meters = 6378137.0 * c
    return meters


def delta_location(lat, lon, bearing, meters):
    """
    Calculates a destination location from a starting location, a bearing and a distance in meters.
    :param lat: Start latitude
    :param lon: Start longitude
    :param bearing: Bearing (North is zero degrees, measured clockwise)
    :param meters: Distance to displace from the starting point
    :return: Tuple with the new latitude and longitude
    """
    delta = meters / 6378137.0
    theta = math.radians(bearing)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lat_r2 = math.asin(math.sin(lat_r) * math.cos(delta) + math.cos(lat_r) * math.sin(delta) * math.cos(theta))
    lon_r2 = lon_r + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(lat_r),
                                math.cos(delta) - math.sin(lat_r) * math.sin(lat_r2))
    return math.degrees(lat_r2), math.degrees(lon_r2)


def delta_degree_to_meters(lat, lon, delta_lat=0, delta_lon=0):
    return haversine_np(lon, lat, lon + delta_lon, lat + delta_lat)


def x_meters_to_degrees(meters, lat, lon):
    _, lon2 = delta_location(lat, lon, 90, meters)
    return abs(lon - lon2)


def y_meters_to_degrees(meters, lat, lon):
    lat2, _ = delta_location(lat, lon, 0, meters)
    return abs(lat - lat2)
