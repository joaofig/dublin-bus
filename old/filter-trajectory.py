import numpy as np
import pandas as pd
import math
import os.path
import matplotlib.pyplot as plt
import mplleaflet


def load_day(day):
    header = ['timestamp', 'line_id', 'direction', 'jrny_patt_id', 'time_frame', 'journey_id', 'operator',
              'congestion', 'lon', 'lat', 'delay', 'block_id', 'vehicle_id', 'stop_id', 'at_stop']
    types = {'timestamp': np.int64,
             'journey_id': np.int32,
             'congestion': np.int8,
             'lon': np.float64,
             'lat': np.float64,
             'delay': np.int8,
             'vehicle_id': np.int32,
             'at_stop': np.int8}
    file_name = 'data/siri.201301{0:02d}.csv'.format(day)
    df = pd.read_csv(file_name, header=None, names=header, dtype=types, parse_dates=['time_frame'],
                     infer_datetime_format=True)
    null_replacements = {'line_id': 0, 'stop_id': 0}
    df = df.fillna(value=null_replacements)
    df['line_id'] = df['line_id'].astype(np.int32)
    df['stop_id'] = df['stop_id'].astype(np.int32)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
    return df


def haversine_np(lon1, lat1, lon2, lat2):
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


def calculate_durations(data_frame, vehicle_id):
    one_second = np.timedelta64(1000000000, 'ns')
    dv = data_frame[data_frame['vehicle_id'] == vehicle_id]
    ts = dv.timestamp.values
    dtd = ts[1:] - ts[:-1]
    dt = np.zeros(len(dtd) + 1)
    dt[1:] = dtd / one_second
    return dt


def calculate_distances(data_frame, vehicle_id):
    dv = data_frame[data_frame['vehicle_id'] == vehicle_id]
    lat = dv.lat.values
    lon = dv.lon.values
    dxm = haversine_np(lon[1:], lat[1:], lon[:-1], lat[:-1])
    dx = np.zeros(len(dxm) + 1)
    dx[1:] = dxm
    return dx


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


def calculate_q(lat, lon, sigma_speed):
    q = np.zeros((4, 4), dtype=np.float)
    q[2, 2] = x_meters_to_degrees(sigma_speed, lat, lon) ** 2
    q[3, 3] = y_meters_to_degrees(sigma_speed, lat, lon) ** 2
    return q


def calculate_r(lat, lon, sigma):
    r = np.zeros((2, 2), dtype=np.float)
    r[0, 0] = x_meters_to_degrees(lat, lon, sigma) ** 2
    r[1, 1] = y_meters_to_degrees(lat, lon, sigma) ** 2
    return r


def calculate_p(lat, lon, sigma, sigma_speed):
    p = np.zeros((4, 4), dtype=np.float)
    p[0, 0] = x_meters_to_degrees(sigma, lat, lon) ** 2
    p[1, 1] = y_meters_to_degrees(sigma, lat, lon) ** 2
    p[2, 2] = x_meters_to_degrees(sigma_speed, lat, lon) ** 2
    p[3, 3] = y_meters_to_degrees(sigma_speed, lat, lon) ** 2
    return p


def calculate_phi(dt):
    """
    Calculates the Φ matrix
    :param dt: Δtᵢ
    :return: The Φ matrix
    """
    phi = np.eye(4)
    phi[0, 2] = dt
    phi[1, 3] = dt
    return phi


def calculate_kalman_gain(p, c, r):
    num = np.matmul(p, np.transpose(c))
    den = np.matmul(c, num) + r
    return np.matmul(num, np.linalg.pinv(den))


def predict_step(prev_x, prev_p, phi, sigma_speed):
    lon = prev_x[0, 0]
    lat = prev_x[1, 0]
    next_x = np.matmul(phi, prev_x)
    next_p = np.matmul(np.matmul(phi, prev_p), np.transpose(phi)) + calculate_q(lat, lon, sigma_speed)
    return next_x, next_p


def update_step(predicted_x, predicted_p, c, y, sigma_x):
    lon = predicted_x[0, 0]
    lat = predicted_x[1, 0]
    r = calculate_r(lat, lon, sigma_x)
    k = calculate_kalman_gain(predicted_p, c, r)
    updated_x = predicted_x + np.matmul(k, y - np.matmul(c, predicted_x))
    identity = np.eye(4)
    updated_p = np.matmul(identity - np.matmul(k, c), predicted_p)
    return updated_x, updated_p


def read_row(t, i):
    r = np.zeros((4, 1), dtype=np.float)
    r[0, 0] = t[i, 1]
    r[1, 0] = t[i, 2]
    return r


def read_observation(t, i):
    r = np.zeros((2, 1), dtype=np.float)
    r[0, 0] = t[i, 1]
    r[1, 0] = t[i, 2]
    return r


def get_line_text(lon0, lat0, lon1, lat1):
    return "\"LINE({0} {1}, {2}, {3})\"".format(lon0, lat0, lon1, lat1)


def convert_speed(deg_per_sec_x, deg_per_sec_y, lat, lon):
    """
    Converts a speed in degrees per second decomposed in longitude and latitude components
    into an absolute value measured in meters per second.
    :param deg_per_sec_x: Speed along the longitude (x) axis
    :param deg_per_sec_y: Speed along the latitude (y) axis
    :param lat: Latitude of the location where the original speed is measured
    :param lon: Longitude of the location where the original speed is measured
    :return: Absolute value of the speed in meters per second.
    """
    ms_x = delta_degree_to_meters(lat, lon, delta_lon=deg_per_sec_x)
    ms_y = delta_degree_to_meters(lat, lon, delta_lat=deg_per_sec_y)

    ms = math.sqrt(ms_x * ms_x + ms_y * ms_y)
    return ms


def run():
    day = load_day(1)
    vehicles = day['vehicle_id'].unique()

    vehicle_id = 43004

    trajectories = {}

    for v in vehicles:
        vehicle_selector = day['vehicle_id'] == v
        day.loc[vehicle_selector, 'dt'] = calculate_durations(day, v)
        day.loc[vehicle_selector, 'dx'] = calculate_distances(day, v)
        # speed_selector = day.loc[vehicle_selector, 'dt'] > 0
        day.loc[vehicle_selector, 'speed'] = day[vehicle_selector].dx / day[vehicle_selector].dt * 3.6

        trajectories[v] = day.loc[vehicle_selector, ['dt', 'lon', 'lat']].values

    t = trajectories[vehicle_id]

    prev_x = read_row(t, 0)
    lat = prev_x[1, 0]
    lon = prev_x[0, 0]
    sigma_x = 10.0
    sigma_s = 100.0
    c = np.zeros((2, 4), dtype=np.float)
    c[0, 0] = 1.0
    c[1, 1] = 1.0
    prev_p = calculate_p(lat, lon, sigma_x, sigma_s)

    # print("observed, filtered")
    # print("lat_obs, lon_obs, lat_flt, lon_flt, speed_flt")

    result = np.zeros((t.shape[0], 5), dtype=np.float64)
    result[0, 0:4] = np.transpose(prev_x)
    result[0, 2:4] = result[0, 0:2]

    for i in range(1, t.shape[0]):
        y = read_observation(t, i)
        phi = calculate_phi(t[i, 0])
        next_x, next_p = predict_step(prev_x, prev_p, phi, sigma_s)
        updated_x, updated_p = update_step(next_x, next_p, c, y, sigma_x)

        speed_x = updated_x[2, 0]   # Longitude speed in degrees per second
        speed_y = updated_x[3, 0]   # Latitude speed in degrees per second

        px = y[0, 0]                # Longitude
        py = y[1, 0]                # Latitude

        ms = convert_speed(speed_x, speed_y, py, px) * 3.6  # Estimated speed in km/h

        # Pack the row data
        result[i, 0:2] = np.transpose(y)
        result[i, 2:4] = np.transpose(updated_x[0:2, 0])
        result[i, 4] = ms

        # print("{0}, {1}, {2}, {3}, {4}".format(*result[i, :]))

        prev_x, prev_p = updated_x, updated_p

    out_columns = ['lon', 'lat', 'flt_lon', 'flt_lat', 'speed_flt']

    df = day.loc[day['vehicle_id'] == vehicle_id, ['timestamp', 'stop_id', 'at_stop', 'delay', 'speed']]
    flt = pd.DataFrame(data=result, columns=out_columns, index=df.index.values)
    df = pd.concat([df, flt], axis=1)

    df.to_csv('data/speed.csv', index=False)

    # print(df.head(10))

    plt.plot(result[:, 0], result[:, 1], "r")
    plt.plot(result[:, 2], result[:, 3], "b")
    mplleaflet.show()


if __name__ == "__main__":
    run()
