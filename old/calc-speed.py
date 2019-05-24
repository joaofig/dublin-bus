import numpy as np
import pandas as pd
import geomath as gm
import kalman
import math


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
    dxm = gm.haversine_np(lon[1:], lat[1:], lon[:-1], lat[:-1])
    dx = np.zeros(len(dxm) + 1)
    dx[1:] = dxm
    return dx


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
    ms_x = gm.delta_degree_to_meters(lat, lon, delta_lon=deg_per_sec_x)
    ms_y = gm.delta_degree_to_meters(lat, lon, delta_lat=deg_per_sec_y)

    ms = math.sqrt(ms_x * ms_x + ms_y * ms_y)
    return ms


def filter_trajectory(trajectory):
    prev_x = read_row(trajectory, 0)
    lat = prev_x[1, 0]
    lon = prev_x[0, 0]

    # Calculate the location standard deviations in degrees
    sigma_x = gm.x_meters_to_degrees(10.0, lat, lon)
    sigma_y = gm.y_meters_to_degrees(10.0, lat, lon)
    sigma = np.array([sigma_x, sigma_y])

    # Calculate the speed standard deviations in degrees per second
    sigma_sx = gm.x_meters_to_degrees(10.0, lat, lon)
    sigma_sy = gm.y_meters_to_degrees(10.0, lat, lon)
    sigma_speed = np.array([sigma_sx, sigma_sy])

    c = np.zeros((2, 4), dtype=np.float)
    c[0, 0] = 1.0
    c[1, 1] = 1.0
    prev_p = kalman.calculate_p(sigma, sigma_speed)

    # Set-up the result array
    result = np.zeros((trajectory.shape[0], 5), dtype=np.float64)
    result[0, 0:4] = np.transpose(prev_x)
    result[0, 2:4] = result[0, 0:2]

    for i in range(1, trajectory.shape[0]):
        y = read_observation(trajectory, i)
        phi = kalman.calculate_phi(trajectory[i, 0])
        next_x, next_p = kalman.predict_step(prev_x, prev_p, phi, sigma_speed)
        updated_x, updated_p = kalman.update_step(next_x, next_p, c, y, sigma)

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

    return result


def result_to_dataframe(result, data_frame, vehicle):
    out_columns = ['lon', 'lat', 'flt_lon', 'flt_lat', 'speed_flt']

    df = data_frame.loc[data_frame['vehicle_id'] == vehicle, ['timestamp', 'stop_id', 'at_stop', 'delay', 'speed']]
    flt = pd.DataFrame(data=result, columns=out_columns, index=df.index.values)
    df = pd.concat([df, flt], axis=1)
    return df


def main():
    day = load_day(1)
    vehicles = day['vehicle_id'].unique()

    trajectories = {}

    for vehicle in vehicles:
        print(vehicle)

        vehicle_selector = day['vehicle_id'] == vehicle
        day.loc[vehicle_selector, 'dt'] = calculate_durations(day, vehicle)
        day.loc[vehicle_selector, 'dx'] = calculate_distances(day, vehicle)
        day.loc[vehicle_selector, 'speed'] = day[vehicle_selector].dx / day[vehicle_selector].dt * 3.6

        trajectory = day.loc[vehicle_selector, ['dt', 'lon', 'lat']].values

        trajectories[vehicle] = trajectory

        filtered = filter_trajectory(trajectory)
        df = result_to_dataframe(filtered, day, vehicle)

        df.to_csv('data/out/speed_{0}.csv'.format(vehicle), index=False)


if __name__ == "__main__":
    main()
