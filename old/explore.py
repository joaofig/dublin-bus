import pandas as pd
import numpy as np


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
    df = pd.read_csv(file_name, header=None, names=header, dtype=types, parse_dates=['time_frame'], infer_datetime_format=True)
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

    #c = 2 * np.arcsin(np.sqrt(a))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    meters = 6372000.0 * c
    return meters


def calculate_durations(data_frame, vehicle_id):
    one_second = np.timedelta64(1000000000, 'ns')
    dv = data_frame[data_frame['vehicle_id']==vehicle_id]
    ts = dv.timestamp.values
    dtd = ts[1:] - ts[:-1]
    dt = np.zeros(len(dtd) + 1)
    dt[1:] = dtd / one_second
    return dt


def calculate_distances(data_frame, vehicle_id):
    dv = data_frame[data_frame['vehicle_id']==vehicle_id]
    lat = dv.lat.values
    lon = dv.lon.values
    dxm = haversine_np(lon[1:], lat[1:], lon[:-1], lat[:-1])
    dx = np.zeros(len(dxm) + 1)
    dx[1:] = dxm
    return dx


def filter_columns(df):
    columns = ['timestamp', 'direction', 'journey_id', 'congestion', 'lon', 'lat', 'delay', 'vehicle_id', 'stop_id', 'at_stop']
    return df[columns]


def run():
    days = None
    for d in range(31):
        print("Day {0}".format(d + 1))

        day = filter_columns(load_day(d + 1))
        day['dt'] = 0.0
        day['dx'] = 0.0
        day['speed'] = 0.0
        if days is None:
            days = day
        else:
            days = days.append(day)

    vehicles = days['vehicle_id'].unique()

    for v in vehicles:
        print("Vehicle {0}".format(v))
        
        vehicle_selector = days['vehicle_id'] == v
        days.loc[vehicle_selector, 'dt'] = calculate_durations(days, v)
        days.loc[vehicle_selector, 'dx'] = calculate_distances(days, v)

    speed_selector = days['dt'] > 0
    days.loc[speed_selector, 'speed'] = days[speed_selector].dx / days[speed_selector].dt * 3.6

    # Filter invalid points (speeds over 100 km/h)
    days = days[days['speed'] < 100.0]

    days.to_csv('data/201301.csv', index=False)


if __name__ == "__main__":
    run()
