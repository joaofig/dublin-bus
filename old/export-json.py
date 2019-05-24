import numpy as np
import pandas as pd
import json


def format_date_time(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S+0000')


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
    df['date_time'] = df['timestamp'].apply(format_date_time)
    return df


def main():
    df = load_day(2)
    # df.apply(create_track_point)

    vehicles = df['vehicle_id'].unique()

    for v in vehicles:
        print(v)

        points = []
        vdf = df[df['vehicle_id'] == v].drop_duplicates(subset=['date_time'])
        for row in vdf.itertuples():
            pt = {'time': row.date_time,
                  'point': 'POINT({0} {1})'.format(row.lon, row.lat),
                  'id': "\\x0001"
                  }
            points.append(pt)

        with open('data/json/vehicle_{0}.json'.format(v), 'w') as f:
            f.write(json.dumps(points))


if __name__ == '__main__':
    main()
