import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx


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


def create_track_point(row):
    return gpxpy.gpx.GPXTrackPoint(latitude=row.lat, longitude=row.lon, time=row.timestamp)


def main():
    df = load_day(2)
    # df.apply(create_track_point)

    vehicles = df['vehicle_id'].unique()

    for v in vehicles:
        gpx = gpxpy.gpx.GPX()
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        print(v)

        for row in df[df['vehicle_id'] == v].itertuples():
            gpx_segment.points.append(create_track_point(row))

        with open('data/gpx/vehicle_{0}.gpx'.format(v), 'w') as f:
            f.write(gpx.to_xml())


if __name__ == '__main__':
    main()
