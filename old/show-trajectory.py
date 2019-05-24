import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import mplleaflet


def load_day(month):
    file_name = 'data/siri.201301{:02d}.csv'.format(month)
    column_names = ['timestamp', 'line_id', 'direction', 'journey_pattern_id',
                    'time_frame', 'journey_id', 'operator', 'congestion', 'lon', 'lat',
                    'delay', 'block_id', 'vehicle_id', 'stop_id', 'stop']
    df = pd.read_csv(file_name, header=None, names=column_names)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
    return df


def run():
    df = load_day(1)
    trip = df.loc[df['line_id'] == 27, ['timestamp', 'lat', 'lon', 'stop']]
    stop = trip.loc[trip['stop'] == 1, ['timestamp', 'lat', 'lon']]
    print(df['line_id'].unique())

    plt.plot(stop['lon'], stop['lat'], 'rs')
    # print(stop.head())
    # plt.plot(trip['lon'], trip['lat'])
    mplleaflet.show()


if __name__ == "__main__":
    # print(os.getcwd())
    run()
