import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import mplleaflet
from shapely.geometry import Point
from functools import partial
from shapely.ops import transform
from shapely.ops import cascaded_union
from shapely.ops import transform
from shapely.ops import cascaded_union
import pyproj
import math
from sklearn.cluster import DBSCAN
import os.path


def pandas_load_day(day):
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
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
    return df


def create_radian_columns(df):
    df['rad_lng'] = np.radians(df['lon'].values)
    df['rad_lat'] = np.radians(df['lat'].values)
    return df


def buffer_in_meters(lng, lat, radius):
    proj_meters = pyproj.Proj(init='epsg:3857')
    proj_latlng = pyproj.Proj(init='epsg:4326')

    project_to_meters = partial(pyproj.transform, proj_latlng, proj_meters)
    project_to_latlng = partial(pyproj.transform, proj_meters, proj_latlng)

    pt_latlng = Point(lng, lat)
    pt_meters = transform(project_to_meters, pt_latlng)

    buffer_meters = pt_meters.buffer(radius)
    buffer_latlng = transform(project_to_latlng, buffer_meters)
    return buffer_latlng


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


def generate_blob_clusters(df, eps_in_meters=50):
    # Group the observations by cluster identifier
    groups = df.groupby('Cluster')
    clusters = list()
    blobs = list()
    counts = list()

    for cluster_id, points in groups:
        if cluster_id >= 0:
            buffer_radius = eps_in_meters * 0.6
            buffers = [buffer_in_meters(lon, lat, buffer_radius)
                       for lon, lat in zip(points['lon'], points['lat'])]
            blob = cascaded_union(buffers)
            blobs.append(blob)
            clusters.append(cluster_id)
            counts.append(len(points))

    # Create the GeoDataFrame from the cluster numbers and blobs
    data = {'cluster': clusters, 'polygon': blobs, 'count': counts}

    cluster_gdf = gpd.GeoDataFrame(pd.DataFrame(data), geometry='polygon')
    cluster_gdf.crs = {'init': 'epsg:4326'}
    return cluster_gdf


def get_move_ability_array(source: np.ndarray, index: int, window=4) -> np.ndarray:
    source_size = source.shape[0]
    n = 2 * window + 1
    if 0 <= index < window:
        # The index lies on the left window
        result = np.full(n, source[0])
        result[window - index:] = source[:window + index + 1]
    elif source_size - window <= index < source_size:
        # The index lies in the right window
        result = np.full(n, source[-1])
        a = source_size - index - 1
        result[:window + a + 1] = source[index - window:index + a + 1]
    else:
        result = source[index - window:index + window + 1]
    return result


def trajectory_curve_distance(lats: np.ndarray, lons: np.ndarray) -> float:
    distance = 0.0
    for i in range(1, lats.shape[0]):
        distance += haversine_np(lons[i-1], lats[i-1], lons[i], lats[i])
    return distance


def trajectory_direct_distance(latitudes: np.ndarray, longitudes: np.ndarray) -> float:
    return haversine_np(longitudes[0], latitudes[0], longitudes[-1], latitudes[-1])


def get_move_ability(df: pd.DataFrame, columns=['lat', 'lon']) -> np.ndarray:
    locations = np.transpose(df[columns].values)

    if locations.shape[1] < 9:
        return np.zeros(locations.shape[1])

    move_ability = []
    for i in range(locations.shape[1]):
        lats = get_move_ability_array(locations[0, :], i)
        lons = get_move_ability_array(locations[1, :], i)

        curve_dist = trajectory_curve_distance(lats, lons)
        direct_dist = trajectory_direct_distance(lats, lons)

        if curve_dist > 0.0:
            move_ability.append(direct_dist / curve_dist)
        else:
            move_ability.append(0)

    return np.array(move_ability)


def density_cluster(df, eps_in_meters=50):
    num_samples = 15
    earth_perimeter = 40070000.0  # In meters
    eps_in_radians = eps_in_meters / earth_perimeter * (2 * math.pi)

    db_scan = DBSCAN(eps=eps_in_radians, min_samples=num_samples, metric='haversine')
    return db_scan.fit_predict(df[['rad_lat', 'rad_lng']])


def show_blob_map(data_frame):
    gdf = generate_blob_clusters(data_frame)
    ax = gdf.geometry.plot(linewidth=2.0, color='red', edgecolor='red', alpha=0.5)
    # plt.show()
    mplleaflet.show(fig=ax.figure)


def run():
    # a = np.arange(100)
    # for i in range(100):
    #     print(i, get_move_ability_array(a, i))
    # for i in range(1, 32):
    #     print(i)
    #     day = pandas_load_day(i)
    #     stops = day.loc[day['at_stop'] == 1, ['lat', 'lon', 'stop_id']]
    #
    #     file_name = 'data/stops{0:02d}.csv'.format(i)
    #     stops.to_csv(file_name, index=False)

    day = 3

    out_file_name = 'data/move_ability_201301{0:02d}.csv'.format(day)

    if os.path.exists(out_file_name):
        types = {'timestamp': np.int64,
                 'journey_id': np.int32,
                 'congestion': np.int8,
                 'lon': np.float64,
                 'lat': np.float64,
                 'delay': np.int8,
                 'vehicle_id': np.int32,
                 'at_stop': np.int8}
        df = pd.read_csv(out_file_name, dtype=types, parse_dates=['time_frame'],
                         infer_datetime_format=True)
    else:
        df = pandas_load_day(day)
        df['move_ability'] = 0.0

        vehicles = df['vehicle_id'].unique()

        for v in vehicles:
            print(v)
            vehicle = df[df['vehicle_id'] == v]
            move_ability = get_move_ability(vehicle)
            df.loc[df['vehicle_id'] == v, 'move_ability'] = move_ability

        df.to_csv(out_file_name, index=False)

    hist = df.loc[df['move_ability'] >= 0.0, 'move_ability'].plot.kde()
    plt.show(hist)

    low_move_ability = (df['move_ability'] >= 0.0) & (df['move_ability'] < 0.3)
    ma = df[low_move_ability].copy()

    ma = create_radian_columns(ma)
    ma['Cluster'] = density_cluster(ma, eps_in_meters=25)

    # Filter out the noise points and retain only the clusters
    cluster_points = ma.loc[ma['Cluster'] > -1, ['Cluster', 'lat', 'lon']]

    show_blob_map(cluster_points)
    # plt.scatter(df.loc[low_move_ability, 'lon'], df.loc[low_move_ability, 'lat'])
    # mplleaflet.show()


if __name__ == '__main__':
    run()
