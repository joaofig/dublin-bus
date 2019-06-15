import numpy as np
import pandas as pd

from geo.geomath import num_haversine, vec_haversine


class DataCleaner(object):

    def __init__(self,
                 ts_col: str = "Timestamp",
                 lat_col: str = "Lat",
                 lon_col: str = "Lon",
                 dx_col: str = "dx",
                 dt_col: str = "dt",
                 speed_col: str = "v",
                 one_second: int = 1000000):
        self.ts_col = ts_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.dx_col = dx_col
        self.dt_col = dt_col
        self.speed_col = speed_col
        self.one_second = one_second

    def calculate_dt(self,
                     df: pd.DataFrame) -> pd.DataFrame:
        df[self.dt_col] = df[self.ts_col].diff()
        df[self.dt_col] = df[self.dt_col].fillna(value=0.0)
        df[self.dt_col] = df[self.dt_col] / self.one_second
        return df

    def calculate_dx(self,
                     df: pd.DataFrame) -> pd.DataFrame:
        lat0 = df[self.lat_col][:-1].to_numpy()
        lon0 = df[self.lon_col][:-1].to_numpy()
        lat1 = df[self.lat_col][1:].to_numpy()
        lon1 = df[self.lon_col][1:].to_numpy()
        dist = vec_haversine(lat0, lon0, lat1, lon1)
        df[self.dx_col] = np.insert(dist, 0, 0.0)
        return df

    def calculate_speed(self,
                        df: pd.DataFrame) -> pd.DataFrame:
        dx = df[self.dx_col].to_numpy()
        dt = df[self.dt_col].to_numpy()
        v = np.zeros_like(dx)
        zi = dt > 0
        v[zi] = dx[zi] / dt[zi] * 3.6
        df[self.speed_col] = v
        return df

    def calculate_anomalies(self,
                            df: pd.DataFrame,
                            max_speed: float):
        df = self.calculate_dt(df)
        df = self.calculate_dx(df)
        df = self.calculate_speed(df)
        anom = df[df[self.speed_col] > max_speed]
        return df, anom

    def remove_anomaly(self,
                       df: pd.DataFrame,
                       anom: pd.DataFrame) -> pd.DataFrame:
        i1 = df.index.get_loc(anom.index[0])
        i0 = i1 - 1
        i2 = i1 + 1
        idx2 = df.index[i2]
        idx1 = df.index[i1]
        idx0 = df.index[i0]

        # Recalculate the time difference
        df.loc[idx2, self.dt_col] += df.loc[idx1, self.dt_col]

        # Recalculate the distance
        lat1 = df.loc[idx0, self.lat_col]
        lon1 = df.loc[idx0, self.lon_col]
        lat2 = df.loc[idx2, self.lat_col]
        lon2 = df.loc[idx2, self.lon_col]

        df.loc[idx2, self.dx_col] = num_haversine(lat1, lon1, lat2, lon2)

        # Recalculate the speed
        df.loc[idx2, self.speed_col] = df.loc[idx2, self.dx_col] / \
                                       df.loc[idx2, self.dt_col] * 3.6
        return df
