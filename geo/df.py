import numpy as np
import pandas as pd

from typing import List

from geo.geomath import num_haversine, vec_haversine


# Both methods below were taken from
# https://medium.com/unit8-machine-learning-publication/
# from-pandas-wan-to-pandas-master-4860cf0ce442

def mem_usage(df: pd.DataFrame) -> str:
    """
    This method styles the memory usage of a DataFrame to be readable as MB.
    Parameters
    ----------
    df: pd.DataFrame
        Data frame to measure.
    Returns
    -------
    str
        Complete memory usage as a string formatted for MB.
    """
    return f'{df.memory_usage(deep=True).sum() / 1024 ** 2 : 3.2f} MB'


def categorize_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.copy(deep=True).astype({col: 'category' for col in columns})


def convert_df(df: pd.DataFrame, deep_copy: bool = True) -> pd.DataFrame:
    """
    Automatically converts columns that are worth stored as
    ``categorical`` dtype.
    Parameters
    ----------
    df: pd.DataFrame
        Data frame to convert.
    deep_copy: bool
        Whether or not to perform a deep copy of the original data frame.
    Returns
    -------
    pd.DataFrame
        Optimized copy of the input data frame.
    """
    return df.copy(deep=deep_copy).astype({
        col: 'category' for col in df.columns
        if df[col].nunique() / df[col].shape[0] < 0.5})


class DataCleaner(object):
    """
    Specialized data cleaner for the Dublin Bus data set.
    """

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
        """
        Calculates the consecutive duration in seconds.
        :param df: Input DataFrame
        :return: DataFrame with added 'dt' column.
        """
        df[self.dt_col] = df[self.ts_col].diff()
        df[self.dt_col] = df[self.dt_col].fillna(value=0.0)
        df[self.dt_col] = df[self.dt_col] / self.one_second
        return df

    def calculate_dx(self,
                     df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the consecutive distance in meters.
        :param df: Input DataFrame
        :return: DataFrame with added 'dx' column.
        """
        lat0 = df[self.lat_col][:-1].to_numpy()
        lon0 = df[self.lon_col][:-1].to_numpy()
        lat1 = df[self.lat_col][1:].to_numpy()
        lon1 = df[self.lon_col][1:].to_numpy()
        dist = vec_haversine(lat0, lon0, lat1, lon1)
        df[self.dx_col] = np.insert(dist, 0, 0.0)
        return df

    def get_max_speed(self, df: pd.DataFrame) -> float:
        """
        Calculates the maximum speed using the Tukey box plot algorithm
        :param df: Source DataFrame
        :return: Speed at the top whisker of the box plot
        """
        q = df[self.speed_col].quantile([.25, .5, .75])
        iqr = q.loc[0.75] - q.loc[0.25]
        return q.loc[0.75] + 1.5 * iqr

    def calculate_speed(self,
                        df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the consecutive average speeds in km/h.
        :param df: Input DataFrame
        :return: DataFrame with added 'v' column.
        """
        dx = df[self.dx_col].to_numpy()
        dt = df[self.dt_col].to_numpy()
        v = np.zeros_like(dx)
        zi = dt > 0
        v[zi] = dx[zi] / dt[zi] * 3.6
        df[self.speed_col] = v
        return df

    @staticmethod
    def get_type1_anomalies(df: pd.DataFrame) -> pd.DataFrame:
        return df[df['dx'] == 0.0]

    def calculate_derived_columns(self,
                                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all the derived columns: 'dt', 'dx' and 'v'.
        :param df: Input DataFrame
        :return: DataFrame with added columns.
        """
        df = self.calculate_dt(df)
        df = self.calculate_dx(df)
        df = self.calculate_speed(df)
        return df

    def calculate_type2_anomalies(self,
                                  df: pd.DataFrame,
                                  max_speed: float):
        return df[df[self.speed_col] > max_speed]

    def fix_type1_anomaly(self,
                          df: pd.DataFrame,
                          idx: int) -> pd.DataFrame:
        """
        Fixes a type-1 anomaly
        :param df: Source DataFrame
        :param idx: Anomaly index
        :return: DataFrame with corrected anomaly
        """
        i1 = df.index.get_loc(idx)
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

        # Recalculate the speed in km/h
        df.loc[idx2, self.speed_col] = df.loc[idx2, self.dx_col] / \
                                       df.loc[idx2, self.dt_col] * 3.6
        return df

    def fix_type1_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fixes all type-1 anomalies
        :param df: Source DataFrame
        :return: DataFrame with corrected type-1 anomalies
        """
        df = df[df['dx'] > 0.0].copy()
        df = self.calculate_derived_columns(df)
        return df

    def fix_type2_anomaly(self,
                          df: pd.DataFrame,
                          anom: pd.DataFrame) -> pd.DataFrame:
        """
        Fixes a type-2 anomaly
        :param df: Source DataFrame
        :param anom: Anomaly DataFrame - only the first will be fixed
        :return: DataFrame with corrected anomaly
        """
        return self.fix_type1_anomaly(df, anom.index[0])

    def fix_type2_anomalies(self,
                            df: pd.DataFrame,
                            max_speed: float) -> (pd.DataFrame, pd.DataFrame):
        """
        Detects and fixes type-2 anomalies (time travellers?)
        :param df: Source DataFrame
        :param max_speed: Maximum allowed speed for the bus
        :return: Tuple with the cleaned DataFrame and the anomaly DataFrame
        """
        anomalies = None
        type2 = self.calculate_type2_anomalies(df, max_speed)
        while type2.shape[0] > 0:
            df = self.fix_type2_anomaly(df, type2)
            idx = type2.index[0]
            row = df.loc[idx:idx].copy()
            if anomalies is None:
                anomalies = row
            else:
                anomalies = pd.concat([anomalies, row])
            df = df.drop(index=idx)
            type2 = self.calculate_type2_anomalies(df, max_speed)
        return df, anomalies
