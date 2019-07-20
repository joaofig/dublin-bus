import numpy as np
import pandas as pd

from geo.geomath import num_haversine, vec_haversine


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

        # Calculate the forward differences of latitude and longitude.
        # One of the conditions for a type-1 anomaly to occur is to have both
        # differences equal to zero.
        df['dLat'] = df['Lat'].diff()
        df['dLon'] = df['Lon'].diff()
        df['dLat'] = df['dLat'].fillna(0.0)
        df['dLon'] = df['dLon'].fillna(0.0)

        # Now, shift both differences forward and backward so we can test for
        # the type-1 anomaly using a single row. This will make the test a
        # one-liner, and we can later remove the unnecessary columns.
        df['dLatPrev'] = df['dLat'].shift(periods=1, fill_value=0.0)
        df['dLonPrev'] = df['dLon'].shift(periods=1, fill_value=0.0)
        df['dLatNext'] = df['dLat'].shift(periods=-1, fill_value=0.0)
        df['dLonNext'] = df['dLon'].shift(periods=-1, fill_value=0.0)

        anomalies = (df['dLat'] == 0.0) \
                    & (df['dLon'] == 0.0) \
                    & (df['dLatPrev'] != 0.0) \
                    & (df['dLonPrev'] != 0.0) \
                    & (df['dLatNext'] != 0.0) \
                    & (df['dLonNext'] != 0.0)
        df['type1'] = False
        df.loc[anomalies, 'type1'] = True

        df = df.drop(['dLat', 'dLon', 'dLatPrev', 'dLonPrev', 'dLatNext',
                      'dLonNext'], axis=1)
        return df

    def get_anomalies(self,
                      df: pd.DataFrame,
                      max_speed: float) -> pd.DataFrame:
        return df[df[self.speed_col] > max_speed]

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

    def calculate_anomalies(self,
                            df: pd.DataFrame,
                            max_speed: float):
        df = self.calculate_derived_columns(df)
        anomalies = self.get_anomalies(df, max_speed)
        return df, anomalies

    def fix_anomaly(self,
                    df: pd.DataFrame,
                    anom: pd.DataFrame) -> pd.DataFrame:
        """
        Fixes a type-1 anomaly
        :param df: Source DataFrame
        :param anom: Anomaly DataFrame - only the first will be fixed
        :return: DataFrame with corrected anomaly
        """
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

        # Recalculate the speed in km/h
        df.loc[idx2, self.speed_col] = df.loc[idx2, self.dx_col] / \
                                       df.loc[idx2, self.dt_col] * 3.6
        return df

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
        df = self.get_type1_anomalies(df)
        anomalies = df[df['type1']]
        for idx in anomalies.index:
            df = self.fix_type1_anomaly(df, idx)
        df = df[~df['type1']]
        df = df.drop(['type1'], axis=1)
        return df

