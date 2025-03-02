from typing import Tuple
import pandas as pd
from helpers.helpers import Helper
from helpers.utils import pipeline_step
from preprocessing.metadata.dataclasses import Event, Sensor
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import pickle
from ydata_profiling import ProfileReport, compare
from helpers.constants import dataframe_metadata
from preprocessing.common.preprocessing_pipeline import Preprocessing
from visualization.plotter import DataPlotter
from sklearn.preprocessing import StandardScaler


class WFCPreprocessing(Preprocessing):

    def __init__(self, event_id: int):
        super().__init__(event_id)
        self.logger = Helper.get_logger(name="WFAPreprocessing")
        #self.weeks = 2

    def process(self):
        df = self.load_dataframe()

        df_avg = self.keep_columns(columns_to_keep="avg", df=df)
        self.sensor_metadata = self.load_sensor_data(df=df_avg)
        print(self.sensor_metadata)
        df_avg = self.filter_status_rows(df=df)
        #self.plot_columns(df=df_avg)
        #self.pdf_report(df=df_avg)
        useless_columns = [
            "sensor_46_avg",  # constant zero
            "sensor_49_avg",  # constant zero
            "sensor_1_avg",  # angle
            "sensor_2_avg",  # angle
            "sensor_5_avg",  # angle
            "sensor_42_avg",
            "sensor_26_avg"# angle
        ]
        #df_avg = df_avg.drop(columns=useless_columns)
        #df_avg = self.subtract_ambient_temperature(df=df_avg)
        #df_avg = df_avg.drop(columns=["sensor_0_avg"])
        df_avg = self.interpolate_from_sixth(df=df_avg)
        #df_normalized = self.normalize_sensor_data(df=df_avg)
        #self.pdf_report(df=df_normalized)

        df_train, df_test = self.split_dataframe_by_timestamp_advanced(df=df_avg, timestamp=self.event.event_start)
        #self.pdf_report(df=df_train, name="train.html")
        #self.pdf_report(df=df_test, name="test.html")
        train_path = f"res/training_data/WFA/{self.event_id}.pkl"
        test_path = f"res/testing_data/WFA/{self.event_id}.pkl"
        df_train.reset_index().to_pickle(train_path)
        df_test.reset_index().to_pickle(test_path)

    @pipeline_step("Split into Train/Test")
    def split_dataframe_by_timestamp_advanced(self, df: pd.DataFrame, timestamp):
        """
        Splits the DataFrame into two parts:
          - df_before: all rows with 'time_stamp' earlier than two weeks before the given timestamp.
          - df_between: rows from two weeks before the given timestamp up to (but not including) the given timestamp.

        Parameters:
            df (pd.DataFrame): Input DataFrame with a 'time_stamp' column.
            timestamp (str or pd.Timestamp): Reference timestamp.

        Returns:
            tuple: (df_before, df_between)
        """
        timestamp = pd.to_datetime(timestamp)
        lower_bound = timestamp - pd.Timedelta(weeks=2)

        # Ensure 'time_stamp' column is in datetime format.
        if not pd.api.types.is_datetime64_any_dtype(df['time_stamp']):
            df['time_stamp'] = pd.to_datetime(df['time_stamp'])

        # Optionally copy only columns from the sixth onward if needed.
        # df_subset = df.iloc[:, 5:].copy()  # Not used in splitting

        # Split the DataFrame:
        df_before = df[df['time_stamp'] < lower_bound].copy()
        df_between = df[(df['time_stamp'] >= lower_bound) & (df['time_stamp'] < timestamp)].copy()

        return df_before, df_between

    @pipeline_step("Interpolate")
    def interpolate_from_sixth(self, df: pd.DataFrame):
        df_copy = df.copy()
        cols_to_interpolate = df_copy.columns[5:]
        df_copy[cols_to_interpolate] = df_copy[cols_to_interpolate].interpolate(method='linear', axis=0)
        return df_copy
