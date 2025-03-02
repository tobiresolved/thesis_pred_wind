from typing import Tuple
import pandas as pd
from helpers.helpers import Helper
from helpers.utils import pipeline_step
from preprocessing.metadata.dataclasses import Event, Sensor, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import pickle
from ydata_profiling import ProfileReport, compare
from helpers.constants import dataframe_metadata
import numpy as np
from dataclasses import dataclass
from visualization.plotter import DataPlotter

from sklearn.preprocessing import StandardScaler


class Preprocessing:

    def __init__(self, event_id: int):
        self.df = None
        self.event_id = event_id
        self.event = Event.from_json(event_id=self.event_id)
        self.dataset = Dataset.get_from_pickle(event_id=self.event_id)
        self.df_train = None
        self.df_prediction = None
        self.sensor_metadata = None

    @pipeline_step("Load Dataframe")
    def load_dataframe(self) -> pd.DataFrame:
        df_pickle_location = Helper.get_dataframe_pickle_location(dataset_id=self.event_id)
        with open(df_pickle_location, "rb") as file:
            df = pickle.load(file)
            return df

    @pipeline_step("Load Sensor Data")
    def load_sensor_data(self, df) -> dict:
        columns_to_use = df.columns[5:].tolist()
        sensor_metadata = Sensor.from_columns(columns=columns_to_use, event_id=self.event_id)
        return sensor_metadata

    @pipeline_step("Keep Columns")
    def keep_columns(self, columns_to_keep, df: pd.DataFrame) -> pd.DataFrame:

        metadata_columns = df.columns[:5].tolist()
        if isinstance(columns_to_keep, str):
            suffixes = [columns_to_keep]
        else:
            suffixes = columns_to_keep
        sensor_columns = [
            col for col in df.columns[5:]
            if any(col.endswith(suffix) for suffix in suffixes)
        ]
        cols_to_keep = metadata_columns + sensor_columns
        df = df[cols_to_keep]
        return df

    @pipeline_step("Plot Columns")
    def plot_columns(self, df: pd.DataFrame):
        plotter = DataPlotter(event=self.event)
        for col in df.columns[5:]:
            self.logger.info(f"Plotting column: {col}")
            plotter.plot_timeseries(
                timestamp_series=df["time_stamp"],
                sensor_series=df[col],
                sensor=self.sensor_metadata[col]
            )

    @pipeline_step("Filter Normal Rows")
    def filter_status_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = df[df["status_type_id"].isin([0, 2])]
        return filtered_df

    @pipeline_step("Generate PDF-Report")
    def pdf_report(self, df: pd.DataFrame, name: str) -> None:
        file_location = Helper.get_pdf_report_location(dataset_id=self.event_id)
        file_location = file_location + name
        definitions = dataframe_metadata
        sensor_descriptions = {
            sensor_key: sensor_obj
            for sensor_key, sensor_obj in self.sensor_metadata.items()
        }

        definitions.update(sensor_descriptions)

        profile = ProfileReport(
            df,
            title=str(self.event),
            minimal=True,
            explorative=False,
            tsmode=True,
            sortby="time_stamp",
            variables={"descriptions": definitions}
        )
        profile.to_file(file_location)

        self.logger.info(f"PDF report created at: {file_location}")


    @pipeline_step("Rolling Mean")
    def compute_rolling_mean(self, window: int = 3) -> None:
        # Create a copy of the original DataFrame so that self.df remains unchanged
        df_copy = self.df.copy()

        # Convert the first column (timestamp) to datetime in the copied DataFrame
        timestamp_col = df_copy.columns[0]
        df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

        # Select the sensor columns (6th column onward)
        sensor_cols = df_copy.columns[5:].tolist()

        # Set the timestamp as the index on the copy for rolling computation
        df_copy = df_copy.set_index(timestamp_col)

        # Compute the rolling mean on the sensor columns using the specified window
        rolling_mean_df = df_copy[sensor_cols].rolling(window=window).mean()

        # Reset the index so that the timestamp becomes a regular column again
        rolling_mean_df = rolling_mean_df.reset_index()

        # Save the rolling mean DataFrame for later use without altering self.df
        self.rolling_mean_df = rolling_mean_df


    @pipeline_step("Denoising Fourier Transform")
    def denoise_fourier(self, cutoff: float = 0.1) -> "PreprocessingPipeline":
        """
        Denoise the sensor data using a Fourier transform.

        Assumptions:
          - The first five columns are metadata.
          - Sensor data starts from the 6th column.

        The method applies a low-pass filter in the frequency domain by setting to zero all frequency
        components with absolute frequency greater than the specified cutoff. The cutoff is defined
        in units of cycles per sample (with the default assuming unit sampling intervals).

        Parameters:
            cutoff (float): The frequency threshold above which components will be removed (default: 0.1).

        Returns:
            PreprocessingPipeline: The updated pipeline with a new attribute `denoised_df` containing
            the denoised sensor data.
        """
        # Create a copy to preserve the original sensor_df
        denoised_df = self.df.copy()

        # Process each sensor column (columns 6 onward)
        for col in denoised_df.columns[5:]:
            signal = denoised_df[col].values
            N = len(signal)

            # Compute the FFT of the signal
            fft_vals = np.fft.fft(signal)

            # Get the corresponding frequencies (assume unit spacing)
            freqs = np.fft.fftfreq(N, d=1)

            # Zero out all frequency components with absolute frequency > cutoff
            fft_vals[np.abs(freqs) > cutoff] = 0

            # Inverse FFT to obtain the denoised signal and take the real part
            denoised_signal = np.fft.ifft(fft_vals).real

            # Replace the original column data with the denoised signal
            denoised_df[col] = denoised_signal

        # Store the denoised DataFrame for later use
        self.denoised_df = denoised_df
        self.logger.info(f"Denoising complete using cutoff frequency: {cutoff}")
        return self

    @pipeline_step("Index")
    def get_switch_index(self, column: pd.Series) -> int:
        """
        Return the index where the boolean column switches between False and True.

        Assumes that the column switches exactly once.

        Parameters:
            column (pd.Series): A pandas Series (e.g., df["train_test"]) containing boolean values.

        Returns:
            int: The index where the column value changes.

        Raises:
            ValueError: If the number of switches is not exactly one.
        """
        # Convert boolean values to integers (False -> 0, True -> 1)
        int_values = column.astype(int)

        # Compute the difference between consecutive values; non-zero indicates a change
        diff = int_values.diff().fillna(0)

        # Get the indices where a switch occurs
        switch_indices = diff[diff != 0].index.tolist()

        if len(switch_indices) != 1:
            raise ValueError(f"Expected exactly one switch, but found {len(switch_indices)} switches.")

        return switch_indices[0]

    @pipeline_step("Index")
    def split_df(self, df: pd.DataFrame, index: int) -> tuple:

        train_df = df.iloc[:index]
        test_df = df.iloc[index:]

        return train_df, test_df


    @pipeline_step("Compare Train/Test")
    def compare(self, df_train: pd.DataFrame, df_prediction: pd.DataFrame) -> "PreprocessingPipeline":
        file_location = Helper.get_pdf_comparison_location(dataset_id=self.event_id)
        definitions = dataframe_metadata
        sensor_descriptions = {
            sensor_key: sensor_obj
            for sensor_key, sensor_obj in self.sensor_metadata.items()
        }

        definitions.update(sensor_descriptions)
        profile_train = ProfileReport(
            df_train,
            title="Train Report",
            minimal=True,
            explorative=False,
            tsmode=True,
            variables={"descriptions": definitions}
        )

        profile_prediction = ProfileReport(
            df_prediction,
            title=f"Prediction Report for {str(self.event)}",
            minimal=True,
            explorative=False,
            tsmode=True,
            variables={"descriptions": definitions}
        )

        comparison_report = profile_train.compare(profile_prediction)
        comparison_report.to_file(file_location)




    @pipeline_step("Clean")
    def handle_missing_values(self):
        pass

    @pipeline_step("Aggregate")
    def aggregate_hourly(self):
        pass



    @pipeline_step("Normalize")
    def normalize_sensor_data(self):
        pass

