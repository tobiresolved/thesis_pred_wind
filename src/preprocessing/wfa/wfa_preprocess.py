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
from preprocessing.common.preprocess import Preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class WFAPreprocess(Preprocess):

    def __init__(self, event_id: int):
        super().__init__(event_id)

    @pipeline_step("Process", logger="WFA_PREPROCESS")
    def process(self, df_path: str):
        effective_wind_avg = Sensor(
            name="effective_wind_avg",
            description="Effective wind speed (average)",
            unit="m/s",
            is_angle=False,
            is_counter=False,
            stat_type="avg"
        )
        self.sensor_metadata["effective_wind_avg"] = effective_wind_avg

        # Create Sensor object for effective wind (standard deviation)
        effective_wind_std = Sensor(
            name="effective_wind_std",
            description="Effective wind speed (standard deviation)",
            unit="m/s",
            is_angle=False,
            is_counter=False,
            stat_type="std"
        )
        self.sensor_metadata["effective_wind_std"] = effective_wind_std

        # Create Sensor object for effective wind (minimum)
        effective_wind_min = Sensor(
            name="effective_wind_min",
            description="Effective wind speed (minimum)",
            unit="m/s",
            is_angle=False,
            is_counter=False,
            stat_type="min"
        )
        self.sensor_metadata["effective_wind_min"] = effective_wind_min

        # Create Sensor object for effective wind (maximum)
        effective_wind_max = Sensor(
            name="effective_wind_max",
            description="Effective wind speed (maximum)",
            unit="m/s",
            is_angle=False,
            is_counter=False,
            stat_type="max"
        )
        self.sensor_metadata["effective_wind_max"] = effective_wind_max
        df = self.load_dataframe(df_path=df_path)
        #df = self.filter_status_rows(df=df)
        df = self.interpolate_sensordata(df=df)
        df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        df = self.add_effective_wind(df=df)
        dp = DataPlotter(event=self.event)
        dp.sensor_metadata.update({"effective_wind_avg": effective_wind_avg})
        dp.plot_sensor_correlation_by_status_loop(df=df, sensor_x="effective_wind_avg", sensor_y="sensor_52_avg")
        dp.plot_sensor_correlation(df=df, sensor_x="wind_speed_3_avg", sensor_y="sensor_52_avg")
        df_train, df_validation = self.split_by_train_test(df=df)
        useless_columns = [
            "status_type_id",  # constant zero
            "train_test",  # constant zero
            "id",  # angle
            "asset_id",  # angle
            "time_stamp",  # pitch angle
            "sensor_1_avg",
            "sensor_2_avg",
            "wind_speed_3_avg",
            "wind_speed_4_avg",
            "wind_speed_3_max",
            "wind_speed_3_min",
            "wind_speed_3_std",
            "sensor_46_avg",  # constant zero
            "sensor_49_avg",
        ]
        df_train = df_train.drop(columns=useless_columns)
        df_validation = df_validation.drop(columns=useless_columns)

        df_train_X, df_train_Y, = self.split_dataframe(df_train)
        print(df_train_X)
        df_validation_X, df_validation_Y, = self.split_dataframe(df_validation)
        print(df_validation_X)
        # Instantiate the model

        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model on the training set
        model.fit(df_train_X, df_train_Y)

        # Predict on the validation set
        predictions = model.predict(df_validation_X)

        self.plot_actual_vs_predicted_by_column(df_actual=df_validation_Y, predictions=predictions)

        # Evaluate the model using Mean Squared Error
        mse = mean_squared_error(df_validation_Y, predictions)
        print("Validation Mean Squared Error:", mse)


    def split_by_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(df.columns)
        df_true = df[df["train_test"] == False].copy()
        df_false = df[df["train_test"] == True].copy()
        return df_true, df_false

    def split_dataframe(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        selected_cols = [
            'sensor_0_avg', 'effective_wind_avg', 'effective_wind_min', 'effective_wind_max', 'effective_wind_std'
        ]

        # Create a DataFrame with the selected columns
        df_selected = df[selected_cols].copy()

        # Create a DataFrame with all other columns (drop the selected ones)
        df_other = df.drop(columns=selected_cols)

        return df_selected, df_other

    def add_effective_wind(self, df: pd.DataFrame) -> pd.DataFrame:
        relative_direction_col = "sensor_2_avg"
        relative_direction_radians = np.radians(df[relative_direction_col])

        df["effective_wind_avg"] = df["wind_speed_3_avg"] * np.cos(relative_direction_radians)
        df["effective_wind_min"] = df["wind_speed_3_min"] * np.cos(relative_direction_radians)
        df["effective_wind_max"] = df["wind_speed_3_max"] * np.cos(relative_direction_radians)
        df["effective_wind_std"] = df["wind_speed_3_std"] * np.cos(relative_direction_radians)

        return df

    def plot_actual_vs_predicted_by_column(self, df_actual: pd.DataFrame, predictions: np.ndarray) -> None:
        """
        For each column in df_actual, plots a line chart comparing the actual sensor values
        to the predicted values from the predictions array.

        Parameters:
          - df_actual: pd.DataFrame containing the actual sensor values.
          - predictions: np.ndarray of predicted sensor values with the same shape as df_actual.

        Each plot is created in its own figure.
        """
        # Ensure the predictions array matches the shape of the DataFrame
        if predictions.shape != df_actual.shape:
            raise ValueError("The shape of predictions must match the shape of the actual DataFrame.")

        # Iterate over each column
        for sensor in df_actual.columns:
            # Get the index of the column in the DataFrame
            col_index = df_actual.columns.get_loc(sensor)

            # Extract actual values and predicted values for the current sensor
            actual_values = df_actual[sensor]
            predicted_values = predictions[:, col_index]

            # Create a new figure for this sensor
            plt.figure(figsize=(10, 5))
            plt.plot(df_actual.index, actual_values, label="Actual", color="blue")
            plt.plot(df_actual.index, predicted_values, label="Predicted", color="red", linestyle="--", linewidth=0.2)
            plt.xlabel("Sample Index")
            plt.ylabel(sensor)
            sensor_ = Sensor.from_json(sensor_name=sensor, event_id=self.event_id)
            plt.title(f"Actual vs Predicted: {sensor} {sensor_.description}")
            plt.legend()
            plt.grid(True)
            plt.show()