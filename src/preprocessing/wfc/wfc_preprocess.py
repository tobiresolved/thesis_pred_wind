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


class WFCPreprocess(Preprocess):

    def __init__(self, event_id: int):
        super().__init__(event_id)

    @pipeline_step("Process", logger="WFA_PREPROCESS")
    def process(self, df_path: str):
        df = self.load_dataframe(df_path=df_path)
        #df = self.filter_status_rows(df=df)
        #df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        dp = DataPlotter(event=self.event)
        #dp.plot_timeseries(sensor_series=df["sensor_178_avg"], sensor=self.sensor_metadata["sensor_178_avg"])
        #dp.plot_timeseries(sensor_series=df["wind_speed_237_avg"], sensor=self.sensor_metadata["wind_speed_237_avg"])
        #dp.plot_timeseries(sensor_series=df["sensor_145_avg"], sensor=self.sensor_metadata["sensor_145_avg"])

        #dp.plot_sensor_correlation_status_type(df=df, sensor_x="wind_speed_237_avg", sensor_y="sensor_178_avg")
        #dp.plot_sensor_correlation_by_status_loop(df=df, sensor_x="wind_speed_235_avg", sensor_y="sensor_145_avg")
        #dp.plot_sensor_correlation_status_type(df=df, sensor_x="wind_speed_237_avg", sensor_y="sensor_144_avg")

        df_train, df_validation = self.split_by_train_test(df=df)
        useless_columns = [
            "status_type_id",  # constant zero
            "train_test",  # constant zero
            "id",  # angle
            "asset_id",  # angle
            "time_stamp"
        ]
        df_train = df_train.drop(columns=useless_columns)
        df_validation = df_validation.drop(columns=useless_columns)

        df_train_X, df_train_Y, = self.split_dataframe(df_train)
        print(df_train_X)
        df_validation_X, df_validation_Y, = self.split_dataframe(df_validation)
        print(df_validation_X)
        # Instantiate the model
        import tensorflow as tf
        from tensorflow.keras import layers, models

        # Example: A simple Sequential model with hidden layers
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(3,)),  # Input layer, 3 features
            layers.Dense(32, activation='relu'),  # Hidden layer
            layers.Dense(16, activation='relu'),  # Another hidden layer
            layers.Dense(1)  # Output layer (predicts 1 value)
        ])

        # Compile the model for a regression task
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        # Train the model
        model.fit(
            df_train_X,  # Features (3 columns)
            df_train_Y,  # Target (1 column)
            epochs=10,  # Number of epochs, feel free to adjust
            batch_size=32,  # Batch size
            validation_split=0.1  # Use 10% of training data for validation
        )

        # Predict on the validation set (or any other dataset)
        predictions = model.predict(df_validation_X)
        print(predictions)
        predictions = pd.Series(predictions.flatten(), name='sensor_146_avg')

        self.plot_actual_vs_predicted_by_column(df_actual=df_validation_Y, predictions=predictions)

        print(predictions)

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
            "wind_speed_235_avg", "sensor_7_avg", "sensor_144_avg"
        ]

        # Create a DataFrame with the selected columns
        df_selected = df[selected_cols].copy()

        # Create a DataFrame with all other columns (drop the selected ones)
        df_other = df.drop(columns=selected_cols)
        df_other = df_other["sensor_146_avg"]

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
        plt.figure(figsize=(10, 5))
        plt.plot(df_actual.index, df_actual, label="Actual", color="blue")
        plt.plot(df_actual.index, predictions, label="Predicted", color="red", linestyle="--")
        plt.xlabel("Sample Index")
        plt.ylabel("Value")  # or a more specific label if available
        plt.title("Actual vs Predicted")
        plt.legend()
        plt.grid(True)
        plt.show()

