from typing import Tuple
import pandas as pd
import os
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
from models.autoencoder.auto_encoder import SensorAutoencoder
from models.random_forest.sensor_random_forest import SensorRandomForest
from preprocessing.wfa.const import INPUT_COLUMNS, OUTPUT_COLUMNS


class WFAPreprocess(Preprocess):

    def __init__(self, event_id: int):
        super().__init__(event_id)


    @pipeline_step("Create Training-Data")
    def create_normal_data(self, df_path):
        df = self.load_dataframe(df_path=df_path)
        #df = self.add_effective_wind(df=df)
        df = self.interpolate_sensordata(df=df)
        df = self.filter_status_rows(df=df, mode="normal")
        df = self.filter_normal_sequences(df=df, sequence_length=36)

        save_folder = "res/normal_data/"
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, f"WFA/{self.event_id}.pkl")
        df.to_pickle(save_path)
        print(f"Normal training data saved to {save_path}")

        return df

    def create_anomaly_data(self, df_path, validation_frame=1000):
        df = self.load_dataframe(df_path=df_path)
        #df = self.add_effective_wind(df=df)
        df = self.interpolate_sensordata(df=df)
        df = self.filter_status_rows(df=df, mode="anomaly")
        return df


    def create_validation_test_data(self, df_path, validation_frame=144):
        df = self.load_dataframe(df_path=df_path)
        #df = self.add_effective_wind(df=df)
        df = self.interpolate_sensordata(df=df)
        id1 = self.event.event_start_id-validation_frame
        id2 = self.event.event_end_id
        id3 = self.event.event_end_id
        df_validation, df_test = self.split_by_ids(df=df, id1=id1, id2=id2, id3=id3)
        return df_validation, df_test

    def filter_normal_sequences(self, df: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
        """
        Filters the DataFrame to only include rows that belong to a sequence
        of at least `sequence_length` consecutive 'id' values.

        Assumes that the DataFrame is already sorted by the 'id' column.

        Args:
            df (pd.DataFrame): The input DataFrame with an 'id' column.
            sequence_length (int): The minimum required length of a consecutive sequence.

        Returns:
            pd.DataFrame: A DataFrame containing only the rows from sequences that meet
                          the consecutive 'id' length requirement.
        """
        segments = []
        current_segment = [df.iloc[0]]

        for i in range(1, len(df)):
            # If the current row continues the consecutive sequence, add it.
            if df.iloc[i]["id"] == df.iloc[i - 1]["id"] + 1:
                current_segment.append(df.iloc[i])
            else:
                # Check if the finished segment is long enough.
                if len(current_segment) >= sequence_length:
                    segments.append(pd.DataFrame(current_segment))
                current_segment = [df.iloc[i]]

        # Check the final segment.
        if len(current_segment) >= sequence_length:
            segments.append(pd.DataFrame(current_segment))

        # Concatenate valid segments; return empty DataFrame if none meet the criteria.
        if segments:
            filtered_df = pd.concat(segments, ignore_index=True)
        else:
            filtered_df = pd.DataFrame(columns=df.columns)

        return filtered_df
    @pipeline_step("Process", logger="WFA_PREPROCESS")
    def process(self, df_path, intervals_before=1000):
        df = self.load_dataframe(df_path=df_path)

        #df = self.keep_columns(df=df, columns_to_keep="avg", columns_to_exclude=None)
        df = self.add_effective_wind(df=df)
        df = self.interpolate_sensordata(df=df)
        #df = self.smooth_sensor_columns(df=df, window=36, method="mean")
        print(int(self.event.event_start_id))
        df_train, df_test = self.split_by_id(df=df, split_id=(int(self.event.event_start_id)-intervals_before))
        df_train = self.filter_status_rows(df=df_train)
        #
        #print(df_train.shape)
        #print(df_test.shape)

        #sa = SensorRandomForest(input_columns=INPUT_COLUMNS, output_columns=OUTPUT_COLUMNS, event_id=self.event_id)
       # sa.train(train_df=df_train)
        #predicted_df = sa.predict(df=df_test)
        #sa.plot_actual_vs_predicted_with_difference(actual_df=df_test, predicted_df=predicted_df)

        return df_train, df_test

    @pipeline_step("Process", logger="WFATESTTRAIN")
    def process_test(self, df_path):
        df = self.load_dataframe(df_path=df_path)
        #df = self.keep_columns(df=df, columns_to_keep="avg", columns_to_exclude=None)
        #df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        #df = self.filter_status_rows(df=df)
        return df

    def split_by_id(self, df: pd.DataFrame, split_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into two parts based on the given split_id value in the "id" column.

        Args:
            df (pd.DataFrame): The input DataFrame. Must contain an "id" column.
            split_id (int): The value in the "id" column at which to split the DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the first part (rows with id < split_id)
                                                and the second part (rows with id >= split_id).
        """
        df_first = df[df["id"] < split_id].copy()
        df_second = df[df["id"] >= split_id].copy()
        return df_first, df_second

    from typing import Tuple
    import pandas as pd

    def split_by_ids(self, df: pd.DataFrame, id1: int, id2: int, id3: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        first_segment = df[(df["id"] >= id1) & (df["id"] < id2)].copy()
        second_segment = df[(df["id"] >= id2) & (df["id"] < id3)].copy()
        return first_segment, second_segment

    def split_by_index(self, df: pd.DataFrame, split_index: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into two parts based on the given split_index.

        Args:
            df (pd.DataFrame): The input DataFrame.
            split_index (int): The index at which to split the DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the first part (rows before the index)
                                                and the second part (rows from the index onward).
        """
        df_first = df.iloc[:split_index].copy()
        df_second = df.iloc[split_index:].copy()
        return df_first, df_second

        """""
        print(df.columns)
        df = df.head(5000)
        for sensor in df.columns[5:]:
            dp.plot_timeseries_with_status(
                df=df,
                outside_temp_col="sensor_0_avg",
                wind_speed_col="wind_speed_3_avg",
                rotor_rpm_col="sensor_52_avg",
                status_col="status_type_id",
                sensor_col=sensor

            )
        """""

    @pipeline_step("Process", logger="WFA_PREPROCESS")
    def process_2(self, df_path: str):
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
        df = self.filter_status_rows(df=df)
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