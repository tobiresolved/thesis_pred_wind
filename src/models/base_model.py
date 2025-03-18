from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


class BaseSensorModel(ABC):
    def __init__(self, input_columns: List[str], output_columns: List[str] = None, window_output_len: int = 5):
        """
        Initializes the base model for sensor data.

        Args:
            input_columns (List[str]): List of sensor column names for current input features.
            output_columns (List[str], optional): List of sensor column names to predict.
                If None, defaults to input_columns.
            window_output_len (int): Number of previous output rows to use.
        """
        self.input_columns = input_columns
        self.output_columns = output_columns if output_columns is not None else input_columns
        self.window_output_len = window_output_len

        # Initialize scalers (shared by many models).
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        # Build the model (subclasses must implement this).
        self._build_model()

    @abstractmethod
    def _build_model(self) -> None:
        """
        Abstract method to build the underlying model architecture.
        Subclasses must override this method.
        """
        pass

    @abstractmethod
    def prepare(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Abstract method for preparing samples from data. This method should be used by both the training and
        prediction methods to ensure that the data is formatted consistently.

        Returns:
            Tuple containing:
              - X_current: np.ndarray of current input samples.
              - X_past: np.ndarray of past output samples.
              - Y: np.ndarray of output samples.
              - indices: List of indices associated with the samples.
        """
        pass

    @abstractmethod
    def train(self, train_dfs: Union[pd.DataFrame, List[pd.DataFrame]]) -> None:
        """
        Abstract method for training the model.
        Subclasses should implement their own training logic using the data prepared by `prepare`.
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for making predictions.
        This method should return a DataFrame containing the predictions.
        """
        pass

    def plot_individual_sensor_plots(self, actual_df: pd.DataFrame, predicted_df: Union[pd.DataFrame, Tuple],
                                     sensor_info: dict, event,
                                     outside_temp_col: str, wind_speed_col: str,
                                     rotor_rpm_col: str, status_col: str, columns=None) -> None:
        """
        For each sensor in self.output_columns, creates a figure with four subplots:
          1) Top subplot: outside temperature, wind speed, and rotor RPM.
          2) Second subplot: status over time shown with colored segments.
          3) Third subplot: actual vs. predicted values for the current sensor.
          4) Fourth subplot: difference between actual and predicted values.

        Additionally, vertical lines are drawn in each subplot at the positions
        corresponding to event.event_start_id and event.event_end_id, marking the slice of interest.

        In the difference subplot, a horizontal line is drawn at 0 and the subplot's height is doubled.

        Parameters:
          - actual_df: DataFrame containing the ground truth values plus additional columns.
          - predicted_df: DataFrame (or tuple with DataFrame as first element) of predicted sensor values.
          - sensor_info: Dictionary mapping sensor column names to descriptive names.
          - event: Object with attributes event_start_id and event_end_id.
          - outside_temp_col, wind_speed_col, rotor_rpm_col, status_col: Column names in actual_df for the respective values.
        """
        # Ensure predicted_df is a DataFrame
        if isinstance(predicted_df, tuple):
            predicted_df = predicted_df[0]

        # Extract actual sensor data using the prepare method.
        # The prepare method returns (X, some_val, Y_actual, indices)
        _, _, Y_actual, indices = self.prepare(actual_df)

        actual_sensor_df = pd.DataFrame(Y_actual, columns=self.output_columns)
        actual_sensor_df.index = indices

        # Make sure the predicted dataframe has the same index.
        predicted_sensor_df = predicted_df.copy()
        predicted_sensor_df.index = indices

        # Define grid ratios: the last (difference) subplot will be twice as high.
        grid_ratios = [1, 1, 1, 2]

        # Loop over each output sensor to create individual figures.
        self.output_columns = self.output_columns if columns is None else columns
        for sensor in self.output_columns:
            # Create a new figure with 4 subplots and custom height ratios.
            fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(12, 13),
                                     gridspec_kw={'height_ratios': grid_ratios})
            ax_top, ax_status, ax_sensor, ax_diff = axes

            # 1) TOP SUBPLOT: Plot outside temperature, wind speed, and rotor RPM.
            ax_top.plot(actual_df.index, actual_df[outside_temp_col],
                        label=outside_temp_col, linestyle="--", linewidth=0.8)
            ax_top.plot(actual_df.index, actual_df[wind_speed_col],
                        label=wind_speed_col, linestyle="--", linewidth=0.8)
            ax_top.plot(actual_df.index, actual_df[rotor_rpm_col],
                        label=rotor_rpm_col, linestyle="--", linewidth=0.8)
            ax_top.set_title("Outside Temperature, Wind Speed, and Rotor RPM")
            ax_top.set_ylabel("Values")
            ax_top.legend()
            # Add event slice markers (vertical lines).
            ax_top.axvline(x=event.event_start_id, color="gray", linestyle="--")
            ax_top.axvline(x=event.event_end_id, color="gray", linestyle="--")

            # 2) SECOND SUBPLOT: Plot status over time with colored segments.
            temp_df = actual_df.copy()
            temp_df["status_segment"] = (temp_df[status_col] != temp_df[status_col].shift()).cumsum()
            # Define a color mapping for status values (customize if needed).
            status_colors = {
                0: "green",
                1: "red",
                2: "blue",
                3: "orange",
                4: "purple",
                5: "brown"
            }
            for _, group in temp_df.groupby("status_segment"):
                seg_status = group[status_col].iloc[0]
                color = status_colors.get(seg_status, "gray")
                ax_status.plot(group.index, group[status_col], color=color, linewidth=0.5)
            ax_status.set_title("Status Over Time")
            ax_status.set_ylabel("Status")
            ax_status.set_ylim(-0.5, 5.5)
            # Add event slice markers.
            ax_status.axvline(x=event.event_start_id, color="gray", linestyle="--")
            ax_status.axvline(x=event.event_end_id, color="gray", linestyle="--")

            # 3) THIRD SUBPLOT: Plot actual vs. predicted for the current sensor.
            ax_sensor.plot(actual_sensor_df.index, actual_sensor_df[sensor],
                           label=f"{sensor_info.get(sensor, sensor)} Actual", color="blue")
            ax_sensor.plot(predicted_sensor_df.index, predicted_sensor_df[sensor],
                           label="Predicted", color="red", linestyle="--")
            ax_sensor.set_title(f"Sensor: {sensor} - Actual vs. Predicted")
            ax_sensor.set_ylabel(sensor)
            ax_sensor.legend()
            # Add event slice markers.
            ax_sensor.axvline(x=event.event_start_id, color="gray", linestyle="--")
            ax_sensor.axvline(x=event.event_end_id, color="gray", linestyle="--")

            from statsmodels.nonparametric.smoothers_lowess import lowess
            import numpy as np

            # Compute the difference between actual and predicted values.
            difference = actual_sensor_df[sensor] - predicted_sensor_df[sensor]

            # Create a time index for regression. Using a sequential index works if data is evenly spaced.
            x = np.arange(len(difference))

            # Use LOWESS to compute a robust trend line.
            # frac controls the fraction of data used when estimating each local regression;
            # adjust it to increase or decrease the smoothing.
            lowess_result = lowess(difference, x, frac=0.1, it=3)  # robust iterations set by it=3
            trend = lowess_result[:, 1]

            # Plot the robust trend line.
            ax_diff.plot(actual_sensor_df.index, trend,
                         label="Robust Trend (Actual - Predicted)", color="green", linewidth=0.7)

            # Plot a horizontal line at 0.
            ax_diff.axhline(y=0, color="black", linestyle="--")
            ax_diff.set_title(f"Sensor: {sensor} - Robust Trend (Actual - Predicted)")
            ax_diff.set_xlabel("Time")
            ax_diff.set_ylabel("Difference")
            ax_diff.legend()

            # Add event slice markers.
            ax_diff.axvline(x=event.event_start_id, color="gray", linestyle="--")
            ax_diff.axvline(x=event.event_end_id, color="gray", linestyle="--")

            # Add event slice markers.
            ax_diff.axvline(x=event.event_start_id, color="gray", linestyle="--")
            ax_diff.axvline(x=event.event_end_id, color="gray", linestyle="--")

            plt.tight_layout()
            plt.show()


    def save_prediction_statistics(self, actual_df: pd.DataFrame, predicted_df: Union[pd.DataFrame, Tuple],
                                   output_file: str, event=None) -> None:
        """
        Computes prediction statistics (MAPE and MAE) for each output sensor and appends a summary line
        with the event's details to separate CSV files for each sensor in the format:
        event.event_id;MAPE;MAE;event.is_anomaly;event.event_description

        Each sensor gets its own file, named as <output_file>_<sensor_name>.csv.
        """
        if isinstance(predicted_df, tuple):
            predicted_df = predicted_df[0]

        # Use the abstract prepare method to extract ground truth samples.
        _, _, Y_actual, indices = self.prepare(actual_df)
        stats = {}

        for i, col in enumerate(self.output_columns):
            actual_values = Y_actual[:, i]
            predicted_values = predicted_df[col].values
            if len(actual_values) != len(predicted_values):
                raise ValueError(
                    f"Mismatch for {col}: actual {len(actual_values)} vs predicted {len(predicted_values)}"
                )
            mae = mean_absolute_error(actual_values, predicted_values)
            mask = actual_values != 0
            mape = (np.mean(np.abs((actual_values[mask] - predicted_values[mask]) / actual_values[mask])) * 100
                    if np.sum(mask) > 0 else np.nan)
            stats[col] = {"MAPE": mape, "MAE": mae}

        if event is not None:
            for col in self.output_columns:
                sensor_stats = stats[col]
                summary_data = {
                    "event_id": [event.event_id],
                    "MAPE": [sensor_stats["MAPE"]],
                    "MAE": [sensor_stats["MAE"]],
                    "is_anomaly": [event.is_anomaly],
                    "event_description": [event.event_description]
                }
                summary_df = pd.DataFrame(summary_data)
                sensor_file = f"{output_file}_{col}.csv"
                file_exists = os.path.exists(sensor_file)
                summary_df.to_csv(sensor_file, index=False, sep=";", mode="a", header=not file_exists)
                print(f"Event summary for sensor '{col}' appended to {sensor_file}")
        else:
            print("No event provided; summary not appended.")
