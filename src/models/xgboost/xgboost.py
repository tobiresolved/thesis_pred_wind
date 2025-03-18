import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from preprocessing.metadata.dataclasses import Event, Sensor, Dataset

class SensorXGBoost_V2:
    def __init__(self, input_columns, output_columns=None, n_estimators=100, max_depth=3, learning_rate=0.1, event_id=None):
        """
        Initializes the XGBoost model for sensor data.

        Args:
            input_columns (list): List of column names to use as input.
            output_columns (list, optional): List of column names to use as output.
                If None, defaults to input_columns.
            n_estimators (int): Number of trees.
            max_depth (int): Maximum tree depth.
            learning_rate (float): Learning rate.
            event_id: Optional event identifier.
        """
        self.input_columns = input_columns
        self.output_columns = output_columns if output_columns is not None else input_columns
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.event_id = event_id
        self._build_model()

    def _build_model(self):
        """Builds the XGBoost model wrapped for multi-output regression."""
        # XGBoost regressor instance
        xgb_regressor = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            objective='reg:squarederror'
        )
        # Wrap in MultiOutputRegressor to handle multiple outputs if needed.
        self.model = MultiOutputRegressor(xgb_regressor)

    def train(self, train_dfs):
        """
        Trains the XGBoost model using one or more training DataFrames.

        Args:
            train_dfs (pd.DataFrame or list of pd.DataFrame): Training data. If a list is provided,
                the DataFrames will be concatenated before training.
        """
        # Concatenate if multiple DataFrames are provided.
        if isinstance(train_dfs, list):
            train_df = pd.concat(train_dfs, ignore_index=True)
        else:
            train_df = train_dfs

        X_train = train_df[self.input_columns].values
        Y_train = train_df[self.output_columns].values

        self.model.fit(X_train, Y_train)

    def predict(self, df):
        """
        Uses the XGBoost model to predict the sensor data.

        Args:
            df (pd.DataFrame): DataFrame containing the data to predict.
                               Must include the columns specified in `input_columns`.

        Returns:
            pd.DataFrame: Predicted data with columns specified in `output_columns` and the same index.
        """
        X = df[self.input_columns].values
        predictions = self.model.predict(X)
        return pd.DataFrame(predictions, columns=self.output_columns, index=df.index)

    def plot_actual_vs_predicted_with_difference(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame, sensor_info: dict) -> None:
        """
        For each sensor column (using output_columns if available, otherwise input_columns), plots a three-panel figure:
          - Top panel: Overlaid time series of actual and predicted sensor data.
          - Middle panel: The status over time from the 'status_type_id' column in actual_df.
          - Bottom panel: The difference between actual and predicted values.
        The 'status_type_id' column is plotted with a reset index to align with the other plots.

        Args:
            actual_df (pd.DataFrame): DataFrame containing the actual sensor data. Must include the columns specified
                                      in self.output_columns (or self.input_columns if output_columns is None)
                                      and a 'status_type_id' column.
            predicted_df (pd.DataFrame): DataFrame containing the predicted sensor data.
            sensor_info (dict): A dictionary where keys are sensor column names and values contain sensor metadata,
                                e.g., description.
        """
        plot_columns = self.output_columns if self.output_columns is not None else self.input_columns

        actual_df = actual_df.reset_index(drop=True)
        predicted_df = predicted_df.reset_index(drop=True)

        for col in plot_columns:
            fig, (ax_top, ax_status, ax_bottom) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

            # Top panel: actual vs predicted sensor data
            ax_top.plot(actual_df.index, actual_df[col],
                        label=f"{sensor_info[col].description} {col}", color="blue")
            ax_top.plot(predicted_df.index, predicted_df[col],
                        label="Predicted", color="red", linestyle="--")
            ax_top.set_title(f"Sensor: {col} - Actual vs Predicted")
            ax_top.legend()

            # Middle panel: status over time
            ax_status.set_title("Status Over Time")
            ax_status.set_ylabel("Status ID")
            if 'status_type_id' in actual_df.columns:
                temp_df = actual_df.copy().reset_index(drop=True)
                temp_df["status_segment"] = (temp_df["status_type_id"] != temp_df["status_type_id"].shift()).cumsum()
                status_colors = {0: "green", 1: "red", 2: "blue", 3: "orange", 4: "purple", 5: "gray"}
                for _, group in temp_df.groupby("status_segment"):
                    seg_status = group["status_type_id"].iloc[0]
                    color = status_colors.get(seg_status, "black")
                    ax_status.plot(group.index, group["status_type_id"], color=color, linewidth=0.5)
                ax_status.set_ylim(-0.5, 5.5)
            else:
                ax_status.text(0.5, 0.5, "'status_type_id' not found", transform=ax_status.transAxes,
                               horizontalalignment="center", verticalalignment="center")

            # Bottom panel: difference plot
            difference = actual_df[col] - predicted_df[col]
            ax_bottom.plot(actual_df.index, difference, label="Difference (Actual - Predicted)", color="green")
            ax_bottom.set_title("Difference")
            ax_bottom.legend()
            ax_bottom.set_xlabel("Time")

            plt.tight_layout()
            plt.show()

    def save_prediction_statistics(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame, output_file: str) -> None:
        """
        Computes and saves statistics for each sensor column comparing actual and predicted sensor data.
        The statistics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score for each sensor.

        Args:
            actual_df (pd.DataFrame): DataFrame containing the actual sensor data. Must include the columns specified
                                      in self.output_columns (or self.input_columns if output_columns is None).
            predicted_df (pd.DataFrame): DataFrame containing the predicted sensor data.
            output_file (str): Path to the file where the statistics should be saved (e.g., "prediction_statistics.csv").
        """
        stats = {}
        plot_columns = self.output_columns if self.output_columns is not None else self.input_columns

        for col in plot_columns:
            actual = actual_df[col].values
            predicted = predicted_df[col].values
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            stats[col] = {"MAE": mae, "MSE": mse, "R2": r2}

        stats_df = pd.DataFrame(stats).T  # Transpose so each sensor is a row
        stats_df.to_csv(output_file, index=True)
        print(f"Statistics saved to {output_file}")
