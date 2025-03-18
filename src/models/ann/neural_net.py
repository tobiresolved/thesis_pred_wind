import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense


class SensorNeuralNet:
    def __init__(self, input_columns, output_columns=None, epochs=10, batch_size=32, event_id=None):
        """
        Initializes the neural network model for sensor data with scaling.

        Args:
            input_columns (list): List of column names to use as input.
            output_columns (list, optional): List of column names to use as output.
                If None, defaults to input_columns.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            event_id: Optional event identifier.
        """
        self.input_columns = input_columns
        self.output_columns = output_columns if output_columns is not None else input_columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.event_id = event_id

        # Initialize scalers for inputs and outputs (global scaling)
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        self._build_model()

    def _build_model(self):
        """Builds a simple feedforward neural network model."""
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(len(self.input_columns),)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(len(self.output_columns)))
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, train_dfs):
        """
        Trains the neural network model using one or more training DataFrames.

        Args:
            train_dfs (pd.DataFrame or list of pd.DataFrame): Training data. If a list is provided,
                the DataFrames will be concatenated before training.
        """
        # Concatenate if multiple DataFrames are provided
        if isinstance(train_dfs, list):
            train_df = pd.concat(train_dfs, ignore_index=True)
        else:
            train_df = train_dfs

        # Extract features and targets
        X_train = train_df[self.input_columns].values
        Y_train = train_df[self.output_columns].values

        # Fit the scalers on the global training data and transform the data
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        Y_train_scaled = self.output_scaler.fit_transform(Y_train)

        # Train the model
        self.model.fit(X_train_scaled, Y_train_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, df):
        """
        Uses the neural network model to predict the sensor data.

        Args:
            df (pd.DataFrame): DataFrame containing the data to predict.
                               Must include the columns specified in `input_columns`.

        Returns:
            pd.DataFrame: Predicted data with columns specified in `output_columns` and the same index.
                        The predicted values are inverse-transformed to the original scale.
        """
        # Scale the input data using the previously fitted scaler
        X = df[self.input_columns].values
        X_scaled = self.input_scaler.transform(X)

        # Get scaled predictions and inverse transform them to original scale
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.output_scaler.inverse_transform(predictions_scaled)
        return pd.DataFrame(predictions, columns=self.output_columns, index=df.index)

    def plot_actual_vs_predicted_with_difference(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame,
                                                 sensor_info: dict) -> None:
        """
        For each sensor column (using output_columns if available, otherwise input_columns), plots a three-panel figure:
          - Top panel: Overlaid time series of actual and predicted sensor data.
          - Middle panel: The status over time from the 'status_type_id' column in actual_df.
          - Bottom panel: The difference between actual and predicted values.
        Assumes that the predicted data is already in the original scale (i.e., inverse-transformed).
        The actual data is used as provided.

        Args:
            actual_df (pd.DataFrame): DataFrame containing the actual sensor data. Must include the columns specified
                                      in self.output_columns (or self.input_columns if output_columns is None)
                                      and a 'status_type_id' column.
            predicted_df (pd.DataFrame): DataFrame containing the predicted sensor data in original scale.
            sensor_info (dict): A dictionary where keys are sensor column names and values contain sensor metadata,
                                e.g., description.
        """
        plot_columns = self.output_columns if self.output_columns is not None else self.input_columns

        # Reset indices for alignment
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
        Assumes that the predicted data is already in the original scale (i.e., inverse-transformed).
        The statistics include:
            - MAPE: Mean Absolute Percentage Error (expressed as a percentage),
            - MAE: Mean Absolute Error.

        Args:
            actual_df (pd.DataFrame): DataFrame containing the actual sensor data. Must include the columns specified
                                      in self.output_columns (or self.input_columns if output_columns is None).
            predicted_df (pd.DataFrame): DataFrame containing the predicted sensor data in original scale.
            output_file (str): Path to the file where the statistics should be saved (e.g., "prediction_statistics.csv").
        """
        stats = {}
        plot_columns = self.output_columns if self.output_columns is not None else self.input_columns

        for col in plot_columns:
            actual = actual_df[col].values
            predicted = predicted_df[col].values
            mae = mean_absolute_error(actual, predicted)

            # Compute MAPE while avoiding division by zero
            actual_array = np.array(actual)
            predicted_array = np.array(predicted)
            mask = actual_array != 0
            if np.sum(mask) == 0:
                mape = np.nan
            else:
                mape = np.mean(np.abs((actual_array[mask] - predicted_array[mask]) / actual_array[mask])) * 100

            stats[col] = {
                "MAPE": mape,
                "MAE": mae
            }

        stats_df = pd.DataFrame(stats).T  # Transpose so each sensor is a row
        stats_df.to_csv(output_file, index=True)
        print(f"Statistics saved to {output_file}")

    def save_model(self, path: str) -> None:
        """
        Saves the Keras model and scalers to the specified directory.

        Args:
            path (str): Directory path where the model and scalers will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        # Save Keras model
        model_path = os.path.join(path, "model.h5")
        self.model.save(model_path)
        # Save scalers using pickle
        scaler_path_input = os.path.join(path, "input_scaler.pkl")
        scaler_path_output = os.path.join(path, "output_scaler.pkl")
        with open(scaler_path_input, "wb") as f:
            pickle.dump(self.input_scaler, f)
        with open(scaler_path_output, "wb") as f:
            pickle.dump(self.output_scaler, f)
        print(f"Model and scalers saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Loads the Keras model and scalers from the specified directory.

        Args:
            path (str): Directory path from which the model and scalers will be loaded.
        """
        model_path = os.path.join(path, "model.h5")
        scaler_path_input = os.path.join(path, "input_scaler.pkl")
        scaler_path_output = os.path.join(path, "output_scaler.pkl")
        self.model = load_model(model_path)
        with open(scaler_path_input, "rb") as f:
            self.input_scaler = pickle.load(f)
        with open(scaler_path_output, "rb") as f:
            self.output_scaler = pickle.load(f)
        print(f"Model and scalers loaded from {path}")


import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from typing import Tuple


import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

class SensorNeuralNet_V2:
    def __init__(self, input_columns, output_columns=None, epochs=10, batch_size=32, event_id=None):
        """
        Initializes the neural network model for sensor data using continuous sequences.
        Each training example is built from a sliding window of 6 consecutive rows (based on a continuous "id" column).
        The window is flattened into one vector (with the most recent time step as the target output).

        Args:
            input_columns (list): List of column names to use as input.
            output_columns (list, optional): List of column names to use as output.
                If None, defaults to input_columns.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            event_id: Optional event identifier.
        """
        self.input_columns = input_columns
        self.output_columns = output_columns if output_columns is not None else input_columns
        self.epochs = epochs
        self.batch_size = batch_size
        self.event_id = event_id

        # Initialize scalers for inputs and outputs
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        self._build_model()

    def _build_model(self):
        """
        Builds a simple feedforward neural network.
        Note that the input layer shape is adjusted to (6 * number_of_features,)
        since each input is a flattened window of 6 rows.
        """
        num_features = len(self.input_columns)
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(6 * num_features,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(len(self.output_columns)))
        self.model.compile(optimizer='adam', loss='mse')

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepares training examples from the dataframe by finding continuous sequences.
        A continuous sequence is defined by rows where the "id" column increases by 1 at each step.
        Only sequences of length >= 6 are used.
        For each continuous sequence, sliding windows of 6 rows are extracted.

        Each input X is formed by flattening a window of 6 rows (order: oldest to most recent)
        and the corresponding output Y is the sensor measurement(s) at the most recent (6th) row.

        Args:
            df (pd.DataFrame): Input DataFrame. Must contain an "id" column.

        Returns:
            X (np.ndarray): Array of shape (n_examples, 6 * n_features).
            Y (np.ndarray): Array of shape (n_examples, n_output_features).
            indices (list): List of indices (e.g., the id of the last row in each window) for later reference.
        """
        # Ensure the DataFrame is sorted by the "id" column
        df_sorted = df.sort_values("id").reset_index(drop=True)
        sequences = []
        current_seq = [df_sorted.iloc[0]]
        for i in range(1, len(df_sorted)):
            # Check if the current row is continuous with the previous row
            if df_sorted.loc[i, "id"] == df_sorted.loc[i - 1, "id"] + 1:
                current_seq.append(df_sorted.iloc[i])
            else:
                # End current sequence and start a new one if sequence length >= 6
                if len(current_seq) >= 6:
                    sequences.append(pd.DataFrame(current_seq))
                current_seq = [df_sorted.iloc[i]]
        if len(current_seq) >= 6:
            sequences.append(pd.DataFrame(current_seq))

        X_list, Y_list, indices = [], [], []
        for seq in sequences:
            arr_in = seq[self.input_columns].values  # shape (L, n_features)
            arr_out = seq[self.output_columns].values  # shape (L, n_output_features)
            # Create sliding windows of length 6
            for i in range(5, len(arr_in)):
                window = arr_in[i - 5:i + 1]  # rows i-5 to i (total 6 rows)
                X_list.append(window.flatten())  # flatten into one vector
                Y_list.append(arr_out[i])  # target is the sensor measurement at the last row
                indices.append(seq.iloc[i]["id"])
        X = np.array(X_list)
        Y = np.array(Y_list)
        return X, Y, indices

    def train(self, train_dfs):
        """
        Trains the neural network model using continuous sequences extracted from the training DataFrame(s).

        Args:
            train_dfs (pd.DataFrame or list of pd.DataFrame): Training data.
                Must contain an "id" column indicating time steps.
        """
        if isinstance(train_dfs, list):
            train_df = pd.concat(train_dfs, ignore_index=True)
        else:
            train_df = train_dfs

        # Prepare sequences from the training data
        X_train, Y_train, _ = self.prepare_sequences(train_df)

        # Fit the scalers on the training examples and transform the data
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        Y_train_scaled = self.output_scaler.fit_transform(Y_train)

        # Train the model
        self.model.fit(X_train_scaled, Y_train_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Uses the neural network model to predict sensor data on continuous sequences extracted from the DataFrame.
        Predictions are made for each valid sliding window.

        Args:
            df (pd.DataFrame): DataFrame containing the data to predict.
                Must include an "id" column and the sensor columns specified in input_columns.

        Returns:
            predicted_df (pd.DataFrame): DataFrame with predicted sensor values (in original scale) for each window.
            indices (list): List of the "id" values corresponding to the predicted outputs.
        """
        # Prepare sequences from the prediction data
        X_pred, _, indices = self.prepare_sequences(df)
        X_pred_scaled = self.input_scaler.transform(X_pred)
        predictions_scaled = self.model.predict(X_pred_scaled)
        predictions = self.output_scaler.inverse_transform(predictions_scaled)
        predicted_df = pd.DataFrame(predictions, columns=self.output_columns)
        return predicted_df, indices

    def plot_actual_vs_predicted_with_difference(self, actual_df: pd.DataFrame, predicted_df: pd.DataFrame,
                                                 sensor_info: dict) -> None:
        """
        Plots for each sensor column a three-panel figure:
          - Top panel: Overlaid time series of actual and predicted sensor data.
          - Middle panel: The status over time from the 'status_type_id' column in actual_df.
          - Bottom panel: The difference between actual and predicted values.
        Assumes that the predicted data is already in the original scale.
        (The actual_df is expected to include a 'status_type_id' column if status plotting is desired.)

        Args:
            actual_df (pd.DataFrame): DataFrame containing the actual sensor data. Must include the columns specified
                                      in output_columns (or input_columns if output_columns is None) and a 'status_type_id' column.
            predicted_df (pd.DataFrame): DataFrame containing the predicted sensor data in original scale.
            sensor_info (dict): A dictionary where keys are sensor column names and values contain sensor metadata.
        """
        plot_columns = self.output_columns if self.output_columns is not None else self.input_columns

        actual_df = actual_df.reset_index(drop=True)
        predicted_df = predicted_df.reset_index(drop=True)

        for col in plot_columns:
            fig, (ax_top, ax_status, ax_bottom) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            ax_top.plot(actual_df.index, actual_df[col], label=f"{sensor_info[col].description} {col}", color="blue")
            ax_top.plot(predicted_df.index, predicted_df[col], label="Predicted", color="red", linestyle="--")
            ax_top.set_title(f"Sensor: {col} - Actual vs Predicted")
            ax_top.legend()

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

            difference = actual_df[col] - predicted_df[col]
            ax_bottom.plot(actual_df.index, difference, label="Difference (Actual - Predicted)", color="green")
            ax_bottom.set_title("Difference")
            ax_bottom.legend()
            ax_bottom.set_xlabel("Time")

            plt.tight_layout()
            plt.show()

    def save_prediction_statistics(self, actual_df: pd.DataFrame, predicted_df, output_file: str) -> None:
        """
        Computes and saves statistics for each sensor column comparing actual and predicted sensor data.
        The statistics include:
            - MAPE: Mean Absolute Percentage Error (expressed as a percentage),
            - MAE: Mean Absolute Error.

        The method applies the same sliding-window (sequence) extraction to the actual_df so that the
        ground truth targets match the predictions from the model.

        Args:
            actual_df (pd.DataFrame): DataFrame containing the raw actual sensor data.
            predicted_df (pd.DataFrame or tuple): DataFrame containing the predicted sensor data in original scale,
                                                  or a tuple (predicted_df, indices).
            output_file (str): File path to save the statistics CSV.
        """
        # If predicted_df is a tuple, extract the DataFrame portion.
        if isinstance(predicted_df, tuple):
            predicted_df = predicted_df[0]

        # Extract the sliding-window targets (Y) from the actual data
        # This ensures that we are comparing predictions for the last row of each window.
        _, Y_actual, _ = self.prepare_sequences(actual_df)

        stats = {}
        # Loop over each output sensor column. Use the order defined in self.output_columns.
        for i, col in enumerate(self.output_columns):
            actual_values = Y_actual[:, i]
            predicted_values = predicted_df[col].values

            # Check that the lengths match
            if len(actual_values) != len(predicted_values):
                raise ValueError(f"Mismatch in number of samples for column {col}: "
                                 f"actual {len(actual_values)} vs predicted {len(predicted_values)}")

            mae = mean_absolute_error(actual_values, predicted_values)
            # Compute MAPE while avoiding division by zero
            actual_array = np.array(actual_values)
            predicted_array = np.array(predicted_values)
            mask = actual_array != 0
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((actual_array[mask] - predicted_array[mask]) / actual_array[mask])) * 100
            else:
                mape = np.nan
            stats[col] = {"MAPE": mape, "MAE": mae}

        stats_df = pd.DataFrame(stats).T  # each sensor is a row
        stats_df.to_csv(output_file, index=True)
        print(f"Statistics saved to {output_file}")

    def save_model(self, path: str) -> None:
        """
        Saves the Keras model and scalers to the specified directory.

        Args:
            path (str): Directory path where the model and scalers will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, "model.h5")
        self.model.save(model_path)
        scaler_path_input = os.path.join(path, "input_scaler.pkl")
        scaler_path_output = os.path.join(path, "output_scaler.pkl")
        with open(scaler_path_input, "wb") as f:
            pickle.dump(self.input_scaler, f)
        with open(scaler_path_output, "wb") as f:
            pickle.dump(self.output_scaler, f)
        print(f"Model and scalers saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Loads the Keras model and scalers from the specified directory.

        Args:
            path (str): Directory path from which the model and scalers will be loaded.
        """
        model_path = os.path.join(path, "model.h5")
        scaler_path_input = os.path.join(path, "input_scaler.pkl")
        scaler_path_output = os.path.join(path, "output_scaler.pkl")
        self.model = load_model(model_path)
        with open(scaler_path_input, "rb") as f:
            self.input_scaler = pickle.load(f)
        with open(scaler_path_output, "rb") as f:
            self.output_scaler = pickle.load(f)
        print(f"Model and scalers loaded from {path}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class SensorNeuralNet_V2_AR:
    def __init__(self, input_columns: List[str], output_columns: List[str] = None,
                 window_output_len: int = 5, epochs: int = 10, batch_size: int = 16, event_id=None):
        """
        Initializes the autoregressive neural network model.
        Each training sample is built as:
            X = [ current input sensor values ] + [ flattened past output sensor values (from the last window_output_len steps) ]
            Y = output sensor values at the current time step.

        Args:
            input_columns (List[str]): List of column names for current input features.
            output_columns (List[str], optional): List of column names for output sensor values.
                If None, defaults to input_columns.
            window_output_len (int): Number of previous output rows to use.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            event_id: Optional event identifier.
        """
        self.input_columns = input_columns
        self.output_columns = output_columns if output_columns is not None else input_columns
        self.window_output_len = window_output_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.event_id = event_id

        # Initialize scalers for the input and output parts
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        self._build_model()

    def _build_model(self):
        """
        Builds a simple feedforward neural network.
        The input shape is:
            len(input_columns) + window_output_len * len(output_columns)
        """
        n_input = len(self.input_columns)
        n_output = len(self.output_columns)
        input_shape = n_input + self.window_output_len * n_output
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(n_output))
        self.model.compile(optimizer='adam', loss='mse')

    def _prepare_samples_single(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Prepares autoregressive samples from a single DataFrame.
        The DataFrame must contain an "id" column indicating time steps.
        The method splits the series into continuous segments (where "id" increases by 1)
        and, for each segment of length L >= window_output_len+1, for each time t (t >= window_output_len):

            X = [ input_columns at time t ] + [ flattened output_columns from time t-window_output_len to t-1 ]
            Y = output_columns at time t

        Returns:
            X (np.ndarray): Array of shape (n_samples, len(input_columns) + window_output_len * len(output_columns)).
            Y (np.ndarray): Array of shape (n_samples, len(output_columns)).
            indices (List): List of "id" values corresponding to the target rows.
        """
        df_sorted = df.sort_values("id").reset_index(drop=True)
        segments = []
        current_seg = [df_sorted.iloc[0]]
        for i in range(1, len(df_sorted)):
            if df_sorted.loc[i, "id"] == df_sorted.loc[i - 1, "id"] + 1:
                current_seg.append(df_sorted.iloc[i])
            else:
                if len(current_seg) >= self.window_output_len + 1:
                    segments.append(pd.DataFrame(current_seg))
                current_seg = [df_sorted.iloc[i]]
        if len(current_seg) >= self.window_output_len + 1:
            segments.append(pd.DataFrame(current_seg))

        X_list, Y_list, indices = [], [], []
        for seg in segments:
            seg_input = seg[self.input_columns].values  # shape (L, n_input)
            seg_output = seg[self.output_columns].values  # shape (L, n_output)
            L = len(seg)
            for t in range(self.window_output_len, L):
                current_inputs = seg_input[t]  # current input, shape (n_input,)
                past_outputs = seg_output[t - self.window_output_len:t]  # shape (window_output_len, n_output)
                X_sample = np.concatenate([current_inputs, past_outputs.flatten()])
                Y_sample = seg_output[t]
                X_list.append(X_sample)
                Y_list.append(Y_sample)
                indices.append(seg.iloc[t]["id"])
        X = np.array(X_list)
        Y = np.array(Y_list)
        return X, Y, indices

    def prepare_autoregressive_samples(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]]) -> Tuple[
        np.ndarray, np.ndarray, List]:
        """
        Prepares autoregressive samples from a single DataFrame or a list of DataFrames.
        When provided a list, each DataFrame is treated as an independent time series.

        Returns:
            X (np.ndarray), Y (np.ndarray), indices (List)
        """
        if isinstance(dfs, list):
            X_all, Y_all, indices_all = [], [], []
            for df in dfs:
                X, Y, indices = self._prepare_samples_single(df)
                if len(X) > 0:
                    X_all.append(X)
                    Y_all.append(Y)
                    indices_all.extend(indices)
            if len(X_all) == 0:
                return np.empty((0, 0)), np.empty((0, 0)), []
            X_total = np.vstack(X_all)
            Y_total = np.vstack(Y_all)
            return X_total, Y_total, indices_all
        else:
            return self._prepare_samples_single(dfs)

    def train(self, train_dfs: Union[pd.DataFrame, List[pd.DataFrame]]):
        """
        Trains the autoregressive neural network using samples extracted from the training DataFrame(s).
        The training data must contain an "id" column. When a list is provided, each DataFrame is processed independently.
        """
        X_train, Y_train, _ = self.prepare_autoregressive_samples(train_dfs)
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        Y_train_scaled = self.output_scaler.fit_transform(Y_train)
        self.model.fit(X_train_scaled, Y_train_scaled, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """
        Predicts output sensor values for each sample extracted from the DataFrame.
        The same autoregressive sample creation is applied as in training.

        Args:
            df (pd.DataFrame): DataFrame containing the data to predict.
                Must include an "id" column and the columns in input_columns.

        Returns:
            predicted_df (pd.DataFrame): DataFrame with predicted sensor values (in original scale) for each sample.
            indices (List): List of "id" values corresponding to each prediction.
        """
        X_pred, _, indices = self.prepare_autoregressive_samples(df)
        X_pred_scaled = self.input_scaler.transform(X_pred)
        predictions_scaled = self.model.predict(X_pred_scaled)
        predictions = self.output_scaler.inverse_transform(predictions_scaled)
        predicted_df = pd.DataFrame(predictions, columns=self.output_columns)
        return predicted_df, indices

    def plot_actual_vs_predicted_with_difference(self, actual_df: pd.DataFrame,
                                                 predicted_df: Union[pd.DataFrame, Tuple], sensor_info: dict) -> None:
        """
        Plots, for each sensor column, a three-panel figure:
          - Top panel: Overlaid time series of actual and predicted sensor data.
          - Middle panel: The status over time from the 'status_type_id' column in actual_df.
          - Bottom panel: The difference between actual and predicted values.
        The method applies the same autoregressive sample extraction to actual_df so that the ground truth targets match predictions.
        If predicted_df is a tuple, the DataFrame portion is extracted.

        Args:
            actual_df (pd.DataFrame): Raw actual sensor data. Must contain an "id" column and the sensor columns.
            predicted_df (pd.DataFrame or Tuple): Predicted sensor data (in original scale) for each sample, or tuple (df, indices).
            sensor_info (dict): Dictionary mapping sensor column names to metadata (e.g., description).
        """
        if isinstance(predicted_df, tuple):
            predicted_df = predicted_df[0]
        _, Y_actual, seq_indices = self.prepare_autoregressive_samples(actual_df)
        actual_plot_df = pd.DataFrame(Y_actual, columns=self.output_columns)
        actual_plot_df.index = seq_indices
        predicted_plot_df = predicted_df.copy()
        predicted_plot_df.index = seq_indices

        plot_columns = self.output_columns if self.output_columns is not None else self.input_columns
        for col in plot_columns:
            fig, (ax_top, ax_status, ax_bottom) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            ax_top.plot(actual_plot_df.index, actual_plot_df[col],
                        label=f"{sensor_info.get(col, col)}", color="blue")
            ax_top.plot(predicted_plot_df.index, predicted_plot_df[col],
                        label="Predicted", color="red", linestyle="--")
            ax_top.set_title(f"Sensor: {col} - Actual vs Predicted")
            ax_top.legend()

            ax_status.set_title("Status Over Time")
            ax_status.set_ylabel("Status ID")
            if 'status_type_id' in actual_df.columns:
                status_values = []
                for seq_id in seq_indices:
                    row = actual_df[actual_df["id"] == seq_id]
                    if not row.empty:
                        status_values.append(row.iloc[0]["status_type_id"])
                    else:
                        status_values.append(np.nan)
                ax_status.plot(seq_indices, status_values, color="black", linewidth=1)
                ax_status.set_ylim(-0.5, 5.5)
            else:
                ax_status.text(0.5, 0.5, "'status_type_id' not found", transform=ax_status.transAxes,
                               horizontalalignment="center", verticalalignment="center")

            difference = actual_plot_df[col] - predicted_plot_df[col]
            ax_bottom.plot(actual_plot_df.index, difference,
                           label="Difference (Actual - Predicted)", color="green")
            ax_bottom.set_title("Difference")
            ax_bottom.legend()
            ax_bottom.set_xlabel("id")

            plt.tight_layout()
            plt.show()

    def save_prediction_statistics(self, actual_df: pd.DataFrame, predicted_df: Union[pd.DataFrame, Tuple],
                                   output_file: str) -> None:
        """
        Computes and saves statistics for each sensor column comparing actual and predicted sensor data.
        The statistics include:
            - MAPE: Mean Absolute Percentage Error (expressed as a percentage),
            - MAE: Mean Absolute Error.
        The method applies the same autoregressive sample extraction to actual_df so that the ground truth targets match predictions.

        Args:
            actual_df (pd.DataFrame): Raw actual sensor data.
            predicted_df (pd.DataFrame or Tuple): Predicted sensor data (in original scale) for each sample, or tuple (df, indices).
            output_file (str): File path to save the statistics CSV.
        """
        if isinstance(predicted_df, tuple):
            predicted_df = predicted_df[0]
        _, Y_actual, _ = self.prepare_autoregressive_samples(actual_df)
        stats = {}
        for i, col in enumerate(self.output_columns):
            actual_values = Y_actual[:, i]
            predicted_values = predicted_df[col].values
            if len(actual_values) != len(predicted_values):
                raise ValueError(
                    f"Mismatch in samples for {col}: actual {len(actual_values)} vs predicted {len(predicted_values)}")
            mae = mean_absolute_error(actual_values, predicted_values)
            mask = actual_values != 0
            mape = np.mean(
                np.abs((actual_values[mask] - predicted_values[mask]) / actual_values[mask])) * 100 if np.sum(
                mask) > 0 else np.nan
            stats[col] = {"MAPE": mape, "MAE": mae}
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv(output_file, index=True)
        print(f"Statistics saved to {output_file}")

