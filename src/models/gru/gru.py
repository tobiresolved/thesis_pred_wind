import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.optimizers import Adam


class SensorNeuralNet_GRU:
    def __init__(self, input_columns: List[str], window_len: int = 10,
                 epochs: int = 10, batch_size: int = 16, event_id=None):
        """
        Initializes the GRU-based sensor neural network.
        Each training sample is built as:
            X = sensor readings over a sliding window (of length window_len)
            Y = sensor readings at the next time step.

        Args:
            input_columns (List[str]): List of sensor column names.
            window_len (int): Length of the sliding window (number of time steps).
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            event_id: Optional event identifier.
        """
        self.input_columns = input_columns
        self.window_len = window_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.event_id = event_id

        # Initialize scaler for sensor data
        self.scaler = StandardScaler()

        self._build_model()

    def _build_model(self):
        """
        Builds a GRU-based neural network.
        Input shape: (window_len, n_features) where n_features = len(input_columns)
        """
        n_features = len(self.input_columns)
        input_layer = Input(shape=(self.window_len, n_features), name="sequence_input")
        gru_out = GRU(64, activation='tanh', name="gru_layer")(input_layer)
        dense1 = Dense(64, activation='relu', name="dense1")(gru_out)
        output = Dense(n_features, name="output")(dense1)

        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer=Adam(), loss='mse')
        self.model.summary()

    def _prepare_samples_single(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Prepares sliding window samples from a single DataFrame.
        The DataFrame must contain an "id" column indicating time steps.
        For each continuous segment (where "id" increases by 1) of length L >= window_len+1:
            For each time t (t >= window_len):
                X = sensor readings from time t-window_len to t-1 (shape: (window_len, n_features))
                Y = sensor readings at time t (shape: (n_features,))

        Returns:
            X (np.ndarray): Array of shape (n_samples, window_len, n_features).
            Y (np.ndarray): Array of shape (n_samples, n_features).
            indices (List): List of "id" values corresponding to the target rows.
        """
        df_sorted = df.sort_values("id").reset_index(drop=True)
        segments = []
        current_seg = [df_sorted.iloc[0]]
        for i in range(1, len(df_sorted)):
            if df_sorted.loc[i, "id"] == df_sorted.loc[i - 1, "id"] + 1:
                current_seg.append(df_sorted.iloc[i])
            else:
                if len(current_seg) >= self.window_len + 1:
                    segments.append(pd.DataFrame(current_seg))
                current_seg = [df_sorted.iloc[i]]
        if len(current_seg) >= self.window_len + 1:
            segments.append(pd.DataFrame(current_seg))

        X_list, Y_list, indices = [], [], []
        for seg in segments:
            seg_values = seg[self.input_columns].values  # shape (L, n_features)
            L = len(seg)
            for t in range(self.window_len, L):
                X_sample = seg_values[t - self.window_len:t]  # (window_len, n_features)
                Y_sample = seg_values[t]  # (n_features,)
                X_list.append(X_sample)
                Y_list.append(Y_sample)
                indices.append(seg.iloc[t]["id"])
        X = np.array(X_list)
        Y = np.array(Y_list)
        return X, Y, indices

    def prepare_samples(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Prepares sliding window samples from a single DataFrame or a list of DataFrames.
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
                return np.empty((0, 0, 0)), np.empty((0, 0)), []
            X_total = np.vstack(X_all)
            Y_total = np.vstack(Y_all)
            return X_total, Y_total, indices_all
        else:
            return self._prepare_samples_single(dfs)

    def train(self, train_dfs: Union[pd.DataFrame, List[pd.DataFrame]]):
        """
        Trains the GRU network using samples extracted from the training DataFrame(s).
        The training DataFrame must contain an "id" column.
        """
        X_train, Y_train, _ = self.prepare_samples(train_dfs)

        # Scale the entire training data (flattening the sequence for scaling, then reshape)
        n_samples, seq_len, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled_2d = self.scaler.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled_2d.reshape(n_samples, seq_len, n_features)
        Y_train_scaled = self.scaler.transform(Y_train)

        self.model.fit(X_train_scaled, Y_train_scaled,
                       epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """
        Predicts sensor values for each sample extracted from the DataFrame.
        The DataFrame must include an "id" column and the sensor columns.

        Returns:
            predicted_df (pd.DataFrame): DataFrame with predicted sensor values (in original scale).
            indices (List): List of "id" values corresponding to each prediction.
        """
        X_pred, _, indices = self.prepare_samples(df)
        n_samples, seq_len, n_features = X_pred.shape
        X_pred_2d = X_pred.reshape(-1, n_features)
        X_pred_scaled_2d = self.scaler.transform(X_pred_2d)
        X_pred_scaled = X_pred_scaled_2d.reshape(n_samples, seq_len, n_features)

        predictions_scaled = self.model.predict(X_pred_scaled)
        predictions = self.scaler.inverse_transform(predictions_scaled)
        predicted_df = pd.DataFrame(predictions, columns=self.input_columns)
        return predicted_df, indices

    def plot_actual_vs_predicted(self, actual_df: pd.DataFrame, predicted_df: Union[pd.DataFrame, Tuple],
                                 sensor_info: dict) -> None:
        """
        Plots actual vs. predicted sensor data for each sensor column.
        The actual and predicted values are obtained via sliding window sample extraction.

        Args:
            actual_df (pd.DataFrame): Raw sensor data with an "id" column.
            predicted_df (pd.DataFrame or Tuple): Predicted sensor data (or tuple with DataFrame).
            sensor_info (dict): Dictionary mapping sensor column names to metadata.
        """
        if isinstance(predicted_df, tuple):
            predicted_df = predicted_df[0]
        _, Y_actual, seq_indices = self.prepare_samples(actual_df)
        actual_plot_df = pd.DataFrame(Y_actual, columns=self.input_columns)
        actual_plot_df.index = seq_indices
        predicted_plot_df = predicted_df.copy()
        predicted_plot_df.index = seq_indices

        for col in self.input_columns:
            plt.figure(figsize=(12, 6))
            plt.plot(actual_plot_df.index, actual_plot_df[col], label=f"{sensor_info.get(col, col)} Actual",
                     color="blue")
            plt.plot(predicted_plot_df.index, predicted_plot_df[col], label="Predicted", color="red", linestyle="--")
            plt.title(f"Sensor: {col} - Actual vs. Predicted")
            plt.xlabel("id")
            plt.ylabel(col)
            plt.legend()
            plt.show()

    def save_prediction_statistics(self, actual_df: pd.DataFrame, predicted_df: Union[pd.DataFrame, Tuple],
                                   output_file: str) -> None:
        """
        Computes and saves prediction statistics (MAPE and MAE) comparing actual and predicted sensor data.

        Args:
            actual_df (pd.DataFrame): Raw sensor data.
            predicted_df (pd.DataFrame or Tuple): Predicted sensor data.
            output_file (str): File path to save the statistics CSV.
        """
        if isinstance(predicted_df, tuple):
            predicted_df = predicted_df[0]
        _, Y_actual, _ = self.prepare_samples(actual_df)
        stats = {}
        for i, col in enumerate(self.input_columns):
            actual_values = Y_actual[:, i]
            predicted_values = predicted_df[col].values
            if len(actual_values) != len(predicted_values):
                raise ValueError(
                    f"Mismatch for {col}: actual {len(actual_values)} vs predicted {len(predicted_values)}")
            mae = mean_absolute_error(actual_values, predicted_values)
            mask = actual_values != 0
            mape = np.mean(
                np.abs((actual_values[mask] - predicted_values[mask]) / actual_values[mask])) * 100 if np.sum(
                mask) > 0 else np.nan
            stats[col] = {"MAPE": mape, "MAE": mae}
        stats_df = pd.DataFrame(stats).T
        stats_df.to_csv(output_file, index=True)
        print(f"Statistics saved to {output_file}")
