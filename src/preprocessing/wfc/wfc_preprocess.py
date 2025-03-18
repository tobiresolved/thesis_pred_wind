from typing import Tuple
import pandas as pd
from helpers.utils import pipeline_step
import matplotlib.pyplot as plt
import numpy as np
from visualization.plotter import DataPlotter
from preprocessing.common.preprocess import Preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from preprocessing.wfc.const import INPUT_SENSORS, OUTPUT_SENSORS, EXCEPT_COLUMNS
from models.lstm.rnn import RecurrentSequenceModel
from matplotlib.backends.backend_pdf import PdfPages


class WFCPreprocess(Preprocess):

    def __init__(self, event_id: int):
        super().__init__(event_id)

    @pipeline_step("Process", logger="WFALOGGER")
    def process_away(self, df_path: str):
        df = self.load_dataframe(df_path=df_path)
        df = self.keep_columns(df=df, columns_to_keep="avg", columns_to_exclude=EXCEPT_COLUMNS)

        df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        df_train, df_test = self.split_by_train_test(df=df)
        df_train = self.filter_status_rows(df=df_train)
        #df_train = self.filter_normal_sequences(df=df_train)
        nan_rows = df_train.isnull().any(axis=1).sum()
        total_rows = df_train.shape[0]

        # Print the result
        print(f"{nan_rows} out of {total_rows} rows contain NaN values.")
        rf = RandomForest()
        X, y, xt, yt = rf.preprocess(
            train_df=df_train,
            test_df=df_test,
            input_columns=INPUT_SENSORS,
            output_columns=None
        )
        rf.train(X=X, y=y)
        yp = rf.predict(X=xt)
        self.plot_actual_vs_predicted_by_column(df_actual=yt, predictions=yp, pdf_filename=f"res/plots/WFA/{self.event_id}")


        print(df_train.shape)
        print(df_test.shape)
        print(df_test.head())


    def process(self, df_path: str):
        df = self.load_dataframe(df_path=df_path)
        df = self.keep_columns(df=df, columns_to_keep="avg", columns_to_exclude=EXCEPT_COLUMNS)
        df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        df = self.change_normal_rows(df=df)
        df = self.add_turbine_standstill(df=df)
        df_train, df_test = self.split_by_train_test(df=df)
        df_first1000 = df.head(10000)
        dp = DataPlotter(event=self.event)
        lstm_ae = SensorLSTMAutoencoder(input_columns=INPUT_SENSORS, event_id=self.event_id, time_steps=36)
        # 3. Mask the training DataFrame.
        df_train_masked = lstm_ae.mask_scaled_train_df(df_train)

        # 4. Train the autoencoder using the masked training data.
        lstm_ae.train(train_df=df_train_masked, epochs=8, batch_size=6, validation_split=0.2)

        # 5. Once training is done, predict/reconstruct on test data.
        # For test data, you might simply scale without masking (unless required).
        predicted_test = lstm_ae.predict(df_test)

        # Optionally, plot the results.
        lstm_ae.plot_actual_vs_predicted_with_difference(
            actual_df=df_test,
            predicted_df=predicted_test
        )



        for sensor in INPUT_SENSORS:
            dp.plot_timeseries_with_status(
                df=df,
                outside_temp_col="sensor_177_avg",
                wind_speed_col="wind_speed_236_avg",
                rotor_rpm_col="sensor_144_avg",
                status_col="status_type_id",
                sensor_col=sensor
            )








    @pipeline_step("Process", logger="WFALOGGER")
    def process_3(self, df_path: str):
        df = self.load_dataframe(df_path=df_path)
        df = self.keep_columns(df=df, columns_to_keep="avg", columns_to_exclude=EXCEPT_COLUMNS)
        df = self.filter_normal_sequences(df=df)
        df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        print(df.shape)
        df = self.filter_status_rows(df=df)


        df_train, df_validation = self.split_by_train_test(df=df)

        useless_columns = [
            "status_type_id",  # constant zero
            "train_test",  # constant zero
            "asset_id",  # angle
            "time_stamp"
        ]
        df_train = df_train.drop(columns=useless_columns)
        df_validation = df_validation.drop(columns=useless_columns)

        df_train_X, df_train_Y, = self.split_dataframe_id(df_train)
        dfs_train_X = self.split_into_chunks_6_hours(df=df_train_X)
        dfs_train_Y = self.split_into_chunks_6_hours(df=df_train_Y)
        for df in dfs_train_X:
            df.drop(columns=["id"], inplace=True)
        for df in dfs_train_Y:
            df.drop(columns=["id"], inplace=True)
        print("!!")
        print(len(dfs_train_X))
        print(dfs_train_X[0].shape)
        print(len(dfs_train_Y))
        print(dfs_train_Y[0].shape)
        #print(df_train_X)
        #print(df_train_Y)
        df_validation_X, df_validation_Y, = self.split_dataframe_id(df_validation)
        dfs_validation_X = self.split_into_chunks_6_hours(df=df_validation_X)
        dfs_validation_Y = self.split_into_chunks_6_hours(df=df_validation_Y)
        for df in dfs_validation_X:
            df.drop(columns=["id"], inplace=True)
        for df in dfs_validation_Y:
            df.drop(columns=["id"], inplace=True)
        print("!!")
        print(len(dfs_validation_X))
        print(dfs_validation_X[0].shape)
        print(len(dfs_validation_Y))
        print(dfs_validation_Y[0].shape)

        rnn = RecurrentSequenceModel(input_shape=dfs_train_X[0].shape, output_shape=dfs_train_Y[0].shape)
        rnn.train(X_train_df_list=dfs_train_X, Y_train_df_list=dfs_train_Y)
        prediction = rnn.predict(X_df_list=dfs_validation_X)

        self.plot_actual_vs_predicted_by_column_list(actual_dfs=dfs_validation_Y, predicted_dfs=prediction)
        """""
        model = RandomForestRegressor(n_estimators=10, random_state=42)

        # Train the model on the training set
        model.fit(df_train_X, df_train_Y)

        # Predict on the validation set
        predictions = model.predict(df_validation_X)

        self.plot_actual_vs_predicted_by_column(df_actual=df_validation_Y, predictions=predictions)

        # Evaluate the model using Mean Squared Error
        mse = mean_squared_error(df_validation_Y, predictions)
        print("Validation Mean Squared Error:", mse)


        
        df_first_10000 = df.head(3000)
        dp = DataPlotter(event=self.event)
        
        for sensor in INPUT_SENSORS:
            dp.plot_timeseries_with_status(
                df=df_first_10000,
                outside_temp_col="sensor_177_avg",
                wind_speed_col="wind_speed_236_avg",
                rotor_rpm_col="sensor_144_avg",
                status_col="status_type_id",
                sensor_col=sensor
            )
        """""

    @pipeline_step("Process", logger="WFA_PREPROCESS")
    def process_1(self, df_path: str):
        df = self.load_dataframe(df_path=df_path)
        # df = self.filter_status_rows(df=df)
        # df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        dp = DataPlotter(event=self.event)
        # dp.plot_timeseries(sensor_series=df["sensor_178_avg"], sensor=self.sensor_metadata["sensor_178_avg"])
        # dp.plot_timeseries(sensor_series=df["wind_speed_237_avg"], sensor=self.sensor_metadata["wind_speed_237_avg"])
        # dp.plot_timeseries(sensor_series=df["sensor_145_avg"], sensor=self.sensor_metadata["sensor_145_avg"])

        # dp.plot_sensor_correlation_status_type(df=df, sensor_x="wind_speed_237_avg", sensor_y="sensor_178_avg")
        # dp.plot_sensor_correlation_by_status_loop(df=df, sensor_x="wind_speed_235_avg", sensor_y="sensor_145_avg")
        # dp.plot_sensor_correlation_status_type(df=df, sensor_x="wind_speed_237_avg", sensor_y="sensor_144_avg")

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


        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model on the training set
        model.fit(df_train_X, df_train_Y)

        # Predict on the validation set
        predictions = model.predict(df_validation_X)

        self.plot_actual_vs_predicted_by_column(df_actual=df_validation_Y, predictions=predictions)

        # Evaluate the model using Mean Squared Error
        mse = mean_squared_error(df_validation_Y, predictions)
        print("Validation Mean Squared Error:", mse)


    @pipeline_step("Process", logger="WFA_PREPROCESS")
    def process_2(self, df_path: str):
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
        from tensorflow.keras import layers

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
        #predictions = pd.Series(predictions.flatten(), name='sensor_178_avg')

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
        df_train = df[df["train_test"] == False].copy()
        df_test = df[df["train_test"] == True].copy()
        return df_train, df_test

    @pipeline_step("Keep Columns")
    def split_dataframe_id(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # Create a copy of the input sensors list and add the "id" column
        input_cols = INPUT_SENSORS.copy()
        input_cols.append("id")
        df_selected = df[input_cols].copy()

        # Create a copy of the output sensors list and add the "id" column
        output_cols = OUTPUT_SENSORS.copy()
        output_cols.append("id")
        df_other = df[output_cols].copy()

        return df_selected, df_other

    def split_dataframe(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        selected_cols = [
            "wind_speed_235_avg", "sensor_7_avg", "sensor_144_avg"
        ]
        selected_cols = INPUT_SENSORS
        selected_cols.append("id")
        # Create a DataFrame with the selected columns
        df_selected = df[selected_cols].copy()

        # Create a DataFrame with all other columns (drop the selected ones)
        df_other = df.drop(columns=INPUT_SENSORS)
        other_cols = OUTPUT_SENSORS
        other_cols.append("id")
        df_other = df_other[other_cols]

        return df_selected, df_other

    def add_effective_wind(self, df: pd.DataFrame) -> pd.DataFrame:
        relative_direction_col = "sensor_2_avg"
        relative_direction_radians = np.radians(df[relative_direction_col])

        df["effective_wind_avg"] = df["wind_speed_3_avg"] * np.cos(relative_direction_radians)
        df["effective_wind_min"] = df["wind_speed_3_min"] * np.cos(relative_direction_radians)
        df["effective_wind_max"] = df["wind_speed_3_max"] * np.cos(relative_direction_radians)
        df["effective_wind_std"] = df["wind_speed_3_std"] * np.cos(relative_direction_radians)

        return df

    def plot_actual_vs_predicted_by_column_list(self, actual_dfs: list, predicted_dfs: list) -> None:
        """
        Expects:
          - actual_dfs: a list of DataFrames with the actual values,
                        each DataFrame should have the same columns.
          - predicted_dfs: a list of DataFrames with the predicted values,
                           in the same order as actual_dfs. These DataFrames may not have column names.

        The method concatenates all DataFrames in each list (vertically) and then plots,
        for each column, the actual and predicted time series.
        """
        # Ensure both lists have the same number of DataFrames.
        if len(actual_dfs) != len(predicted_dfs):
            raise ValueError("The lists of actual and predicted DataFrames must be the same length.")

        # Optionally, verify that each corresponding pair has the same number of columns.
        for i in range(len(actual_dfs)):
            if actual_dfs[i].shape[1] != predicted_dfs[i].shape[1]:
                raise ValueError(f"DataFrame at index {i} differs in column count between actual and predicted.")

        # Concatenate the list of DataFrames vertically (ignoring the original index).
        df_actual_concat = pd.concat(actual_dfs, axis=0, ignore_index=True)
        df_predicted_concat = pd.concat(predicted_dfs, axis=0, ignore_index=True)

        # Assign predicted DataFrame the same column names as the actual DataFrame,
        # because neural network outputs may not have column names.
        df_predicted_concat.columns = df_actual_concat.columns

        if df_actual_concat.shape != df_predicted_concat.shape:
            raise ValueError("After concatenation, actual and predicted DataFrames must have the same shape.")

        # Plot column by column.
        for column in df_actual_concat.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df_actual_concat.index, df_actual_concat[column], label="Actual", color="blue")
            plt.plot(df_predicted_concat.index, df_predicted_concat[column], label="Predicted", color="red",
                     linestyle="--")
            plt.xlabel("Sample Index")
            plt.ylabel(column)
            plt.title(f"Actual vs Predicted for {column}")
            plt.legend()
            plt.grid(True)
            plt.show()

        # Optionally, verify that each corresponding pair has the same number of columns.
        for i in range(len(actual_dfs)):
            if actual_dfs[i].shape[1] != predicted_dfs[i].shape[1]:
                raise ValueError(f"DataFrame at index {i} differs in column count between actual and predicted.")

        # Concatenate the list of DataFrames vertically (ignoring the original index).
        df_actual_concat = pd.concat(actual_dfs, axis=0, ignore_index=True)
        df_predicted_concat = pd.concat(predicted_dfs, axis=0, ignore_index=True)

        if df_actual_concat.shape != df_predicted_concat.shape:
            raise ValueError("After concatenation, actual and predicted DataFrames must have the same shape.")

        # Plot column by column
        for column in df_actual_concat.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(df_actual_concat.index, df_actual_concat[column], label="Actual", color="blue")
            plt.plot(df_predicted_concat.index, df_predicted_concat[column], label="Predicted", color="red",
                     linestyle="--")
            plt.xlabel("Sample Index")
            plt.ylabel(column)
            plt.title(f"Actual vs Predicted for {column}")
            plt.legend()
            plt.grid(True)
            plt.show()

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def plot_actual_vs_predicted_by_column(self, df_actual: pd.DataFrame, predictions: np.ndarray,
                                           pdf_filename: str = "plots.pdf") -> None:
        # Ensure the predictions array matches the shape of the DataFrame
        if predictions.shape != df_actual.shape:
            raise ValueError("The shape of predictions must match the shape of the actual DataFrame.")

        # Create a PdfPages object to save all plots in one PDF file.
        with PdfPages(pdf_filename) as pdf:
            # Iterate over each column in the DataFrame
            for column in df_actual.columns:
                print(f"Plotting for column: {column}")
                fig = plt.figure(figsize=(10, 5))
                # Plot actual values
                plt.plot(df_actual.index, df_actual[column], label="Actual", color="blue")
                # Get the index of the current column to select corresponding predicted values
                col_idx = df_actual.columns.get_loc(column)
                plt.plot(df_actual.index, predictions[:, col_idx], label="Predicted", color="red", linestyle="--")
                plt.xlabel("Sample Index")
                plt.ylabel("Value")
                plt.title(
                    f"{self.sensor_metadata[column].name} - {self.sensor_metadata[column].description} - {self.sensor_metadata[column].unit}")
                plt.legend()
                plt.grid(True)

                # Save the current figure into the PDF
                pdf.savefig(fig)
                plt.close(fig)  # Close the figure to free memory

        print(f"All plots have been saved to {pdf_filename}")




