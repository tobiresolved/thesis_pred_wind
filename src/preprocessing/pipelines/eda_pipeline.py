from typing import Tuple
import pandas as pd
from helpers.helpers import Helper
from helpers.utils import pipeline_step
from preprocessing.metadata.dataclasses import Event, Sensor
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import pickle


class EDAPipeline:

    def __init__(self, event_id: int):
        self.logger = Helper.get_logger("EDA")
        self.event_id = event_id
        self.df = self._read_dataset()
        self.update_column_names()
        self.event = Event.from_json(event_id=self.event_id)
        self.sensor_data = self._load_sensor_data()

    @pipeline_step(step_name="Read Dataset")
    def _read_dataset(self) -> pd.DataFrame:
        file_path = Helper.get_dataset_file_location(dataset_id=self.event_id)
        self.df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        return self.df

    def _load_sensor_data(self) -> dict:
            sensor_data = {}
            for sensor_column in self.df.columns[5:]:
                sensor_data[sensor_column] = Sensor.from_json(sensor_name=sensor_column, event_id=self.event_id)
            return sensor_data

    @pipeline_step("Update Column Names")
    def update_column_names(self) -> "EDAHandler":
        stat_types = ("std", "min", "max", "avg")
        updated_columns = {
            col: f"{col}_avg" if (not col.endswith(stat_types) and col.startswith("sensor")) else col
            for col in self.df.columns
        }
        self.df.rename(columns=updated_columns, inplace=True)
        return self

    @pipeline_step("Format DataFrame")
    def format_df(self) -> "EDAHandler":
        self.df['time_stamp'] = pd.to_datetime(self.df['time_stamp'])
        self.df['train_test'] = self.df['train_test'].map({'train': False, 'prediction': True})
        self.df['status_type_id'] = self.df['status_type_id'].astype('category')
        sensor_columns = self.df.columns[5:]
        self.df[sensor_columns] = self.df[sensor_columns].astype('float32')
        self.sensor_df = self.df[sensor_columns]
        self.logger.info(f"DataFrame size after format: {self.df.size}")
        return self

    @pipeline_step("Dump Everything")
    def dump_df_event_sensors(self) -> "EDAHandler":
        pickle_location = Helper.get_pickle_location(dataset_id=self.event_id, obj="DATAFRAME")
        with open(pickle_location, "wb") as file:
            pickle.dump(self.df, file=file)
        pickle_location = Helper.get_pickle_location(dataset_id=self.event_id, obj="EVENTS")
        with open(pickle_location, "wb") as file:
            pickle.dump(self.event, file=file)
        pickle_location = Helper.get_pickle_location(dataset_id=self.event_id, obj="SENSORS")
        with open(pickle_location, "wb") as file:
            pickle.dump(self.sensor_data, file=file)
        return self

    @pipeline_step("Subtract Ambient Temperature")
    def subtract_ambient_temperature_2(
            self,
            temperature_sensors: list = None,
            ambient_sensor: str = "sensor_0_avg"
    ) -> "EDAHandler":
        """
        Subtracts the ambient temperature sensor from each temperature sensor, row by row,
        and creates a new DataFrame containing these differences. This operation is performed
        on a slice of the DataFrame that corresponds to 2 weeks before and 2 weeks after the
        event index (given in 10-minute intervals).

        Args:
            event: An object or structure containing the 'event_start_id', which denotes the
                   central row index around which to slice data.
            temperature_sensors (list): A list of sensor column names to process.
                                        If None, a default set of temperature sensors will be used.
            ambient_sensor (str): The column name of the ambient temperature sensor.

        Returns:
            self (EDAHandler): For method chaining.
        """
        import pandas as pd

        # 1. Define default temperature sensors if none are provided
        if temperature_sensors is None:
            temperature_sensors = [
                'sensor_0_avg', 'sensor_6_avg', 'sensor_7_avg', 'sensor_8_avg', 'sensor_9_avg',
                'sensor_10_avg', 'sensor_11_avg', 'sensor_12_avg', 'sensor_13_avg', 'sensor_14_avg',
                'sensor_15_avg', 'sensor_16_avg', 'sensor_17_avg', 'sensor_19_avg', 'sensor_20_avg',
                'sensor_21_avg', 'sensor_35_avg', 'sensor_36_avg', 'sensor_37_avg', 'sensor_38_avg',
                'sensor_39_avg', 'sensor_40_avg', 'sensor_41_avg', 'sensor_43_avg', 'sensor_53_avg'
            ]

        # 2. Check for columns that actually exist in self.df
        existing_sensors = [col for col in temperature_sensors if col in self.df.columns]

        if not existing_sensors:
            self.logger.warning("No matching temperature sensor columns found in the DataFrame.")
            return self

        # 3. Ensure the ambient sensor exists
        if ambient_sensor not in self.df.columns:
            self.logger.warning(f"Ambient sensor '{ambient_sensor}' not found in the DataFrame.")
            return self

        # 4. Determine the row index for the event
        event_row = int(self.event.event_start_id)

        # 5. Calculate the number of rows that correspond to 2 weeks, given 10-minute intervals.
        #    1 day = 24 hours * (60 / 10) intervals = 144 intervals
        #    2 weeks = 14 days * 144 intervals = 2016 intervals
        two_weeks_in_rows = 2016

        # 6. Slice the DataFrame to 2 weeks before and 2 weeks after the event_row
        start_row = max(0, event_row - two_weeks_in_rows)
        end_row = min(len(self.df), event_row + two_weeks_in_rows + 1)  # +1 for inclusive slice
        sub_df = self.df.iloc[start_row:end_row]

        if sub_df.empty:
            self.logger.warning(
                "The slice of the DataFrame for the specified event range is empty. "
                "No ambient subtraction performed."
            )
            return self

        # 7. Create a new DataFrame for the ambient-subtracted values
        subtracted_df = pd.DataFrame(index=sub_df.index)

        for sensor in existing_sensors:
            if sensor == ambient_sensor:
                # Optionally skip subtracting the ambient sensor from itself
                continue
            subtracted_df[sensor] = sub_df[sensor] - sub_df[ambient_sensor]

        # 8. (Optional) Rename columns to indicate they've been ambient-subtracted
        #    (Here we keep them the same, but you can rename if desired.)
        #    Example:
        #    subtracted_df.rename(
        #        columns={col: f"{col}_minus_{ambient_sensor}" for col in subtracted_df.columns},
        #        inplace=True
        #    )

        # 9. Store this new sub-DataFrame in the class for later use or plotting
        self.ambient_subtracted_df = subtracted_df

        # 10. Log some information
        self.logger.info(
            f"Created ambient-subtracted DataFrame with {len(subtracted_df.columns)} columns "
            f"from row {start_row} to {end_row}, using '{ambient_sensor}' as the ambient reference."
        )

        return self

    @pipeline_step("Subtract Ambient Temperature")
    def subtract_ambient_temperatures(
            self,
            temperature_sensors: list = None,
            ambient_sensor: str = "sensor_0_avg"
    ) -> "EDAHandler":
        """
        Subtracts the ambient temperature sensor from each temperature sensor, row by row,
        and creates a new DataFrame containing these differences.

        Args:
            temperature_sensors (list): A list of sensor column names to process.
                                        If None, a default set of temperature sensors will be used.
            ambient_sensor (str): The column name of the ambient temperature sensor.

        Returns:
            self (EDAHandler): For method chaining.
        """
        import pandas as pd

        # 1. If no sensors are provided, define a default list
        if temperature_sensors is None:
            temperature_sensors = [
                'sensor_0_avg', 'sensor_6_avg', 'sensor_7_avg', 'sensor_8_avg', 'sensor_9_avg',
                'sensor_10_avg', 'sensor_11_avg', 'sensor_12_avg', 'sensor_13_avg', 'sensor_14_avg',
                'sensor_15_avg', 'sensor_16_avg', 'sensor_17_avg', 'sensor_19_avg', 'sensor_20_avg',
                'sensor_21_avg', 'sensor_35_avg', 'sensor_36_avg', 'sensor_37_avg', 'sensor_38_avg',
                'sensor_39_avg', 'sensor_40_avg', 'sensor_41_avg', 'sensor_43_avg', 'sensor_53_avg'
            ]


        # 2. Check for columns that actually exist in self.df
        existing_sensors = [col for col in temperature_sensors if col in self.df.columns]

        if not existing_sensors:
            self.logger.warning("No matching temperature sensor columns found in the DataFrame.")
            return self

        # 3. Ensure the ambient sensor exists
        if ambient_sensor not in self.df.columns:
            self.logger.warning(f"Ambient sensor '{ambient_sensor}' not found in the DataFrame.")
            return self

        # 4. Create a new DataFrame for the ambient-subtracted values
        subtracted_df = pd.DataFrame(index=self.df.index)

        for sensor in existing_sensors:
            if sensor == ambient_sensor:
                # Optionally, you might skip the ambient sensor itself
                # or store it unchanged. We'll skip subtracting it from itself.
                continue
            # Subtract each row's ambient sensor value from the sensor's value
            subtracted_df[sensor] = self.df[sensor] - self.df[ambient_sensor]

        # 5. (Optional) Rename columns to indicate they've been ambient-subtracted
        #    e.g., "sensor_6_avg_minus_sensor_0_avg"
        subtracted_df = subtracted_df.rename(
            columns={
                col: f"{col}" for col in subtracted_df.columns
            }
        )

        # 6. Store this new DataFrame in the class for later use or plotting
        self.ambient_subtracted_df = subtracted_df
        print(self.ambient_subtracted_df)

        self.logger.info(
            f"Created ambient-subtracted DataFrame with {len(subtracted_df.columns)} columns, "
            f"using '{ambient_sensor}' as the ambient reference."
        )

        return self

    @pipeline_step("Compute Generator-to-Rotor RPM Ratio")
    def compute_generator_to_rotor_ratio(self) -> "EDAHandler":
        """
        Computes the ratio of generator RPM (sensor_18_avg) to rotor RPM (sensor_52_avg),
        adds it as a new column, and plots this ratio over the four-week window
        centered around self.event.event_start_id.

        Returns:
            self (EDAHandler): For method chaining.
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # Ensure required columns exist
        required_sensors = ['sensor_18_avg', 'sensor_52_avg']
        missing_sensors = [col for col in required_sensors if col not in self.df.columns]

        if missing_sensors:
            self.logger.warning(f"Missing required columns: {missing_sensors}. Skipping computation.")
            return self

        # Compute the generator-to-rotor RPM ratio
        #self.df['generator_rotor_ratio'] = self.df['sensor_18_avg'] / self.df['sensor_52_avg']
        self.df['generator_rotor_ratio'] = self.df['sensor_52_avg'] / self.df['sensor_18_avg']
        # Determine the event start index
        event_start_idx = self.event.event_start_id

        # Define the range (2 weeks before and 2 weeks after)
        # Since each index represents a 10-minute frame:
        intervals_per_week = (7 * 24 * 60) // 10  # 10-minute intervals in a week
        start_idx = max(0, event_start_idx - 2 * intervals_per_week)
        end_idx = min(len(self.df) - 1, event_start_idx + 2 * intervals_per_week)

        # Select data within this window
        event_window_df = self.df.iloc[start_idx:end_idx]

        if event_window_df.empty:
            self.logger.warning("No data available for the selected four-week window.")
            return self

        # Plot the generator-to-rotor RPM ratio
        plt.figure(figsize=(12, 6))
        plt.plot(event_window_df.index, event_window_df['generator_rotor_ratio'], label="Generator/Rotor RPM Ratio",
                 color='b')
        plt.xlabel("Index (10-minute intervals)")
        plt.ylabel("Generator-to-Rotor RPM Ratio")
        plt.title("Generator-to-Rotor RPM Ratio Over the Four-Week Window")
        plt.legend()
        plt.grid(True)
        plt.show()

        self.logger.info("Computed and plotted generator-to-rotor RPM ratio for the event window.")

        return self

    @pipeline_step("Add Attributes to DataFrame")
    def add_attrs_to_df(self) -> "EDAHandler":

        event = Event.from_json(event_id=self.event_id)
        self.event = event
        self.df.attrs["event"] = event

        for sensor_column in self.df.columns[5:]:
            self.df.attrs[sensor_column] = Sensor.from_json(
                sensor_name=sensor_column, event_id=self.event_id
            )

        self.logger.info("Successfully added attributes to the DataFrame.")

        return self

    @pipeline_step("Plot Cooling Water Temperature Over Last 4 Weeks")
    def plot_cooling_water_temperature(self) -> "EDAHandler":
        """
        Extracts and plots the temperature of the VCS cooling water (`sensor_10_avg`)
        over the four-week window centered around `self.event.event_start_id`.

        Returns:
            self (EDAHandler): For method chaining.
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # Ensure the sensor column exists
        if 'sensor_10_avg' not in self.df.columns:
            self.logger.warning("Missing required column: sensor_10_avg. Skipping computation.")
            return self

        # Determine the event start index
        event_start_idx = self.event.event_start_id

        # Define the range (2 weeks before and 2 weeks after)
        # Since each index represents a 10-minute frame:
        intervals_per_week = (7 * 24 * 60) // 10  # 10-minute intervals in a week
        start_idx = max(0, event_start_idx - 2 * intervals_per_week)
        end_idx = min(len(self.df) - 1, event_start_idx + 2 * intervals_per_week)

        # Select data within this window
        event_window_df = self.df.iloc[start_idx:end_idx]

        if event_window_df.empty:
            self.logger.warning("No data available for the selected four-week window.")
            return self

        # Plot the cooling water temperature
        plt.figure(figsize=(12, 6))
        plt.plot(event_window_df.index, event_window_df['sensor_10_avg'], label="VCS Cooling Water Temperature",
                 color='r')
        plt.xlabel("Index (10-minute intervals)")
        plt.ylabel("Temperature (°C)")
        plt.title("VCS Cooling Water Temperature Over the Last 4 Weeks")
        plt.legend()
        plt.grid(True)
        plt.show()

        self.logger.info("Plotted VCS cooling water temperature for the event window.")

        return self

    @pipeline_step("Aggregate DataFrame on 2-hour intervals")
    def aggregate_data_2h(self) -> "EDAHandler":
        """
        Groups the DataFrame into 2-hour bins based on the 'time_stamp' column
        and computes the average of all columns except the first five.
        The first five columns are kept using their first value within each bin.
        """

        # Ensure time_stamp is a datetime type
        self.df["time_stamp"] = pd.to_datetime(self.df["time_stamp"])

        # Define which columns to retain (first five) and which to average
        cols_keep = self.df.columns[:5]  # 'time_stamp', 'asset_id', 'id', 'train_test', 'status_type'
        cols_mean = self.df.columns[5:]  # all other columns

        # Set time_stamp as the index for resampling
        self.df.set_index("time_stamp", inplace=True)

        # Aggregate each group of 2 hours:
        #   - Keep the first occurrence of the "cols_keep" columns
        #   - Take the mean of the remaining columns
        agg_dict = {col: "first" for col in cols_keep}
        agg_dict.update({col: "mean" for col in cols_mean})

        aggregated_df = self.df.resample("2H").agg(agg_dict)

        # Reset the index so that 'time_stamp' becomes a column again
        aggregated_df.reset_index(inplace=True)

        # Store the aggregated DataFrame back in self.df
        self.df = aggregated_df

        self.logger.info("Successfully aggregated data on a 2-hour basis.")

        return self

    @pipeline_step("Plot Sensor Data and Status Over Last 2 Weeks Before Event")
    def plot_sensor_data_with_status(self) -> "EDAHandler":
        """
        Extracts and plots each sensor column (columns starting from the sixth column)
        over the two-week window preceding `self.event.event_start_id`. For every sensor,
        a figure is created with two subplots: the top subplot displays the sensor's data,
        while the bottom subplot displays the `status_type_id` data. The event description
        (`self.event.event_description`) is used as the overall title for each figure.

        Returns:
            self (EDAHandler): For method chaining.
        """
        import pandas as pd
        import matplotlib.pyplot as plt

        # Check if the required 'status_type_id' column exists
        if 'status_type_id' not in self.df.columns:
            self.logger.warning("Missing required column: status_type_id. Skipping plot generation.")
            return self

        # Determine the event start index
        event_start_idx = self.event.event_start_id

        # Define the two-week window (each index represents a 10-minute interval)
        intervals_per_week = (7 * 24 * 60) // 10  # 10-minute intervals per week
        window_length = 2 * intervals_per_week
        start_idx = max(0, event_start_idx - window_length)
        end_idx = event_start_idx

        # Select data within this two-week window
        window_df = self.df.iloc[start_idx:]
        if window_df.empty:
            self.logger.warning("No data available for the selected two-week window.")
            return self

        # Iterate over each sensor column (all columns from the sixth column onward)
        sensor_columns = self.df.columns[5:]
        for sensor in sensor_columns:
            # Create a figure with two vertically-stacked subplots sharing the x-axis
            fig, (ax_sensor, ax_status) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Plot the sensor data
            ax_sensor.plot(window_df.index, window_df[sensor], label=sensor, color='b')
            ax_sensor.set_ylabel(self.sensor_data[sensor].description)
            ax_sensor.legend(loc='upper right')
            ax_sensor.grid(True)

            # Mark the event start index on the sensor data subplot
            ax_sensor.axvline(x=event_start_idx, color='r', linestyle='--', label="Event Start")

            # Plot the status_type_id as discrete points using a scatter plot (no connecting line)
            ax_status.scatter(window_df.index, window_df['status_type_id'], label="Status Type ID",
                              color='g', marker='o')
            ax_status.set_ylabel("Status Type ID")
            ax_status.set_xlabel("Index (10-minute intervals)")
            ax_status.legend(loc='upper right')
            ax_status.grid(True)

            # Mark the event start index on the status subplot
            ax_status.axvline(x=event_start_idx, color='r', linestyle='--', label="Event Start")

            # Set the overall title using the event description
            plt.suptitle(self.event.event_description, fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            self.logger.info(f"Plotted sensor data for {sensor} with status overlay and event marker.")

        return self

    def get_sensor_description(self, column_name: str) -> str:
        """
        Retrieves the sensor description from the sensor_data dictionary.

        Args:
            column_name (str): The name of the sensor column.

        Returns:
            str: The sensor's description if found; otherwise, the column name itself.
        """
        sensor = self.sensor_data.get(column_name)
        if sensor:
            return sensor.description
        return f"No description found for {column_name}"

    @pipeline_step("Identify and Visualize Strong Correlations")
    def identify_strong_correlations(self, threshold: float = 0.7, plot: bool = True) -> "EDAHandler":
        """
        Identifies, logs, and visualizes strong correlations between sensors in the DataFrame.

        Args:
            threshold (float): Minimum absolute correlation value to consider a pair as strongly correlated.
            plot (bool): Whether to visualize the correlation matrix as a heatmap.

        Returns:
            EDAHandler: The instance of EDAHandler for method chaining.
        """

        # Calculate the correlation matrix
        correlation_matrix = self.sensor_df.corr()

        # Extract strong correlation pairs
        strong_pairs = correlation_matrix.unstack().reset_index()
        strong_pairs.columns = ['Sensor 1', 'Sensor 2', 'Correlation']
        strong_pairs = strong_pairs[(strong_pairs['Correlation'].abs() > threshold) &
                                    (strong_pairs['Sensor 1'] != strong_pairs['Sensor 2'])]

        # Remove duplicate pairs (e.g., (A, B) and (B, A))
        strong_pairs['pair'] = strong_pairs.apply(lambda x: tuple(sorted([x['Sensor 1'], x['Sensor 2']])), axis=1)
        strong_pairs = strong_pairs.drop_duplicates(subset=['pair']).drop('pair', axis=1)

        # Replace sensor names with descriptions using self.get_sensor_description
        strong_pairs['Sensor 1'] = strong_pairs['Sensor 1'].apply(self.get_sensor_description)
        strong_pairs['Sensor 2'] = strong_pairs['Sensor 2'].apply(self.get_sensor_description)

        # Log the result
        self.logger.info(f"Strong correlations (threshold={threshold}):")
        self.logger.info(f"\n{strong_pairs.to_string(index=False)}")

        # Visualization
        if plot:
            plt.figure(figsize=(12, 8))

            # Replace matrix labels with descriptions for visualization
            labeled_correlation_matrix = correlation_matrix.rename(columns=self.get_sensor_description,
                                                                   index=self.get_sensor_description)

            sns.heatmap(labeled_correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
            plt.title('Sensor Correlation Matrix')
            plt.show()

        # Optionally store the result in the instance if needed for further use
        self.strong_correlations = strong_pairs

        return self

    @pipeline_step("Keep Columns and Calculate Differences")
    def keep_columns_and_calculate_difference(self) -> "EDAHandler":
        cols_to_keep = ['wind_speed_3_avg', 'sensor_18_avg', 'sensor_52_avg']
        missing_cols = [col for col in cols_to_keep if col not in self.df.columns]
        if missing_cols:
            self.logger.error(f"Missing columns in DataFrame: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        self.df = self.df[cols_to_keep]
        self.df['diff_wind_sensor_18'] = self.df['wind_speed_3_avg'] - self.df['sensor_18_avg']
        self.df['diff_wind_sensor_52'] = self.df['wind_speed_3_avg'] - self.df['sensor_52_avg']

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['diff_wind_sensor_18'], label='Difference: Wind Speed vs Sensor 18', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Difference')
        plt.title('Wind Speed Difference with Sensor 18 Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['diff_wind_sensor_52'], label='Difference: Wind Speed vs Sensor 52', color='green')
        plt.xlabel('Time Step')
        plt.ylabel('Difference')
        plt.title('Wind Speed Difference with Sensor 52 Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        self.logger.info(f"Columns retained and differences calculated: {cols_to_keep}")



    def keep_columns(self, columns_to_keep: str = "avg") -> "EDAPipeline":
        """
        Keep the first 5 columns (metadata) and any columns ending with the specified suffix.
        Discard all other columns.

        Parameters:
            columns_to_keep (str): The suffix of the columns to keep (default is "avg").

        Returns:
            Tuple[bool, EDAHandler]: Status (True/False) and the updated EDAHandler.
        """
        try:
            # Always keep the first 5 columns


            # Also keep columns (beyond index 4) that end with the specified suffix
            cols_to_keep = [col for col in self.sensor_df.columns if col.endswith(columns_to_keep)]

            # Filter the DataFrame in place
            self.sensor_df = self.sensor_df[cols_to_keep]

            # Log the operation
            self.logger.info(f"Columns retained: {cols_to_keep}")
            return self  # Return success status and updated handler

        except Exception as e:
            # Log the error and return False
            self.logger.error(f"Error in keep_columns: {e}")
            return self

    def generate_pdf_report(self) -> "EDAHandler":
        """
        Generate an EDA report using pandas-profiling.
        """

        try:
            # Create a profile report
            sensor_columns = self.df.columns[5:]
            self.sensor_df = self.df[sensor_columns]
            descriptions = {
                col: sensor.description
                for col, sensor in self.df.attrs.items()
                if isinstance(sensor, Sensor)
            }
            profile = ProfileReport(
                self.sensor_df,
                title=f"DataFrame Report for Event: {self.event_id}",
                tsmode=True,
                minimal=True,
                variables=descriptions
            )

            # Define output path
            output_path = Helper.get_pdf_file_location(dataset_id=self.event_id) + f"_report_{self.event_id}.pdf"

            # Export the report to PDF
            profile.to_file(output_path)

            self.logger.info(f"PDF report generated and saved at {output_path}.")

            return self  # Success

        except Exception as e:
            self.logger.error(f"Error generating PDF report: {e}")
            return self  # Failure

    def pdf_Report(self) -> Tuple[bool, "EDAHandler"]:
        """
        Generate a PDF report containing important statistics about the DataFrame, including:
        - Event information from metadata.
        - Amount of values in training and prediction datasets.
        - Binary anomaly attribute.
        - Summary of each column in a tabular format.

        Returns:
            Tuple[bool, EDAHandler]: Status (True/False) and the updated EDAHandler.
        """
        try:
            from fpdf import FPDF
            import pandas as pd

            output_path = Helper.get_pdf_file_location(dataset_id=self.event_id) + f"_report_{self.event_id}.pdf"
            event_id = self.event_id

            class PDF(FPDF):
                def __init__(self, orientation="L", unit="mm", format="A4"):
                    super().__init__(orientation, unit, format)

                def header(self):
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 10, f'DataFrame Report for Event: {event_id}', align='C', ln=True)
                    self.ln(10)

                def chapter_title(self, title):
                    self.set_font('Arial', 'B', 12)
                    self.cell(0, 10, title, ln=True)
                    self.ln(5)

                def chapter_body(self, body):
                    self.set_font('Arial', '', 10)
                    self.multi_cell(0, 10, body)
                    self.ln()

                def add_table(self, data):
                    self.set_font('Arial', '', 10)
                    col_widths = [50, 80, 40, 40, 40, 40]  # Adjust column widths
                    for row in data:
                        for idx, item in enumerate(row):
                            self.cell(col_widths[idx], 10, str(item), border=1)
                        self.ln()

            pdf = PDF()
            pdf.add_page()

            # Add event info from metadata
            event = self.df.attrs.get("event")
            pdf.chapter_title("Event Information")
            pdf.chapter_body(str(event))

            # Data split information
            train_data = self.df[self.df['train_test'] == False]
            prediction_data = self.df[self.df['train_test'] == True]

            # Count 'status_type_id' values
            train_status_counts = train_data['status_type_id'].value_counts()
            prediction_status_counts = prediction_data['status_type_id'].value_counts()

            pdf.chapter_title("Data Split Information")
            pdf.chapter_body(f"Training Data Count:\n{train_status_counts}\n\n"
                             f"Prediction Data Count:\n{prediction_status_counts}")

            # Add summary of every column
            pdf.chapter_title("Column Summary")
            column_summary = [["Column Name", "Description", "Mean", "Min", "Max", "Std Dev"]]

            for col in self.df.columns[5:]:
                # Retrieve description from Sensor objects or fallback to default
                sensor_metadata = self.df.attrs.get(col)
                if isinstance(sensor_metadata, Sensor):
                    description = sensor_metadata.description
                else:
                    description = 'No description'

                if pd.api.types.is_numeric_dtype(self.df[col]):
                    mean = round(self.df[col].mean(), 4)
                    min_val = round(self.df[col].min(), 4)
                    max_val = round(self.df[col].max(), 4)
                    std_dev = round(self.df[col].std(), 4)
                else:
                    mean = min_val = max_val = std_dev = 'N/A'

                column_summary.append([col, description, mean, min_val, max_val, std_dev])

            pdf.add_table(column_summary)

            # Save PDF
            pdf.output(output_path)
            self.logger.info(f"PDF report generated and saved at {output_path}.")

            return True, self  # Success

        except Exception as e:
            self.logger.error(f"Error in generate_pdf_report: {e}")
            return False, self  # Failure

    def split_and_analyze_data(self) -> "EDAHandler":
        """
        Split the DataFrame into training and prediction datasets based on the 'train_test' column.
        Print and log the counts of 'status_type_id' categories in both subsets.

        Returns:
            DatasetHandler: Returns self for method chaining.
        """
        # Split the DataFrame into train and prediction sets

        train_data = self.df[self.df['train_test'] == False]
        prediction_data = self.df[self.df['train_test'] == True]
        self.X_train = train_data.loc[5:]
        self.X_prediction = prediction_data.loc[5:]
        print(self.X_train)

        # Count and print 'status_type_id' values in train_data
        train_status_counts = train_data['status_type_id'].value_counts()
        print("Training Data Status Type Counts:")
        print(train_status_counts)
        self.logger.info(f"Training Data Status Type Counts:\n{train_status_counts}")

        # Count and print 'status_type_id' values in prediction_data
        prediction_status_counts = prediction_data['status_type_id'].value_counts()
        print("Prediction Data Status Type Counts:")
        print(prediction_status_counts)
        self.logger.info(f"Prediction Data Status Type Counts:\n{prediction_status_counts}")


        return self

    def plot(self, df) -> "EDAPipeline":

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        pdf_path = Helper.get_pdf_file_report(dataset_id=self.event_id)

        try:
            plt.ion()  # Enable interactive mode so it won't block on show()

            # 1) Get the Event object from attrs
            event = self.event
            # Build a title string with any relevant event details
            event_title = (
                f"Event {event.event_id}: {event.event_label} --- {event.event_description}\n"
                f"Start: {event.event_start}  |  End: {event.event_end}"
            )


            sensor_columns = df.columns
            batch_size = 5  # Adjust batch size as needed

            # Open the existing PDF to append plots
            with PdfPages(pdf_path) as pdf:
                for i in range(0, len(sensor_columns), batch_size):
                    batch_cols = sensor_columns[i: i + batch_size]

                    # Create a figure with len(batch_cols) subplots, stacked vertically
                    fig, axes = plt.subplots(
                        nrows=len(batch_cols),
                        ncols=1,
                        figsize=(20, 3 * len(batch_cols)),  # wide & tall enough for stacked subplots
                        sharex=True
                    )

                    # If there's only 1 subplot, make axes a list for uniform handling
                    if len(batch_cols) == 1:
                        axes = [axes]

                    # Loop through each column in this batch
                    # Loop through each column in this batch
                    # Assume:
                    #   - df is your *sliced* DataFrame containing only sensor columns (and the same index as self.df).
                    #   - self.df has columns like "time_stamp" and "train_test" that do not exist in df.
                    #   - event.event_start_id and event.event_end_id are valid indices in self.df (and thus in df.index).

                    # Loop through each column in this batch
                    for ax, col in zip(axes, batch_cols):
                        sensor = self.sensor_data[col]  # Retrieve sensor metadata

                        # -----------------------------------------------------
                        # 1. Plot the entire (sliced) sensor time series in blue
                        # -----------------------------------------------------
                        # X-axis: time_stamp from the original DataFrame, using df.index
                        # Y-axis: the sensor data from the sliced DataFrame
                        ax.plot(
                            self.df["time_stamp"].loc[df.index],
                            df[col],
                            linewidth=0.5,
                            color="blue",
                            label=col
                        )

                        # -----------------------------------------------------------------
                        # 2. Plot rows where train_test == True in green (within the slice)
                        # -----------------------------------------------------------------
                        # First, mask based on train_test in the original DF, but restricted to df.index
                        mask_prediction = self.df["train_test"].loc[df.index] == True

                        ax.plot(
                            self.df["time_stamp"].loc[df.index][mask_prediction],
                            df[col][mask_prediction],
                            color="green",
                            linewidth=0.5,
                            label=f"{sensor.description} (prediction)"
                        )

                        # ---------------------------------------------------------------------
                        # 3. Highlight the portion between event_start_id and event_end_id in red
                        # ---------------------------------------------------------------------
                        # Ensure the event indices are in the slice. We'll build a mask that says:
                        #  "Use rows from df.index that fall between event_start_id and event_end_id."
                        event_mask = (df.index >= event.event_start_id) & (df.index <= event.event_end_id)

                        if event_mask.any():
                            ax.plot(
                                self.df["time_stamp"].loc[df.index][event_mask],
                                df[col][event_mask],
                                color="red",
                                linewidth=0.5,
                                label=f"{col} (event)"
                            )

                        # 4. Set labels, grid, legend, etc.
                        ax.set_ylabel(f"{sensor.description}\n[{sensor.unit}]", fontsize=9)
                        ax.grid(True)
                        ax.legend(fontsize=8)

                    # 5. Title & X-axis label
                    axes[0].set_title(event_title, fontsize=10)
                    axes[-1].set_xlabel("Time Stamp")

                    plt.tight_layout()

                    # 6. Save the figure to the PDF
                    pdf.savefig(fig)
                    plt.close(fig)

            # Log success
            self.logger.info(f"Plots successfully appended to PDF: {pdf_path}")
            return self

        except Exception as e:
            self.logger.error(f"Error in plot method: {e}")
            return self

