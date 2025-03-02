import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.metadata.dataclasses import Event, Sensor
from helpers.utils import pipeline_step
from helpers.helpers import Helper
from helpers.constants import dataframe_metadata
from ydata_profiling import ProfileReport
import matplotlib.patches as mpatches

class DataReport:
    @staticmethod
    def create_pdf_report(self, df) -> "BasePipeline":
        file_location = Helper.get_pdf_report_location(dataset_id=self.event_id)

        definitions = dataframe_metadata
        sensor_descriptions = {
            sensor_key: sensor_obj
            for sensor_key, sensor_obj in self.sensor_metadata.items()
        }

        definitions.update(sensor_descriptions)

        profile = ProfileReport(
            df,
            title="Short Report",
            minimal=True,
            explorative=False,
            tsmode=True,
            sortby="time_stamp",
            variables={"descriptions": definitions}
        )
        profile.to_file(file_location)

        self.logger.info(f"PDF report created at: {file_location}")
        return self

    @staticmethod
    def compare(self, first_df, second_df) -> "BasePipeline":

        file_location = Helper.get_pdf_comparison_location(dataset_id=self.event_id)

        self.df_train = self.df[self.df["train_test"] == False].copy()

        definitions = dataframe_metadata
        sensor_descriptions = {
            sensor_key: sensor_obj
            for sensor_key, sensor_obj in self.sensor_metadata.items()
        }

        definitions.update(sensor_descriptions)

        profile_train = ProfileReport(
            first_df,
            title="Train Report",
            minimal=True,
            explorative=False,
            tsmode=True,
            # type_schema=type_schema,
            sortby="time_stamp",
            variables={"descriptions": definitions}
        )

        profile_prediction = ProfileReport(
            second_df,
            title=f"Prediction Report for {str(self.event)}",
            minimal=True,
            explorative=False,
            tsmode=True,
            # type_schema=type_schema,
            sortby="time_stamp",
            variables={"descriptions": definitions}
        )

        comparison_report = profile_train.compare(profile_prediction)
        comparison_report.to_file(file_location)
        return self



class DataPlotter:

    def __init__(self, event: Event):
        self.event = event
        self.wind_farm = Helper.wind_farm_for_dataset(dataset_id=self.event.event_id)
        self.sensor_metadata = Sensor.all_as_dict(event_id=event.event_id)

    def plot_two_sensor_correlations(self, df: pd.DataFrame, sensor_x: str, sensor_y1: str, sensor_y2: str) -> None:
        # Create two side-by-side subplots with a shared x-axis
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharex=True)

        # Plot the first sensor pair on the left subplot
        axes[0].scatter(df[sensor_x], df[sensor_y1], color="blue", alpha=0.6, edgecolors="w", linewidth=0.5, s=2)
        axes[0].set_xlabel(f"{sensor_x} {self.sensor_metadata[sensor_x].description}")
        axes[0].set_ylabel(f"{sensor_y1} {self.sensor_metadata[sensor_y1].description}")
        axes[0].set_title(f"Wind Farm {self.wind_farm}: {sensor_y1} vs {sensor_x}")
        axes[0].grid(True)

        # Plot the second sensor pair on the right subplot
        axes[1].scatter(df[sensor_x], df[sensor_y2], color="blue", alpha=0.6, edgecolors="w", linewidth=0.5, s=2)
        axes[1].set_xlabel(f"{sensor_x} {self.sensor_metadata[sensor_x].description}")
        axes[1].set_ylabel(f"{sensor_y2} {self.sensor_metadata[sensor_y2].description}")
        axes[1].set_title(f"Wind Farm {self.wind_farm}: {sensor_y2} vs {sensor_x}")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_sensor_correlation_3d(self, df: pd.DataFrame, sensor_x: str, sensor_y: str, sensor_z: str) -> None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Create a 3D scatter plot with smaller points (s controls the marker size)
        ax.scatter(df[sensor_x], df[sensor_y], df[sensor_z],
                   color="blue", alpha=0.6, edgecolors="w", linewidth=0.5, s=2)

        # Set axis labels and title
        ax.set_xlabel(f"{sensor_x} {self.sensor_metadata[sensor_x].description}")
        ax.set_ylabel(f"{sensor_y} {self.sensor_metadata[sensor_y].description}")
        ax.set_zlabel(f"{sensor_z} {self.sensor_metadata[sensor_z].description}")
        ax.set_title(f"3D Scatter Wind Farm {self.wind_farm}: {sensor_z} vs {sensor_x} vs {sensor_y}")

        plt.show()

    def plot_sensor_correlation(self, df: pd.DataFrame, sensor_x: str, sensor_y: str) -> None:
        plt.figure(figsize=(10, 6))

        # Create a scatter plot with sensor_x on the x-axis and sensor_y on the y-axis.
        plt.scatter(df[sensor_x], df[sensor_y], color="blue", alpha=0.6, edgecolors="w", linewidth=0.5, s=2)

        # Label the axes and add a title
        plt.xlabel(f"{sensor_x} {self.sensor_metadata[sensor_x].description}")
        plt.ylabel(f"{sensor_y} {self.sensor_metadata[sensor_y].description}")
        plt.title(f"Wind Farm {self.wind_farm}: {sensor_y} vs {sensor_x}")
        plt.grid(True)

        plt.show()

    def plot_scatter(self, df):
        sensor_mapping = Helper.get_scatter_plot_sensors_direction(self.event.event_id)
        for sensor_x, sensor_y in sensor_mapping:
            self.plot_sensor_correlation(df, sensor_x, sensor_y)

        #sensor_mapping = Helper.get_scatter_plot_sensors(self.event.event_id)
        #for sensor_x, sensor_y in sensor_mapping:
        #    self.plot_sensor_correlation(df, sensor_x, sensor_y)

    def plot_scatter_3d(self, df):
        sensor_mapping = Helper.get_scatter_plot_sensors_3d_rotor(self.event.event_id)
        for sensor_x, sensor_y, sensor_z in sensor_mapping:
            self.plot_sensor_correlation_3d(df, sensor_x, sensor_y, sensor_z)

    def plot_sensor_correlation_status_type(self, df: pd.DataFrame, sensor_x: str, sensor_y: str) -> None:
        plt.figure(figsize=(15, 10))

        # Define a new color mapping with more vibrant, high-contrast colors
        colors = {
            0: "dodgerblue",  # Normal
            1: "crimson",  # Derated
            2: "limegreen",  # Idling
            3: "gold",  # Service
            4: "mediumorchid",  # Down
            5: "turquoise"  # Other
        }

        # Map status_type_id to custom labels
        status_labels = {
            0: "Normal",
            1: "Derated",
            2: "Idling",
            3: "Service",
            4: "Down",
            5: "Other"
        }

        # Loop through each unique status and plot its points with the designated color
        for status, color in colors.items():
            subset = df[df["status_type_id"] == status]
            plt.scatter(
                subset[sensor_x],
                subset[sensor_y],
                color=color,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                s=10,
                label=status_labels[status]
            )

        # Label the axes and add a title
        plt.xlabel(f"{sensor_x} {self.sensor_metadata[sensor_x].description} {self.sensor_metadata[sensor_x].unit}")
        plt.ylabel(f"{sensor_y} {self.sensor_metadata[sensor_y].description} {self.sensor_metadata[sensor_y].unit}")
        plt.title(f"Wind Farm {self.wind_farm}: {sensor_y} vs {sensor_x}")
        plt.grid(True)
        plt.legend(title="Status")
        plt.show()

    def plot_sensor_correlation_by_status_loop(self, df: pd.DataFrame, sensor_x: str, sensor_y: str) -> None:
        # Define a new color mapping with more vibrant, high-contrast colors
        colors = {
            0: "dodgerblue",  # Normal
            1: "crimson",  # Derated
            2: "limegreen",  # Idling
            3: "gold",  # Service
            4: "mediumorchid",  # Down
            5: "turquoise"  # Other
        }

        # Map status_type_id to custom labels
        status_labels = {
            0: "Normal",
            1: "Derated",
            2: "Idling",
            3: "Service",
            4: "Down",
            5: "Other"
        }

        for status, color in colors.items():
            subset = df[df["status_type_id"] == status]
            if subset.empty:
                continue  # Skip plotting if no data for this status

            plt.figure(figsize=(15, 10))
            plt.scatter(
                subset[sensor_x],
                subset[sensor_y],
                color=color,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                s=10
            )

            # Label the axes and add a title that includes the status label
            plt.xlabel(f"{sensor_x} {self.sensor_metadata[sensor_x].description} {self.sensor_metadata[sensor_x].unit}")
            plt.ylabel(f"{sensor_y} {self.sensor_metadata[sensor_y].description} {self.sensor_metadata[sensor_y].unit}")
            plt.title(f"Wind Farm {self.wind_farm}: {sensor_y} vs {sensor_x} - {status_labels[status]}")
            plt.grid(True)
            plt.show()

    def plot_timeseries(
            self,
            sensor_series: pd.Series,
            sensor: Sensor
    ) -> None:

        plt.figure(figsize=(15, 6))

        title = f"{self.event.event_label.upper()} - {self.event.event_description}" or f"Event {self.event.event_id}"
        plt.title(title)

        # 1) Plot the entire time series in blue
        plt.plot(sensor_series, label="Sensor Data", color="blue", linewidth=0.5)

        # 2) Highlight the event window in red
        start_idx = self.event.event_start_id
        end_idx = self.event.event_end_id

        # We assume start_idx and end_idx are valid integer positions
        # within the series. We slice using .iloc
        plt.plot(
            sensor_series.iloc[start_idx: end_idx + 1],
            label="Event Window",
            color="red",
            linewidth=0.5
        )

        # Add labels and legend
        plt.xlabel(f"Plot of {sensor.name} {sensor.description.upper()} over time")
        plt.ylabel(f"{sensor.description} - {sensor.stat_type} - {sensor.unit}")
        plt.legend()

        # Display the plot
        plt.show()
