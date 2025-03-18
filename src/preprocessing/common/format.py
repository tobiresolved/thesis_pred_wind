import pandas as pd
from helpers.helpers import Helper
from helpers.utils import pipeline_step
from preprocessing.metadata.dataclasses import Event, Sensor
import pickle
from ydata_profiling import ProfileReport
import ydata_profiling
import json
from helpers.constants import dataframe_metadata


class FormatDataframe:

    def __init__(self, event_id: int, sensor_dict: dict):
        self.logger = Helper.get_logger("Format Dataframe")
        self.event_id = event_id


    def process(self, pdf_report: bool):
        df = self.read_dataset()
        df = self.update_column_names(df=df)
        #self.check_right_column_names(df=df)
        df = self.format_dataframe(df=df)
        if pdf_report:
            self.create_pdf_report(df=df)
        pickle_path = self.dump_dataframe(df=df)
        return pickle_path

    @pipeline_step(step_name="Read Dataset", logger="Format Dataframe")
    def read_dataset(self) -> pd.DataFrame:
        file_path = Helper.get_dataset_file_location(dataset_id=self.event_id)
        df = pd.read_csv(filepath_or_buffer=file_path, sep=";")
        return df

    @pipeline_step("Update Column Names", logger="Format Dataframe")
    def update_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        stat_types = ("std", "min", "max", "avg")
        updated_columns = {
            col: f"{col}_avg" if (not col.endswith(stat_types) and col.startswith("sensor")) else col
            for col in df.columns
        }
        df.rename(columns=updated_columns, inplace=True)
        return df

    @pipeline_step("Load Sensor Data", logger="Format Dataframe")
    def load_sensor_data_for_df(self, df: pd.DataFrame) -> dict:
        columns_to_use = df.columns[5:].tolist()
        sensor_metadata = Sensor.from_columns(columns=columns_to_use, event_id=self.event_id)
        return sensor_metadata

    def check_right_column_names(self, df: pd.DataFrame):

        columns_to_use = df.columns[5:].tolist()
        sensor_keys = list(self.sensor_metadata.keys())

        # Check if the number of columns matches the number of sensor keys
        if len(columns_to_use) != len(sensor_keys):
            raise ValueError(
                f"Column count mismatch: {len(columns_to_use)} columns in df (from index 5 onward) "
                f"vs {len(sensor_keys)} sensor keys."
            )

        # Check that each sensor key is present in the columns_to_use
        missing_keys = [key for key in sensor_keys if key not in columns_to_use]
        if missing_keys:
            raise ValueError(
                f"The following sensor keys are missing in the DataFrame columns: {missing_keys}"
            )

    @pipeline_step("Format DataFrame", logger="Format Dataframe")
    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate Memory Usage
        memory_bytes = df.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 ** 2)
        self.logger.info(f"DataFrame memory usage after format: {memory_mb:.2f} MB")

        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        df["train_test"] = df["train_test"].map({"train": False, "prediction": True})
        df["status_type_id"] = df["status_type_id"].astype("category")
        sensor_columns = df.columns[5:]
        df[sensor_columns] = df[sensor_columns].astype("float32")
        # Calculate Memory Usage Again
        memory_bytes = df.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 ** 2)
        self.logger.info(f"DataFrame memory usage after format: {memory_mb:.2f} MB")

        return df

    @pipeline_step("Create PDF-Report", logger="Format Dataframe")
    def create_pdf_report(self, df) -> None:
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

    @pipeline_step("Dump DataFrame", logger="Format Dataframe")
    def dump_dataframe(self, df: pd.DataFrame) -> str:

        df_pickle_location = Helper.get_dataframe_pickle_location(dataset_id=self.event_id, step="formatted")

        with open(df_pickle_location, "wb") as file:
            pickle.dump(df, file)

        self.logger.info(f"Dumped DataFrame to {df_pickle_location}")
        return df_pickle_location
