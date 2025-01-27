import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from helpers.helpers import Helper
from preprocessing.data_handling.df_operations import DFOperations
from preprocessing.data_models.dataclasses import Event, Sensor
from visualization.plotter import Plotter
from logging import getLogger


class DatasetHandler:

    def __init__(self, event_id: int):
        self.logger = getLogger()
        self.event_id = event_id
        self.df = self.read_dataset()

    def __repr__(self):
        msg = (f"{self.__class__.__name__}("
               f"event_id={self.event_id}, "
               f"df_rows={len(self.df)}")
        self.logger.info(msg=msg)

    def read_dataset(self) -> pd.DataFrame:
        file_path = Helper.get_dataset_file_location(dataset_id=self.event_id)
        return pd.read_csv(filepath_or_buffer=file_path, sep=";")

    def format_df(self) -> None:
        self.df['time_stamp'] = pd.to_datetime(self.df['time_stamp'])
        self.df['train_test'] = self.df['train_test'].map({'train': False, 'prediction': True})
        self.df['status_type_id'] = self.df['status_type_id'].astype('category')
        sensor_columns = self.df.columns[5:]
        self.df[sensor_columns] = self.df[sensor_columns].astype('float32')
        self.logger.info(self.df.size)

    def dump_df(self) -> None:
        output_path = Helper.get_pickle_location(dataset_id=self.event_id)
        self.df.to_pickle(output_path)

    def add_attrs_to_df(self) -> None:
        event = Event.from_json(event_id=self.event_id)
        self.df.attrs["event"] = event
        for sensor_column in self.df.columns[5:]:
            self.df.attrs[sensor_column] = Sensor.from_json(sensor_name=sensor_column, event_id=self.event_id)

