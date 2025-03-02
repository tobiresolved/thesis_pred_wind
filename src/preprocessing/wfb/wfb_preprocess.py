from typing import Tuple
import pandas as pd
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


class WFBPreprocess(Preprocess):

    def __init__(self, event_id: int):
        super().__init__(event_id)

    @pipeline_step("Process", logger="WFA_PREPROCESS")
    def process(self, df_path: str):
        df = self.load_dataframe(df_path=df_path)
        df = self.filter_status_rows(df=df)
        df = self.smooth_sensor_columns(df=df, window=6, method="mean")
        dp = DataPlotter(event=self.event)
        dp.plot_scatter(df=df)
