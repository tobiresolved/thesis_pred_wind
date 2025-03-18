import datetime
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import pickle
from helpers.helpers import Helper
import pandas as pd


@dataclass
class Dataset:
    time_stamp: pd.Series
    train_test: pd.Series
    status_type_id: pd.Series
    sensordata: pd.DataFrame

    @classmethod
    def get_from_pickle(cls, event_id: int) -> "Dataset":

        df_pickle_location = Helper.get_dataframe_pickle_location(dataset_id=event_id)

        # Load the DataFrame from the pickle file
        with open(df_pickle_location, "rb") as file:
            df = pickle.load(file)

        # Extract the required columns
        time_stamp = df["time_stamp"]
        train_test = df["train_test"]
        status_type_id = df["status_type_id"]

        # Assume sensor data is in all columns starting from the 6th column (i.e. index 5 onward)
        sensordata = df.iloc[:, 5:]

        return cls(
            time_stamp=time_stamp,
            train_test=train_test,
            status_type_id=status_type_id,
            sensordata=sensordata
        )


@dataclass(frozen=True)
class Event:
    event_id: int
    event_label: str
    event_start: datetime
    event_start_id: int
    event_end: datetime
    event_end_id: int
    event_description: Optional[str]
    is_anomaly: bool

    def __str__(self) -> str:
        return (
            f"Event("
            f"id={self.event_id}, "
            f"label={self.event_label!r}, "
            f"start={self.event_start.format()}, "
            f"start_id={self.event_start_id}, "
            f"end={self.event_end.format()}, "
            f"end_id={self.event_end_id}, "
            f"is_anomaly={self.is_anomaly}\n"
            f"description={self.event_description!r}"
            f")"
        )

    @classmethod
    def from_json(cls, event_id: int) -> "Event":
        wind_farm = Helper.wind_farm_for_dataset(dataset_id=event_id)
        event_base_json_path = os.getenv("EVENTS_BASE_JSON_PATH")
        event_json_path = event_base_json_path.format(wind_farm=wind_farm)

        with open(event_json_path, "r") as event_file:
            data = json.load(event_file)

        event_data = data.get(str(event_id))
        if not event_data:
            raise ValueError(f"Event ID {event_id} not found in metadata for wind farm {wind_farm}.")

        return cls(
            event_id=event_id,
            event_label=event_data["event_label"],
            event_start=event_data["event_start"],
            event_start_id=int(event_data["event_start_id"]),
            event_end=event_data["event_end"],
            event_end_id=int(event_data["event_end_id"]),
            event_description=event_data.get("event_description"),
            is_anomaly=event_data["is_anomaly"]
        )

    @classmethod
    def all_as_dict(cls, wind_farm: str) -> Dict[str, "Event"]:
        datasets = Helper.datasets_for_wind_farm(wind_farm=wind_farm)
        datasets_dict = {}
        for dataset in datasets:
            sensor_base_json_path = os.getenv("SENSORS_BASE_JSON_PATH")
            sensor_json_path = sensor_base_json_path.format(wind_farm=wind_farm)
            sensor_json_path = "res/metadata/events/WFB.json"
            with open(sensor_json_path, "r", encoding="utf-8") as sensor_file:
                data = json.load(sensor_file)

            for sensor_key, sensor_data in data.items():
                print(sensor_key)
                print(sensor_data)
                datasets_dict[sensor_key] = cls(
                    event_id=sensor_key,
                    event_label=sensor_data["event_label"],
                    event_start=sensor_data["event_start"],
                    event_start_id=int(sensor_data["event_start_id"]),
                    event_end=sensor_data["event_end"],
                    event_end_id=int(sensor_data["event_end_id"]),
                    event_description=sensor_data.get("event_description"),
                    is_anomaly=sensor_data["is_anomaly"]
                )
            return datasets_dict


@dataclass(frozen=True)
class Sensor:
    name: str
    description: str
    stat_type: str
    unit: str
    is_angle: bool
    is_counter: bool

    def __str__(self) -> str:
        return (
            f"description={self.description!r}, "
            f"unit={self.unit!r}, "
            f"is_angle={self.is_angle!r}, "
            f"is_counter={self.is_counter!r}"
        )

    @classmethod
    def from_json(cls, sensor_name: str, event_id: int) -> "Sensor":
        wind_farm = Helper.wind_farm_for_dataset(dataset_id=event_id)
        sensor_base_json_path = os.getenv("SENSORS_BASE_JSON_PATH")
        sensor_json_path = sensor_base_json_path.format(wind_farm=wind_farm)

        with open(sensor_json_path, "r", encoding="utf-8") as sensor_file:
            data = json.load(sensor_file)

        sensor_data = data.get(sensor_name)
        if not sensor_data:
            sensor_data = data.get(sensor_name + "_avg")

        return cls(
            name=sensor_data["name"],
            description=sensor_data["description"],
            stat_type=sensor_data["stat_type"],
            unit=sensor_data["unit"],
            is_angle=sensor_data["is_angle"],
            is_counter=sensor_data["is_counter"]
        )

    @classmethod
    def all_as_dict(cls, event_id: int) -> Dict[str, "Sensor"]:
        wind_farm = Helper.wind_farm_for_dataset(dataset_id=event_id)
        sensor_base_json_path = os.getenv("SENSORS_BASE_JSON_PATH")
        sensor_json_path = sensor_base_json_path.format(wind_farm=wind_farm)

        with open(sensor_json_path, "r", encoding="utf-8") as sensor_file:
            data = json.load(sensor_file)

        sensors = {}
        for sensor_key, sensor_data in data.items():
            sensors[sensor_key] = cls(
                name=sensor_data["name"],
                description=sensor_data["description"],
                stat_type=sensor_data["stat_type"],
                unit=sensor_data["unit"],
                is_angle=sensor_data["is_angle"],
                is_counter=sensor_data["is_counter"]
            )
        return sensors

    @classmethod
    def all_as_dict_wind_farm(cls, wind_farm: str) -> Dict[str, "Sensor"]:
        sensor_base_json_path = os.getenv("SENSORS_BASE_JSON_PATH")
        sensor_json_path = sensor_base_json_path.format(wind_farm=wind_farm)

        with open(sensor_json_path, "r", encoding="utf-8") as sensor_file:
            data = json.load(sensor_file)

        sensors = {}
        for sensor_key, sensor_data in data.items():
            sensors[sensor_key] = cls(
                name=sensor_data["name"],
                description=sensor_data["description"],
                stat_type=sensor_data["stat_type"],
                unit=sensor_data["unit"],
                is_angle=sensor_data["is_angle"],
                is_counter=sensor_data["is_counter"]
            )
        return sensors

    @classmethod
    def from_columns(cls, columns: list[str], event_id: int) -> dict[str, "Sensor"]:
        sensors = {}
        for col in columns:
            try:
                sensors[col] = cls.from_json(col, event_id)
            except ValueError:
                # If the sensor is not defined in JSON, decide how to handle:
                # - skip silently
                # - or raise again
                # For now, we'll skip.
                pass
        return sensors
