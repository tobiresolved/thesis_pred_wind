import datetime
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from helpers.helpers import Helper


@dataclass(frozen=True)
class Event:
    event_id: int
    event_label: str
    event_start: datetime
    event_start_id: int
    event_end: datetime
    event_end_id: int
    event_description: Optional[str]

    @classmethod
    def from_json(cls, event_id: int) -> "Event":
        wind_farm = Helper.wind_farm_for_dataset(dataset_id=event_id)
        event_base_json_path = os.getenv("EVENT_BASE_PATH_JSON")
        event_json_path = event_base_json_path.format(wind_farm=wind_farm)

        with open(event_json_path, "r") as event_file:
            data = json.load(event_file)

        event_data = data.get(event_id)
        if not event_data:
            raise ValueError(f"Event ID {event_id} not found in metadata for wind farm {wind_farm}.")

        return cls(
            event_id=event_data["event_id"],
            event_label=event_data["event_label"],
            event_start=event_data["event_start"],
            event_start_id=event_data["event_start_id"],
            event_end=event_data["event_end"],
            event_end_id=event_data["event_end_id"],
            event_description=event_data.get("event_description")
        )


@dataclass(frozen=True)
class Sensor:
    name: str
    description: str
    stat_type: str
    unit: str
    is_angle: bool
    is_counter: bool

    @classmethod
    def from_json(cls, sensor_name: str, event_id: int) -> "Sensor":
        wind_farm = Helper.wind_farm_for_dataset(dataset_id=event_id)
        sensor_base_json_path = os.getenv("SENSOR_BASE_PATH_JSON")
        sensor_json_path = sensor_base_json_path.format(wind_farm=wind_farm)

        with open(sensor_json_path, "r", encoding="utf-8") as sensor_file:
            data = json.load(sensor_file)

        sensor_data = data.get(sensor_name)
        if not sensor_data:
            raise KeyError(f"Sensor '{sensor_name}' not found in JSON file.")

        return cls(
            name=sensor_data["name"],
            description=sensor_data["description"],
            stat_type=sensor_data["stat_type"],
            unit=sensor_data["unit"],
            is_angle=sensor_data["is_angle"],
            is_counter=sensor_data["is_counter"],
        )
