import csv
import json
import os
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STAT_TYPE_MAPPING: Dict[str, str] = {
    "std_dev": "std",
    "minimum": "min",
    "maximum": "max",
    "average": "avg"
}


def csv_to_dict(csv_file_path: str, delimiter: str = ";") -> List[Dict[str, str]]:
    """Reads a CSV file and returns a list of dictionaries representing each row."""
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv_reader]


def events_csv_to_json(csv_file_path: str, json_file_path: str, key_field: str = "event_id") -> None:
    data_dict = {}
    rows = csv_to_dict(csv_file_path)

    for row_number, row in enumerate(rows, start=1):
        if key_field not in row:
            raise ValueError(f"Row {row_number} is missing the key field '{key_field}'.")

        key = row.pop(key_field)
        if key in data_dict:
            raise ValueError(f"Duplicate key '{key}' found at row {row_number}.")

        row["is_anomaly"] = row.get("event_label") == "anomaly"

        data_dict[key] = row

    with open(json_file_path, mode="w", encoding="utf-8") as json_file:
        json.dump(data_dict, json_file, indent=4, ensure_ascii=False)

    logger.info(f"Converted '{csv_file_path}' to '{json_file_path}' with {len(data_dict)} events.")


def sensors_csv_to_json(
    csv_file_path: str,
    json_file_path: str,
    stat_type_mapping: Dict[str, str] = STAT_TYPE_MAPPING
) -> None:

    sensor_dict = {}
    rows = csv_to_dict(csv_file_path)

    for row_number, row in enumerate(rows, start=1):
        sensor_name = row.get("sensor_name", "")
        statistics_type = row.get("statistics_type", "")
        description = row.get("description", "")
        unit = row.get("unit", "")
        is_angle = row.get("is_angle", "False").lower() == "true"
        is_counter = row.get("is_counter", "False").lower() == "true"

        if not sensor_name:
            raise ValueError(f"Row {row_number}: 'sensor_name' is missing.")

        statistics_types = [stat.strip() for stat in statistics_type.split(",") if stat.strip()]

        for stat_type in statistics_types:
            formatted_stat_type = stat_type_mapping.get(stat_type, stat_type)
            sensor_key = f"{sensor_name}_{formatted_stat_type}" if formatted_stat_type else sensor_name

            if sensor_key in sensor_dict:
                raise ValueError(f"Row {row_number}: Duplicate sensor key '{sensor_key}' found.")

            sensor_dict[sensor_key] = {
                "name": sensor_key,
                "description": description,
                "stat_type": formatted_stat_type,
                "unit": unit,
                "is_angle": is_angle,
                "is_counter": is_counter
            }

    with open(json_file_path, mode="w", encoding="utf-8") as json_file:
        json.dump(sensor_dict, json_file, indent=4, ensure_ascii=False)

    logger.info(f"Converted '{csv_file_path}' to '{json_file_path}' with {len(sensor_dict)} sensors.")


def set_config():
    wind_farms = ["A", "B", "C"]
    for wind_farm in wind_farms:
        logger.info("----------")
        logger.info(f"Start the Event and Sensor Conversion for Wind Farm: {wind_farm}")
        try:
            event_info_csv_path = os.getenv("EVENT_INFO_BASE_CSV_PATH", "").format(wind_farm=wind_farm)
            event_info_json_path = os.getenv("EVENTS_BASE_JSON_PATH", "").format(wind_farm=wind_farm)

            feature_description_csv_path = os.getenv("FEATURE_DESCRIPTION_BASE_CSV_PATH", "").format(wind_farm=wind_farm)
            feature_description_json_path = os.getenv("SENSORS_BASE_JSON_PATH", "").format(wind_farm=wind_farm)

            events_csv_to_json(event_info_csv_path, event_info_json_path)
            logger.info(f"Event metadata JSON for {wind_farm} saved to {event_info_json_path}")

            sensors_csv_to_json(feature_description_csv_path, feature_description_json_path)
            logger.info(f"Sensor data JSON for {wind_farm} saved to {feature_description_json_path}")

        except Exception as e:
            logger.error(f"Failed to process data for {wind_farm}: {e}")


if __name__ == "__main__":
    set_config()
