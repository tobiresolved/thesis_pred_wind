import pandas as pd

from helpers.helpers import Helper
from preprocessing.metadata.dataclasses import Event, Sensor
from preprocessing.wfa.wfa_preprocess import WFAPreprocess
from models.random_forest.random_forest_v2 import SensorRandomForest_V2
from models.ann.neural_net import SensorNeuralNet, SensorNeuralNet_V2, SensorNeuralNet_V2_AR
from preprocessing.wfa.const import INPUT_ENVIRONMENTAL, OUTPUT_MECHANICAL, OUTPUT_TEMPERATURE, INPUT_ENVIRONMENTAL_PLUS_MECHANICAL, INPUT_WFB, TEMPERATURE_WFB
from models.gru.gru import SensorNeuralNet_GRU
from models.lstm.lstm import LSTM_Model
from models.random_forest.random_forest_v2 import RandomForest
import pickle
from preprocessing.common.format import FormatDataframe

class WFA:

    def __init__(self):
        self.event_ids = [0, 3, 10, 13, 14, 17, 22, 24, 25, 26, 38, 40, 42, 45, 51, 68, 69, 71, 72, 73, 84, 92]
        self.event_metadata = Event.all_as_dict(wind_farm="B")
        self.sensor_metadata = Sensor.all_as_dict_wind_farm(wind_farm="B")

    def create_normal_data(self, training_ids: list=None) -> list[pd.DataFrame]:
        normal_train_dfs = []
        for event_id in training_ids:
            #format_df = FormatDataframe(event_id=event_id, sensor_dict=self.sensor_metadata)
            #pickle_path = format_df.process(pdf_report=False)
            #pickle_path = Helper.get_dataframe_pickle_location(dataset_id=event_id, step="formatted")
            #wfa_preprocess = WFAPreprocess(event_id=event_id)
            #df_train = wfa_preprocess.create_normal_data(df_path=pickle_path)
            pickle_path = f"res/normal_data/WFA/{event_id}.pkl"
            with open(pickle_path, "rb") as file:
                df_train = pickle.load(file=file)
                normal_train_dfs.append(df_train)
        return normal_train_dfs

    def make_predictions(self, model, predictions_ids: list=None):
        for event_id in predictions_ids:
            pickle_path = Helper.get_dataframe_pickle_location(dataset_id=event_id, step="formatted")
            wfa_preprocess = WFAPreprocess(event_id=event_id)
            df_validation, df_test = wfa_preprocess.create_validation_test_data(df_path=pickle_path,
                                                                                validation_frame=52991)

            prediction_temp = model.predict(df_validation)
            model.save_prediction_statistics(actual_df=df_validation, predicted_df=prediction_temp, event=self.event_metadata[str(event_id)], output_file="res/model_results/lstm/lstm.csv")
            print(self.event_metadata[str(event_id)])
            model.plot_individual_sensor_plots(actual_df=df_validation, predicted_df=prediction_temp, sensor_info=self.sensor_metadata, event=self.event_metadata[str(event_id)],
                           outside_temp_col="sensor_8_avg", wind_speed_col="wind_speed_61_avg", rotor_rpm_col="sensor_25_avg", status_col="status_type_id", columns = ["sensor_51_avg", "sensor_52_avg"])

    def lstm(self):
        anomaly_test = [68, 22, 72, 73, 0, 26, 40, 42, 10, 45, 84]
        normal_test = [71, 14, 92, 51]
        normal_train = [25, 69, 13, 24, 3, 17, 38]
        # wfb
        normal_train = [82, 86, 87]
        test_ids = [34, 7, 19]
        validation_dfs = {}
        test_dfs = {}
        normal_dfs = {}
        anomaly_dfs = {}
        test_ids = anomaly_test + normal_test
        test_ids = [34, 7, 19]
        test_ids = [ 27, 53, 77, 19, 21, 7, 34]
        for test_id in test_ids:
            fd = FormatDataframe(event_id=test_id, sensor_dict=None)
            fd.process(pdf_report=False)
        normal_train_dfs = self.create_normal_data(training_ids=normal_train)
        #lstm = LSTM_Model(input_columns=INPUT_WFB, output_columns=TEMPERATURE_WFB)
        #lstm.train(normal_train_dfs)
        #lstm.save_model(folder="res/models/lstm")
        lstm = LSTM_Model.load_model(folder="res/models/lstm", input_columns=INPUT_WFB, output_columns=TEMPERATURE_WFB)

        self.make_predictions(model=lstm, predictions_ids=test_ids)



    def format_dfs(self):
        anomaly_test = [68, 22, 72, 73, 0, 26, 40, 42, 10, 45, 84]
        normal_test = [71, 14, 92, 51]
        normal_train = [25, 69, 13, 24, 3, 17, 38]
        train_dfs = {}
        validation_dfs = {}
        test_dfs = {}
        normal_dfs = {}
        anomaly_dfs = {}
        test_ids = anomaly_test + normal_test

        for event_id in normal_train:
            # Uncomment and use this if you use FormatDataframe:
            format_df = FormatDataframe(event_id=event_id, sensor_dict=self.sensor_metadata)
            pickle_path = format_df.process(pdf_report=False)
            pickle_path = Helper.get_dataframe_pickle_location(dataset_id=event_id, step="formatted")
            wfa_preprocess = WFAPreprocess(event_id=event_id)
            df_train = wfa_preprocess.create_normal_data(df_path=pickle_path)
            train_dfs[event_id] = df_train

        normal_train_dfs = [train_dfs[event_id] for event_id in train_dfs]
        nn1 = SensorNeuralNet(input_columns=INPUT_ENVIRONMENTAL_PLUS_MECHANICAL, output_columns=OUTPUT_TEMPERATURE)
        #nn1.train(normal_train_dfs)
        nn1.save_model(path="src/models/ann/model")

        for event_id in test_ids:
            pickle_path = Helper.get_dataframe_pickle_location(dataset_id=event_id, step="formatted")
            wfa_preprocess = WFAPreprocess(event_id=event_id)
            df_normal = wfa_preprocess.create_normal_data(df_path=pickle_path)
            df_anomaly = wfa_preprocess.create_anomaly_data(df_path=pickle_path)
            normal_dfs[event_id] = df_normal
            anomaly_dfs[event_id] = df_anomaly
            df_validation, df_test = wfa_preprocess.create_validation_test_data(df_path=pickle_path, validation_frame=1000)
            validation_dfs[event_id] = df_validation
            test_dfs[event_id] = df_test

            #nn1 = SensorNeuralNet(input_columns=INPUT_ENVIRONMENTAL, output_columns=OUTPUT_MECHANICAL)
            #nn1.load_model(path="src/models/ann/model")
            prediction_temp = nn1.predict(test_dfs[event_id])
            nn1.plot_actual_vs_predicted_with_difference(actual_df=test_dfs[event_id], predicted_df=prediction_temp, sensor_info=self.sensor_metadata)
            nn1.save_prediction_statistics(predicted_df=prediction_temp, actual_df=normal_dfs[event_id],
                                           output_file=f"res/test_statistics/nn/normal/{event_id}_normal.csv")
            prediction_temp = nn1.predict(anomaly_dfs[event_id])
            nn1.save_prediction_statistics(predicted_df=prediction_temp, actual_df=anomaly_dfs[event_id],
                                           output_file=f"res/test_statistics/nn/normal/{event_id}_anomaly.csv")

    def format_dfs_2(self):
        anomaly_test = [68, 22, 72, 73, 0, 26, 40, 42, 10, 45, 84]
        normal_test = [71, 27, 14, 92, 51]
        normal_train = [25, 69, 13, 24, 3, 17, 38]
        train_dfs = {}
        validation_dfs = {}
        test_dfs = {}
        normal_dfs = {}
        anomaly_dfs = {}
        test_ids = normal_test + anomaly_test
        for event_id in normal_train:
            # Uncomment and use this if you use FormatDataframe:
            # format_df = FormatDataframe(event_id=event_id, sensor_dict=self.sensor_metadata)
            # pickle_path = format_df.process(pdf_report=False)
            pickle_path = Helper.get_dataframe_pickle_location(dataset_id=event_id, step="formatted")
            wfa_preprocess = WFAPreprocess(event_id=event_id)
            df_train = wfa_preprocess.create_normal_data(df_path=pickle_path)
            train_dfs[event_id] = df_train

        normal_train_dfs = [train_dfs[event_id] for event_id in train_dfs]
        nn1 = SensorNeuralNet_V2_AR(input_columns=INPUT_ENVIRONMENTAL_PLUS_MECHANICAL, output_columns=OUTPUT_TEMPERATURE, window_output_len=1)
        nn1.train(normal_train_dfs)

        for event_id in test_ids:
            pickle_path = Helper.get_dataframe_pickle_location(dataset_id=event_id, step="formatted")
            wfa_preprocess = WFAPreprocess(event_id=event_id)
            df_normal = wfa_preprocess.create_normal_data(df_path=pickle_path)
            df_anomaly = wfa_preprocess.create_anomaly_data(df_path=pickle_path)
            normal_dfs[event_id] = df_normal
            anomaly_dfs[event_id] = df_anomaly
            df_validation, df_test = wfa_preprocess.create_validation_test_data(df_path=pickle_path, validation_frame=1000)
            validation_dfs[event_id] = df_validation
            test_dfs[event_id] = df_test

            #nn1 = SensorNeuralNet(input_columns=INPUT_ENVIRONMENTAL, output_columns=OUTPUT_MECHANICAL)
            #nn1.load_model(path="src/models/ann/model")
            prediction_temp, indices = nn1.predict(normal_dfs[event_id])
            nn1.save_prediction_statistics(predicted_df=prediction_temp, actual_df=normal_dfs[event_id],
                                           output_file=f"res/test_statistics/nn/normal_v2/{event_id}_normal.csv")
            prediction_temp, indices = nn1.predict(anomaly_dfs[event_id])
            nn1.save_prediction_statistics(predicted_df=prediction_temp, actual_df=anomaly_dfs[event_id],
                                           output_file=f"res/test_statistics/nn/normal_v2/{event_id}_anomaly.csv")

            nn1.plot_actual_vs_predicted_with_difference()


        """""
        for event_id in anomaly_test:
            pickle_path = Helper.get_dataframe_pickle_location(dataset_id=event_id, step="formatted")
            wfa_preprocess = WFAPreprocess(event_id=event_id)
            df_validation, df_test = wfa_preprocess.create_validation_test_data(df_path=pickle_path,
                                                                                validation_frame=1000)
            validation_dfs[event_id] = df_validation
            test_dfs[event_id] = df_test

            #nn1 = SensorNeuralNet(input_columns=INPUT_ENVIRONMENTAL, output_columns=OUTPUT_MECHANICAL)
            #nn1.load_model(path="src/models/ann/model/")
            prediction_temp = nn1.predict(test_dfs[event_id])
            nn1.save_prediction_statistics(predicted_df=prediction_temp, actual_df=test_dfs[event_id],
                                           output_file=f"res/test_statistics/nn/anomaly/{event_id}.csv")
        """""





        #prediction_temp = nn1.predict(test_dfs[10])
        #nn1.plot_actual_vs_predicted_with_difference(predicted_df=prediction_temp, actual_df=test_dfs[10], sensor_info=self.sensor_metadata)
        #nn1.save_prediction_statistics(predicted_df=prediction_temp, actual_df=test_dfs[10], output_file="res/test_statistics/nn/10.csv")
        #rf1 = SensorRandomForest_V2(input_columns=INPUT_ENVIRONMENTAL, output_columns=OUTPUT_MECHANICAL)
        #rf1.train(normal_train_dfs)
        #prediction_mech = rf1.predict(test_dfs[45])
        #rf1.plot_actual_vs_predicted_with_difference(predicted_df=prediction_mech, actual_df=test_dfs[45], sensor_info=self.sensor_metadata)
        #rf2 = SensorRandomForest_V2(input_columns=INPUT_ENVIRONMENTAL, output_columns=OUTPUT_TEMPERATURE)
        #rf2.train(normal_train_dfs)
        #prediction_temp = rf2.predict(test_dfs[45])
        #rf2.plot_actual_vs_predicted_with_difference(predicted_df=prediction_temp, actual_df=test_dfs[45], sensor_info=self.sensor_metadata)




