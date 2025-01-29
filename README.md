# thesis_pred_wind
Implementation of Algorithms for Predictive Maintenance of Wind Power Systems

The project is split up into different functional pipelines

1) Project Setup: 
After importing the dataset, this script will extract the information from the events and sensors and save them into JSON Files. 
Those files are modeled with dataclasses to access the attributes as needed when plotting or doing operations later.

2) EDA-Pipeline:
This Pipeline acts as a basic Exploratory Data Analysis (EDA) Pipeline of the datasets. 
The goal here is to format the data and get an overview. The dataset is into a pandas-df from the csv file and then formatted to save memory. 
Later the metadata from 1) is added to the df and its dumped to a pickle file. In the end two methods will be applied,
which first generate a pdf report of the dataset with all the metadata for the Event, as well as describing the sensors and then plot the sensor data for an overview.

3) MLOps-Pipeline:
In this pipeline, the preprocessing of the data happens, which will later feed into the ML Models.

4) Model-Pipeline:
Here the training of the models happens.