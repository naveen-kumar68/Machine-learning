import pandas as pd
import numpy as np

def load_data(file):
    data=pd.read.csv(file)
    return data

def aqi_dataset_one_attri(data):
    X=data[["PM2.5"]]
    y=data["AQI"]
    return X,y

def aqi_dataset_all_attri(data):
    X=data[["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]]
    y=data["AQI"]
    return X,y

def k_means_(data):
    X=data[["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]]
    return X


