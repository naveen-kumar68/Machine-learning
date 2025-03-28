import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score 
from sklearn.metrics import davies_bouldin_score 
from preprocessing import *
from linear_reg import *
from knn_model import *

data = pd.read_csv(r"C:\Users\navee\Documents\AQI_Bangalore.csv")

X,y=aqi_dataset_one_attri(data)

print(data.info(),data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

#### A1,A2

accuracy,rmse,mse,mape,r_2_score,rmse_train,mse_train,mape_train,r_2_score_train=train_LR(X_train, X_test, y_train, y_test)

print("Accuracy: ",accuracy)
print()
print("For Test Data")

print("RMSE:",rmse)
print("MSE:",mse)
print("MAPE:",mape)
print("R_2_Score:",r_2_score)

print()
print("For Train Data")

print("RMSE for train:",rmse_train)
print("MSE for train :",mse_train)
print("MAPE for train :",mape_train)
print("R_2_Score for train :",r_2_score_train)

#####  A3

X,y=aqi_dataset_all_attri(data)

print(data.info(),data.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

accuracy,rmse,mse,mape,r_2_score,rmse_train,mse_train,mape_train,r_2_score_train=train_LR(X_train, X_test, y_train, y_test)

print("Accuracy: ",accuracy)
print()
print("For Test Data")

print("RMSE:",rmse)
print("MSE:",mse)
print("MAPE:",mape)
print("R_2_Score:",r_2_score)

print()
print("For Train Data")

print("RMSE for train:",rmse_train)
print("MSE for train :",mse_train)
print("MAPE for train :",mape_train)
print("R_2_Score for train :",r_2_score_train)

## A4,A5

X_=k_means_(data)

labels,cluster_labels,Sh_scores,ch_scores,db_indexs=k_means_cluster(X_)

print(labels)
print(cluster_labels)


print("Silhouette_score",Sh_scores)
print("CH Score",ch_scores)
print("DB Index",db_indexs)


####3 A6
k_values_plot(X_)

###   A7

distortion_plot(X_)

