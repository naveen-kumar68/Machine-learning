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

def train_LR(X_train, X_test, y_train, y_test):
 
    reg = LinearRegression().fit(X_train, y_train) 
    y_train_pred = reg.predict(X_train) 
    y_test_pred=reg.predict(X_test)

    accuracy=reg.score(X_test, y_test)

    rmse=root_mean_squared_error(y_test,y_test_pred)
    mse=mean_squared_error(y_test,y_test_pred)
    mape=mean_absolute_percentage_error(y_test,y_test_pred)
    r_2_score=r2_score(y_test,y_test_pred)

    rmse_train=root_mean_squared_error(y_train,y_train_pred)
    mse_train=mean_squared_error(y_train,y_train_pred)
    mape_train=mean_absolute_percentage_error(y_train,y_train_pred)
    r_2_score_train=r2_score(y_train,y_train_pred)

    return accuracy,rmse,mse,mape,r_2_score,rmse_train,mse_train,mape_train,r_2_score_train


