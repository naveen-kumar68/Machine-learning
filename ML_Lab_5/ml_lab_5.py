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

data = pd.read_csv(r"C:\Users\navee\Documents\AQI_Bangalore.csv")
print(data.info(),data.head())

X=data[["PM2.5"]]
y=data["AQI"]


###   A1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
reg = LinearRegression().fit(X_train, y_train) 
y_train_pred = reg.predict(X_train) 
y_test_pred=reg.predict(X_test)
print(y_train_pred)

print("Accuracy:",reg.score(X_test, y_test))

### A2 

rmse=root_mean_squared_error(y_test,y_test_pred)
print("RMSE:",rmse)

mse=mean_squared_error(y_test,y_test_pred)
print("MSE:",mse)

mape=mean_absolute_percentage_error(y_test,y_test_pred)
print("MAPE:",mape)

r_2_score=r2_score(y_test,y_test_pred)
print("R_2_Score:",r_2_score)

##Train Data

print()
print("For Train Data")

rmse=root_mean_squared_error(y_train,y_train_pred)
print("RMSE:",rmse)

mse=mean_squared_error(y_train,y_train_pred)
print("MSE:",mse)

mape=mean_absolute_percentage_error(y_train,y_train_pred)
print("MAPE:",mape)

r_2_score=r2_score(y_train,y_train_pred)
print("R_2_Score:",r_2_score)

print()
print("For all Atrributes")

##   A3

## All Attributes

X=data[["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]]
y=data["AQI"]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
reg = LinearRegression().fit(X_train, y_train) 
y_train_pred = reg.predict(X_train) 
y_test_pred=reg.predict(X_test)
print(y_train_pred)

print("Accuracy:",reg.score(X_test, y_test))

rmse=root_mean_squared_error(y_test,y_test_pred)
print("RMSE:",rmse)

mse=mean_squared_error(y_test,y_test_pred)
print("MSE:",mse)

mape=mean_absolute_percentage_error(y_test,y_test_pred)
print("MAPE:",mape)

r_2_score=r2_score(y_test,y_test_pred)
print("R_2_Score:",r_2_score)

##Train Data
print()
print("For Train Data")

rmse=root_mean_squared_error(y_train,y_train_pred)
print("RMSE:",rmse)

mse=mean_squared_error(y_train,y_train_pred)
print("MSE:",mse)

mape=mean_absolute_percentage_error(y_train,y_train_pred)
print("MAPE:",mape)

r_2_score=r2_score(y_train,y_train_pred)
print("R_2_Score:",r_2_score)


print()
print("K-Means CLustering")

## A4

X_train=data[["PM2.5","PM10","NO","NO2","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"]]

kmeans = KMeans(n_clusters=2, random_state=0, 
n_init="auto").fit(X_train)  
kmeans.labels_ 
kmeans.cluster_centers_ 
print(kmeans.labels_)
print(kmeans.cluster_centers_)

###  A5

kmeans = KMeans(n_clusters=2, random_state=42).fit(X_train) 
print("Silhouette_score",silhouette_score(X_train, kmeans.labels_))
print("CH Score",calinski_harabasz_score(X_train, kmeans.labels_) )
print("DB Index",davies_bouldin_score(X_train, kmeans.labels_))

## A6

sh_score=[]
ch_score=[]
db_index=[]
for k in range(2,12):
   kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train) 
   sh_score.append(silhouette_score(X_train, kmeans.labels_))
   ch_score.append(calinski_harabasz_score(X_train, kmeans.labels_))
   db_index.append(davies_bouldin_score(X_train, kmeans.labels_))

plt.figure(figsize=(10,6))
plt.plot(range(2,12),sh_score,marker="o",linestyle="-",color="blue",label="SH Score")
plt.plot(range(2,12),ch_score,marker="o",linestyle="-",color="green",label="CH Score")
plt.plot(range(2,12),db_index,marker="o",linestyle="-",color="red",label="DB Index")

plt.xlabel("No.of Clusters:(k)")
plt.ylabel("Scores")
plt.show()

### A7

distortions=[]
for k in range(2, 20):  
 kmeans = KMeans(n_clusters=k).fit(X_train) 
 distortions.append(kmeans.inertia_) 
plt.plot(distortions) 
plt.show()