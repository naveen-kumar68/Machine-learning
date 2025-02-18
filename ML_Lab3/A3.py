import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report

data = pd.read_csv(r"C:\Users\navee\Downloads\diabetes.csv")
print(data.info(),data.head())

feature_vec0=data["Glucose"]
feature_vec1=data["BloodPressure"]

print(feature_vec0)
print(feature_vec1)

feature_0=np.array(feature_vec0,dtype=float)
feature_1=np.array(feature_vec1,dtype=float)

for r in range(1,11):
 minkowski_distance=np.sum(np.abs(feature_0-feature_1)**r)**(1/r)
 print("Minkwoski Distance for r:{r}",minkowski_distance)

###### BY importing distance
minkowski_dist=[]
for r in range(1,11):
 minkowski_distance=distance.minkowski(feature_0,feature_1,r)
 print(f"Minkwoski Distance for r={r}={minkowski_distance}")
 minkowski_dist.append(minkowski_distance)

###plot distance

plt.plot(range(1,11),minkowski_dist,marker="o",linestyle="-",color="blue")
plt.xlabel("Feature_0")
plt.ylabel("Feature_1")
plt.show()